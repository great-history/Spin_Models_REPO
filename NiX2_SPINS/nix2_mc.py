# mc.py
import numpy as np
from nix2_numba_kernels import (
    _mc_sweep_numba, _total_energy_numba,
    _topological_charge_triangular_lattice,
    _topological_charge_rectangular_lattice,
    _electric_polarization_numba,
)

# ==========================================
# conversion
# ==========================================
KB_MEV_PER_K = 0.08617333262  # 玻尔兹曼常数 (meV/K)
MU_S_MEV_PER_T = 0.1157676    # 有效磁矩 (meV/T)，假设 g=2, S=1 (可根据具体材料调整)
HBAR_MEV_PS = 0.6582119569    # meV·ps

# ==========================================
# adaptive
# ==========================================
_ACCEPT_TARGET_LO = 0.30
_ACCEPT_TARGET_HI = 0.60
_ANGLE_ADJUST = 1.05
_ANGLE_MIN = 0.01
_ANGLE_MAX = 0.50
_ADAPT_EVERY = 10          # 每 N 个 sweep 调整一次提案角

# ==========================================
# classes
# ==========================================
class LatticeNiX2:
    def __init__(self, a_vecs, d_vecs, Nx, Ny):
        self.a_vecs = np.array(a_vecs, dtype=np.float64)
        self.d_vecs = np.array(d_vecs, dtype=np.float64)
        self.Ns = len(d_vecs)
        self.Nx = Nx
        self.Ny = Ny
        self.N_total = Nx * Ny * self.Ns

    def get_cartesian_coords(self):
        coords = np.zeros((self.Nx, self.Ny, self.Ns, 2))
        for x in range(self.Nx):
            for y in range(self.Ny):
                for b in range(self.Ns):
                    f1, f2 = self.d_vecs[b]
                    r = (x + f1) * self.a_vecs[0] + (y + f2) * self.a_vecs[1]
                    coords[x, y, b] = r
        return coords

class SpinModelNiX2:
    def __init__(self, Ns, A_ani, B_field=np.array([0.0, 0.0, 0.0]), B_biquad=0.0):
        if Ns <= 0:
            raise ValueError("Ns must be positive.")
        self.Ns = Ns
            
        self.A_ani = np.ones(Ns, dtype=np.float64) * A_ani if np.ndim(A_ani) == 0 \
                                        else np.array(A_ani, dtype=np.float64)

        self.zeeman = np.array(B_field, dtype=np.float64) * MU_S_MEV_PER_T

        self.B_biquad = B_biquad
        if B_biquad == 0.0:
            self.with_biquad = False
        else:
            self.with_biquad = True
        
        self.bonds = [[] for _ in range(Ns)]
        self.Mtensors = [[] for _ in range(Ns)]

    def update_zeeman(self, B_field=np.array([0.0, 0.0, 0.0])):
        self.zeeman = np.array(B_field, dtype=np.float64) * MU_S_MEV_PER_T

    def add_bond_matrix(self, s1, s2, offset, J_matrix, CI=True):
        # CI = True : system has inversion symmetry
        dx, dy = offset
        J = np.array(J_matrix, dtype=np.float64)
        self.bonds[s1].append((s2, dx, dy, J))
        
        # NiX2 has inversion symmetry, automatically add the reverse bond if necessary
        if CI and (s1 != s2 or dx != 0 or dy != 0):
            self.bonds[s2].append((s1, -dx, -dy, J.T))

    # ========================================================
    # M tensors for computing electric polarization
    # ========================================================
    def add_M_tensor(self, s1, s2, offset, M_tensor, CI=True):
        dx, dy = offset
        M = np.array(M_tensor, dtype=np.float64)
        self.Mtensors[s1].append((s2, dx, dy, M))

        # NiX2 has inversion symmetry, automatically add the reverse tensor if necessary
        if CI and (s1 != s2 or dx != 0 or dy != 0):
            self.Mtensors[s2].append((s1, -dx, -dy, -M)) # caution : add a negative sign

    def build_numba_arrays(self):
        self.num_bonds = np.array([len(b) for b in self.bonds], dtype=np.int32)
        max_bonds = max(self.num_bonds)
        if max_bonds == 0:
            raise ValueError("add bonds before building numba arrays !")

        # =======================================
        # self.bond_vectors[si,j,:] = (sk, dx, dy)
        #       the j-th bond of site si is connected to site sk with offset vector (dx, dy)
        # self.bond_matrices[si,j,:,:] = J
        #       the exchange matrix for the j-th bond of site si is 3x3 matrix J
        # =======================================
        self.bond_vectors = np.zeros((self.Ns, max_bonds, 3), dtype=np.int32)
        self.bond_matrices = np.zeros((self.Ns, max_bonds, 3, 3), dtype=np.float64)

        for si in range(self.Ns):
            for j, (sk, dx, dy, Jmat) in enumerate(self.bonds[si]):
                self.bond_vectors[si, j, 0] = sk
                self.bond_vectors[si, j, 1] = dx
                self.bond_vectors[si, j, 2] = dy
                self.bond_matrices[si, j] = Jmat

        if any(len(mt) > 0 for mt in self.Mtensors):
            print("Building M tensor arrays for electric polarization.")
            self.build_Mtensor_arrays()
    
    # ========================================================
    # Build arrays for M tensors for electric polarization
    # ========================================================
    def build_Mtensor_arrays(self):
        self.Mtensor_bond_vectors = np.zeros((self.Ns, 6, 3), dtype=np.int32)
        self.Mtensor_matrices = np.zeros((self.Ns, 6, 3, 3), dtype=np.float64)

        for si in range(self.Ns):
            for j, (sk, dx, dy, M) in enumerate(self.Mtensors[si]):
                self.Mtensor_bond_vectors[si, j, 0] = sk
                self.Mtensor_bond_vectors[si, j, 1] = dx
                self.Mtensor_bond_vectors[si, j, 2] = dy
                self.Mtensor_matrices[si, j] = M

class MonteCarloNiX2:
    def __init__(self, lattice:LatticeNiX2, spin_model:SpinModelNiX2, seed=None):
        """
            if seed is None, the random sequence is fixed ( for reproducibility ), otherwise it is determined by the given seed.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.lat = lattice
        self.ham = spin_model
        self.ham.build_numba_arrays()
        self.spin_config = self._random_spins() # random initial spin configuration
        
        # temperatures & magnetic field
        self.current_temperature_in_K = None
        self.current_temperature_in_meV = None
        self.current_B_field = None

        # adaptive proposal for MC
        self._auto_angle = True
        self._proposal_angle = None
        self._sweeps_since_adjust = 0

    def _random_spins(self):
        # generate a random spin configuration for 3D spins on a 2D lattice
        phi = self._rng.uniform(0, 2 * np.pi, size=(self.lat.Nx, self.lat.Ny, self.lat.Ns))
        costheta = self._rng.uniform(-1, 1, size=(self.lat.Nx, self.lat.Ny, self.lat.Ns))
        sintheta = np.sqrt(1 - costheta**2)

        # 使用 np.ascontiguousarray 确保内存连续，最大化 Numba 速度
        spins = np.stack((sintheta * np.cos(phi), sintheta * np.sin(phi), costheta), axis=-1).astype(np.float64)

        return np.ascontiguousarray(spins)
    
    def set_proposal_angle(self, proposal_angle: float | None = None, T_in_meV: float | None = None):
        if proposal_angle is None:
            self._auto_angle = True
            self._proposal_angle = min(_ANGLE_MAX, max(0.02, 0.20 * np.sqrt(T_in_meV)))
        else:
            self._auto_angle = False
            self._proposal_angle = proposal_angle

    def _adapt_angle(self, acceptance):
        """
            按接受率反馈调节提案角，目标区间 [0.30, 0.60]。
        """
        self._sweeps_since_adjust += 1
        if self._sweeps_since_adjust < _ADAPT_EVERY:
            return
        self._sweeps_since_adjust = 0
        if self._proposal_angle is None:
            self._proposal_angle = 0.2
        if acceptance > _ACCEPT_TARGET_HI:
            self._proposal_angle = min(_ANGLE_MAX, self._proposal_angle * _ANGLE_ADJUST)
        elif acceptance < _ACCEPT_TARGET_LO:
            self._proposal_angle = max(_ANGLE_MIN, self._proposal_angle / _ANGLE_ADJUST)

    def mc_sweep(self, T_meV, reset_probability=0.02):
        N_total = self.lat.N_total
        rnd = self._rng.random((N_total, 7))

        acc = _mc_sweep_numba(
            self.spin_config, T_meV, 
            self.lat.Nx, self.lat.Ny, self.lat.Ns,
            float(self._proposal_angle), float(reset_probability), rnd,
            self.ham.num_bonds, self.ham.bond_vectors, self.ham.bond_matrices,
            self.ham.A_ani, self.ham.zeeman, self.ham.B_biquad,
        )
        
        if self._auto_angle:
            self._adapt_angle(acc)
        
        return acc

    def get_magnetization(self):
        M_vec = np.sum(self.spin_config, axis=(0,1,2)) / self.lat.N_total
        return np.linalg.norm(M_vec), M_vec

    def electric_polarization(self):
        return _electric_polarization_numba(
            self.spin_config, self.lat.Nx, self.lat.Ny, 
            self.ham.Mtensor_bond_vectors, self.ham.Mtensor_matrices
        )

    def total_energy(self):
        return _total_energy_numba(
            self.spin_config, self.lat.Nx, self.lat.Ny,
            self.ham.num_bonds, self.ham.bond_vectors, self.ham.bond_matrices,
            self.ham.A_ani, self.ham.zeeman, self.ham.B_biquad,
        )

    def spin_structure_factor(self, component_resolved=False):
        """ 
            compute spin structure factor by FFT :
            S(q) = 1/N Σ_{α=x,y,z} |(Σ_b Σ_r S_{b,α}(r) e^{-iq·(r+τ_b)}|²。
        """
        # component_resolved=True returns (Nx, Ny, 3) intensities.
        Nx, Ny, Ns, _ = self.spin_config.shape
        h = np.fft.fftfreq(Nx)
        k = np.fft.fftfreq(Ny)
        F_total = np.zeros((Nx, Ny, 3), dtype=np.complex128)
        for b in range(Ns):
            f1, f2 = self.lat.d_vecs[b]
            # Fractional basis positions require the same signed frequencies
            # as the returned q grid, including negative-frequency FFT bins.
            phase = np.exp(-2j*np.pi*(f1*h[:, None] + f2*k[None, :]))
            for comp in range(3): # x,y,z
                F_total[:, :, comp] += np.fft.fft2(self.spin_config[:, :, b, comp]) * phase
        # Optional last axis contains Sx, Sy, Sz; default remains their sum.
        S_components = (F_total * np.conj(F_total)).real / (Nx * Ny * Ns)
        S = S_components if component_resolved else S_components.sum(axis=2)
        q1 = h * Nx   # Signed FFT indices, in native (unshifted) order.
        q2 = k * Ny
        return q1, q2, S

    def topological_charge(self):
        """Return total topological charge Q and its square Q**2.

            Compute susceptibility from thermal averages: (<Q**2> - <Q>**2) / T.
            T is in energy units (k_B times temperature).
            Spins must be unit vectors on a periodic triangular or rectangular lattice.
        """
        if self.lat.Ns == 1:
            return _topological_charge_triangular_lattice(
                self.spin_config, self.lat.Nx, self.lat.Ny,
            )
        else:
            return _topological_charge_rectangular_lattice(
                self.spin_config, self.lat.Nx, self.lat.Ny,
            )

    @staticmethod
    def _validate_sampling(T, equip_steps, calc_steps, sample_interval):
        if T <= 0.0:
            raise ValueError("统计温度必须大于零")
        if equip_steps < 0 or calc_steps <= 0 or sample_interval <= 0:
            raise ValueError("equip_steps >= 0、calc_steps > 0、sample_interval > 0")

    # Compatibility name for existing notebooks.
    mc_sweep_with_biquad = mc_sweep
