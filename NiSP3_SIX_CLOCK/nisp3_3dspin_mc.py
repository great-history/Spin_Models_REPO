# mc.py
import numpy as np
from nisp3_3dxyz_numba_kernel import (
    _mc_sweep_3dxyz_numba, _total_energy_numba, _neel_vectors_numba
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
# 2D Lattice Class
# ==========================================
class Lattice2D:
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


# ==========================================
# Class for Spin Model : 3D XYZ spins
# ==========================================
class SpinModel3DXYZ:
    def __init__(self, Ns, *, Dani:float=0.0, K2:float=0.0, K6:float=0.0):
        if Ns <= 0:
            raise ValueError("Ns must be positive.")
        self.Ns = Ns
        self.Dani = Dani # out-of-plane anisotropy
        self.K2 = K2
        self.K6 = K6
        self.bonds = [[] for _ in range(Ns)]

    def add_bond_matrix(self, s1, s2, offset, J_matrix, *, CI):
        # CI = True : system has inversion symmetry
        dx, dy = offset
        J = np.array(J_matrix, dtype=np.float64)
        self.bonds[s1].append((s2, dx, dy, J))
        
        if CI and (s1 != s2 or dx != 0 or dy != 0):
            self.bonds[s2].append((s1, -dx, -dy, J.T))

    def build_numba_arrays(self):
        self.num_bonds = np.array([len(b) for b in self.bonds], dtype=np.int32)
        max_bonds = max(self.num_bonds)
        if max_bonds == 0:
            raise ValueError("add bonds before building numba arrays !")
        
        self.bond_vectors = np.zeros((self.Ns, max_bonds, 3), dtype=np.int32)
        self.bond_matrices = np.zeros((self.Ns, max_bonds, 3, 3), dtype=np.float64)

        for si in range(self.Ns):
            for j, (sk, dx, dy, Jmat) in enumerate(self.bonds[si]):
                self.bond_vectors[si, j, 0] = sk
                self.bond_vectors[si, j, 1] = dx
                self.bond_vectors[si, j, 2] = dy
                self.bond_matrices[si, j] = Jmat


# ==========================================
# Class for Monte Carlo simulation : 3D XYZ spins
# ==========================================
# ==========================================
# Class for Monte Carlo simulation : 3D XYZ spins
# ==========================================
class MonteCarlo3DXYZ:
    def __init__(self, lattice:Lattice2D, spin_model:SpinModel3DXYZ, seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.lat = lattice
        self.ham = spin_model
        self.ham.build_numba_arrays()

        self.spin_config = self._random_spins() # random initial spin configuration
        
        # temperatures & magnetic field
        self.current_temperature_in_K = None
        self.current_temperature_in_meV = None

        # adaptive proposal for MC
        self._auto_angle = True
        self._proposal_angle = None
        self._sweeps_since_adjust = 0

        # neel vector is the order parameter for the zigzag patterns
        self.set_neel_eta()

    def _random_spins(self):
        # generate a random spin configuration for 3D spins on a 2D lattice
        phi = self._rng.uniform(0, 2 * np.pi, size=(self.lat.Nx, self.lat.Ny, self.lat.Ns))
        costheta = self._rng.uniform(-1, 1, size=(self.lat.Nx, self.lat.Ny, self.lat.Ns))
        sintheta = np.sqrt(1 - costheta**2)

        # 使用 np.ascontiguousarray 确保内存连续，最大化 Numba 速度
        # spins shape: (Nx, Ny, Ns, 3) for 3D XYZ spins
        spins = np.stack((sintheta * np.cos(phi), sintheta * np.sin(phi), costheta), axis=-1)

        return np.ascontiguousarray(spins, dtype=np.float64)
    
    def set_proposal_angle(self, proposal_angle: float | None = None, 
                           T_in_meV: float | None = None):
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
        
        acc = _mc_sweep_3dxyz_numba(
            self.spin_config, 
            T_meV, 
            self.lat.Nx, self.lat.Ny, self.lat.Ns,
            self.ham.num_bonds, self.ham.bond_vectors, self.ham.bond_matrices,
            rnd,
            float(reset_probability), float(self._proposal_angle),
            Dani=self.ham.Dani, k2=self.ham.K2, k6=self.ham.K6,
        )
        
        if self._auto_angle:
            self._adapt_angle(acc)
        
        return acc
    
    def get_magnetization(self):
        # spins shape: (Nx, Ny, Ns, 3) for 3D XYZ spins, average over the first three dimensions to get the 3D magnetization vector
        M_vec = np.sum(self.spin_config, axis=(0,1,2)) / self.lat.N_total
        return np.linalg.norm(M_vec), M_vec
    
    def set_neel_eta(self):
        """Set the three zigzag templates for the rectangular four-site basis."""
        nx, ny, ns = self.lat.Nx, self.lat.Ny, self.lat.Ns
        if ns != 4:
            raise ValueError("This method require the rectangular four-site basis.")
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError("This method requires even Nx and Ny for the zigzag pattern because of the periodic boundary conditions.")
        
        # three zigzag patterns for the four-site basis
        signs = np.array([
                            [1,  1, -1, -1],
                            [1, -1, -1, -1],
                            [1,  1,  1, -1],
                        ], dtype=np.float64)
        
        # Initialize the Neel eta array with the zigzag patterns
        self._neel_eta = np.zeros((3, nx, ny, ns))
        
        for ix in range(nx):
            for iy in range(ny):
                parity = 1.0 if (ix + iy) % 2 == 0 else -1.0
                for b in range(ns):
                    self._neel_eta[0, ix, iy, b] = signs[0, b]
                    self._neel_eta[1, ix, iy, b] = parity * signs[1, b]
                    self._neel_eta[2, ix, iy, b] = parity * signs[2, b]

    def get_neel_vectors(self):
        return _neel_vectors_numba(self.spin_config, self._neel_eta)
    
    def total_energy(self):
        return _total_energy_numba(
            self.spin_config,
            self.lat.Nx, self.lat.Ny, self.lat.Ns,
            self.ham.num_bonds, self.ham.bond_vectors, self.ham.bond_matrices,
            Dani=self.ham.Dani, k2=self.ham.K2, k6=self.ham.K6,
        )