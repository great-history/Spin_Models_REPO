# ==========================================
# numba kernels for mc
# ==========================================
from numba import njit
import numpy as np
HBAR_MEV_PS = 0.6582119569

@njit(cache=True)
def _local_energy_update_numba(x, y, b, # location of spin_i
                                           spin_i_old, spin_i_new, spin_config,
                                           Nx, Ny, num_bonds_list, 
                                           bond_vectors, bond_matrices, 
                                           A_ani, zeeman, B_biquad=0.0):
    E_old = 0.0
    E_new = 0.0

    n_bonds = num_bonds_list[b]
    for i in range(n_bonds):
        neighbor_b = bond_vectors[b, i, 0]
        dx = bond_vectors[b, i, 1]
        dy = bond_vectors[b, i, 2]

        # periodic boundary conditions
        nx = (x + dx) % Nx
        ny = (y + dy) % Ny

        S_j = spin_config[nx, ny, neighbor_b]
        Jmat = bond_matrices[b, i]

        # 1. exchange matrix : spin_i . (J . S_j) expansion
        J_Sj_x = Jmat[0,0]*S_j[0] + Jmat[0,1]*S_j[1] + Jmat[0,2]*S_j[2]
        J_Sj_y = Jmat[1,0]*S_j[0] + Jmat[1,1]*S_j[1] + Jmat[1,2]*S_j[2]
        J_Sj_z = Jmat[2,0]*S_j[0] + Jmat[2,1]*S_j[1] + Jmat[2,2]*S_j[2]

        E_old += spin_i_old[0]*J_Sj_x + spin_i_old[1]*J_Sj_y + spin_i_old[2]*J_Sj_z
        E_new += spin_i_new[0]*J_Sj_x + spin_i_new[1]*J_Sj_y + spin_i_new[2]*J_Sj_z

        # 4. biquadratic exchange ( only 6 nearest-neighbors )
        if B_biquad != 0.0 and i < 6:
            E_old += B_biquad * (spin_i_old[0]*S_j[0] + spin_i_old[1]*S_j[1] + spin_i_old[2]*S_j[2])**2
            E_new += B_biquad * (spin_i_new[0]*S_j[0] + spin_i_new[1]*S_j[1] + spin_i_new[2]*S_j[2])**2
    
    # 2. single-ion anisotropy (A_ani < 0 corresponds to easy axis along Z)
    E_old += A_ani[b] * (spin_i_old[2] * spin_i_old[2])
    E_new += A_ani[b] * (spin_i_new[2] * spin_i_new[2])

    # 3. Zeeman energy (already converted to meV externally)
    E_old -= (zeeman[0]*spin_i_old[0] + zeeman[1]*spin_i_old[1] + zeeman[2]*spin_i_old[2])
    E_new -= (zeeman[0]*spin_i_new[0] + zeeman[1]*spin_i_new[1] + zeeman[2]*spin_i_new[2])

    return E_new - E_old

@njit(cache=True)
def _mc_sweep_numba(spin_config, T_meV, 
                    Nx, Ny, Ns, 
                    proposal_angle, reset_probability, rnd,
                    num_bonds_list, bond_vectors, bond_matrices,
                    A_ani, zeeman, B_biquad=0.0):
    """
        perform a single Monte Carlo sweep at the given temperature T_meV.
        rnd : pre-generated random numbers for the sweep, shape (N_total, 7)
        why 7 ?  [ x, y, b, proposal type, angle/height, azimuth, Metropolis acceptance ]
        there are two types of proposals for updating a spin : Reset or Small-Angle-Rotation.
    """
    N_total = Nx * Ny * Ns
    accepted = 0

    for i in range(N_total):
        # choose a random spin to update
        x = int(rnd[i, 0] * Nx)
        y = int(rnd[i, 1] * Ny)
        b = int(rnd[i, 2] * Ns)

        S_old = spin_config[x, y, b]

        # choose a proposal type : Reset or Small-Angle-Rotation
        if rnd[i, 3] < reset_probability:
            z = rnd[i, 4] * 2.0 - 1.0
            phi = rnd[i, 5] * 2.0 * np.pi
            sin_theta = np.sqrt(1.0 - z*z)
            S_new = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), z])
        else:
            # rotate around S_old 
            if abs(S_old[2]) < 0.9:
                e1x = -S_old[1]
                e1y = S_old[0]
                e1z = 0.0
            else:
                e1x = 0.0
                e1y = -S_old[2]
                e1z = S_old[1]
            e1norm = np.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
            e1x /= e1norm
            e1y /= e1norm
            e1z /= e1norm

            # e2 = S_old x e1
            e2x = S_old[1]*e1z - S_old[2]*e1y
            e2y = S_old[2]*e1x - S_old[0]*e1z
            e2z = S_old[0]*e1y - S_old[1]*e1x

            alpha = (rnd[i, 4] * 2.0 - 1.0) * proposal_angle
            phi = rnd[i, 5] * 2.0 * np.pi
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            tx = np.cos(phi)*e1x + np.sin(phi)*e2x
            ty = np.cos(phi)*e1y + np.sin(phi)*e2y
            tz = np.cos(phi)*e1z + np.sin(phi)*e2z
            S_new = np.array([ca*S_old[0] + sa*tx,
                              ca*S_old[1] + sa*ty,
                              ca*S_old[2] + sa*tz])

        # energy difference
        dE = _local_energy_update_numba(x, y, b, S_old, S_new, spin_config, Nx, Ny, 
                                                    num_bonds_list, bond_vectors, bond_matrices, 
                                                    A_ani, zeeman, B_biquad)

        # Metropolis 判据
        if dE <= 0.0 or rnd[i, 6] < np.exp(-dE / max(T_meV, 1e-12)):
            spin_config[x, y, b, 0] = S_new[0]
            spin_config[x, y, b, 1] = S_new[1]
            spin_config[x, y, b, 2] = S_new[2]
            accepted += 1

    return accepted / N_total

@njit(cache=True)
def _total_energy_numba(spin_config, Nx, Ny, 
                        num_bonds_list, bond_vectors, bond_matrices, 
                        A_ani, zeeman, B_biquad=0.0):
    """总能量（meV）。每条双向存储的键只计一次（0.5 因子）。"""
    Ns = spin_config.shape[2]
    E = 0.0
    for x in range(Nx):
        for y in range(Ny):
            for b in range(Ns):
                S_i = spin_config[x, y, b]
                n_bonds = num_bonds_list[b]
                for j in range(n_bonds):
                    neighbor_b = bond_vectors[b, j, 0]
                    dx = bond_vectors[b, j, 1]
                    dy = bond_vectors[b, j, 2]
                    S_j = spin_config[(x + dx) % Nx, (y + dy) % Ny, neighbor_b]
                    J = bond_matrices[b, j]
                    J_Sj_x = J[0,0]*S_j[0] + J[0,1]*S_j[1] + J[0,2]*S_j[2]
                    J_Sj_y = J[1,0]*S_j[0] + J[1,1]*S_j[1] + J[1,2]*S_j[2]
                    J_Sj_z = J[2,0]*S_j[0] + J[2,1]*S_j[1] + J[2,2]*S_j[2]
                    # on-site 键（b1==b2 且 δ=0）无反条目：不计 0.5（否则能量减半）
                    if neighbor_b == b and dx == 0 and dy == 0:
                        E += S_i[0]*J_Sj_x + S_i[1]*J_Sj_y + S_i[2]*J_Sj_z
                    else:
                        E += 0.5 * (S_i[0]*J_Sj_x + S_i[1]*J_Sj_y + S_i[2]*J_Sj_z)
                    
                    # bi-quadratic exchange
                    if B_biquad != 0.0 and j < 6:
                        E += 0.5 * B_biquad * (S_i[0] * S_j[0] + S_i[1] * S_j[1] + S_i[2] * S_j[2])**2
                    
                E += A_ani[b] * S_i[2] * S_i[2]
                E -= (zeeman[0]*S_i[0] + zeeman[1]*S_i[1] + zeeman[2]*S_i[2])
    return E

# Compatibility names share the same kernels.
_local_energy_update_with_biquad_numba = _local_energy_update_numba
_mc_sweep_with_biquad_numba = _mc_sweep_numba
_total_energy_with_biquad_numba = _total_energy_numba

@njit(cache=True)
def _topological_charge_triangular_lattice(spin_config, Nx, Ny):
    """
        Return total topological charge Q and its square Q**2.
        Assumes unit spins and periodic boundaries on a triangular lattice.
    """

    def solid_angle(a, b, c):
        numerator = np.dot(a, np.cross(b, c))
        denominator = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
        return 2.0 * np.arctan2(numerator, denominator)

    q = 0.0
    for x in range(Nx):
        xp = (x + 1) % Nx
        for y in range(Ny):
            yp = (y + 1) % Ny
            s00 = spin_config[x, y, 0]
            s10 = spin_config[xp, y, 0]
            s01 = spin_config[x, yp, 0]
            s11 = spin_config[xp, yp, 0]

            # triangular unit cell has two plaquettes
            angle1 = solid_angle(s00, s10, s01)
            angle2 = solid_angle(s10, s11, s01)
            q += angle1
            q += angle2

    Q = q / (4.0 * np.pi)
    return Q, Q**2

@njit(cache=True)
def _topological_charge_rectangular_lattice(spin_config, Nx, Ny):
    """
        Return total topological charge Q and its square Q**2.
        Assumes unit spins and periodic boundaries on a rectangular lattice.
    """

    def solid_angle(a, b, c):
        numerator = np.dot(a, np.cross(b, c))
        denominator = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
        return 2.0 * np.arctan2(numerator, denominator)

    q = 0.0
    for x in range(Nx):
        xp = (x + 1) % Nx
        for y in range(Ny):
            ym = (y - 1) % Ny
            yp = (y + 1) % Ny
            s000 = spin_config[x, y, 0]
            s001 = spin_config[x, y, 1]

            s010 = spin_config[x, yp, 0]
            s0m11 = spin_config[x, ym, 1]

            s110 = spin_config[xp, yp, 0]
            s100 = spin_config[xp, y, 0]

            # Four distinct nearest-neighbor triangles, all counterclockwise.
            angle1 = solid_angle(s000, s001, s010)
            angle2 = solid_angle(s000, s0m11, s001)
            angle3 = solid_angle(s0m11, s100, s001)
            angle4 = solid_angle(s001, s100, s110)

            q += angle1
            q += angle2
            q += angle3
            q += angle4

    Q = q / (4.0 * np.pi)
    return Q, Q**2

@njit(cache=True)
def _electric_polarization_numba(spin_config, Nx, Ny, Mtensor_bond_vectors, Mtensor_matrices):
    """Return polarization per spin, counting each directed bond with weight one half."""
    Polar = np.zeros(3, dtype=np.float64)
    Ns = spin_config.shape[2]
    n_bonds = Mtensor_bond_vectors.shape[1]
    for x in range(Nx):
        for y in range(Ny):
            for b in range(Ns):
                for i in range(n_bonds):
                    neighbor_b, dx, dy = Mtensor_bond_vectors[b, i]
                    S_i = spin_config[x, y, b]
                    S_j = spin_config[(x + dx) % Nx, (y + dy) % Ny, neighbor_b]
                    Polar += np.dot(Mtensor_matrices[b, i], np.cross(S_i, S_j))
    return Polar / (2 * Nx * Ny * Ns)

@njit(cache=True)
def _get_torques_numba(spins, alpha, num_bonds, bond_vectors,
                       bond_matrices, A_ani, zeeman, B_biquad):
    """LLG derivative using only the current site's energy gradient.

    Non-onsite bonds must occur in reciprocal pairs with J_ji = J_ij.T.
    The first six bonds per site are nearest neighbors (biquadratic term).
    Spins are read-only: each site's torque uses the same input configuration.
    """
    Nx, Ny, Ns, _ = spins.shape
    result = np.empty_like(spins)
    factor = 1.0 / (1.0 + alpha*alpha)
    for x in range(Nx):
        for y in range(Ny):
            for b in range(Ns):
                S_i = spins[x, y, b]
                gradient = np.zeros(3, dtype=spins.dtype)
                for j in range(num_bonds[b]):
                    nb, dx, dy = bond_vectors[b, j]
                    nx, ny = (x + dx) % Nx, (y + dy) % Ny
                    S_j = spins[nx, ny, nb]
                    J = bond_matrices[b, j]
                    for u in range(3):
                        for v in range(3):
                            gradient[u] += J[u, v] * S_j[v]
                            if nb == b and dx == 0 and dy == 0:
                                gradient[u] += J[v, u] * S_i[v]

                    if B_biquad != 0.0 and j < 6:
                        dot = S_i[0]*S_j[0] + S_i[1]*S_j[1] + S_i[2]*S_j[2]
                        for u in range(3):
                            gradient[u] += 2.0 * B_biquad * dot * S_j[u]

                gradient[2] += 2.0 * A_ani[b] * S_i[2]
                for u in range(3):
                    gradient[u] -= zeeman[u]

                # All contributions for this site are complete: compute
                # its torque immediately, without a second lattice loop.
                sx, sy, sz = S_i
                hx, hy, hz = -gradient / HBAR_MEV_PS
                px, py, pz = hy*sz-hz*sy, hz*sx-hx*sz, hx*sy-hy*sx
                result[x, y, b, 0] = (px + alpha*(sy*pz-sz*py)) * factor
                result[x, y, b, 1] = (py + alpha*(sz*px-sx*pz)) * factor
                result[x, y, b, 2] = (pz + alpha*(sx*py-sy*px)) * factor
    return result
