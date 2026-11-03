from __future__ import annotations

import numpy as np
from numba import njit

# unit conversion
KB_MEV_PER_K = 0.08617333262
MU_S_MEV_PER_T = 0.1157676  # g=2 and S=1

# ==========================================
# energy
# ==========================================
# @njit(cache=True)
# def _sixfold_energy_numba(spin, k6):
#     x2 = spin[0] * spin[0]
#     y2 = spin[1] * spin[1]
#     return (-k6) * (x2*x2*x2 - 15*x2*x2*y2 + 15*x2*y2*y2 - y2*y2*y2)

@njit(cache=True)
def _local_energy_numba(
    x, y, b, # coordinates and sublattice index of the spin being updated
    spin_i, # proposed new spin vector, call it S_i
    spins, # spin configuration
    nx, ny, num_bonds, bond_vectors, bond_matrices,
    *, Dani=0.0, k2=0.0, k6=0.0,
):
    """
        Energy terms changed by replacing one spin.
        The neighbor list contains both directions of every physical bond.  
        A local update touches each incident bond once, so no factor 1/2 is used here.
    """
    energy = 0.0

    for bond_index in range(num_bonds[b]):
        neighbor_b = bond_vectors[b, bond_index, 0]
        dx = bond_vectors[b, bond_index, 1]
        dy = bond_vectors[b, bond_index, 2]
        spin_j = spins[(x + dx) % nx, (y + dy) % ny, neighbor_b] # periodic boundary conditions
        exchange = bond_matrices[b, bond_index]

        exchange_spin_j_x = (
            exchange[0, 0] * spin_j[0]
            + exchange[0, 1] * spin_j[1]
            + exchange[0, 2] * spin_j[2]
        )
        exchange_spin_j_y = (
            exchange[1, 0] * spin_j[0]
            + exchange[1, 1] * spin_j[1]
            + exchange[1, 2] * spin_j[2]
        )
        exchange_spin_j_z = (
            exchange[2, 0] * spin_j[0]
            + exchange[2, 1] * spin_j[1]
            + exchange[2, 2] * spin_j[2]
        )

        energy += (
            spin_i[0] * exchange_spin_j_x
            + spin_i[1] * exchange_spin_j_y
            + spin_i[2] * exchange_spin_j_z
        )
    
    # anisotropy : out-of-plane (Dani) and in-plane (k2, k6)
    x2 = spin_i[0] * spin_i[0]
    y2 = spin_i[1] * spin_i[1]
    z2 = spin_i[2] * spin_i[2]

    if Dani != 0.0:
        energy -= Dani * z2

    if k2 != 0.0:
        # energy += (-k2) * (spin_i[0] * spin_i[0] - spin_i[1] * spin_i[1])
        energy -= k2 * x2
        
    if k6 != 0.0:
        energy -= k6 * (x2*x2*x2 - 15*x2*x2*y2 + 15*x2*y2*y2 - y2*y2*y2)
    
    
    return energy

@njit(cache=True)
def _total_energy_numba(
    spins,
    nx, ny, ns, num_bonds, bond_vectors, bond_matrices,
    *, Dani=0.0, k2=0.0, k6=0.0,
):
    """ Return total energy """
    # ns = spins.shape[2]
    energy = 0.0

    for x in range(nx):
        for y in range(ny):
            for b in range(ns):
                spin_i = spins[x, y, b]

                # exchange energy part
                for bond_index in range(num_bonds[b]):
                    # get S_j
                    neighbor_b = bond_vectors[b, bond_index, 0]
                    dx = bond_vectors[b, bond_index, 1]
                    dy = bond_vectors[b, bond_index, 2]
                    spin_j = spins[(x + dx) % nx, (y + dy) % ny, neighbor_b]

                    exchange = bond_matrices[b, bond_index]
                    bilinear = 0.0
                    for row in range(3):
                        for col in range(3):
                            bilinear += (
                                spin_i[row] * exchange[row, col] * spin_j[col]
                            )
                    energy += 0.5 * bilinear

                # anisotropy : out-of-plane (Dani) and in-plane (k2, k6)
                x2 = spin_i[0] * spin_i[0]
                y2 = spin_i[1] * spin_i[1]
                z2 = spin_i[2] * spin_i[2]

                if Dani != 0.0:
                    energy -= Dani * z2

                if k2 != 0.0:
                    energy -= k2 * x2

                if k6 != 0.0:
                    energy -= k6 * (x2*x2*x2 - 15*x2*x2*y2 + 15*x2*y2*y2 - y2*y2*y2)

    return energy

# ==========================================
# neel vector
# ==========================================
@njit(cache=True)
def _neel_vectors_numba(spins, neel_eta):
    """ spins: (Nx, Ny, Ns, 2) | neel_eta: (3, Nx, Ny, Ns) """
    nx, ny, nb, nc = spins.shape
    num_components = neel_eta.shape[0]
    vectors = np.zeros((num_components, nc)) # (3, 2) 3 types of Neel vectors & 2D spins

    for a in range(num_components):
        for ix in range(nx):
            for iy in range(ny):
                for b in range(nb):
                    weight = neel_eta[a, ix, iy, b]
                    for c in range(nc):
                        vectors[a, c] += weight * spins[ix, iy, b, c]
    
    for a in range(num_components):
        for c in range(nc):
            vectors[a, c] /= nx * ny * nb
    return vectors

################################################################
# mc sweep
################################################################
@njit(cache=True)
def _mc_sweep_3dxyz_numba(
    spins, # spin configuration
    temperature_meV, # temperature in meV
    nx, ny, ns,
    num_bonds, bond_vectors, bond_matrices,
    random_values,
    reset_probability,
    proposal_angle,*,
    Dani=0.0, k2=0.0, k6=0.0,
):
    """Perform one sweep; spin_mode 0 is Heisenberg and 1 is XY."""
    num_sample = nx * ny * ns
    accepted = 0

    for proposal in range(num_sample): 
        # proposal index is not the spin index, but the number of proposals is equal to the number of spins
        x = int(random_values[proposal, 0] * nx)
        y = int(random_values[proposal, 1] * ny)
        b = int(random_values[proposal, 2] * ns)
        spin_old = spins[x, y, b]
        
        # choose a proposal type : Reset or Small-Angle-Rotation
        if random_values[proposal, 3] < reset_probability:
            z = random_values[proposal, 4] * 2.0 - 1.0
            phi = random_values[proposal, 5] * 2.0 * np.pi
            sin_theta = np.sqrt(1.0 - z*z)
            spin_new = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), z])
        else:
            # rotate around S_old 
            if abs(spin_old[2]) < 0.9:
                e1x = -spin_old[1]
                e1y = spin_old[0]
                e1z = 0.0
            else:
                e1x = 0.0
                e1y = -spin_old[2]
                e1z = spin_old[1]
            
            e1norm = np.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
            e1x /= e1norm
            e1y /= e1norm
            e1z /= e1norm

            # e2 = S_old x e1
            e2x = spin_old[1]*e1z - spin_old[2]*e1y
            e2y = spin_old[2]*e1x - spin_old[0]*e1z
            e2z = spin_old[0]*e1y - spin_old[1]*e1x

            alpha = (random_values[proposal, 4] * 2.0 - 1.0) * proposal_angle
            phi = random_values[proposal, 5] * 2.0 * np.pi
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            tx = np.cos(phi)*e1x + np.sin(phi)*e2x
            ty = np.cos(phi)*e1y + np.sin(phi)*e2y
            tz = np.cos(phi)*e1z + np.sin(phi)*e2z
            spin_new = np.array([ca*spin_old[0] + sa*tx,
                                 ca*spin_old[1] + sa*ty,
                                 ca*spin_old[2] + sa*tz])

        old_energy = _local_energy_numba(
            x, y, b,
            spin_old,
            spins,
            nx, ny,
            num_bonds, bond_vectors, bond_matrices,
            Dani, k2, k6,
        )
        new_energy = _local_energy_numba(
            x, y, b,
            spin_new,
            spins,
            nx, ny,
            num_bonds, bond_vectors, bond_matrices,
            Dani, k2, k6,
        )
        delta_energy = new_energy - old_energy

        if delta_energy <= 0.0 or random_values[proposal, 5] < np.exp(
            -delta_energy / max(temperature_meV, 1.0e-12)
        ):
            spins[x, y, b, 0] = spin_new[0]
            spins[x, y, b, 1] = spin_new[1]
            spins[x, y, b, 2] = spin_new[2]
            accepted += 1

    return accepted / num_sample