from __future__ import annotations

import numpy as np
from numba import njit

# unit conversion
KB_MEV_PER_K = 0.08617333262
MU_S_MEV_PER_T = 0.1157676  # g=2 and S=1

# ==========================================
# energy
# ==========================================
@njit(cache=True)
def _sixfold_energy_numba(spin, k6):
    """
        for unit spins = -K6 sin(theta)^6 cos(6 psi).
    """
    x2 = spin[0] * spin[0]
    y2 = spin[1] * spin[1]
    return (-k6) * (x2*x2*x2 - 15*x2*x2*y2 + 15*x2*y2*y2 - y2*y2*y2)

@njit(cache=True)
def _local_energy_numba(
    x, y, b, # coordinates and sublattice index of the spin being updated
    spin_i, # proposed new spin vector, call it S_i
    spins, # spin configuration
    nx, ny, num_bonds, bond_vectors, bond_matrices,
    *, k2=0.0, k6=0.0,
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
        )
        exchange_spin_j_y = (
            exchange[1, 0] * spin_j[0]
            + exchange[1, 1] * spin_j[1]
        )
        energy += (
            spin_i[0] * exchange_spin_j_x
            + spin_i[1] * exchange_spin_j_y
        )
    
    if k2 != 0.0:
        # energy += (-k2) * (spin_i[0] * spin_i[0] - spin_i[1] * spin_i[1])
        energy -= k2 * (spin_i[0] * spin_i[0])
        
    if k6 != 0.0:
        energy += _sixfold_energy_numba(spin_i, k6)
    
    
    return energy

@njit(cache=True)
def _total_energy_numba(
    spins,
    nx, ny, ns, num_bonds, bond_vectors, bond_matrices,
    *, k2=0.0, k6=0.0,
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
                    for row in range(2):
                        for col in range(2):
                            bilinear += (
                                spin_i[row] * exchange[row, col] * spin_j[col]
                            )
                    energy += 0.5 * bilinear

                # in-plane anisotropy energy part
                if k6 != 0.0:
                    energy += _sixfold_energy_numba(spin_i, k6)
                if k2 != 0.0:
                    # energy -= k2 * (spin_i[0] * spin_i[0] - spin_i[1] * spin_i[1])
                    energy -= k2 * (spin_i[0] * spin_i[0]) # from the paper
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
def _mc_sweep_2dxy_numba(
    spins, # spin configuration
    temperature_meV, # temperature in meV
    nx, ny, ns,
    num_bonds, bond_vectors, bond_matrices,
    random_values,
    reset_probability,
    proposal_angle,*,
    k2=0.0, k6=0.0,
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
        
        if random_values[proposal, 3] < reset_probability:
            phi = 2.0 * np.pi * random_values[proposal, 4]
            spin_new = np.array([np.cos(phi), np.sin(phi)])
        else:
            alpha = (
                2.0 * random_values[proposal, 4] - 1.0
            ) * proposal_angle
            cos_alpha = np.cos(alpha)
            sin_alpha = np.sin(alpha)
            spin_new = np.array(
                [
                    cos_alpha * spin_old[0] - sin_alpha * spin_old[1],
                    sin_alpha * spin_old[0] + cos_alpha * spin_old[1],
                ]
            )

        old_energy = _local_energy_numba(
            x, y, b,
            spin_old,
            spins,
            nx, ny,
            num_bonds, bond_vectors, bond_matrices,
            k2, k6,
        )
        new_energy = _local_energy_numba(
            x, y, b,
            spin_new,
            spins,
            nx, ny,
            num_bonds, bond_vectors, bond_matrices,
            k2, k6,
        )
        delta_energy = new_energy - old_energy

        if delta_energy <= 0.0 or random_values[proposal, 5] < np.exp(
            -delta_energy / max(temperature_meV, 1.0e-12)
        ):
            spins[x, y, b, 0] = spin_new[0]
            spins[x, y, b, 1] = spin_new[1]
            accepted += 1

    return accepted / num_sample