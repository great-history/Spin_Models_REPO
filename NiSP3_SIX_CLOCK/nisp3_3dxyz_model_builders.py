"""NiPS3 XY model from NiPS3_mini_model_test.ipynb; energies in meV."""

import numpy as np
from nisp3_3dspin_mc import Lattice2D, SpinModel3DXYZ

# Indexed by source sublattice, then shell (NN, 2NN, 3NN).
# Each entry is (dx, dy, target_sublattice); both directions are listed.

RECTANGULAR_NEIGHBORS = (
    (   
        # nearest-neighbor
        ((0,0,1),(1,0,1),(0,-1,3)),
        # second-nearest-neighbor
        ((0,0,2),(1,0,2),(1,0,0),(1,-1,2),(0,-1,2),(-1,0,0)),
        # third-nearest-neighbor
        ((0,0,3),(1,-1,3),(-1,-1,3))
     ),
     (
        ((0,0,0),(-1,0,0),(0,0,2)),
        ((0,0,3),(1,0,1),(0,-1,3),(-1,-1,3),(-1,0,1),(-1,0,3)),
        ((1,0,2),(0,-1,2),(-1,0,2))
     ),
     (
        ((0,0,1),(-1,0,3),(0,0,3)),
        ((0,0,0),(-1,0,0),(-1,0,2),(-1,1,0),(0,1,0),(1,0,2)),
        ((1,0,1),(-1,0,1),(0,1,1)),
     ),
    (
        ((0,0,2),(0,1,0),(1,0,2)),
        ((0,0,1),(-1,0,3),(0,1,1),(1,1,1),(1,0,3),(1,0,1)),
        ((0,0,0),(-1,1,0),(1,1,0))
     ),
)

TRIANGULAR_NEIGHBORS = (
    (
        ((-1, -1, 1), (0, -1, 1), (0, 0, 1)),
        ((-1, -1, 0), (-1, 0, 0), (0, -1, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
        ((-1, -2, 1), (-1, 0, 1), (1, 0, 1)),
    ),
    (
        ((0, 0, 0), (0, 1, 0), (1, 1, 0)),
        ((-1, -1, 1), (-1, 0, 1), (0, -1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)),
        ((-1, 0, 0), (1, 0, 0), (1, 2, 0)),
    ),
)

NEIGHBORS = {
    "rectangular": RECTANGULAR_NEIGHBORS,
    "triangular": TRIANGULAR_NEIGHBORS,
}

VECTORS = {
    "rectangular": (
        [[1, 0], [0, np.sqrt(3)]],
        [[0.5, 1/6], [0, 1/3], [0, 2/3], [0.5, 5/6]],
    ),
    "triangular": (
        [[1, 0], [-0.5, np.sqrt(3)/2]],
        [[0, 0], [1/3, 2/3]],
    ),
}

def build_nisp3_lattice(Nx, Ny, *, geometry="rectangular"):
    """
        Four-site rectangular cell or two-site triangular Bravais cell.
        Nx and Ny count cells along the chosen vectors, but not necessarily x and y directions.
    """
    a_vectors, d_vectors = VECTORS[geometry]
    return Lattice2D(a_vectors, d_vectors, Nx=Nx, Ny=Ny)

def build_nisp3_3dxyz_model(J1=-3.8, J3=13.8, *, J2iso=0.2, 
                           Dani= 0.0, K6=1.0, K2=0.0, geometry="rectangular"):
    """
        Build the mini model with scalar J1/J2/J3 exchange and 2d XY spins.
    """
    neighbors_by_sublattice = NEIGHBORS[geometry]
    model = SpinModel3DXYZ(Ns=len(neighbors_by_sublattice), Dani=Dani, K2=K2,K6=K6)
    
    exchanges = [J * np.eye(3) for J in (J1, J2iso, J3)]
    for s1, shells in enumerate(neighbors_by_sublattice):
        for Jmat, neighbors in zip(exchanges, shells):
            for dx, dy, s2 in neighbors:
                model.add_bond_matrix(s1, s2, (dx, dy), Jmat, CI=False)

    # model.model_parameters = dict(J1=J1, J2=J2iso, J3=J3)
    return model
