import numpy as np

from nix2_mc import LatticeNiX2, SpinModelNiX2

# ===========================================
# functions for building exchange tensors
# ===========================================
# construct exchange tensors from Jprime and Kitaev
def exchange_from_hk(Jprime, Kitaev):
    """
        Return the y-bond tensor Jprime*I + Kitaev*u*u.T.

        u = (sqrt(2/3), 0, -sqrt(1/3)), as in the PRL paper.
        Its isotropic part is trace(J)/3 = Jprime + Kitaev/3.
        Rotate the returned tensor explicitly if using another spin frame.
    """
    # kitaev_matrix = np.array([[2/3, 0, -np.sqrt(2)/3],
    #                           [0, 0, 0],
    #                           [-np.sqrt(2)/3, 0, 1/3]])
    
    kitaev_matrix = np.array([[0, 0, 0],
                              [0, 2/3, -np.sqrt(2)/3],
                              [0, -np.sqrt(2)/3, 1/3]])
    return Jprime * np.eye(3) + Kitaev * kitaev_matrix

# construct exchange tensors from isotropic and anisotropic components
def exchange_from_iso_aniso(Jiso, *, Jxx=0.0, Jyy=0.0, Jzz=0.0, Jxy=0.0, Jxz=0.0, Jyz=0.0):
    """
        Return Jiso*I plus a symmetric anisotropic tensor.

        Supply global components for the reference bond: y for rectangular & x for triangular.
        Normally Jxx + Jyy + Jzz = 0 for an isotropic/traceless decomposition.
    """
    anisotropy = np.array([[Jxx, Jxy, Jxz],
                           [Jxy, Jyy, Jyz],
                           [Jxz, Jyz, Jzz]], dtype=float)
    return Jiso * np.eye(3) + anisotropy

# get rotated exchange tensor along different bond directions
def rotate_exchange(J, angle_degrees):
    """
        Rotate a global exchange tensor around z: R @ J @ R.T.
    """
    angle = np.deg2rad(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R @ J @ R.T


# ===========================================
# neighbors
# each entry is (dx, dy, target_sublattice, bond_angle_degrees)
# Shell order is NN, 2NN, 3NN. Both directions are listed explicitly
# ===========================================
RECTANGULAR_NEIGHBORS = (
    (
        ((0, 0, 1, 30), (0, 1, 0, 90), (-1, 0, 1, 150),
         (-1, -1, 1, 210), (0, -1, 0, 270), (0, -1, 1, 330)),
        ((1, 0, 0, 0), (0, 1, 1, 60), (-1, 1, 1, 120),
         (-1, 0, 0, 180), (-1, -2, 1, 240), (0, -2, 1, 300)),
        ((1, 1, 0, 30), (0, 2, 0, 90), (-1, 1, 0, 150),
         (-1, -1, 0, 210), (0, -2, 0, 270), (1, -1, 0, 330)),
    ),
    (
        ((1, 1, 0, 30), (0, 1, 1, 90), (0, 1, 0, 150),
         (0, 0, 0, 210), (0, -1, 1, 270), (1, 0, 0, 330)),
        ((1, 0, 1, 0), (1, 2, 0, 60), (0, 2, 0, 120),
         (-1, 0, 1, 180), (0, -1, 0, 240), (1, -1, 0, 300)),
        ((1, 1, 1, 30), (0, 2, 1, 90), (-1, 1, 1, 150),
         (-1, -1, 1, 210), (0, -2, 1, 270), (1, -1, 1, 330)),
    ),
)

TRIANGULAR_NEIGHBORS = (
    (
        ((1, 0, 0, 0), (1, 1, 0, 60), (0, 1, 0, 120),
         (-1, 0, 0, 180), (-1, -1, 0, 240), (0, -1, 0, 300)),
        ((2, 1, 0, 30), (1, 2, 0, 90), (-1, 1, 0, 150),
         (-2, -1, 0, 210), (-1, -2, 0, 270), (1, -1, 0, 330)),
        ((2, 0, 0, 0), (2, 2, 0, 60), (0, 2, 0, 120),
         (-2, 0, 0, 180), (-2, -2, 0, 240), (0, -2, 0, 300)),
    ),
)

NEIGHBORS = {
    "rectangular": RECTANGULAR_NEIGHBORS,
    "triangular": TRIANGULAR_NEIGHBORS,
}

VECTORS = {
        "rectangular": ([[np.sqrt(3), 0], [0, 1]], [[0, 0], [0.5, 0.5]]),
        "triangular": ([[1, 0], [-0.5, np.sqrt(3)/2]], [[0, 0]]),
    }

# ===========================================
# build NiX2 lattice and spin model functions
# ===========================================
def build_nix2_lattice(Nx, Ny, *, geometry):
    a_vectors, d_vectors = VECTORS[geometry]
    return LatticeNiX2(a_vectors, d_vectors, Nx=Nx, Ny=Ny)


def build_nix2_model(J1, J3=0.0, *, J2iso=0.0, 
                     Azz=0.0, B_biquad=0.0, B_field=(0.0, 0.0, 0.0), 
                     geometry):
    J1mat_bond000 = float(J1) * np.eye(3) if np.ndim(J1) == 0 else np.asarray(J1, dtype=float)
    J3mat_bond000 = float(J3) * np.eye(3) if np.ndim(J3) == 0 else np.asarray(J3, dtype=float)
    flag_J3iso = np.ndim(J3) == 0

    neighbors_by_sublattice = NEIGHBORS[geometry]
    model = SpinModelNiX2(Ns=len(neighbors_by_sublattice), 
                          A_ani=Azz, B_field=B_field, B_biquad=B_biquad)

    # Opposite bonds share a tensor; follow each notebook's reference bond.
    rotation = {
        "rectangular": {90: 0, 30: 120, 150: 240},
        "triangular": {0: 0, 120: 120, 60: 240},
    }[geometry]
    
    # three-fold rotation symmetry for the exchange tensors, i.e., the exchange tensor for a bond rotated by 120 or 240 degrees can be obtained from the reference bond by applying the corresponding rotation
    J1_rotated = {a: rotate_exchange(J1mat_bond000, a) for a in (0, 120, 240)}
    if not flag_J3iso:
        J3_rotated = {a: rotate_exchange(J3mat_bond000, a) for a in (0, 120, 240)}

    for sublattice, shells in enumerate(neighbors_by_sublattice):
        for distance, neighbors in enumerate(shells):
            if distance == 1 and J2iso == 0:
                continue
            for dx, dy, target, angle in neighbors:
                if distance == 0:
                    Jmat = J1_rotated[rotation[angle % 180]]
                elif distance == 1:
                    Jmat = J2iso * np.eye(3)
                else: # distance == 2 (J3)
                    if flag_J3iso:
                        Jmat = J3mat_bond000
                    else:
                        Jmat = J3_rotated[rotation[angle % 180]]
                model.add_bond_matrix(sublattice, target, (dx, dy), Jmat, CI=False)

    # model.build_numba_arrays()
    return model


def prepare_M_tensors(model: SpinModelNiX2, lat: LatticeNiX2, M01, *, lat_geom):
    """Set nearest-neighbor M tensors from M01 (+y rectangular, +x triangular), with odd bond reversal."""
    model.Mtensors = [[] for _ in range(lat.Ns)]
    reference_angle = {"rectangular": 90, "triangular": 0}[lat_geom]
    rotated = {(reference_angle + angle) % 360: rotate_exchange(np.asarray(M01), angle)
               for angle in (0, 120, 240)}
    for source, shells in enumerate(NEIGHBORS[lat_geom]):
        for dx, dy, target, angle in shells[0]:
            # M_ji = -M_ij compensates S_j x S_i = -(S_i x S_j).
            M = rotated[angle] if angle in rotated else -rotated[(angle + 180) % 360]
            model.add_M_tensor(source, target, (dx, dy), M, CI=False)
    if hasattr(model, "Mtensor_matrices"):
        model.build_numba_arrays()
