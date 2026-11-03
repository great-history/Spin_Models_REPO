"""Landau-Lifshitz-Gilbert dynamics; energy in meV and time in ps."""
import numpy as np
from scipy.integrate import solve_ivp
from nix2_numba_kernels import _get_torques_numba


def get_torques(alpha=0.01, spin_config=None, Nx=None, Ny=None, Ns=None,
                nix2_ham=None):
    """Return dS/dt, including biquadratic, anisotropy and field terms.

    Spins have shape (Nx, Ny, Ns, 3). Built bond arrays contain reciprocal
    pairs; the first six neighbors carry the biquadratic term. Optional
    dimensions keep existing notebook calls compatible.
    """
    spins = np.ascontiguousarray(spin_config, dtype=np.float64)
    if spins.shape[2] != nix2_ham.Ns:
        raise ValueError("Spins and Hamiltonian must use the same sublattices.")
    return _get_torques_numba(
        spins, float(alpha), nix2_ham.num_bonds, nix2_ham.bond_vectors,
        nix2_ham.bond_matrices, nix2_ham.A_ani, nix2_ham.zeeman,
        nix2_ham.B_biquad,
    )


get_torque = get_torques


def LLG_relaxation(t, spin_sol, alpha, Nx, Ny, Ns, nix2_ham):
    """Flat solve_ivp RHS; this function does not run the integration itself."""
    spins = np.asarray(spin_sol).reshape(Nx, Ny, Ns, 3)
    return get_torques(alpha, spins, nix2_ham=nix2_ham).ravel()


def relax_spins(spin_config, nix2_ham, alpha=0.5, t_end=100.0,
                rtol=1e-7, atol=1e-9, max_step=0.05):
    """Run damped LLG and return normalized spins without modifying the input."""
    spins = np.array(spin_config, dtype=float, copy=True)
    Nx, Ny, Ns = spins.shape[:3]
    sol = solve_ivp(
        LLG_relaxation, (0, t_end), spins.ravel(),
        args=(alpha, Nx, Ny, Ns, nix2_ham), method="RK45",
        rtol=rtol, atol=atol, max_step=max_step, t_eval=[t_end],
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    relaxed = sol.y[:, -1].reshape(spins.shape).copy()
    relaxed /= np.linalg.norm(relaxed, axis=-1, keepdims=True)
    return np.ascontiguousarray(relaxed)
