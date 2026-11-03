"""Landau--Lifshitz--Gilbert RHS and simulation driver for two sublattices."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .pulses import two_thz_pulses
from .spin_hamiltonian import effective_fields_no_external

HBAR_MEV_PS = 0.6582119569
GAMMA_RAD_PS_T = 0.176085963


def llg_rhs_full(t, state, params: dict, alpha: float, pulse_params: dict, pulse_func,
                 field_func=two_thz_pulses):
    """Evaluate the six-component Landau--Lifshitz--Gilbert equation RHS.

    Parameters
    ----------
    t : float
        Simulation time at which the spin derivatives are evaluated.
    state : array-like of shape (6,)
        Concatenated spin state ``(S1x, S1y, S1z, S2x, S2y, S2z)``.
    params : dict
        Hamiltonian parameters. ``effective_fields_no_external`` returns
        energy gradients in meV. Optional ``hbar_mev_ps`` and
        ``gamma_rad_ps_t`` override the default physical conversion constants.
    alpha : float
        Dimensionless Gilbert damping coefficient.
    pulse_params : dict
        Parameters for THz pulses A and B; these determine the applied
        external magnetic field at time ``t``.
    pulse_func : callable
        Temporal waveform used to evaluate each THz pulse.

    Returns
    -------
    numpy.ndarray of shape (6,)
        Time derivative ``(dS1/dt, dS2/dt)`` from precession and Gilbert
        damping under the combined internal and external effective fields.

    Notes
    -----
    Time is in ps, Hamiltonian parameters are in meV, and external pulse
    fields are in T. Internal energy gradients are converted to angular
    frequencies by dividing by :math:`\hbar`; external fields are converted by
    the gyromagnetic ratio before the two contributions are added.
    """
    s1, s2 = np.asarray(state[:3]), np.asarray(state[3:])
    h1_internal, h2_internal = effective_fields_no_external(s1, s2, params)
    hbar_mev_ps = params.get("hbar_mev_ps", HBAR_MEV_PS)
    gamma_rad_ps_t = params.get("gamma_rad_ps_t", GAMMA_RAD_PS_T)
    omega_1_internal = h1_internal / hbar_mev_ps
    omega_2_internal = h2_internal / hbar_mev_ps
    omega_external = gamma_rad_ps_t * field_func(t, pulse_params, pulse_func)

    def llg(spin, h_eff):
        """Evaluate precession and damping for one sublattice spin.

        Parameters
        ----------
        spin : numpy.ndarray of shape (3,)
            Instantaneous spin vector of one magnetic sublattice.
        h_eff : numpy.ndarray of shape (3,)
            Total effective angular-frequency vector in rad ps^-1.

        Returns
        -------
        numpy.ndarray of shape (3,)
            Time derivative of the spin from LLG precession and damping.
        """
        spin_norm = np.linalg.norm(spin)
        if spin_norm == 0:
            raise ValueError("A zero-length spin is not valid in the LLG equation.")
        precession = np.cross(h_eff, spin)
        damping = np.cross(spin, precession) / spin_norm
        return (precession + alpha * damping) / (1.0 + alpha**2)

    return np.concatenate((
        llg(s1, omega_1_internal + omega_external),
        llg(s2, omega_2_internal + omega_external),
    ))


def run_llg(time, s1_initial, s2_initial, params, alpha, pulse_params, pulse_func,
            field_func=two_thz_pulses, **solver_options):
    """Integrate the two-sublattice LLG dynamics over a specified time grid.

    Parameters
    ----------
    time : array-like
        Monotonically sampled simulation times. Its unit must be consistent
        with the gyromagnetic ratio and THz pulse parameters.
    s1_initial, s2_initial : array-like of shape (3,)
        Initial spin vectors of sublattices 1 and 2.
    params : dict
        Hamiltonian parameters and optional gyromagnetic ratio ``gamma``.
    alpha : float
        Dimensionless Gilbert damping coefficient.
    pulse_params : dict
        Parameters describing THz pulses A and B and their polarizations.
    pulse_func : callable
        Scalar temporal waveform used for each THz pulse.
    **solver_options
        Additional keyword options passed to ``scipy.integrate.solve_ivp``.

    Returns
    -------
    dict
        Time-resolved solution with ``time``, sublattice spins ``S1`` and
        ``S2``, net magnetization ``M = S1 + S2``, and the raw SciPy solver
        result under ``solution``.

    Notes
    -----
    The full six-component system is integrated, allowing the spin components
    to change sign during strong-pulse dynamics.
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a one-dimensional array with at least two entries")
    state0 = np.concatenate((np.asarray(s1_initial, dtype=float), np.asarray(s2_initial, dtype=float)))
    options = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11, "max_step": 0.02}
    options.update(solver_options)
    solution = solve_ivp(
        llg_rhs_full,
        (time[0], time[-1]),
        state0,
        t_eval=time,
        args=(params, alpha, pulse_params, pulse_func, field_func),
        **options,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    s1, s2 = solution.y[:3].T, solution.y[3:].T
    return {"time": time, "S1": s1, "S2": s2, "M": s1 + s2, "solution": solution}


def update_S1S2(state, Stot: float, signs=(1.0, -1.0)):
    """Reconstruct two fixed-length spins from the four-component state.

    Parameters
    ----------
    state : array-like of shape (4,) four components
        ``(S1b, S1c, S2b, S2c)`` for M along c axes
        ``(S1a, S1b, S2a, S2b)`` for M along a axes
    Stot : float
        The total spin length.
    signs : tuple of float, optional
        (1.0, -1.0) for M along c axes, the signs of the a components of S1 and S2.
        (-1.0, 1.0) for M along a axes, the signs of the c components of S1 and S2.

    Returns
    -------
    s1, s2 : tuple of numpy.ndarray of shape (3,)
        Reconstructed spin vectors of sublattices 1 and 2.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (4,):
        raise ValueError("Reduced LLG state must have shape (4,): (S1b, S1c, S2b, S2c).")
    if Stot <= 0:
        raise ValueError("Stot must be positive.")

    signs = np.asarray(signs, dtype=float)
    if signs.shape != (2,) or np.any(~np.isin(signs, (-1.0, 1.0))):
        raise ValueError("signs must contain exactly two values, each +1 or -1.")

    s1b, s1c, s2b, s2c = state
    s1a_sq = Stot**2 - s1b**2 - s1c**2
    s2a_sq = Stot**2 - s2b**2 - s2c**2
    tolerance = 1e-12 * Stot**2
    if s1a_sq < -tolerance or s2a_sq < -tolerance:
        raise ValueError("Reduced state exceeds the fixed Stot.")

    s1 = np.array((signs[0] * np.sqrt(max(s1a_sq, 0.0)), s1b, s1c))
    s2 = np.array((signs[1] * np.sqrt(max(s2a_sq, 0.0)), s2b, s2c))
    return s1, s2


def llg_rhs_reduced(t, state, params: dict, alpha: float, pulse_params: dict, pulse_func,
                    field_func=two_thz_pulses, signs=(1.0, -1.0)):
    """Evaluate the four-component, fixed-spin-length LLG RHS.

    ``state`` is ``(S1b, S1c, S2b, S2c)``.  The a components are reconstructed
    with :func:`update_S1S2`; only the independent b and c
    derivatives are returned.  ``field_func`` accepts the same interface as
    :func:`two_spin_dynamics.pulses.two_thz_pulses`, so it may instead be
    :func:`two_spin_dynamics.pulses.single_thz_pulse` for a single-pulse
    parameter dictionary.
    """
    Stot = params["Stot"]
    s1, s2 = update_S1S2(state, Stot, signs=signs)
    h1_internal, h2_internal = effective_fields_no_external(s1, s2, params)
    hbar_mev_ps = params.get("hbar_mev_ps", HBAR_MEV_PS)
    gamma_rad_ps_t = params.get("gamma_rad_ps_t", GAMMA_RAD_PS_T)
    omega_external = gamma_rad_ps_t * field_func(t, pulse_params, pulse_func)

    def llg(spin, h_eff):
        precession = np.cross(h_eff, spin)
        damping = np.cross(spin, precession) / np.linalg.norm(spin)
        return (precession + alpha * damping) / (1.0 + alpha**2)

    ds1_dt = llg(s1, h1_internal / hbar_mev_ps + omega_external)
    ds2_dt = llg(s2, h2_internal / hbar_mev_ps + omega_external)
    return np.array((ds1_dt[1], ds1_dt[2], ds2_dt[1], ds2_dt[2]))


def run_llg_reduced(time, s1_initial, s2_initial, params, alpha, pulse_params, pulse_func,
                    field_func=two_thz_pulses, signs=None, **solver_options):
    """Integrate LLG in the four-component fixed-length coordinate chart.

    The returned dictionary has the same ``time``, ``S1``, ``S2``, ``M``, and
    ``solution`` entries as :func:`run_llg`, plus ``reduced_state`` with
    columns ``(S1b, S1c, S2b, S2c)``.  ``field_func`` defaults to the project
    two-pulse field builder; pass ``single_thz_pulse`` when using the
    single-pulse parameter format from ``LLG-two-spin-dynamics-single-pulse``.

    This solver holds both spin lengths at ``params[\"Stot\"]`` by construction
    but cannot represent a sign change in either a component.  It raises an
    error if the trajectory leaves the selected branch.
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a one-dimensional array with at least two entries")
    Stot = params["Stot"]
    s1_initial = np.asarray(s1_initial, dtype=float)
    s2_initial = np.asarray(s2_initial, dtype=float)
    if s1_initial.shape != (3,) or s2_initial.shape != (3,):
        raise ValueError("s1_initial and s2_initial must each have shape (3,)")
    if signs is None:
        signs = (np.sign(s1_initial[0]) or 1.0, np.sign(s2_initial[0]) or -1.0)
    state0 = np.array((s1_initial[1], s1_initial[2], s2_initial[1], s2_initial[2]))
    # Validate the initial state and selected branch before integration.
    s1_reconstructed, s2_reconstructed = update_S1S2(
        state0, Stot, signs=signs
    )
    if not (np.allclose(s1_initial, s1_reconstructed, rtol=1e-10, atol=1e-12)
            and np.allclose(s2_initial, s2_reconstructed, rtol=1e-10, atol=1e-12)):
        raise ValueError(
            "Initial spins must have length params['Stot'] and agree with the selected signs."
        )

    options = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11, "max_step": 0.02}
    options.update(solver_options)
    solution = solve_ivp(
        llg_rhs_reduced,
        (time[0], time[-1]),
        state0,
        t_eval=time,
        args=(params, alpha, pulse_params, pulse_func, field_func, signs),
        **options,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    reduced_state = solution.y.T
    s1, s2 = zip(*(update_S1S2(state, Stot, signs=signs)
                   for state in reduced_state))
    s1, s2 = np.asarray(s1), np.asarray(s2)
    return {
        "time": time,
        "S1": s1,
        "S2": s2,
        "M": s1 + s2,
        "reduced_state": reduced_state,
        "solution": solution,
    }
