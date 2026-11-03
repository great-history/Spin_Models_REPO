"""Landau--Lifshitz--Gilbert RHS and simulation driver for two sublattices."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .pulses import two_thz_pulses
from .spin_hamiltonian import effective_fields_no_external

HBAR_MEV_PS = 0.6582119569
GAMMA_RAD_PS_T = 0.176085963

def update_S1S2(sol, Stot):
    S1b, S1c, S2b, S2c = np.asarray(sol, dtype=float)
    S1a_sq = Stot**2 - S1b**2 - S1c**2
    S2a_sq = Stot**2 - S2b**2 - S2c**2

    tolerance = 1e-12 * Stot**2

    if S1a_sq < -tolerance:
        raise ValueError("The S1a exceed the Stot")
    # if S2a_sq < -tolerance:
    #     raise ValueError("The S2a exceed the Stot")
    if S2a_sq < -tolerance:
        raise ValueError("The S2a exceed the Stot")

    S1a =  np.sqrt(max(S1a_sq, 0.0)) # S1a is positive
    S2a = -np.sqrt(max(S2a_sq, 0.0)) # S2a is negative
    S1 = np.array([S1a, S1b, S1c])
    S2 = np.array([S2a, S2b, S2c])
    
    return S1, S2

def LLG_S1S2_dynamics(t, sol, Ham_params, alpha, pulse_params, pulse_func):
    """
        Four-component LLG RHS compatible with scipy.integrate.solve_ivp.
    """
    S1, S2 = update_S1S2(sol, Ham_params["Stot"]) # dimensionless

    # internal fields
    H1_internal, H2_internal = effective_fields_no_external(S1, S2, Ham_params)
    H1_internal = H1_internal / HBAR_MEV_PS # unit is ps^-1
    H2_internal = H2_internal / HBAR_MEV_PS # unit is ps^-1

    # external field
    H_external = two_thz_pulses(t, pulse_params, pulse_func) # unit is Tesla
    H_external = H_external * GAMMA_RAD_PS_T # unit is ps^-1

    def llg(S, H_eff):
        precession = np.cross(H_eff, S)
        damping = np.cross(S, precession) / Ham_params["Stot"]
        return 1 / (1 + alpha**2) * (precession + alpha * damping)

    dS1_dt = llg(S1, H1_internal + H_external)
    dS2_dt = llg(S2, H2_internal + H_external)

    # Only the independent b and c components are integrated.
    return np.array([dS1_dt[1], dS1_dt[2], dS2_dt[1], dS2_dt[2]])