"""Hamiltonian, effective fields, and stationary states for two spin sublattices."""

from __future__ import annotations

import numpy as np


def canted_afm_ground_state(params: dict):
    """Return the zero-field canted-AFM ground state for the two-spin model.

    The easy axis is ``a`` when ``K_a > K_c`` and ``c`` when ``K_c > K_a``.
    The returned spins have magnitude ``params[\"Stot\"]`` and use the branch
    consistent with positive ``D_b``.  The result is intended as the initial
    condition for a small-amplitude LLG simulation.

    Returns
    -------
    tuple
        ``(s1_eq, s2_eq, beta0)`` where ``beta0`` is in radians.
    """
    n = params["nn"]
    spin_length = params["Stot"]
    j_ex = params["Jex"]
    d_b = np.asarray(params["Dvec"], dtype=float)[1]
    k_a, _, k_c = np.asarray(params["Kvec"], dtype=float)

    if np.isclose(k_a, k_c):
        raise ValueError("K_a and K_c are degenerate; choose a ground-state branch explicitly.")

    if k_a > k_c: # net magnetization along c
        beta0 = 0.5 * np.arctan2(n * d_b, n * j_ex + k_a - k_c)
        s1 = spin_length * np.array([np.cos(beta0), 0.0, np.sin(beta0)])
        s2 = spin_length * np.array([-np.cos(beta0), 0.0, np.sin(beta0)])
    else: # net magnetization along a
        beta0 = 0.5 * np.arctan2(n * d_b, n * j_ex - k_a + k_c)
        s1 = spin_length * np.array([np.sin(beta0), 0.0, -np.cos(beta0)])
        s2 = spin_length * np.array([np.sin(beta0), 0.0, np.cos(beta0)])
    return s1, s2, beta0


def effective_fields_no_external(s1, s2, params: dict):
    """
        Calculate the internal effective fields acting on two spin sublattices.

        Parameters
        ----------
        s1, s2 : array-like of shape (3,)
            Spin vectors of sublattices 1 and 2 in the simulation coordinate
            system. Their magnitudes represent the sublattice spin amplitudes.
        params : dict
            Hamiltonian parameters: ``nn`` is the number of nearest neighbours,
            ``Jex`` the exchange coupling, ``Dvec`` the Dzyaloshinskii--Moriya
            interaction vector, and ``Kvec`` the component-wise anisotropy.

        Returns
        -------
        tuple of numpy.ndarray
            ``(H_eff_1, H_eff_2)``, the three-component internal effective fields
            ``-dH/dS1`` and ``-dH/dS2``. External THz fields are not included.
    """
    n, j = params["nn"], params["Jex"]
    da, db, dc = np.asarray(params["Dvec"], dtype=float)
    ka, kb, kc = np.asarray(params["Kvec"], dtype=float)
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)

    d_h_s1 = np.array([
        n * j * s2[0] + n * (dc * s2[1] - db * s2[2]) - 2 * ka * s1[0],
        n * j * s2[1] + n * (da * s2[2] - dc * s2[0]) - 2 * kb * s1[1],
        n * j * s2[2] + n * (db * s2[0] - da * s2[1]) - 2 * kc * s1[2],
    ])
    d_h_s2 = np.array([
        n * j * s1[0] - n * (dc * s1[1] - db * s1[2]) - 2 * ka * s2[0],
        n * j * s1[1] - n * (da * s1[2] - dc * s1[0]) - 2 * kb * s2[1],
        n * j * s1[2] - n * (db * s1[0] - da * s1[1]) - 2 * kc * s2[2],
    ])

    return -d_h_s1, -d_h_s2


def energy(s1, s2, params: dict, external_field=None) -> float:
    """Evaluate the two-sublattice Hamiltonian energy.

    Parameters
    ----------
    s1, s2 : array-like of shape (3,)
        Spin vectors of the two magnetic sublattices.
    params : dict
        Hamiltonian parameters ``nn``, ``Jex``, ``Dvec``, and ``Kvec`` for
        exchange, Dzyaloshinskii--Moriya, and anisotropy contributions.
    external_field : array-like of shape (3,), optional
        Applied magnetic-field vector. When supplied, its Zeeman coupling
        ``-H_ext . (S1 + S2)`` is included.

    Returns
    -------
    float
        Total Hamiltonian energy in the energy unit implied by ``params`` and
        the spin normalization.
    """
    n, j = params["nn"], params["Jex"]
    dvec, kvec = np.asarray(params["Dvec"]), np.asarray(params["Kvec"])
    value = n * j * np.dot(s1, s2) + n * np.dot(dvec, np.cross(s1, s2))
    value -= np.dot(kvec, np.asarray(s1) ** 2 + np.asarray(s2) ** 2)
    if external_field is not None:
        value -= np.dot(external_field, np.asarray(s1) + np.asarray(s2))
    return float(value)
