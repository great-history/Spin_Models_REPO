"""THz pulse envelopes and two-pulse sequence construction."""

from __future__ import annotations

import numpy as np


def gaussian_second_derivative(t: float | np.ndarray, t0: float, sigma: float) -> np.ndarray:
    """Compute the second time derivative of a normalized Gaussian envelope.

    Parameters
    ----------
    t : float or numpy.ndarray
        Observation time(s) at which the pulse envelope is evaluated.
    t0 : float
        Time at which the underlying Gaussian envelope is centered.
    sigma : float
        Temporal standard deviation of the Gaussian; it sets the pulse duration
        and must use the same time unit as ``t`` and ``t0``.

    Returns
    -------
    numpy.ndarray
        Second time derivative of the unit-area Gaussian envelope.  This is a
        scalar pulse-shape factor rather than a magnetic-field vector.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dt = np.asarray(t) - t0
    return (dt**2 - sigma**2) * np.exp(-dt**2 / (2.0 * sigma**2)) / (
        np.sqrt(2.0 * np.pi) * sigma**5 )


def d2gdt2_pulse(
    t: float | np.ndarray, t0: float, sigma: float, h_peak: float
) -> np.ndarray:
    """Evaluate a zero-area, second-derivative Gaussian THz field pulse.

    Parameters
    ----------
    t : float or numpy.ndarray
        Observation time(s) at which the pulse field is evaluated.
    t0 : float
        Arrival time of the pulse centre, where the field reaches ``h_peak``.
    sigma : float
        Characteristic temporal width of the pulse.  It controls the pulse
        duration and must use the same time unit as ``t`` and ``t0``.
    h_peak : float
        Peak signed magnetic-field amplitude at ``t0``.  Its unit is the field
        unit used by the LLG model (for example, T or A/m).

    Returns
    -------
    numpy.ndarray
        Scalar magnetic-field waveform at ``t``.  The waveform integrates to
        zero over time and is later projected onto a polarization direction.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dt = np.asarray(t) - t0
    return h_peak * (1.0 - dt**2 / sigma**2) * np.exp(-dt**2 / (2.0 * sigma**2))


def single_thz_pulse(
    t: float,
    pulse_params: dict,
    pulse_func=d2gdt2_pulse,
) -> np.ndarray:
    """Construct the three-component magnetic field of one THz pulse.

    Parameters
    ----------
    t : float
        Simulation time at which the external magnetic field is evaluated.
    pulse_params : dict
        Parameters of the pulse: ``shape_params`` contains the temporal pulse
        parameters (for example, ``t0``, ``sigma``, and ``h_peak``),
        ``polarization`` is a three-component vector specifying the field
        direction in the simulation coordinate system, and optional ``state``
        specifies whether the physical pulse is active.  The polarization is
        normalized internally, so ``h_peak`` remains the field magnitude.
    pulse_func : callable, optional
        Scalar temporal waveform.  It receives ``t`` and ``shape_params`` and
        returns the signed magnetic-field amplitude.  By default this is
        :func:`d2gdt2_pulse`.

    Returns
    -------
    numpy.ndarray
        External magnetic-field vector ``(H_x, H_y, H_z)`` of the pulse at
        time ``t``.  An inactive pulse returns a zero field.
    """
    state = pulse_params.get("state", "on")
    if state in ("off", 0, False):
        return np.zeros(3, dtype=float)
    if state not in ("on", 1, True):
        raise ValueError("Pulse state must be 'on'/'off' or 1/0")

    polarization = np.asarray(pulse_params["polarization"], dtype=float)
    norm = np.linalg.norm(polarization)
    if polarization.shape != (3,) or norm == 0:
        raise ValueError("Pulse polarization must be a nonzero length-3 vector")

    amplitude = np.asarray(pulse_func(t, **pulse_params["shape_params"]))
    direction = polarization / norm

    # solve_ivp passes scalar t and receives a field vector of shape (3,).
    # Passing a time array is useful for plotting and returns shape (N, 3).
    if amplitude.ndim == 0:
        return amplitude * direction
    return amplitude[..., np.newaxis] * direction


def two_thz_pulses(
    t: float,
    pulse_params: dict,
    pulse_func=d2gdt2_pulse,
) -> np.ndarray:
    """Construct the total three-component THz magnetic field from pulses A and B.

    Parameters
    ----------
    t : float
        Simulation time at which the external magnetic field is evaluated.
    pulse_params : dict
        Parameters for pulses ``"A"`` and ``"B"``.  Each pulse contains
        ``shape_params`` (temporal pulse parameters such as ``t0``, ``sigma``,
        and ``h_peak``), ``polarization`` (a three-component vector giving the
        field direction in the simulation coordinate system), and optional
        ``state`` (whether that physical pulse is active).  The polarization is
        normalized internally, so ``h_peak`` remains the field magnitude.
    pulse_func : callable, optional
        Scalar temporal waveform used for each pulse.  It receives ``t`` and
        the pulse's ``shape_params`` and returns the signed field amplitude.
        By default this is :func:`d2gdt2_pulse`.

    Returns
    -------
    numpy.ndarray
        Total external magnetic-field vector ``(H_x, H_y, H_z)`` acting on the
        two-spin system at time ``t``, expressed in the model's field unit.
    """
    time = np.asarray(t)
    # Scalar ``t`` is used by solve_ivp and returns (3,).  Array time is used
    # for plotting and returns (..., 3), matching single_thz_pulse.
    field = np.zeros(time.shape + (3,), dtype=float)
    for label in ("A", "B"):
        pulse = pulse_params[label]
        state = pulse.get("state", "on")
        if state in ("off", 0, False):
            continue
        if state not in ("on", 1, True):
            raise ValueError(f"Pulse {label} state must be 'on'/'off' or 1/0")

        polarization = np.asarray(pulse["polarization"], dtype=float)
        norm = np.linalg.norm(polarization)
        if polarization.shape != (3,) or norm == 0:
            raise ValueError(f"Pulse {label} polarization must be a nonzero length-3 vector")

        amplitude = np.asarray(pulse_func(t, **pulse["shape_params"]))
        direction = polarization / norm
        if amplitude.ndim == 0:
            field += amplitude * direction
        else:
            field += amplitude[..., np.newaxis] * direction
    return field
