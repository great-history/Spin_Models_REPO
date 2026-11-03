"""One- and two-dimensional FFT utilities for THz signals."""

from __future__ import annotations

import numpy as np


def rfft_spectrum(time, signal, *, normalize: bool = False, remove_mean: bool = False):
    """Calculate the one-sided Fourier spectrum of a uniformly sampled signal.

    Parameters
    ----------
    time : array-like
        Uniform observation-time grid. When expressed in ps, the returned
        frequencies are in THz.
    signal : array-like, shape (N, ...) 
        Real-valued time-domain observable whose first axis is sampled at
        ``time``. For example, a scalar signal has shape ``(N,)`` and a
        three-component magnetization has shape ``(N, 3)``.
    normalize : bool, optional
        If ``True``, scale the returned amplitude spectrum to a maximum of 1.
    remove_mean : bool, optional
        If ``True``, subtract the DC offset before transforming.

    Returns
    -------
    tuple of numpy.ndarray
        ``(frequency, spectrum, amplitude)``: non-negative frequency grid,
        complex Fourier coefficients, and their magnitudes. The trailing
        component axes of ``signal`` are retained; e.g. an input of shape
        ``(N, 3)`` returns spectra of shape ``(N_frequency, 3)``.
    """
    time, signal = np.asarray(time, dtype=float), np.asarray(signal)
    if time.ndim != 1 or signal.ndim == 0 or signal.shape[0] != time.size:
        raise ValueError("time must be one-dimensional and signal.shape[0] must equal time.size")
    if time.size < 2:
        raise ValueError("at least two time samples are required")
    dt = np.diff(time)
    if not np.allclose(dt, dt[0]):
        raise ValueError("time samples must be uniformly spaced")

    values = signal - signal.mean(axis=0, keepdims=True) if remove_mean else signal
    spectrum = np.fft.rfft(values, axis=0)
    amplitude = np.abs(spectrum)
    if normalize and amplitude.max() > 0:
        amplitude = amplitude / amplitude.max()
    frequency = np.fft.rfftfreq(time.size, d=dt[0])
    return frequency, spectrum, amplitude


def fft2_spectrum(delay, time, signal_2d, *, remove_mean: bool = True):
    """Calculate the two-dimensional Fourier spectrum of a delay-time signal.

    Parameters
    ----------
    delay : array-like
        Uniform pump-probe delay grid. In ps, its Fourier-conjugate axis is in
        THz.
    time : array-like
        Uniform observation-time grid. In ps, its Fourier-conjugate axis is
        in THz.
    signal_2d : array-like, shape (delay.size, time.size)
        Observable ``S(delay, time)`` measured or simulated over both axes.
    remove_mean : bool, optional
        If ``True``, remove the global DC background before transformation.

    Returns
    -------
    tuple of numpy.ndarray
        ``(f_delay, f_time, spectrum)``: centered frequency grids conjugate to
        delay and time, plus the complex two-dimensional Fourier spectrum.
    """
    delay, time, signal_2d = np.asarray(delay, dtype=float), np.asarray(time, dtype=float), np.asarray(signal_2d)
    if signal_2d.shape != (delay.size, time.size):
        raise ValueError("signal_2d must have shape (delay.size, time.size)")
    if delay.size < 2 or time.size < 2:
        raise ValueError("each axis needs at least two samples")
    d_delay, d_time = np.diff(delay), np.diff(time)
    if not np.allclose(d_delay, d_delay[0]) or not np.allclose(d_time, d_time[0]):
        raise ValueError("delay and time samples must be uniformly spaced")

    values = signal_2d.copy()
    if remove_mean:
        values -= values.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    f_delay = np.fft.fftshift(np.fft.fftfreq(delay.size, d=d_delay[0]))
    f_time = np.fft.fftshift(np.fft.fftfreq(time.size, d=d_time[0]))
    return f_delay, f_time, spectrum
