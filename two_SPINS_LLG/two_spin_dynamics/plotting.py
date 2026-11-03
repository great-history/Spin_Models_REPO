"""Composable Matplotlib helpers for two-spin THz dynamics.

All plotting helpers accept an existing ``Axes`` where practical.  This makes
it easy to combine pulse, magnetization, and spectral panels in one figure.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np


# Apply explicitly with ``plt.rcParams.update(PLOT_CONFIG)`` in a notebook.
PLOT_CONFIG = {
    "savefig.dpi": 300, 
    "axes.labelsize": 12, 
    "axes.titlesize": 13,
    "legend.fontsize": 10, 
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10,
    "font.size": 10, 
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
}

PLOT_STYLES = {
    "pulse_a": {"color": "tab:orange", "linewidth": 1.2},
    "pulse_b": {"color": "tab:blue", "linewidth": 1.2},
    "pulse_total": {"color": "black", "linestyle": "--", "linewidth": 1.2},
    "component_a": {"color": "tab:red", "linewidth": 1.25},
    "component_b": {"color": "tab:green", "linewidth": 1.25},
    "component_c": {"color": "tab:blue", "linewidth": 1.25},
    "spectrum": {"color": "tab:purple", "linewidth": 1.2},
}

LEGEND_STYLES = {
    "default": {"loc": "upper right", "frameon": False, "fontsize": 10},
    "compact": {"loc": "best", "frameon": False, "fontsize": 9},
    "horizontal": {"loc": "upper center", "ncol": 3, "frameon": False, "fontsize": 9},
    "outside_right": {"bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False, "fontsize": 9},
}

_COMPONENTS = ("a", "b", "c")
_COMPONENT_STYLES = tuple(PLOT_STYLES[f"component_{c}"] for c in _COMPONENTS)


def _validate(time, values, name="values"):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if values.ndim not in (1, 2) or values.shape[0] != time.size:
        raise ValueError(f"{name} must have shape (N,) or (N, K), with N matching time.")
    return time, values


def _component_labels(base):
    base = str(base).strip("$")
    return [rf"${base}_{{{component}}}$" for component in _COMPONENTS]


def _finish(ax, *, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
            grid_alpha=0.3, show_legend=True, legend_config=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid_alpha:
        ax.grid(True, alpha=grid_alpha)
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(**(LEGEND_STYLES["compact"] if legend_config is None else legend_config))
    return ax


def plot_time_series(ax, time, series, labels=None, *, styles=None, title=None,
                     xlabel="Time (ps)", ylabel=None, xlim=None, ylim=None,
                     grid_alpha=0.3, show_legend=True, legend_config=None):
    """Plot generic scalar time traces on an existing axes.

    ``series`` can be ``{label: values}`` or a sequence of values accompanied
    by a matching sequence of ``labels``.
    """
    time = np.asarray(time, dtype=float)
    items = list(series.items()) if isinstance(series, Mapping) else list(zip(labels or (), series))
    if not items:
        raise ValueError("series must contain at least one labelled trace.")
    styles = [{}] if styles is None else styles
    for index, (curve_label, values) in enumerate(items):
        _, values = _validate(time, values, "each time series")
        if values.ndim != 1:
            raise ValueError("plot_time_series accepts scalar traces only.")
        ax.plot(time, values, label=curve_label, **styles[index % len(styles)])
    return _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   grid_alpha=grid_alpha, show_legend=show_legend, legend_config=legend_config)


def plot_pulse(time, field, ax=None, *, label=r"$H_{\mathrm{THz}}(t)$",
               title="Second-derivative Gaussian pulse", xlabel="Time (ps)",
               ylabel="H (T)", xlim=None, ylim=None, shade=True,
               grid_alpha=0.3, legend_config=None):
    """Plot one signed scalar THz pulse; positive/negative lobes may be shaded."""
    time, field = _validate(time, field, "field")
    if field.ndim != 1:
        raise ValueError("plot_pulse expects a scalar field with shape (N,).")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 2.5), dpi=180)
    if shade:
        ax.fill_between(time, field, 0, where=field >= 0, interpolate=True, color="tab:orange", alpha=0.35)
        ax.fill_between(time, field, 0, where=field < 0, interpolate=True, color="tab:blue", alpha=0.35)
    ax.plot(time, field, color="black", linewidth=1.0, label=label)
    ax.axhline(0, color="gray", linewidth=0.8)
    return _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   grid_alpha=grid_alpha, legend_config=legend_config)


def plot_field_components(ax, time, field, *, 
                          title=None, xlabel="Time (ps)", ylabel="THz field H (T)", 
                          base_label="H", components=(0, 1, 2),
                          xlim=None, ylim=None, grid_alpha=0.3, show_zero=False,
                          show_legend=True, legend_config=None):
    """Plot selected components of a vector field ``field[:, (a,b,c)]``."""
    time, field = _validate(time, field, "field")
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError("field must have shape (N, 3).")
    labels = _component_labels(base_label)
    for component in components:
        if component not in range(3):
            raise ValueError("components must be chosen from 0 (a), 1 (b), 2 (c).")
        if show_zero or np.any(field[:, component] != 0):
            ax.plot(time, field[:, component], label=labels[component], **_COMPONENT_STYLES[component])
    ax.axhline(0, color="gray", linewidth=0.8)
    return _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   grid_alpha=grid_alpha, show_legend=show_legend, legend_config=legend_config)


def plot_two_pulses(time, h_a, h_b, ax=None, *, 
                    title="Two-pulse THz sequence",
                    xlabel="Time (ps)", ylabel="H (T)", xlim=None, ylim=None,
                    legend_config=None):
    """Plot pulses A, B, and their scalar sum on one axes."""
    time, h_a = _validate(time, h_a, "h_a")
    _, h_b = _validate(time, h_b, "h_b")
    if h_a.ndim != 1 or h_b.ndim != 1:
        raise ValueError("h_a and h_b must have shape (N,).")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 2.5), dpi=180)
    ax.fill_between(time, h_a, 0, color="tab:orange", alpha=0.20)
    ax.fill_between(time, h_b, 0, color="tab:blue", alpha=0.20)
    ax.plot(time, h_a, label=r"$H_A(t)$", **PLOT_STYLES["pulse_a"])
    ax.plot(time, h_b, label=r"$H_B(t)$", **PLOT_STYLES["pulse_b"])
    ax.plot(time, h_a + h_b, label=r"$H_A(t)+H_B(t)$", **PLOT_STYLES["pulse_total"])
    ax.axhline(0, color="gray", linewidth=0.8)
    return _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   legend_config=legend_config)


def plot_magnetization_components(ax, time, magnetization, *, components=(0, 1, 2),
                                  title=None, xlabel="Time (ps)",
                                  ylabel="Magnetization M (spin units)", base_label="M",
                                  xlim=None, ylim=None, grid_alpha=0.3, show_legend=True,
                                  legend_config=None):
    """Plot selected components of ``magnetization[:, (M_a,M_b,M_c)]``."""
    time, magnetization = _validate(time, magnetization, "magnetization")
    if magnetization.ndim != 2 or magnetization.shape[1] != 3:
        raise ValueError("magnetization must have shape (N, 3).")
    labels = _component_labels(base_label)
    for component in components:
        if component not in range(3):
            raise ValueError("components must be chosen from 0 (a), 1 (b), 2 (c).")
        ax.plot(time, magnetization[:, component], label=labels[component], **_COMPONENT_STYLES[component])
    return _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   grid_alpha=grid_alpha, show_legend=show_legend, legend_config=legend_config)


def plot_magnetization_dynamics(time, magnetization, *, pulse=None,
                                title="Magnetization dynamics", pulse_label="H",
                                figsize=(8, 6.3), dpi=180, xlim=None,
                                pulse_ylim=None, mc_ylim=None, mab_ylim=None,
                                legend_config=None):
    """Create a shared-time-axis pulse / ``M_c`` / ``M_a,M_b`` figure.

    ``pulse`` can be scalar ``(N,)`` or vector ``(N, 3)``.  The function
    returns ``(fig, axes)`` where axes is keyed by ``pulse`` (if present),
    ``mc``, and ``mab``.
    """
    time, magnetization = _validate(time, magnetization, "magnetization")
    if magnetization.ndim != 2 or magnetization.shape[1] != 3:
        raise ValueError("magnetization must have shape (N, 3).")
    has_pulse = pulse is not None
    if has_pulse:
        _, pulse = _validate(time, pulse, "pulse")
        if pulse.ndim == 2 and pulse.shape[1] != 3:
            raise ValueError("a vector pulse must have shape (N, 3).")

    nrows = 3 if has_pulse else 2
    fig, axes_array = plt.subplots(
        nrows, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.0, 1.0] if has_pulse else [1.0, 1.0]},
    )
    fig.suptitle(title)
    axes_array = np.atleast_1d(axes_array)
    axes = {}
    if has_pulse:
        ax_pulse, ax_mc, ax_mab = axes_array
        axes["pulse"] = ax_pulse
        if pulse.ndim == 1:
            plot_pulse(time, pulse, ax=ax_pulse, label=rf"${pulse_label}_{{\mathrm{{THz}}}}$",
                       title=None, xlabel=None, ylabel="THz field H (T)", ylim=pulse_ylim,
                       legend_config=legend_config)
        else:
            plot_field_components(ax_pulse, time, pulse, xlabel=None, ylabel="THz field H (T)",
                                  base_label=pulse_label, ylim=pulse_ylim, legend_config=legend_config)
    else:
        ax_mc, ax_mab = axes_array
    axes["mc"], axes["mab"] = ax_mc, ax_mab
    plot_magnetization_components(ax_mc, time, magnetization, components=(2,), xlabel=None,
                                  ylabel=r"$M_c$ (spin units)", xlim=xlim, ylim=mc_ylim,
                                  legend_config=legend_config)
    plot_magnetization_components(ax_mab, time, magnetization, components=(0, 1),
                                  ylabel=r"$M_a, M_b$ (spin units)", xlim=xlim, ylim=mab_ylim,
                                  legend_config=legend_config)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_fft_spectrum(frequency, spectrum, *, title, label, ax=None,
                      xlabel="Frequency (THz)", ylabel="FFT amplitude",
                      xlim=None, ylim=None, grid_alpha=0.3, legend_config=None):
    """Plot a scalar or ``a,b,c`` one-sided FFT spectrum on an existing axes."""
    frequency = np.asarray(frequency, dtype=float)
    spectrum = np.asarray(spectrum)
    if frequency.ndim != 1:
        raise ValueError("frequency must be one-dimensional.")
    if spectrum.ndim not in (1, 2) or spectrum.shape[0] != frequency.size:
        raise ValueError("spectrum must have shape (Nf,) or (Nf, 3), with Nf matching frequency.")
    if spectrum.ndim == 2 and spectrum.shape[1] != 3:
        raise ValueError("a multi-component spectrum must have exactly three columns (a, b, c).")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), dpi=180)
    amplitude = np.abs(spectrum)
    if amplitude.ndim == 1:
        ax.plot(frequency, amplitude, label=label, **PLOT_STYLES["spectrum"])
    else:
        labels = _component_labels(label) if isinstance(label, str) else list(label)
        if len(labels) != 3:
            raise ValueError("label must be a string or three labels for a three-component spectrum.")
        for component, component_label in enumerate(labels):
            ax.plot(frequency, amplitude[:, component], label=component_label,
                    **_COMPONENT_STYLES[component])
    return _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                   grid_alpha=grid_alpha, legend_config=legend_config)
