from pathlib import Path
from matplotlib.colors import is_color_like
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Default appearance. Importing this module does not change Matplotlib settings.
PLOT_CONFIG = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
}

ZIGZAG_COLORS = ["tab:blue", "tab:orange", "tab:green"]
SPIN_CMAP = "hsv"


def _draw_bonds(ax, coords, mask, neighbors):
    """Draw one neighbor shell, omitting periodic wraparound bonds."""
    nx, ny, num_basis = mask.shape
    segments = []
    for ix, iy, source in np.argwhere(mask):
        for dx, dy, target in neighbors[source]:
            jx = ix + dx
            jy = iy + dy
            if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                continue
            if not mask[jx, jy, target]:
                continue
            # The neighbor list contains both directions; draw each bond once.
            if (ix, iy, source) < (jx, jy, target):
                segments.append([coords[ix, iy, source], coords[jx, jy, target]])
    bonds = LineCollection(segments, colors="0.75", linewidths=0.8, zorder=1)
    ax.add_collection(bonds)


def plot_spin_config(
    spins, lattice, ax=None, title="Spin configuration",
    xlim=None, ylim=None, neighbors=None, figsize=(7, 6),
    arrow_scale=2.5, arrow_width=0.045, colorbar=True,
):
    """Plot XY spin arrows, colored by angle.

    xlim and ylim are Cartesian limits in lattice units.
    For nearest-neighbor bonds, pass neighbors=NEIGHBORS[0].
    """
    # Lattice coordinates, spin angles, and the visible region.
    spins = np.asarray(spins)
    coords = lattice.get_cartesian_coords()

    angles = np.arctan2(spins[..., 1], spins[..., 0])
    angles = np.mod(angles, 2 * np.pi)
    mask = np.ones(spins.shape[:3], dtype=bool)
    if xlim is not None:
        mask &= (coords[..., 0] >= xlim[0]) & (coords[..., 0] <= xlim[1])
    if ylim is not None:
        mask &= (coords[..., 1] >= ylim[0]) & (coords[..., 1] <= ylim[1])
    selected_coords = coords[mask]
    selected_spins = spins[mask]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if neighbors is not None:
        _draw_bonds(ax, coords, mask, neighbors)

    arrows = ax.quiver(
        selected_coords[:, 0], selected_coords[:, 1],
        selected_spins[:, 0], selected_spins[:, 1], angles[mask],
        cmap=SPIN_CMAP, clim=(0, 2 * np.pi),
        angles="xy", scale_units="xy", scale=arrow_scale,
        pivot="mid", units="xy", width=arrow_width, zorder=2,
    )

    # Axes and angle colorbar.
    if xlim is None:
        xlim = (coords[..., 0].min() - 0.6, coords[..., 0].max() + 0.6)
    if ylim is None:
        ylim = (coords[..., 1].min() - 0.6, coords[..., 1].max() + 0.6)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", title=title,
           xlabel="x (lattice units)", ylabel="y (lattice units)")
    if colorbar:
        cbar = ax.figure.colorbar(arrows, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label(r"Spin angle $\phi$ (rad)")
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    return ax


def plot_spin_angles(
    spins, lattice, ax=None, title="Spin-angle map",
    xlim=None, ylim=None, neighbors=None, figsize=(7, 6),
    marker_size=None, colorbar=True,
):
    """Plot colored lattice sites without arrows; optionally add bonds."""
    # Lattice coordinates, spin angles, and the visible region.
    spins = np.asarray(spins)
    coords = lattice.get_cartesian_coords()

    angles = np.arctan2(spins[..., 1], spins[..., 0])
    angles = np.mod(angles, 2 * np.pi)
    mask = np.ones(spins.shape[:3], dtype=bool)
    if xlim is not None:
        mask &= (coords[..., 0] >= xlim[0]) & (coords[..., 0] <= xlim[1])
    if ylim is not None:
        mask &= (coords[..., 1] >= ylim[0]) & (coords[..., 1] <= ylim[1])
    selected_coords = coords[mask]
    if marker_size is None:
        lattice_size = max(spins.shape[:2])
        marker_size = min(120.0, (240.0 / lattice_size) ** 2)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if neighbors is not None:
        _draw_bonds(ax, coords, mask, neighbors)

    points = ax.scatter(
        selected_coords[:, 0], selected_coords[:, 1],
        c=angles[mask], s=marker_size,
        cmap=SPIN_CMAP, vmin=0, vmax=2 * np.pi,
        edgecolors="none", zorder=2,
    )

    # Axes and angle colorbar.
    if xlim is None:
        xlim = (coords[..., 0].min() - 0.6, coords[..., 0].max() + 0.6)
    if ylim is None:
        ylim = (coords[..., 1].min() - 0.6, coords[..., 1].max() + 0.6)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", title=title,
           xlabel="x (lattice units)", ylabel="y (lattice units)")
    if colorbar:
        cbar = ax.figure.colorbar(points, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label(r"Spin angle $\phi$ (rad)")
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    return ax


def plot_curves(
    x, y1, y2=None, ax=None, *, share_y=False,
    xlabel="x", ylabel1="y1", ylabel2="y2",
    label1="Series 1", label2="Series 2", title=None,
    colors=("navy", "firebrick"), markers=("o", "s"),
    figsize=(8, 4), linewidth=1.5, markersize=4,
    xlim=None, ylim1=None, ylim2=None, sort_x=True,
    save_path=None, flag_show=False, dpi=300,
):
    """y1/y2: (points,) or (curves, points). Returns fig, ax1, ax2.

    colors/markers: two entries, for y1 and y2 respectively.
    Each entry is a single style or a list of styles for that group's curves.
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="stable") if sort_x else np.arange(len(x))

    with plt.rc_context(PLOT_CONFIG):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax1 = ax
        fig = ax1.figure
        ax2 = None
        if y2 is not None:
            ax2 = ax1 if share_y else ax1.twinx()

        # Draw each curve directly on its axis.
        lines = []
        for g, (values, labels, axis) in enumerate([(y1, label1, ax1), (y2, label2, ax2)]):
            if axis is None:
                continue
            curves = np.atleast_2d(values)
            for i, curve in enumerate(curves):
                if isinstance(labels, str):
                    label = labels if len(curves) == 1 else f"{labels} {i + 1}"
                else:
                    label = labels[i]
                color = colors[g] if is_color_like(colors[g]) else colors[g][i]
                marker = markers[g]
                if marker is not None and not isinstance(marker, str):
                    marker = marker[i]
                line, = axis.plot(
                    x[order], curve[order], label=label, color=color, marker=marker,
                    linestyle=("-", "--", "-.", ":")[i % 4],
                    linewidth=linewidth, markersize=markersize,
                )
                lines.append(line)

        ax1.set(xlabel=xlabel, ylabel=ylabel1, title=title, xlim=xlim, ylim=ylim1)
        if ax2 is not None and not share_y:
            ax2.set(ylabel=ylabel2, ylim=ylim2)
        ax1.grid(alpha=0.2)
        ax1.legend(handles=lines)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if flag_show:
            plt.show()
    return fig, ax1, ax2
