"""Plot NiX2 spins (Nx, Ny, basis, 3) and observable curves."""

from pathlib import Path
from matplotlib.colors import is_color_like
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

XYZ_COLORS = ["tab:red", "tab:orange", "tab:green"]
SPIN_CMAP = "hsv"

# Reference-inspired saturated colors; positions correspond to (Sz + 1) / 2.
SPIN_TEXTURE_CMAP = LinearSegmentedColormap.from_list(
    "spin_rainbow",
    [(0.00, "#1818ff"), (0.25, "#a6dce6"),
     (0.40, "#20ff00"), (0.55, "#20ff00"),
     (0.75, "#ffff00"), (0.90, "#ff8800"), (1.00, "#ff2200")],
    N=256,
)


# Sign-resolving alternative: negative Sz is cool, positive Sz is warm.
# Retain the original blue/red endpoints and yellow at Sz = +0.5.
# Odd LUT size puts neutral white exactly at Sz = 0 for Normalize(-1, 1).
SPIN_TEXTURE_SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "spin_signed",
    [(0.00, "#1818ff"), (0.25, "#00bfff"),
     (0.375, "#30e8ed"), (0.50, "#ffffff"),
     (0.625, "#ffe680"), (0.75, "#ffff00"),
     (0.90, "#ff8800"), (1.00, "#ff2200")],
    N=257,
)


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
    arrow_scale=1.8, arrow_width=0.025, colorbar=True,
    flag_show=True, save_path=None, cmap=SPIN_TEXTURE_SIGNED_CMAP,
):
    """Plot black XY spin projections over an interpolated Sz background."""
    ax = plot_spin_texture(
        spins, lattice, ax=ax, title=title, xlim=xlim, ylim=ylim,
        figsize=figsize, cmap=cmap, arrow_scale=arrow_scale,
        arrow_width=arrow_width, colorbar=colorbar, neighbors=neighbors,
    )
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path)
    if flag_show:
        plt.show()
    return ax


def plot_spin_texture(
    spins, lattice, ax=None, title="Spin texture",
    xlim=None, ylim=None, figsize=(10, 8), cmap=SPIN_TEXTURE_CMAP,
    arrow_scale=1.8, arrow_width=0.045, colorbar=True, neighbors=None,
):
    """Plot interpolated Sz colors and black arrows retaining the XY projection magnitude."""
    spins = np.asarray(spins)
    coords = lattice.get_cartesian_coords()
    mask = np.ones(spins.shape[:3], dtype=bool)
    if xlim is not None:
        mask &= (coords[..., 0] >= xlim[0]) & (coords[..., 0] <= xlim[1])
    if ylim is not None:
        mask &= (coords[..., 1] >= ylim[0]) & (coords[..., 1] <= ylim[1])
    points = coords.reshape(-1, 2)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    triangulation = Triangulation(points[:, 0], points[:, 1])
    sz_values = spins[..., 2].ravel()
    refined_mesh, refined_sz = UniformTriRefiner(triangulation).refine_field(
        sz_values, triinterpolator=LinearTriInterpolator(triangulation, sz_values),
        subdiv=3,
    )
    background = ax.tripcolor(
        refined_mesh, refined_sz,
        shading="gouraud", cmap=cmap, vmin=-1, vmax=1, zorder=0,
    )
    if neighbors is not None:
        _draw_bonds(ax, coords, mask, neighbors)
    selected_coords = coords[mask]
    selected_spins = spins[mask]
    ax.quiver(
        selected_coords[:, 0], selected_coords[:, 1],
        selected_spins[:, 0], selected_spins[:, 1],
        color="black", angles="xy", scale_units="xy", scale=arrow_scale,
        pivot="mid", units="xy", width=arrow_width, minlength=0, zorder=2,
    )
    if xlim is None:
        xlim = (coords[..., 0].min() - 0.6, coords[..., 0].max() + 0.6)
    if ylim is None:
        ylim = (coords[..., 1].min() - 0.6, coords[..., 1].max() + 0.6)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", title=title,
           xlabel="x (lattice units)", ylabel="y (lattice units)")
    if colorbar:
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.12)
        cbar = ax.figure.colorbar(background, cax=cax)
        cbar.set_label(r"$S_z$")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    return ax


def plot_Sq(
    qx, qy, S_q, *, figsize=None, cmap="inferno",
    xlim=None, ylim=None, vmax=None, colorbar=True,
    title="Final spin configuration", flag_show=True, save_path=None,
):
    """Plot S_q shaped (Nx, Ny) or (Nx, Ny, 3) on matching Cartesian qx, qy grids."""
    sq = np.asarray(S_q)
    component_resolved = sq.ndim == 3
    maps = np.moveaxis(sq, -1, 0) if component_resolved else sq[None, ...]
    labels = [r"$S_x(\mathbf{q})$", r"$S_y(\mathbf{q})$", r"$S_z(\mathbf{q})$"] if component_resolved else [r"$S(\mathbf{q})$"]
    if figsize is None:
        figsize = (16, 5) if component_resolved else (7, 6)
    if vmax is None:
        vmax = float(sq.max()) or 1.0
    with plt.rc_context(PLOT_CONFIG):
        fig, axes = plt.subplots(1, len(maps), figsize=figsize,
                                 squeeze=False, sharey=True, constrained_layout=True)
        for ax, values, label in zip(axes.flat, maps, labels):
            background = ax.pcolormesh(qx, qy, values, shading="auto",
                                       cmap=cmap, vmin=0, vmax=vmax)
            ax.set(aspect="equal", title=label, xlim=xlim, ylim=ylim,
                   xlabel=r"$q_x$ (inverse lattice units)")
            if ax is not axes[0, 0]:
                ax.tick_params(axis="y", left=False, labelleft=False)
        axes[0, 0].set_ylabel(r"$q_y$ (inverse lattice units)")
        if colorbar:
            cax = axes[0, -1].inset_axes([1.04, 0, 0.03, 1])
            colorbar_label = r"$S_\alpha(\mathbf{q})$" if component_resolved else labels[0]
            fig.colorbar(background, cax=cax, label=colorbar_label)
        fig.suptitle(title)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        if flag_show:
            plt.show()
    return fig, axes.ravel()


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
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.12)
        cbar = ax.figure.colorbar(points, cax=cax)
        cbar.set_label(r"Spin angle $\phi$ (rad)")
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    return ax


def plot_neel_order(
    temperatures_K, order, ax=None, order_power=2,
    label=None, linestyle="-", marker="o",
    title="Zigzag order vs temperature", xlim=None, ylim=(0, 1.05),
    figsize=(8, 5), legend=True,
):
    """Plot three thermal Neel moments against temperature in kelvin.

    order must have shape (number of temperatures, 3).
    For order_power=1, supply <|L_a|>; for 2, supply <|L_a|^2>.
    This function does not square or average the supplied values.
    Set label="K6 = 6 meV" to append a run label to each zigzag mode.
    Reuse ax with a different linestyle to compare multiple runs.
    """
    temperatures_K = np.asarray(temperatures_K)
    order = np.asarray(order)
    sort_index = np.argsort(temperatures_K, kind="stable")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for mode, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        curve_label = f"Zigzag {mode + 1}"
        if label is not None:
            curve_label += f", {label}"
        ax.plot(temperatures_K[sort_index], order[sort_index, mode],
                color=color, linestyle=linestyle, marker=marker,
                markersize=3, linewidth=1.3, label=curve_label)
    ylabel = (r"$\langle|\mathbf{L}_a|\rangle$" if order_power == 1 else
              r"$\langle|\mathbf{L}_a|^2\rangle$")
    ax.set(xlabel="Temperature (K)", ylabel=ylabel, title=title,
           xlim=xlim, ylim=ylim)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend()
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
            for axis, side, style in ((ax1, "left", colors[0]), (ax2, "right", colors[1])):
                color = style if is_color_like(style) else style[0]
                axis.yaxis.label.set_color(color)
                axis.tick_params(axis="y", colors=color)
                axis.spines[side].set_color(color)
        ax1.grid(alpha=0.2)
        ax1.legend(handles=lines)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if flag_show:
            plt.show()
    return fig, ax1, ax2


def plot_two_curves(x, y1, y2=None, *, flag_show=True, **kwargs):
    """Compatibility name for plot_curves, with the original show/close behavior.

    See plot_curves for styling options. flag_show=False closes the figure
    to suppress notebook display; the returned figure can still be saved.
    """
    fig, ax1, ax2 = plot_curves(x, y1, y2, flag_show=flag_show, **kwargs)
    if not flag_show:
        plt.close(fig)
    return fig, ax1, ax2
