"""Save spin maps and Neel curves. Samples: (measurements, 3, 2 or 3); temperature in K."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_run_folder(root="nisp3_simulation_results", *, 
                      parameters=None, prefix="NiSP3_NEEL"):
    """Create a timestamped folder and save parameters.json."""
    folder = Path(root) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True)
    with (folder / "parameters.json").open("w", encoding="utf-8") as f:
        json.dump(parameters or {}, f, indent=2, ensure_ascii=False,
                  default=lambda v: str(v) if isinstance(v, Path) else np.asarray(v).tolist())
    return folder

def create_3dxyz_run_folder(root="nisp3_3dxyz_simulation_results", *, 
                      parameters=None, prefix="NiSP3_NEEL"):
    """Create a timestamped folder and save parameters.json."""
    folder = Path(root) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True)
    with (folder / "parameters.json").open("w", encoding="utf-8") as f:
        json.dump(parameters or {}, f, indent=2, ensure_ascii=False,
                  default=lambda v: str(v) if isinstance(v, Path) else np.asarray(v).tolist())
    return folder


def save_temperature_results(folder, mc, index, temperature_K, neel_vectors, *,
                             save_plots=True, dpi=200):
    """Save full spins and Neel samples; plot the XY projection of the spins."""
    folder = Path(folder)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    (folder / "spins_configuration").mkdir(parents=True, exist_ok=True)
    path = folder / "data" / f"{index:05d}.npz"
    spins = np.asarray(mc.spin_config)
    coords = mc.lat.get_cartesian_coords()
    samples = np.asarray(neel_vectors)

    np.savez_compressed(
        path, index=index, temperature_K=temperature_K,
        spins=spins, coordinates=coords, neel_vectors=samples,
        neel_norm_of_mean=np.linalg.norm(samples.mean(axis=0), axis=-1),
    )

    if save_plots:
        xy = coords.reshape(-1, 2)
        # Keep one vector per site, including for XYZ spins.
        vectors = spins[..., :2].reshape(-1, 2)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0]) % (2 * np.pi)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        arrows = ax.quiver(
            xy[:, 0], xy[:, 1], vectors[:, 0], vectors[:, 1], angles,
            cmap="hsv", clim=(0, 2 * np.pi), angles="xy",
            scale_units="xy", scale=2.5, pivot="mid",
        )
        ax.set(aspect="equal", xlabel="x (lattice units)",
               ylabel="y (lattice units)", title=f"Spin XY projection: T = {temperature_K:g} K")
        ax.margins(0.05)
        fig.colorbar(arrows, ax=ax, label="Spin angle (rad)", shrink=0.8)
        fig.savefig(folder / "spins_configuration" / f"{index:05d}.png", dpi=dpi)
        plt.close(fig)

    return path


def save_neel_curve(folder, *, save_plots=True, dpi=200):
    """Call after the temperature loop to export the Neel curve as CSV/PNG."""
    folder = Path(folder)
    rows = []
    for path in (folder / "data").glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            order = np.linalg.norm(data["neel_vectors"].mean(axis=0), axis=-1)
            rows.append([int(data["index"]), float(data["temperature_K"]), *order])
    table = np.asarray(sorted(rows, key=lambda row: row[0]))
    csv_path = folder / "data" / "neel_vs_temperature.csv"
    np.savetxt(csv_path, table, delimiter=",", comments="",
               header="index,temperature_K,norm_mean_L1,norm_mean_L2,norm_mean_L3")

    if save_plots:
        (folder / "neel_vectors").mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for i in range(3):
            ax.plot(table[:, 1], table[:, 2 + i], "o-", ms=3, label=f"Zigzag {i + 1}")
        ax.set(xlabel="Temperature (K)", ylabel=r"$|\langle \mathbf{L}_i \rangle|$",
               title="Neel order vs temperature")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(folder / "neel_vectors" / "neel_vs_temperature.png", dpi=dpi)
        plt.close(fig)
    return csv_path
