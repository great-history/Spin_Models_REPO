"""
    Save NiX2 parameter scans, spin configurations, and arbitrary observables.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from plotting_3d_spins import plot_spin_config, SPIN_TEXTURE_SIGNED_CMAP


def create_run_folder(root="nix2_simulation_results", *, parameters=None,
                      prefix="NiX2_MC"):
    """Create a timestamped run folder and save common parameters as JSON."""
    folder = Path(root) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True)
    with (folder / "parameters.json").open("w", encoding="utf-8") as f:
        json.dump(parameters or {}, f, indent=2, ensure_ascii=False,
                  default=lambda v: str(v) if isinstance(v, Path) else np.asarray(v).tolist())
    return folder


def save_scan_point(folder, mc, index, *, parameters=None, observables=None,
                    save_spins=True, save_plots=True, dpi=200):
    """
        Save one point of any scan, including multidimensional scans.
    """
    folder = Path(folder)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    path = folder / "data" / f"{index:05d}.npz"
    data = {"index": index}
    data.update({f"parameter__{k}": np.asarray(v)
                 for k, v in (parameters or {}).items()})
    data.update({f"observable__{k}": np.asarray(v)
                 for k, v in (observables or {}).items()})
    if save_spins or save_plots:
        spins = np.asarray(mc.spin_config)
        coords = mc.lat.get_cartesian_coords()
    if save_spins:
        data.update(spins=spins, coordinates=coords)
    np.savez_compressed(path, **data)

    if save_plots:
        title = ", ".join(f"{k} = {v}" for k, v in (parameters or {}).items())
        save_spin_plot(folder / "spins_configuration" / f"{index:05d}.png",
                       spins, coords, title=title or f"Point {index}", dpi=dpi)
    return path


def save_temperature_results(folder, mc, index, temperature_K, observables=None,
                             *, parameters=None, **kwargs):
    """Temperature-scan shorthand; kwargs are passed to save_scan_point."""
    params = dict(parameters or {}, temperature_K=temperature_K)
    return save_scan_point(folder, mc, index, parameters=params,
                           observables=observables, **kwargs)


def save_field_results(folder, mc, index, B_field_T, observables=None,
                       *, parameters=None, **kwargs):
    """Field-scan shorthand; B_field_T may be a scalar or [Bx, By, Bz].

    A scalar denotes the signed field along your chosen scan direction.
    This records the supplied field; it does not update the Hamiltonian.
    """
    params = dict(parameters or {}, B_field_T=B_field_T)
    return save_scan_point(folder, mc, index, parameters=params,
                           observables=observables, **kwargs)


def load_scan_point(path):
    """Return index, parameters, observables, and optional spins/coordinates."""
    with np.load(path, allow_pickle=False) as data:
        result = {"index": int(data["index"]), "parameters": {}, "observables": {}}
        for key in data.files:
            if key.startswith("parameter__"):
                result["parameters"][key.removeprefix("parameter__")] = data[key]
            elif key.startswith("observable__"):
                result["observables"][key.removeprefix("observable__")] = data[key]
            elif key != "index":
                result[key] = data[key]
    return result


def save_spin_plot(path, spins, coordinates, *, title="Spin configuration", dpi=200,
                   figsize=(10, 7), arrow_scale=1.8, arrow_width=0.025,
                   cmap=SPIN_TEXTURE_SIGNED_CMAP):
    """Save black XY spin arrows over an interpolated Sz background."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lattice = SimpleNamespace(get_cartesian_coords=lambda: np.asarray(coordinates))
    ax = plot_spin_config(
        spins, lattice, title=title, figsize=figsize, cmap=cmap,
        arrow_scale=arrow_scale, arrow_width=arrow_width, flag_show=False,
    )
    fig = ax.figure
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_observable_curve(folder, observable, *, x="temperature_K",
                          reducer=lambda values: np.mean(values, axis=0),
                          x_component=None, select=None, labels=None, name=None,
                          xlabel=None, ylabel=None, save_plots=True, dpi=200):
    """Export a scalar/vector observable versus a scan parameter as CSV/PNG.
    """
    folder = Path(folder)
    rows = []
    for path in sorted((folder / "data").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            params = {k.removeprefix("parameter__"): data[k]
                      for k in data.files if k.startswith("parameter__")}
            if select is not None and not select(params):
                continue
            xvalue = params[x] if x_component is None else params[x][x_component]
            values = data[f"observable__{observable}"]
            values = values if reducer is None else reducer(values)
            rows.append([int(data["index"]), float(xvalue), *np.asarray(values).ravel()])
    if not rows:
        return None
    table = np.asarray(sorted(rows, key=lambda row: row[0]))
    xname = x if x_component is None else f"{x}_{x_component}"
    if labels is None:
        labels = ([observable] if table.shape[1] == 3 else
                  [f"{observable}_{i}" for i in range(table.shape[1] - 2)])
    name = name or f"{observable}_vs_{xname}"
    csv_path = folder / "data" / f"{name}.csv"
    np.savetxt(csv_path, table, delimiter=",", comments="",
               header=",".join(["index", xname, *labels]))
    if save_plots:
        (folder / "observables").mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for i, label in enumerate(labels):
            ax.plot(table[:, 1], table[:, i + 2], "o-", ms=3, label=label)
        ax.set(xlabel=xlabel or xname, ylabel=ylabel or observable)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(folder / "observables" / f"{name}.png", dpi=dpi)
        plt.close(fig)
    return csv_path
