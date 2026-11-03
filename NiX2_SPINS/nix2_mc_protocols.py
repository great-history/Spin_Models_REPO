"""Simple NiX2 temperature scans: equilibrate, measure, then save each point."""

import numpy as np

from nix2_mc import KB_MEV_PER_K, MU_S_MEV_PER_T, MonteCarloNiX2
from nix2_mc_output import create_run_folder, save_temperature_results, save_observable_curve


OBSERVABLE_ALIASES = {
    "Q": "topological_charge", 
    "E": "energy",
    "P": "electric_polarization", 
    "M": "magnetization",
    "Cv": "heat_capacity", 
    "susceptibility": "chi",
    "magnetization_norm": "magnetization",
    "topological_charge_squared": "topological_charge",
}


def set_temperatures(TK_init, TK_final, cooling_method="linear", *,
                     cooling_rate=0.9, num_steps=10):
    """Return temperatures in K, including the final endpoint only for linear scans."""
    if cooling_method == "linear":
        return np.linspace(TK_init, TK_final, num_steps).tolist()
    # Without these bounds a geometric cooling loop may never terminate.
    if cooling_method != "geometric" or not (TK_init >= TK_final > 0 and 0 < cooling_rate < 1):
        raise ValueError("Geometric cooling needs initial >= final > 0 and 0 < rate < 1.")
    temperatures = []
    t = TK_init
    while t > TK_final:
        temperatures.append(t)
        t *= cooling_rate
    return temperatures


def set_temperature_schedule(TK_init_list, TK_final_list,
                             cooling_method_list, cooling_rate_list):
    """Join segments; for linear segments, rate is the number of points."""
    temperatures = []
    for start, stop, method, rate in zip(
            TK_init_list, TK_final_list, cooling_method_list, cooling_rate_list):
        temperatures.extend(set_temperatures(
            start, stop, method, cooling_rate=rate, num_steps=int(rate)))
    return temperatures


def make_measurements(mc, observables=("energy", "magnetization"), *, extra_observables=None):
    """measure some observables from the Monte Carlo simulation."""
    names = list(dict.fromkeys(OBSERVABLE_ALIASES.get(name, name) for name in observables))
    if "heat_capacity" in names and "energy" not in names:
        names.append("energy")
    if "chi" in names and "topological_charge" not in names:
        names.append("topological_charge")
    if mc.lat.Ns not in (1, 2) and "topological_charge" in names:
        raise ValueError("Q requires triangular (Ns=1) or rectangular (Ns=2) geometry.")
    available = {
        "energy": lambda mc: {"energy": mc.total_energy()},
        "magnetization": lambda mc: dict(zip(
            ("magnetization_norm", "magnetization"), mc.get_magnetization())),
        "topological_charge": lambda mc: dict(zip(
            ("topological_charge", "topological_charge_squared"), mc.topological_charge())),
        "electric_polarization": lambda mc: {"electric_polarization": mc.electric_polarization()},
    }
    measurements = {name: available[name] for name in names
                    if name not in ("heat_capacity", "chi", "spin_structure_factor")}
    measurements.update(extra_observables or {})
    return measurements


def run_mc_annealing(mc, temperatures_K_list, run_folder=None, *,
                     num_eq_sweeps=15000, num_meas_sweeps=5000,
                     converg_per_every=10, meas_per_every=5,
                     observables=["energy", "magnetization", "heat_capacity"],
                     extra_observables=None, Nmuc=1, save_plots=True, verbose=True):
    """Equilibrate, measure, and save NiX2 results at each temperature."""
    temperatures_K_list = np.asarray(temperatures_K_list, dtype=float)
    if np.any(temperatures_K_list <= 0) or num_eq_sweeps < 0 or min(num_meas_sweeps, meas_per_every, converg_per_every) <= 0:
        raise ValueError("Use positive temperatures and sampling intervals, eq >= 0, meas > 0.")
    names = list(dict.fromkeys(OBSERVABLE_ALIASES.get(name, name) for name in observables))
    measurements = make_measurements(mc, names, extra_observables=extra_observables)
    results = {"temperatures_K_list": temperatures_K_list, "samples": {}, "means": {}, "stds": {},
               "convergence_data": []}
    sweep_mc = mc.mc_sweep

    for index, temperature_K in enumerate(temperatures_K_list):
        temperature_meV = temperature_K * KB_MEV_PER_K
        mc.current_temperature_in_K = temperature_K
        mc.current_temperature_in_meV = temperature_meV
        mc.current_B_field = mc.ham.zeeman / MU_S_MEV_PER_T
        if mc._proposal_angle is None:
            mc.set_proposal_angle(T_in_meV=temperature_meV)
        mc._sweeps_since_adjust = 0
        convergence = []
        samples = {}

        # ================================
        # equilibration sweeps
        # ================================
        for sweep in range(num_eq_sweeps):
            sweep_mc(temperature_meV)
            if sweep % converg_per_every == 0:
                convergence.append(mc.total_energy())

        # ================================
        # measurement sweeps
        # ================================
        for sweep in range(num_meas_sweeps):
            sweep_mc(temperature_meV)
            if sweep % meas_per_every == 0:
                for name, measure in measurements.items():
                    values = measure(mc)
                    if not isinstance(values, dict):
                        values = {name: values}
                    for key, value in values.items():
                        samples.setdefault(key, []).append(np.array(value, copy=True))

        # ================================
        # statistical analysis of samples
        # ================================
        samples = {name: np.asarray(values) for name, values in samples.items()}
        means = {name: values.mean(axis=0) for name, values in samples.items()}
        stds = {name: values.std(axis=0) for name, values in samples.items()}
        if "chi" in names:
            # Match the notebook: fluctuations of |Q| per magnetic unit cell.
            means["chi"] = np.abs(samples["topological_charge"]).std()**2 / (Nmuc**4 * temperature_meV)
        if "heat_capacity" in names:
            # Cv per spin in units of k_B; energy samples are total energies in meV.
            means["heat_capacity"] = stds["energy"]**2 / (mc.lat.N_total * temperature_meV**2)
        if "spin_structure_factor" in names:
            q1, q2, structure = mc.spin_structure_factor()
            results.setdefault("spin_structure_factor", []).append(structure)
            results.update(q1=q1, q2=q2)
        for name, values in samples.items():
            results["samples"].setdefault(name, []).append(values)
        for name, value in means.items():
            results["means"].setdefault(name, []).append(value)
        for name, value in stds.items():
            results["stds"].setdefault(name, []).append(value)
        results["convergence_data"].append(np.asarray(convergence))

        # ================================
        # save data
        # ================================
        if run_folder is not None:
            data = dict(samples, convergence_energy=np.asarray(convergence))
            data.update({f"{name}_mean": value for name, value in means.items()})
            data.update({f"{name}_std": value for name, value in stds.items()})
            for name in ("chi", "heat_capacity"):
                if name in means:
                    data[name] = means[name]
            if "spin_structure_factor" in names:
                data.update(q1=q1, q2=q2, spin_structure_factor=structure)
            save_temperature_results(
                run_folder, mc, index, temperature_K, data,
                parameters={"B_field_T": mc.current_B_field, "Nmuc": Nmuc}, save_plots=save_plots)
            
        # ================================
        # print results
        # ================================
        if verbose:
            summary = ", ".join(
                f"{name} = {np.array2string(np.asarray(value), precision=6)}"
                for name, value in means.items()
            )
            print(f"T = {temperature_K:.2f} K, {summary}")

    for group in ("means", "stds"):
        results[group] = {name: np.asarray(values) for name, values in results[group].items()}
    for short, name in (("Q", "topological_charge"), ("E", "energy"),
                        ("P", "electric_polarization"), ("M", "magnetization"),
                        ("chi", "chi"), ("Cv", "heat_capacity")):
        if name in results["means"]:
            results[f"{short}_data"] = results["means"][name] / Nmuc**2 if short == "Q" else results["means"][name]
    if "spin_structure_factor" in results:
        results["spin_structure_factor"] = np.asarray(results["spin_structure_factor"])
    if run_folder is not None:
        for name, values in results["means"].items():
            if values.ndim <= 2:  # Scalar/vector curves; do not flatten maps.
                save_observable_curve(run_folder, f"{name}_mean", reducer=None,
                                      name=f"{name}_vs_temperature_K", ylabel=name,
                                      save_plots=save_plots)
    return results


def run_one_mc(Nx, Ny, model_parameters, seed, temperatures_K_list,
               num_eq_sweeps=15000, num_meas_sweeps=5000,
               converg_per_every=10, meas_per_every=5, *,
               observables=("energy", "magnetization", "heat_capacity"),
               extra_observables=None, Nmuc=1,
               save_plots=True, verbose=False,
               geometry=None, M01=None, proposal_angle=None,
               root="nix2_simulation_results"):
    """Build an independent NiX2 run, save its results, and return its unique folder."""
    from secrets import randbits
    from uuid import uuid4
    from nix2_model_builders import build_nix2_lattice, build_nix2_model, prepare_M_tensors

    temperatures_K_list = list(temperatures_K_list)
    model_parameters = dict(model_parameters)
    stored_geometry = model_parameters.pop("geometry", "rectangular")
    geometry = stored_geometry if geometry is None else geometry
    if "J1mat" in model_parameters:
        model_parameters.setdefault("J1", model_parameters.pop("J1mat"))
    seed = randbits(128) if seed is None else int(seed)
    options = dict(converg_per_every=converg_per_every, meas_per_every=meas_per_every,
                   observables=list(observables), extra_observables=extra_observables,
                   Nmuc=Nmuc, save_plots=save_plots, verbose=verbose)
    if M01 is None and any(name in ("P", "electric_polarization") for name in options["observables"]):
        raise ValueError("Pass M01 when measuring electric polarization.")

    lattice = build_nix2_lattice(Nx, Ny, geometry=geometry)
    model = build_nix2_model(**model_parameters, geometry=geometry)
    if M01 is not None:
        prepare_M_tensors(model, lattice, M01, lat_geom=geometry)
    mc = MonteCarloNiX2(lattice, model, seed=seed)
    if temperatures_K_list:
        mc.set_proposal_angle(proposal_angle=proposal_angle,
                              T_in_meV=temperatures_K_list[0] * KB_MEV_PER_K)

    parameters = dict(model_parameters, Nx=Nx, Ny=Ny, geometry=geometry,
                      seed=seed, M01=M01, proposal_angle=proposal_angle,
                      temperatures_K_list=temperatures_K_list,
                      num_eq_sweeps=num_eq_sweeps, num_meas_sweeps=num_meas_sweeps)
    parameters.update({k: v for k, v in options.items() if k != "extra_observables"})
    parameters["extra_observables"] = list(options["extra_observables"] or {})
    folder = create_run_folder(root, prefix=f"NiX2_MC_{uuid4().hex}", parameters=parameters)
    run_mc_annealing(mc, temperatures_K_list, folder,
                     num_eq_sweeps=num_eq_sweeps, num_meas_sweeps=num_meas_sweeps,
                     **options)
    return folder
