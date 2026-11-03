"""
    Protocols for Monte Carlo simulations of NiSP3 spin models.
"""

import numpy as np

KB_MEV_PER_K = 0.08617333262
from nisp3_mc_output import save_temperature_results, save_neel_curve


def set_temperatures(TK_init:float, TK_final:float, 
                     cooling_method="linear", *, cooling_rate=0.9, num_steps=10):
    if TK_init < TK_final:
        raise ValueError("initial T must be greater than final T")
    if TK_final < 0:
        raise ValueError("final T must be non-negative")
    
    temperatures_in_K = []
    # temperatures_in_meV = []
    if cooling_method == "geometric":
        t = TK_init
        while t > TK_final:
            temperatures_in_K.append(t)
            # temperatures_in_meV.append(KB_MEV_PER_K * t)
            t *= cooling_rate
    elif cooling_method == "linear":
        new_temperatures = [TK_init - i * (TK_init - TK_final) / (num_steps - 1) for i in range(num_steps)]
        temperatures_in_K.extend(new_temperatures)

    return temperatures_in_K


def set_temperature_schedule(TK_init_list, TK_final_list, 
                             cooling_method_list, cooling_rate_list):
    """
        join temperature segments in K. For linear segments, rate means number of points.
    """
    TK_list = []
    for i in range(len(TK_init_list)):
        if cooling_method_list[i] == "linear":
            temperatures = set_temperatures(
                TK_init_list[i], TK_final_list[i], "linear",
                num_steps=cooling_rate_list[i],
            )
        elif cooling_method_list[i] == "geometric":
            temperatures = set_temperatures(
                TK_init_list[i], TK_final_list[i], "geometric",
                cooling_rate=cooling_rate_list[i],
            )
        else:
            continue
        TK_list.extend(temperatures)
    return TK_list


def run_mc_annealing(mc, temperatures_K, run_folder=None, *,
                     num_eq_sweeps=15000, num_meas_sweeps=5000,
                     converg_per_every=10, meas_per_every=5, save_plots=True,
                     verbose=True):
    """Run each temperature in K; update mc spins in place and return measurements."""
    results = {
        "temperatures_K": [],
        "neel_norm_of_mean": [],
        "neel_samples_by_temperature": [],
        "neel_squared_temperature_list": [],
        "convergence_data": [],
    }
    mc.set_proposal_angle(T_in_meV=temperatures_K[0] * KB_MEV_PER_K)

    for index, temperature_K in enumerate(temperatures_K):
        temperature_meV = temperature_K * KB_MEV_PER_K
        convergence = []
        neel_vectors = []

        # Equilibration sweeps.
        for sweep in range(num_eq_sweeps):
            mc.mc_sweep(temperature_meV)
            if sweep % converg_per_every == 0:
                convergence.append(mc.total_energy())

        # Measurement sweeps.
        for sweep in range(num_meas_sweeps):
            mc.mc_sweep(temperature_meV)
            if sweep % meas_per_every == 0:
                neel_vectors.append(mc.get_neel_vectors().copy())

        samples = np.array(neel_vectors)
        results["temperatures_K"].append(temperature_K)
        results["neel_norm_of_mean"].append(np.linalg.norm(samples.mean(axis=0), axis=-1))
        results["neel_samples_by_temperature"].append((temperature_meV, samples))
        results["neel_squared_temperature_list"].append(
            (temperature_meV, np.sum(samples**2, axis=(1, 2)).mean())
        )
        results["convergence_data"].append(np.array(convergence))
        if verbose:
            print(f"T = {temperature_K:.2f} K, E = {mc.total_energy():.6f} meV")

        if run_folder is not None:
            save_temperature_results(
                run_folder, mc, index, temperature_K, samples, save_plots=save_plots,
            )

    # Save the full curve once, after all temperatures finish.
    if run_folder is not None:
        save_neel_curve(run_folder, save_plots=save_plots)
    return results


def run_one_mc(Nx, Ny, K2, K6, seed, temperatures_K, 
               num_eq_sweeps, num_meas_sweeps, 
               converg_per_every=10, meas_per_every=5):
    from nisp3_mini_model_builders import build_nisp3_lattice, build_nisp3_mini_model
    from nisp3_mini_mc import MonteCarlo2DXY
    from nisp3_mc_output import create_run_folder

    # 每個進程建立自己的模型和 spins
    lattice = build_nisp3_lattice(Nx, Ny, geometry="rectangular")
    model = build_nisp3_mini_model(
        J1=-3.8, J2iso=0.2, J3=13.8,
        K2=K2, K6=K6,
    )
    mc = MonteCarlo2DXY(lattice, model, seed=seed)

    folder = create_run_folder(
        prefix=f"NiSP3_K2_{K2}_K6_{K6}",
        parameters={
            "K2": K2, "K6": K6, "seed": seed,
            "Nx": Nx, "Ny": Ny,
            "temperatures_K": temperatures_K,
            "num_eq_sweeps": num_eq_sweeps,
            "num_meas_sweeps": num_meas_sweeps,
        },
    )

    run_mc_annealing(
        mc, temperatures_K, folder,
        num_eq_sweeps=num_eq_sweeps,
        num_meas_sweeps=num_meas_sweeps,
        converg_per_every=converg_per_every, 
        meas_per_every=meas_per_every, 
        save_plots=True,
        verbose=False, # set to False for parallel execution
    )

    
    return folder
