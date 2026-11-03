"""Minimal A-on/B-off semiclassical LLG simulation using ``spin_dynamics``."""

import numpy as np
import matplotlib.pyplot as plt

from two_spin_dynamics.llg import canted_afm_ground_state, run_llg
from two_spin_dynamics.pulses import d2gdt2_pulse


params = {
    "Stot": 2.5,
    "nn": 6,
    "Jex": 4.6,
    "Dvec": np.array([0.0, 0.1066, 0.0]),
    "Kvec": np.array([0.00905, 0.0, 0.9847 / (15.0 + 62.38) + 0.00265]),
    "gamma": 1.0,  # Replace by a correctly unit-converted value for physical ps dynamics.
}
alpha = 0.01
time = np.linspace(0.0, 20.0, 2000)  # ps

pulse_params = {
    "A": {
        "shape_params": {"t0": 0.0, "sigma": 0.5, "h_peak": 0.15},
        "polarization": np.array([1.0, 0.0, 0.0]),
        "state": "on",
    },
    "B": {
        "shape_params": {"t0": 4.4, "sigma": 0.5, "h_peak": 0.15},
        "polarization": np.array([1.0, 0.0, 0.0]),
        "state": "off",
    },
}

s1_0, s2_0, theta = canted_afm_ground_state(params)
result = run_llg(time, s1_0, s2_0, params, alpha, pulse_params, d2gdt2_pulse)

fig, ax = plt.subplots(figsize=(8, 3.8), dpi=180)
for component, (color, label) in enumerate(zip(("tab:red", "tab:green", "tab:blue"), (r"$M_a$", r"$M_b$", r"$M_c$"))):
    ax.plot(result["time"], result["M"][:, component], color=color, label=label)
ax.set(xlabel="Time (ps)", ylabel="Net magnetization", title="LLG: pulse A on, pulse B off")
ax.grid(alpha=0.3)
ax.legend()
plt.show()

print(f"Canted angle theta = {np.degrees(theta):.5f} deg")
