# Spin Models

Research notebooks and Python modules for classical spin simulations, magnetic order, and driven spin dynamics. The repository brings together Monte Carlo studies of NiPS₃ and NiX₂ with semiclassical two-spin Landau–Lifshitz–Gilbert (LLG) simulations of THz-driven antiferromagnets.

## Projects

| Directory | Physical system | Main calculations |
| --- | --- | --- |
| [NiSP3_SIX_CLOCK](NiSP3_SIX_CLOCK/README.md) | NiPS₃-inspired spins on a honeycomb lattice | XY and XYZ models, sixfold anisotropy, Monte Carlo annealing, and zigzag Néel order |
| [NiX2_SPINS](NiX2_SPINS/readme.md) | NiX₂ triangular-lattice magnets, with a focus on NiI₂ | Magnetic textures, electric polarization, topological charge, LLG relaxation, and magnon spectra |
| [two_SPINS_LLG](two_SPINS_LLG/readme.md) | Canted two-sublattice antiferromagnets | Single- and two-pulse THz response, magnetization dynamics, and two-dimensional spectra |

The directory name `NiSP3_SIX_CLOCK` is retained as used in the code; the material discussed there is NiPS₃.

### NiPS₃: six-clock physics

Classical spins on a two-dimensional honeycomb lattice interact through first-, second-, and third-neighbor exchange. Twofold and sixfold anisotropies control the preferred spin directions. The notebooks compare planar XY spins with three-component XYZ spins and examine the temperature dependence of zigzag order across anisotropy parameters.

### NiX₂: magnetic textures and excitations

Two model families are included: an exchange-tensor mini model and a JKJ3B model with Heisenberg, Kitaev, third-neighbor, and biquadratic interactions. Monte Carlo simulations explore thermal observables and low-energy spin configurations. Additional workflows analyze electric polarization, topological charge, proper-screw order, and excitations around relaxed states.

### Two-spin LLG: THz-driven dynamics

A semiclassical two-sublattice model combines exchange, Dzyaloshinskii–Moriya interaction, anisotropy, and applied magnetic pulses. The Python modules provide the Hamiltonian, pulse shapes, LLG integration, plotting, and Fourier analysis. Notebooks cover single-pulse response, two-pulse response, and nonlinear 2D THz spectra. Mathematica notebooks contain supporting derivations.

## Repository structure

```text
Spin_Models_REPO/
├── readme.md
├── NiSP3_SIX_CLOCK/
│   ├── nisp3_mini_model*.ipynb       # XY model and parallel scans
│   ├── nisp3_3dxyz_model.ipynb       # Three-component spin model
│   ├── nisp3_*.py                   # Models, kernels, and MC protocols
│   ├── plotting_mc.py
│   └── Figures_NiSP3/
├── NiX2_SPINS/
│   ├── nix2_*.ipynb                 # MC, polarization, topology, magnons
│   ├── nix2_*.py                    # Models, kernels, MC, and LLG
│   ├── plotting_3d_spins.py
│   ├── Figures_NiX2/
│   └── nix2_simulation_results/
└── two_SPINS_LLG/
    ├── run_single_pulse.py
    ├── LLG-two-spin-dynamics*.ipynb
    ├── two_spin_dynamics/           # Reusable dynamics and analysis modules
    └── mma_derivation/              # Mathematica derivations
```

## Getting started
### Explore the notebooks

Launch JupyterLab from the project directory you want to explore, for example:

```bash
cd NiX2_SPINS
python -m jupyterlab
```

Use the environment created above as the notebook kernel and keep the kernel working directory in the corresponding project folder so that local imports and relative output paths resolve correctly.

| Topic | Notebook |
| --- | --- |
| NiPS₃ XY model | [nisp3_mini_model.ipynb](NiSP3_SIX_CLOCK/nisp3_mini_model.ipynb) |
| NiPS₃ anisotropy scans | [nisp3_mini_model_parallel.ipynb](NiSP3_SIX_CLOCK/nisp3_mini_model_parallel.ipynb) |
| NiPS₃ XYZ model | [nisp3_3dxyz_model.ipynb](NiSP3_SIX_CLOCK/nisp3_3dxyz_model.ipynb) |
| NiX₂ polarization | [nix2_mini_model_polarization.ipynb](NiX2_SPINS/nix2_mini_model_polarization.ipynb) |
| NiX₂ topology | [nix2_mini_model_topological_charge.ipynb](NiX2_SPINS/nix2_mini_model_topological_charge.ipynb) |
| NiX₂ JKJ3B Monte Carlo | [nix2_JKJ3B_model_mc.ipynb](NiX2_SPINS/nix2_JKJ3B_model_mc.ipynb) |
| NiX₂ relaxation and magnons | [nix2_JKJ3B_LLG_Magnon.ipynb](NiX2_SPINS/nix2_JKJ3B_LLG_Magnon.ipynb) |
| Single-pulse spin dynamics | [LLG-two-spin-dynamics-single-pulse.ipynb](two_SPINS_LLG/LLG-two-spin-dynamics-single-pulse.ipynb) |
| Two-pulse spin dynamics | [LLG-two-spin-dynamics-two-pulses.ipynb](two_SPINS_LLG/LLG-two-spin-dynamics-two-pulses.ipynb) |
| 2D THz spectra | [LLG-two-spin-dynamics-2D-THz-spectra.ipynb](two_SPINS_LLG/LLG-two-spin-dynamics-2D-THz-spectra.ipynb) |

These are research workflows with parameters configured in notebook cells. Inspect lattice sizes, temperature schedules, sampling counts, and output directories before running a full scan. In `nisp3_mini_model.ipynb`, the imports from `nisp3_mc_protocols` refer to an earlier module name; update those imports to `nisp3_mini_mc_protocols` before running that notebook.

## Results and reproducibility

The Monte Carlo workflows provide helpers for saving run parameters, numerical data, and plots in project-specific result directories. Selected figures and existing simulation outputs are included in the project folders; see their READMEs for model conventions and examples.

For comparisons between runs, record the interaction parameters, lattice size, boundary conditions, temperature schedule, equilibration and measurement sweeps, and random seed. For LLG calculations, also record units, damping, pulse parameters, integration settings, and Fourier conventions. Check convergence before interpreting quantitative results.
