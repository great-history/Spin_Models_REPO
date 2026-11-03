# Two-Spin LLG Dynamics

Semiclassical Landau-Lifshitz-Gilbert (LLG) simulations of a canted antiferromagnet represented by two sublattice spins. The model includes exchange, Dzyaloshinskii-Moriya interaction, anisotropy, and THz magnetic-field pulses. Single- and two-pulse simulations explore magnetization dynamics, nonlinear response, and 2D THz spectra.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [Single pulse](LLG-two-spin-dynamics-single-pulse.ipynb) | Spin dynamics driven by one THz pulse |
| [Two pulses](LLG-two-spin-dynamics-two-pulses.ipynb) | Two-pulse response and delay dependence |
| [Bisector polarization](LLG-two-spin-dynamics-bisector.ipynb) | Excitation with bisector-polarized fields |
| [2D THz spectra](LLG-two-spin-dynamics-2D-THz-spectra.ipynb) | Nonlinear response in the frequency domain |

## Usage

Open a notebook in Jupyter, set the model and pulse parameters, and run the cells in order. Start with the single-pulse notebook.

The [two_spin_dynamics](two_spin_dynamics/) package contains the Hamiltonian, LLG solver, pulse definitions, spectral analysis, and plotting functions. Symbolic derivations are in [mma_derivation](mma_derivation/).

Dependencies: NumPy, SciPy, Matplotlib, and Jupyter.
