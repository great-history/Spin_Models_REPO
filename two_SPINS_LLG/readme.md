# Two-spin semiclassical LLG simulation workflow for 2D THz spectra

## Purpose and scope

This workflow specifies a reproducible semiclassical simulation of the nonlinear response of a canted antiferromagnet driven by two phase-stable THz magnetic-field pulses.  The model contains two uniform sublattice spins, evolves them with the Landau-Lifshitz-Gilbert (LLG) equation, and converts the nonlinear magnetization response into a two-dimensional (2D) THz spectrum.

It is designed for the ErFeO3 / YFeO3-style model in the accompanying papers.  It deliberately describes the workflow and acceptance checks only; implementation details and code are out of scope.

## 1. Fix conventions before any calculation

Record these choices in a run configuration file or a notebook header.  Do not mix conventions between runs.

| Item | Recommended convention |
| --- | --- |
| Crystallographic axes | Cartesian `(a, b, c)` mapped to `(x, y, z)` |
| Sublattice spin length | `|S1| = |S2| = S`; check it remains constant numerically |
| Exchange convention | Antiferromagnet: `J > 0` in `n J S1 . S2` |
| Effective DM vector | `D = (0, Db, 0)`, i.e. only the `b` component is retained in the uniform two-spin model |
| Equilibrium order | staggered moment mainly along `a`; weak net moment mainly along `c` |
| THz pulse B | centred at `t = 0` |
| THz pulse A | centred at `t = tau`; explicitly state whether positive `tau` means A arrives after or before B |
| Measured quantity | a selected component of `M(t,tau) = S1 + S2`, projected on the THz detection polarization |
| Fourier convention | state the signs and units in `exp(+/- i 2 pi f t)`; retain positive and negative excitation frequencies |

Use one internally consistent unit system.  A practical choice is energy in meV, time in ps, field in T, and an explicitly converted gyromagnetic ratio in rad ps^-1 T^-1.  Verify the conversion once by reproducing a known small-amplitude qFM or qAFM frequency.

## 2. Define the static two-sublattice model

Use the Hamiltonian

$$
\mathcal H_0 = nJ\,\mathbf S_1\!\cdot\!\mathbf S_2
+ n\,\mathbf D\!\cdot\!(\mathbf S_1\!\times\!\mathbf S_2)
- \sum_{i=1,2}\left(K_a S_{ia}^2+K_c S_{ic}^2\right),
$$

with `D = (0, Db, 0)`.  Add the two-pulse Zeeman term

$$
\mathcal H_Z(t,\tau)=-\gamma\,[\mathbf H_A(t,\tau)+\mathbf H_B(t)]\cdot(\mathbf S_1+\mathbf S_2).
$$

For each material and temperature, specify `S`, `n`, `J`, `Db`, `Ka`, `Kc`, the Gilbert damping `alpha`, and `gamma`.  If temperature dependence is represented only through anisotropy, state that modelling assumption clearly.

### Static acceptance checks

1. Minimize `H0` subject to fixed `|S1|` and `|S2|`, or relax the damped LLG dynamics in zero field.
2. Confirm the equilibrium state is antiferromagnetic along `a` with the expected small canting into `c`.
3. Confirm `M_eq = S1_eq + S2_eq` is along the expected weak-ferromagnetic direction.
4. Compare the numerical canting angle against the analytic small-angle result implied by `Db/J` and the anisotropies.
5. Perturb the equilibrium state weakly and recover the expected qFM and qAFM resonance frequencies and polarizations.

Do not start nonlinear calculations until all five checks pass.

## 3. Construct the THz fields

Define a single-cycle pulse as the second time derivative of a Gaussian envelope, with zero time integral.  Specify its peak field, width, carrier-free waveform convention, and polarization vector.

For every delay `tau`, use the same pulse waveform and scale:

$$
\mathbf H_{AB}(t,\tau)=\mathbf H_A(t-\tau)+\mathbf H_B(t).
$$

Run at least these polarization cases:

1. `H || a` to preferentially drive qFM.
2. `H || c` to preferentially drive qAFM.
3. `H || ac` bisector to drive both modes and expose SFG/DFG mixing.
4. Optional azimuthal-angle sweep for 2D THz polarimetry.

Use identical pulse amplitudes for A and B initially.  Later, sweep amplitude to verify perturbative order: single-mode third-order signals should have cubic scaling, while the intermode SFG/DFG or upconversion features of interest have the expected second-order scaling in the appropriate regime.

## 4. Integrate the semiclassical spin dynamics

For each spin, solve the Gilbert form of LLG,

$$
\frac{d\mathbf S_i}{dt}
=-\frac{\gamma}{1+\alpha^2}
\left[\mathbf S_i\times\mathbf H_i^{\rm eff}
+\frac{\alpha}{|\mathbf S_i|}\mathbf S_i\times
 \left(\mathbf S_i\times\mathbf H_i^{\rm eff}\right)\right],
\qquad
\mathbf H_i^{\rm eff}=-\frac{1}{\gamma}\frac{\partial\mathcal H}{\partial\mathbf S_i}.
$$

The derivatives of the DM interaction must have opposite signs for the two sublattices:

$$
\frac{\partial}{\partial\mathbf S_1}\left[\mathbf D\cdot(\mathbf S_1\times\mathbf S_2)\right]
=\mathbf S_2\times\mathbf D,
\qquad
\frac{\partial}{\partial\mathbf S_2}\left[\mathbf D\cdot(\mathbf S_1\times\mathbf S_2)\right]
=\mathbf D\times\mathbf S_1.
$$

Start every trajectory from the same zero-field equilibrium state.  Integrate a time window that includes: pre-pulse baseline, both pulses for all delays, and several qFM/qAFM dephasing times after the last pulse.  Use a maximum time step sufficiently smaller than the highest magnon period and pulse width.  Store `S1(t)`, `S2(t)`, and `M(t)` at a common uniform output grid even if an adaptive integrator is used internally.

### Numerical acceptance checks

1. With `alpha = 0` and no field, energy and both spin lengths stay constant to the selected tolerance.
2. With damping and no field, a perturbed state relaxes toward the equilibrium state.
3. Tightening the integration tolerance and reducing the maximum time step do not materially move peak frequencies or integrated peak amplitudes.
4. The field-free baseline contains no spurious coherent oscillation above the chosen numerical-noise threshold.

## 5. Isolate the nonlinear two-pulse response

For every delay `tau`, calculate four otherwise identical trajectories:

| Run | Fields present |
| --- | --- |
| `0` | neither pulse |
| `A` | pulse A only |
| `B` | pulse B only |
| `AB` | both pulses |

From the detected magnetization component, form the nonlinear response

$$
M_{\rm NL}(t,\tau)=M_{AB}(t,\tau)-M_A(t,\tau)-M_B(t)+M_0.
$$

Equivalently, if every trace is first written as a deviation from `M0`, the last term is already implicit.  This subtraction is essential: it removes all linear contributions and leaves only the response requiring both THz pulses.

Keep the raw four traces as diagnostic output.  In particular, inspect several individual delays before transforming: pulse-overlap artefacts, incorrect delay placement, and incomplete baseline subtraction are much easier to identify in time domain.

## 6. Sample the two time axes

Build a rectangular data array `M_NL[t_index, tau_index]`.

- **Detection time `t`:** sample finely enough for the pulse and the largest emitted frequency.  Its total post-pulse duration sets detection-frequency resolution.
- **Inter-pulse delay `tau`:** sample finely enough to avoid aliasing at the largest coherence frequency.  Its full span sets excitation-frequency resolution.
- **Delay range:** include both signs of `tau` when rephasing/non-rephasing pathways are desired.  A one-sided scan may be used only if its interpretation is explicitly limited.
- **Common temporal origin:** use a consistent zero relative to pulse B and preserve it in all four calculations.

Before the 2D transform, optionally remove the pre-pulse baseline and apply separable, documented apodization windows along `t` and `tau`.  Report the window type and any zero-padding.  Zero-padding may smooth the displayed grid but must not be presented as increased physical resolution.

## 7. Transform to the 2D THz spectrum

Calculate the complex 2D spectrum

$$
S(f_t,f_\tau)=\mathcal F_{t,\tau}\{M_{\rm NL}(t,\tau)\}.
$$

Preserve complex data, then plot magnitude, real part, imaginary part, and phase as needed.  Label axes by physical frequency in THz, not raw FFT indices.  The `t` transform gives the emitted/detection frequency `f_t`; the `tau` transform gives the excitation/coherence frequency `f_tau`.

For direct comparison across polarizations or amplitudes, apply one normalization policy to the entire dataset.  Do not normalize each spectrum independently when comparing relative signal strengths.

### Peak-assignment checklist

1. Locate diagonal qFM and qAFM nonlinear features at their individual mode frequencies.
2. With both modes driven, look for emitted components near
   `f_qAFM + f_qFM` (SFG) and `|f_qAFM - f_qFM|` (DFG).
3. Verify that these intermode peaks vanish, within numerical tolerance, when only qFM or only qAFM is driven.
4. For the ErFeO3 upconversion case, identify the off-diagonal feature with qFM excitation and qAFM emission; scan temperature or parameters through `2 f_qFM = f_qAFM` to test resonant enhancement.
5. Use the signed `f_tau` axis to distinguish rephasing, non-rephasing, pump-probe, and two-quantum pathways where relevant.

## 8. Validate physical, not merely numerical, behaviour

Perform the following controlled tests, changing one ingredient at a time.

| Test | Expected result |
| --- | --- |
| Set THz amplitude small | Linear FID reproduces qFM/qAFM frequencies and selection rules |
| Turn off one pulse | `M_NL` vanishes |
| Set `Db = 0` | Weak-ferromagnetic canting and the canting-enabled pathways disappear or change according to the remaining anisotropy model |
| Drive only `a` or only `c` | Only the corresponding single-mode response remains at linear order |
| Drive along `ac` | qFM and qAFM are both present; SFG/DFG can appear |
| Sweep field amplitude | Fitted power laws match the assigned nonlinear order over the perturbative range |
| Sweep damping | Linewidths and time-domain decay change consistently without shifting the undamped resonance incorrectly |
| Refine time/delay grids | Peak positions and physical conclusions converge |

## 9. Recommended execution order

1. Establish units and derive `H_eff` independently for both spins.
2. Find and validate the zero-field equilibrium state.
3. Simulate a weak single-pulse FID for `H || a` and `H || c`.
4. Validate qFM/qAFM mode frequencies and polarizations.
5. Introduce a second pulse, but inspect raw `AB`, `A`, `B`, and `M_NL` traces at a few delays.
6. Run a coarse `(t, tau)` scan and verify the expected 2D peak locations.
7. Converge time step, integration tolerance, time window, delay increment, and delay range.
8. Perform amplitude and polarization scans; only then interpret nonlinear peak intensities.
9. Run targeted parameter/temperature scans for the desired resonance condition.
10. Archive configuration, raw trajectories, nonlinear time-domain array, complex 2D spectrum, plotting settings, and validation figures together.

## 10. Minimum reproducibility record per final spectrum

Store the following with every published or notebook-final spectrum:

- Hamiltonian and all parameter values, including the sign and direction of `D`.
- Unit conversions and the numerical value used for `gamma`.
- Pulse waveform, field amplitude, width, polarizations, and delay convention.
- Integrator type, tolerances, maximum step, output time grid, and damping.
- The four component trajectories (`0`, `A`, `B`, `AB`) or a path to them.
- Exact nonlinear-subtraction formula.
- `t` and `tau` ranges, increments, apodization, zero-padding, and Fourier sign convention.
- Detection polarization/projection and normalization policy.
- Results of the convergence and physical validation checks above.

Following this record makes a 2D peak traceable to a specific nonlinear spin trajectory rather than to a plotting or FFT convention.
