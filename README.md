# Quadrupole Magnetic Field Optimisation

Numerical optimisation of a quadrupole magnet modelled as four infinite
straight current-carrying wires. The wire angular positions are optimised
so that the resulting field best approximates an ideal linear quadrupole
field inside the beam aperture.

## Features

* **Biot-Savart Solver:** Vectorized 2D magnetic field calculation for infinite straight wires.
* **Least-Squares Gradient Estimation:** Analytical closed-form estimation of the field gradient ($G$) across a polar sampling grid, offering high numerical stability over single-point forward differences.
* **Pure NumPy Nelder-Mead:** A custom, dependency-free implementation of the simplex optimization algorithm.
* **Multipole Spectrum Analysis:** Fast Fourier Transform (FFT) analysis on a circular probe path ($N_\phi = 2048$) to decompose normal ($b_n$) and skew ($a_n$) multipole coefficients.
* **Scaling Law Verification:** Numerical verification of the theoretical scaling law $\frac{|b_n(r)|}{|b_2(r)|} \propto \left(\frac{r}{r_0}\right)^{n-2}$ for allowed higher-order harmonics ($n = 6, 10, 14$).


## Physics background

A quadrupole magnet focuses a charged-particle beam in one transverse plane
while defocusing it in the other. The ideal field satisfies:

Bx = G·y,   By = G·x


where G [T/m] is the field gradient. This study models the magnet as four
wires on a circle of radius r₀ and asks: *what wire positions minimise the
deviation from this ideal field?*

Key results:
- Fourfold rotational symmetry + alternating currents → only multipole orders
  n = 2, 6, 10, 14, … are non-zero; all others vanish **by symmetry**, not by optimisation.
- The leading perturbation beyond the quadrupole (n = 2) is the 12-pole (n = 6),
  scaling as (r/r₀)⁴.
- Manufacturing tolerance: wire angles must be held to within **Δφ < 0.1°**.



## Repository structure

| File Name | Description |
|------|----------------------|
|quadrupole_optimize.py       | Main script — run this |
|quadrupole_paper_en.pdf      | Compiled PDF (13 pages) |
|fig1_field_geometry.png      | Wire positions + field lines + beam region |
|fig2_multipole_spectrum.png  | Allowed vs. forbidden harmonics (bar chart) |
|fig3_scaling_law.png      |    \|bₙ\|/\|b₂\| ∝ (r/r₀)ⁿ⁻² verification |
|fig4_convergence_sensitivity.png | Nelder-Mead history + angular tolerance |




## Running the code

**Requirements:** Python 3.9+, NumPy, Matplotlib — no other dependencies.

```bash
pip install numpy matplotlib
python quadrupole_optimize.py
```

The script prints the optimised wire angles, field gradient, final cost,
multipole amplitudes, and scaling law slopes to stdout, then saves all four
figures to the current directory.



## What the code does

| Step | Function | Description |
|------|----------|-------------|
| 1 | `biot_savart_wire` | 2-D Biot-Savart field of one infinite wire |
| 2 | `superposed_field` | Sum over four wires with alternating currents |
| 3 | `make_polar_grid` | Polar sampling grid inside the beam aperture |
| 4 | `least_squares_gradient` | Analytic best-fit gradient G (closed form) |
| 5 | `relative_residual_cost` | Dimensionless figure of merit C ∈ [0, 1] |
| 6 | `nelder_mead` | Derivative-free simplex optimiser (pure NumPy) |
| 7 | `multipole_spectrum` | FFT-based multipole decomposition |

All "magic numbers" are declared as named constants at the top of the file
(e.g. `WIRE_CIRCLE_RADIUS`, `NM_CONVERGENCE_TOL`). Results are fully
reproducible: `RANDOM_SEED = 42`.



## Key parameters

| Constant | Value | Meaning |
|----------|-------|---------|
| `WIRE_CURRENT` | 1000 A | Current in each wire |
| `WIRE_CIRCLE_RADIUS` | 50 mm | Radius of wire placement circle |
| `BEAM_RADIUS` | 10 mm | Beam aperture radius |
| `APERTURE_TO_WIRE_RATIO` | 0.20 | R_beam / r₀ |
| `RANDOM_SEED` | 42 | Fixed for reproducibility |



## References

1. H. Wiedemann, *Particle Accelerator Physics*, 4th ed., Springer, 2015.  
2. S. Y. Lee, *Accelerator Physics*, 2nd ed., World Scientific, 2004.  
3. E. D. Courant, M. S. Livingston, H. S. Snyder, *Phys. Rev.* 88, 1190 (1952).  
4. D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., Cambridge, 2017.  
5. J. A. Nelder and R. Mead, *Comput. J.* 7, 308 (1965).  
6. J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, 1999.
