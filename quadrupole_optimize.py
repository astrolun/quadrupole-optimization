"""
Quadrupole Magnetic Field Optimisation
=======================================
Models a quadrupole magnet as four infinite straight current-carrying wires
placed on a circle. Numerically optimises wire angles so that the resulting
field best approximates an ideal linear quadrupole field inside the beam aperture.

Physics background
------------------
- Biot-Savart (2-D):     B = mu0*I / (2*pi*rho^2) * (-dy, dx)
- Ideal quadrupole:      Bx = G*y,  By = G*x
- Multipole expansion:   By + i*Bx = sum_n (b_n + i*a_n)(r/R_ref)^(n-1) e^{i(n-1)phi}
- Symmetry selection:    fourfold rotation + alternating currents
                         => only n = 2, 6, 10, 14, ... harmonics are non-zero

Usage
-----
    python quadrupole_optimize.py

Outputs
-------
    fig1_field_geometry.png
    fig2_multipole_spectrum.png
    fig3_scaling_law.png
    fig4_convergence_sensitivity.png

Dependencies: numpy, matplotlib  (standard scientific Python stack)

Author: Lun
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ── Physical constants ─────────────────────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7   # permeability of free space [T·m/A]

# ── Magnet geometry ────────────────────────────────────────────────────────────
WIRE_CURRENT      = 1000.0   # magnitude of each wire current [A]
WIRE_CIRCLE_RADIUS = 0.05    # radius of the wire placement circle [m]
BEAM_RADIUS        = 0.01    # beam aperture radius [m]

# Ratio used in analytic estimates; stored as a named constant for clarity.
APERTURE_TO_WIRE_RATIO = BEAM_RADIUS / WIRE_CIRCLE_RADIUS   # = 0.20

# ── Sampling grid ──────────────────────────────────────────────────────────────
GRID_POINTS_PER_AXIS = 20    # polar grid: N radii × N angles = N² sample points

# The inner radius of the sampling grid is offset from zero.
# At r=0 the ideal field is also zero, contributing no useful information.
GRID_INNER_FRACTION = 0.05   # inner radius = GRID_INNER_FRACTION * BEAM_RADIUS

# ── Nelder-Mead hyperparameters ────────────────────────────────────────────────
NM_INITIAL_STEP      = 0.05    # initial simplex step size [rad]
NM_CONVERGENCE_TOL   = 1e-10   # stop when simplex spread < this value
NM_MIN_ITERATIONS    = 50      # do not check convergence before this iteration
NM_MAX_ITERATIONS    = 5000    # hard cap on iterations

# ── FFT multipole analysis ─────────────────────────────────────────────────────
FFT_PROBE_POINTS    = 2048    # number of equally-spaced probe points on circle
FFT_DEFAULT_RADIUS  = 0.5 * BEAM_RADIUS   # default probe radius for spectrum

# ── Scaling-law scan ───────────────────────────────────────────────────────────
SCALING_NUM_RADII   = 40      # number of log-spaced probe radii
SCALING_RMIN_FACTOR = 0.01    # innermost probe: SCALING_RMIN_FACTOR * WIRE_CIRCLE_RADIUS
SCALING_RMAX_FACTOR = 0.40    # outermost probe: SCALING_RMAX_FACTOR * WIRE_CIRCLE_RADIUS
SCALING_ORDERS      = [6, 10, 14]   # multipole orders to verify

# ── Numerical safety floors ────────────────────────────────────────────────────
# These prevent division-by-zero in otherwise well-posed calculations.
BIOT_SAVART_R2_FLOOR  = 1e-20   # minimum r² in Biot-Savart [m²]; ~10 nm from wire axis
COST_NORMALIZER_FLOOR = 1e-30   # minimum denominator in cost function [T²]
FFT_QUAD_AMP_FLOOR    = 1e-30   # minimum quadrupole amplitude when normalising

# ── Optimiser reproducibility ──────────────────────────────────────────────────
RANDOM_SEED              = 42
INITIAL_ANGLE_PERTURB_DEG = 5.0   # uniform perturbation applied to each start angle [deg]

# Nominal "upright" quadrupole configuration [deg]; optimiser starts near here.
NOMINAL_ANGLES_DEG = [45.0, 135.0, 225.0, 315.0]

# Alternating current pattern: +I, -I, +I, -I (enforces fourfold symmetry).
WIRE_CURRENT_SIGNS = [+1, -1, +1, -1]


# ── Helper: tolerance band used in sensitivity scan ───────────────────────────
SENSITIVITY_SCAN_RANGE_DEG  = 5.0    # scan ±5° around optimum
SENSITIVITY_SCAN_POINTS     = 201    # resolution of scan


# =============================================================================
# Core physics functions
# =============================================================================

def biot_savart_wire(field_x, field_y, wire_x, wire_y, current):
    """
    Magnetic field (Bx, By) of a single infinite straight wire in 2-D.

    Uses the Biot-Savart law reduced to the transverse plane:
        B = mu0 * I / (2*pi*rho^2) * (-delta_y, delta_x)

    Parameters
    ----------
    field_x, field_y : array_like
        Coordinates of field evaluation points [m].
    wire_x, wire_y : float
        Wire position [m].
    current : float
        Wire current [A] (positive = out of page).

    Returns
    -------
    Bx, By : ndarray
        Magnetic field components [T].
    """
    delta_x = field_x - wire_x
    delta_y = field_y - wire_y
    rho_squared = np.maximum(delta_x**2 + delta_y**2, BIOT_SAVART_R2_FLOOR)
    prefactor = (MU0 * current) / (2 * np.pi * rho_squared)
    Bx = -prefactor * delta_y
    By =  prefactor * delta_x
    return Bx, By


def superposed_field(wire_angles_rad, field_x, field_y):
    """
    Total magnetic field (Bx, By) from all four wires at field points.

    Parameters
    ----------
    wire_angles_rad : array_like, shape (4,)
        Angular positions of the wires on WIRE_CIRCLE_RADIUS [rad].
    field_x, field_y : ndarray
        Coordinates of field evaluation points [m].

    Returns
    -------
    Bx, By : ndarray
        Total field components [T].
    """
    Bx = np.zeros_like(field_x, dtype=np.float64)
    By = np.zeros_like(field_y, dtype=np.float64)
    for angle, sign in zip(wire_angles_rad, WIRE_CURRENT_SIGNS):
        wire_x = WIRE_CIRCLE_RADIUS * np.cos(angle)
        wire_y = WIRE_CIRCLE_RADIUS * np.sin(angle)
        bx, by = biot_savart_wire(field_x, field_y, wire_x, wire_y,
                                  sign * WIRE_CURRENT)
        Bx += bx
        By += by
    return Bx, By


# =============================================================================
# Sampling grid
# =============================================================================

def make_polar_grid(n_points, inner_fraction, outer_radius):
    """
    Build a flattened polar sampling grid that excludes the origin.

    The radial range is [inner_fraction * outer_radius, outer_radius] and the
    angular range covers [0, 2*pi) uniformly.

    Parameters
    ----------
    n_points : int
        Number of radii and angles (grid is n_points × n_points).
    inner_fraction : float
        Fraction of outer_radius used as the innermost radius.
    outer_radius : float
        Outermost radius [m].

    Returns
    -------
    px, py : ndarray, shape (n_points**2,)
        Flattened Cartesian coordinates of sample points [m].
    """
    radii  = np.linspace(inner_fraction * outer_radius, outer_radius, n_points)
    angles = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    R_mat, T_mat = np.meshgrid(radii, angles)
    px = (R_mat * np.cos(T_mat)).ravel()
    py = (R_mat * np.sin(T_mat)).ravel()
    return px, py


# =============================================================================
# Gradient extraction and cost function
# =============================================================================

def least_squares_gradient(wire_angles_rad, sample_x, sample_y):
    """
    Estimate the quadrupole field gradient G [T/m] by least squares.

    Minimises ||B_sim - B_ideal(G)||^2 over G analytically.
    The closed-form solution (from dL/dG = 0) is:

        G = sum(Bx*y + By*x) / sum(x^2 + y^2)

    Parameters
    ----------
    wire_angles_rad : array_like, shape (4,)
    sample_x, sample_y : ndarray

    Returns
    -------
    G : float
        Best-fit gradient [T/m].
    """
    Bx, By = superposed_field(wire_angles_rad, sample_x, sample_y)
    numerator   = np.sum(Bx * sample_y + By * sample_x)
    denominator = np.sum(sample_x**2  + sample_y**2)
    return numerator / denominator


def relative_residual_cost(wire_angles_rad, sample_x, sample_y):
    """
    Dimensionless relative residual error between simulated and ideal fields.

        C = sum[(Bx_sim - G*y)^2 + (By_sim - G*x)^2]
            / sum[(G*y)^2 + (G*x)^2]

    C = 0 : perfect quadrupole field.
    C = 1 : zero correlation with ideal field.

    The function is independent of WIRE_CURRENT, MU0, and WIRE_CIRCLE_RADIUS,
    which makes it well-conditioned for any choice of physical parameters.

    Parameters
    ----------
    wire_angles_rad : array_like, shape (4,)
    sample_x, sample_y : ndarray

    Returns
    -------
    cost : float   in [0, 1]
    """
    G = least_squares_gradient(wire_angles_rad, sample_x, sample_y)
    Bx, By = superposed_field(wire_angles_rad, sample_x, sample_y)
    ideal_Bx = G * sample_y
    ideal_By = G * sample_x
    residual   = np.mean((Bx - ideal_Bx)**2 + (By - ideal_By)**2)
    normalizer = np.mean(ideal_Bx**2 + ideal_By**2)
    return residual / (normalizer + COST_NORMALIZER_FLOOR)


# =============================================================================
# Nelder-Mead simplex optimiser (pure NumPy — no scipy dependency)
# =============================================================================

def nelder_mead(objective, x0, extra_args=(), convergence_tol=NM_CONVERGENCE_TOL,
                max_iter=NM_MAX_ITERATIONS, min_iter=NM_MIN_ITERATIONS,
                initial_step=NM_INITIAL_STEP):
    """
    Minimise objective(x, *extra_args) using the Nelder-Mead simplex method.

    Implements the standard reflect / expand / contract / shrink steps
    (Nelder & Mead, 1965). No external optimisation library is required.

    Parameters
    ----------
    objective : callable
        Function to minimise; signature f(x, *extra_args) -> float.
    x0 : ndarray, shape (n,)
        Starting point.
    extra_args : tuple
        Additional arguments forwarded to objective.
    convergence_tol : float
        Stop when max(scores) - min(scores) < convergence_tol.
    max_iter : int
        Hard iteration limit.
    min_iter : int
        Do not test convergence before this many iterations.
    initial_step : float
        Step size used to build the initial simplex around x0.

    Returns
    -------
    best_x : ndarray
        Best solution found.
    best_score : float
        Objective value at best_x.
    cost_history : list of float
        Best cost at each iteration (useful for convergence plots).
    """
    n_dims = len(x0)

    # Build initial simplex: x0 as first vertex, then x0 offset by
    # initial_step along each coordinate axis.
    simplex = np.zeros((n_dims + 1, n_dims), dtype=np.float64)
    simplex[0] = x0
    for dim in range(n_dims):
        perturbed = x0.copy()
        perturbed[dim] += initial_step
        simplex[dim + 1] = perturbed

    scores = np.array([objective(vertex, *extra_args) for vertex in simplex])
    cost_history = []

    for iteration in range(max_iter):
        # Sort vertices from best (lowest cost) to worst (highest cost).
        order   = np.argsort(scores)
        simplex = simplex[order]
        scores  = scores[order]
        cost_history.append(float(scores[0]))

        if iteration > min_iter and (scores[-1] - scores[0]) < convergence_tol:
            break

        centroid = simplex[:-1].mean(axis=0)   # centroid of all but the worst

        # ── Reflection ────────────────────────────────────────────────────────
        reflected   = centroid + (centroid - simplex[-1])
        score_refl  = objective(reflected, *extra_args)

        if score_refl < scores[0]:
            # Better than current best: try expansion.
            expanded    = centroid + 2.0 * (centroid - simplex[-1])
            score_exp   = objective(expanded, *extra_args)
            if score_exp < score_refl:
                simplex[-1], scores[-1] = expanded, score_exp
            else:
                simplex[-1], scores[-1] = reflected, score_refl

        elif score_refl < scores[-2]:
            # Better than second-worst: accept reflection.
            simplex[-1], scores[-1] = reflected, score_refl

        else:
            # ── Contraction ───────────────────────────────────────────────────
            if score_refl < scores[-1]:
                simplex[-1], scores[-1] = reflected, score_refl
            contracted   = centroid + 0.5 * (simplex[-1] - centroid)
            score_cont   = objective(contracted, *extra_args)
            if score_cont < scores[-1]:
                simplex[-1], scores[-1] = contracted, score_cont
            else:
                # ── Shrink ────────────────────────────────────────────────────
                best_vertex = simplex[0]
                simplex = best_vertex + 0.5 * (simplex - best_vertex)
                scores  = np.array([objective(v, *extra_args) for v in simplex])

    return simplex[0], scores[0], cost_history


# =============================================================================
# Multipole spectrum analysis
# =============================================================================

def multipole_spectrum(wire_angles_rad, probe_radius=FFT_DEFAULT_RADIUS,
                       num_probe_points=FFT_PROBE_POINTS):
    """
    Normalised multipole spectrum on a circular probe at radius probe_radius.

    The complex signal  f(phi) = By(phi) + i*Bx(phi)  is sampled uniformly and
    Fourier-transformed.  FFT bin k corresponds to multipole order n = k + 1.
    All amplitudes are normalised to the quadrupole component (n=2, bin k=1).

    Implementation note: np.fft.fft (full complex FFT) is used rather than
    np.fft.rfft because NumPy 2.x rfft rejects complex-valued input arrays.

    Parameters
    ----------
    wire_angles_rad : array_like, shape (4,)
    probe_radius : float
        Radius of the probe circle [m].
    num_probe_points : int
        Number of equally-spaced sample points on the circle.

    Returns
    -------
    multipole_orders : ndarray of int
        n = 1, 2, ..., num_probe_points//2 + 1
    normalised_amplitudes : ndarray of float
        |b_n| / |b_2| for each order.
    """
    phi_values = np.linspace(0.0, 2 * np.pi, num_probe_points, endpoint=False)
    probe_x    = probe_radius * np.cos(phi_values)
    probe_y    = probe_radius * np.sin(phi_values)

    Bx, By = superposed_field(wire_angles_rad, probe_x, probe_y)

    # Complex multipole signal: convention By + i*Bx (standard in accelerator physics)
    complex_signal = (By + 1j * Bx).astype(np.complex128)

    raw_fft      = np.fft.fft(complex_signal)
    half_spectrum = raw_fft[:num_probe_points // 2 + 1]
    amplitudes    = np.abs(half_spectrum) / (num_probe_points / 2)
    amplitudes[0]  /= 2   # DC term counted once
    amplitudes[-1] /= 2   # Nyquist term counted once

    multipole_orders = np.arange(1, len(amplitudes) + 1, dtype=int)
    # Quadrupole is at bin k=1, which is order n=2
    quadrupole_amplitude = amplitudes[1]
    normalised_amplitudes = amplitudes / (quadrupole_amplitude + FFT_QUAD_AMP_FLOOR)

    return multipole_orders, normalised_amplitudes


def is_symmetry_allowed(multipole_order):
    """Return True if multipole_order satisfies n ≡ 2 (mod 4) and n >= 2."""
    return (multipole_order >= 2) and ((multipole_order - 2) % 4 == 0)


# =============================================================================
# Figures
# =============================================================================

def plot_field_geometry(wire_angles_rad, output_path="fig1_field_geometry.png"):
    """Field lines and beam-region field magnitude."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: wire circle, wire positions, field streamlines ────────────
    circle_theta = np.linspace(0, 2 * np.pi, 300)
    r0_mm = WIRE_CIRCLE_RADIUS * 1e3   # convert to mm for axis labels
    ax_left.plot(r0_mm * np.cos(circle_theta), r0_mm * np.sin(circle_theta),
                 'k--', lw=0.8, label=f'Wire circle $r_0 = {r0_mm:.0f}$ mm')

    wire_colors = ['red', 'blue', 'red', 'blue']
    for wire_idx, (angle, color, sign) in enumerate(
            zip(wire_angles_rad, wire_colors, WIRE_CURRENT_SIGNS)):
        wx_mm = r0_mm * np.cos(angle)
        wy_mm = r0_mm * np.sin(angle)
        sign_str = '+' if sign > 0 else '−'
        ax_left.plot(wx_mm, wy_mm, 'o', color=color, ms=10,
                     label=f'Wire {wire_idx+1} ({sign_str}$I_0$)')

    # Streamplot on a regular grid
    grid_extent_mm = WIRE_CIRCLE_RADIUS * 1.1 * 1e3
    grid_1d = np.linspace(-grid_extent_mm, grid_extent_mm, 100)
    Xg_mm, Yg_mm = np.meshgrid(grid_1d, grid_1d)
    Bx_grid, By_grid = superposed_field(wire_angles_rad,
                                        Xg_mm / 1e3, Yg_mm / 1e3)
    ax_left.streamplot(grid_1d, grid_1d, By_grid, Bx_grid,
                       density=1.2, color='gray', linewidth=0.6, arrowsize=0.6)

    beam_circle = plt.Circle((0, 0), BEAM_RADIUS * 1e3,
                             color='gold', fill=False, lw=2, label='Beam aperture')
    ax_left.add_patch(beam_circle)
    ax_left.set_xlim(-70, 70);  ax_left.set_ylim(-70, 70)
    ax_left.set_aspect('equal')
    ax_left.set_xlabel('x [mm]');  ax_left.set_ylabel('y [mm]')
    ax_left.set_title('Wire positions and magnetic field lines')
    ax_left.legend(fontsize=7, loc='upper right')

    # ── Right panel: field magnitude inside beam aperture ─────────────────────
    beam_grid_1d = np.linspace(-BEAM_RADIUS, BEAM_RADIUS, 80)
    Xb, Yb = np.meshgrid(beam_grid_1d, beam_grid_1d)
    inside_aperture = (Xb**2 + Yb**2) < BEAM_RADIUS**2
    Bxb, Byb = superposed_field(wire_angles_rad, Xb, Yb)
    B_magnitude = np.sqrt(Bxb**2 + Byb**2)
    B_magnitude[~inside_aperture] = np.nan

    im = ax_right.pcolormesh(Xb * 1e3, Yb * 1e3, B_magnitude,
                              shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax_right, label='|B| [T]')

    arrow_skip = 6
    ax_right.quiver(Xb[::arrow_skip, ::arrow_skip] * 1e3,
                    Yb[::arrow_skip, ::arrow_skip] * 1e3,
                    Bxb[::arrow_skip, ::arrow_skip],
                    Byb[::arrow_skip, ::arrow_skip],
                    color='white', scale=None, width=0.004)
    ax_right.set_xlim(-BEAM_RADIUS * 1e3, BEAM_RADIUS * 1e3)
    ax_right.set_ylim(-BEAM_RADIUS * 1e3, BEAM_RADIUS * 1e3)
    ax_right.set_aspect('equal')
    ax_right.set_xlabel('x [mm]');  ax_right.set_ylabel('y [mm]')
    ax_right.set_title('Field magnitude inside beam aperture')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_multipole_spectrum(wire_angles_rad,
                            output_path="fig2_multipole_spectrum.png"):
    """Bar chart of normalised multipole amplitudes."""
    multipole_orders, norm_amplitudes = multipole_spectrum(wire_angles_rad)
    orders_to_show = multipole_orders[:20]
    amplitudes_to_show = norm_amplitudes[:20]

    bar_colors = ['green' if is_symmetry_allowed(n) else 'red'
                  for n in orders_to_show]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(orders_to_show, np.abs(amplitudes_to_show),
           color=bar_colors, edgecolor='k', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 10)
    ax.set_xlabel('Multipole order $n$')
    ax.set_ylabel('$|b_n| / |b_2|$')
    ax.set_title('Multipole spectrum — green: symmetry-allowed, red: forbidden by symmetry')
    ax.set_xticks(orders_to_show)
    ax.axhline(1e-10, color='gray', lw=0.8, ls='--',
               label='Numerical noise floor $\\sim 10^{-10}$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scaling_law(wire_angles_rad, scaling_data, probe_radii,
                     output_path="fig3_scaling_law.png"):
    """Log-log plot of |b_n|/|b_2| vs r/r0 for allowed harmonics."""
    theory_slope = {n: n - 2 for n in SCALING_ORDERS}
    line_styles   = ['-o', '-s', '-^']
    plot_colors   = ['tab:blue', 'tab:orange', 'tab:green']
    normalised_radii = probe_radii / WIRE_CIRCLE_RADIUS

    fig, ax = plt.subplots(figsize=(7, 5))
    for n_order, style, color in zip(SCALING_ORDERS, line_styles, plot_colors):
        amplitudes = np.array(scaling_data[n_order])
        valid = amplitudes > 1e-15
        ax.plot(normalised_radii[valid], amplitudes[valid], style,
                color=color, ms=4,
                label=f'$n={n_order}$  (theory slope {theory_slope[n_order]})')
        fitted_slope, log_intercept = np.polyfit(
            np.log(normalised_radii[valid]), np.log(amplitudes[valid]), 1)
        r_endpoints = np.array([normalised_radii[valid][0],
                                 normalised_radii[valid][-1]])
        ax.plot(r_endpoints, np.exp(log_intercept) * r_endpoints**fitted_slope,
                '--', color=color, lw=1.5,
                label=f'  fit slope = {fitted_slope:.3f}')

    ax.set_xscale('log');  ax.set_yscale('log')
    ax.set_xlabel('$r_\\mathrm{probe} / r_0$')
    ax.set_ylabel('$|b_n| / |b_2|$')
    ax.set_title('Scaling law: $|b_n|/|b_2| \\propto (r/r_0)^{n-2}$')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', ls=':', lw=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_convergence_and_sensitivity(cost_history, optimal_angles, optimal_cost,
                                     sample_x, sample_y,
                                     output_path="fig4_convergence_sensitivity.png"):
    """Convergence curve (left) and angular sensitivity of Wire 1 (right)."""
    fig, (ax_conv, ax_sens) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: cost vs iteration ───────────────────────────────────────────────
    ax_conv.semilogy(cost_history, lw=1.5, color='steelblue')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Cost $\\mathcal{C}$')
    ax_conv.set_title('Nelder-Mead convergence history')
    ax_conv.grid(True, which='both', ls=':', lw=0.5)

    # ── Right: cost vs angular perturbation of Wire 1 ────────────────────────
    perturbations_deg = np.linspace(-SENSITIVITY_SCAN_RANGE_DEG,
                                     SENSITIVITY_SCAN_RANGE_DEG,
                                     SENSITIVITY_SCAN_POINTS)
    perturbed_costs = []
    for delta_deg in perturbations_deg:
        angles_perturbed = optimal_angles.copy()
        angles_perturbed[0] = optimal_angles[0] + np.radians(delta_deg)
        perturbed_costs.append(relative_residual_cost(angles_perturbed,
                                                       sample_x, sample_y))

    ax_sens.semilogy(perturbations_deg, perturbed_costs, lw=1.5, color='tomato')
    ax_sens.axvline(0, color='k', lw=0.8, ls='--')
    ax_sens.axhline(optimal_cost, color='gray', lw=0.8, ls=':',
                    label=f'Optimum  $\\mathcal{{C}} = {optimal_cost:.2e}$')
    ax_sens.set_xlabel('Angular perturbation of Wire 1 [deg]')
    ax_sens.set_ylabel('Cost $\\mathcal{C}$')
    ax_sens.set_title('Cost sensitivity — manufacturing tolerance $\\Delta\\phi < 0.1^\\circ$')
    ax_sens.legend(fontsize=8)
    ax_sens.grid(True, which='both', ls=':', lw=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    # ── Build sampling grid ───────────────────────────────────────────────────
    sample_x, sample_y = make_polar_grid(
        n_points=GRID_POINTS_PER_AXIS,
        inner_fraction=GRID_INNER_FRACTION,
        outer_radius=BEAM_RADIUS,
    )

    # ── Perturbed starting point (fixed seed for reproducibility) ─────────────
    rng = np.random.default_rng(RANDOM_SEED)
    perturbations_rad = np.radians(
        rng.uniform(-INITIAL_ANGLE_PERTURB_DEG, INITIAL_ANGLE_PERTURB_DEG, 4)
    )
    start_angles_rad = np.radians(NOMINAL_ANGLES_DEG) + perturbations_rad

    # ── Optimise ──────────────────────────────────────────────────────────────
    optimal_angles, optimal_cost, cost_history = nelder_mead(
        objective=relative_residual_cost,
        x0=start_angles_rad,
        extra_args=(sample_x, sample_y),
    )
    optimal_angles = optimal_angles % (2 * np.pi)
    optimal_gradient = least_squares_gradient(optimal_angles, sample_x, sample_y)

    # ── Report ─────────────────────────────────────────────────────────────────
    print("=" * 54)
    print("Optimised wire configuration")
    print("=" * 54)
    for idx, (angle_rad, sign) in enumerate(zip(optimal_angles, WIRE_CURRENT_SIGNS)):
        sign_str = f"+{WIRE_CURRENT:.0f}" if sign > 0 else f"-{WIRE_CURRENT:.0f}"
        print(f"  Wire {idx+1}  I = {sign_str} A   angle = {np.degrees(angle_rad):.4f} deg")
    print(f"\n  Field gradient G = {optimal_gradient:.6f} T/m")
    print(f"  Final cost     C = {optimal_cost:.4e}")
    print(f"  Iterations       = {len(cost_history)}")
    print("=" * 54)

    # ── Multipole report ──────────────────────────────────────────────────────
    print("\nMultipole amplitudes at r_probe = 0.5 * R_beam:")
    print(f"  {'n':>3}   {'|b_n|/|b_2|':>14}   Status")
    orders_report, norms_report = multipole_spectrum(
        optimal_angles, probe_radius=0.5 * BEAM_RADIUS
    )
    for n in [1, 2, 3, 4, 6, 10, 14]:
        amp = norms_report[n - 1]
        status = "allowed" if is_symmetry_allowed(n) else "FORBIDDEN"
        print(f"  {n:3d}   {amp:14.3e}   {status}")

    # ── Scaling law ───────────────────────────────────────────────────────────
    probe_radii = np.logspace(
        np.log10(SCALING_RMIN_FACTOR * WIRE_CIRCLE_RADIUS),
        np.log10(SCALING_RMAX_FACTOR * WIRE_CIRCLE_RADIUS),
        SCALING_NUM_RADII,
    )
    scaling_data = {n: [] for n in SCALING_ORDERS}
    for rp in probe_radii:
        _, b = multipole_spectrum(optimal_angles, probe_radius=rp)
        for n in SCALING_ORDERS:
            scaling_data[n].append(float(b[n - 1]))

    print("\nScaling law  |b_n|/|b_2|  ~  (r/r0)^(n-2):")
    print(f"  {'n':>3}   {'Theory slope':>14}   {'Fitted slope':>12}")
    for n in SCALING_ORDERS:
        y_vals = np.array(scaling_data[n])
        x_vals = probe_radii / WIRE_CIRCLE_RADIUS
        valid = y_vals > 0
        fitted_slope, _ = np.polyfit(np.log(x_vals[valid]),
                                      np.log(y_vals[valid]), 1)
        print(f"  {n:3d}   {n-2:14d}   {fitted_slope:+12.3f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_field_geometry(optimal_angles)
    plot_multipole_spectrum(optimal_angles)
    plot_scaling_law(optimal_angles, scaling_data, probe_radii)
    plot_convergence_and_sensitivity(
        cost_history, optimal_angles, optimal_cost, sample_x, sample_y
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
