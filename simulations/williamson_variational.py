#!/usr/bin/env python3
"""
Variational Knot Energy — Williamson Model
============================================

Instead of FDTD (which loses topology), compute the minimum energy
of field configurations constrained to specific knot topologies.

Key insight: for a photon confined to a tube following a knot curve,
the energy depends on:
    - The knot curve length L(p,q,R,r)
    - The tube cross-section radius a
    - The field amplitude A (set by charge normalization)
    - The oscillation frequency ω (set by the resonance condition)

The self-consistency condition (Williamson's extended Maxwell in
steady state) creates a relationship between these parameters that
selects specific energies for each topology.

Energy components for a field tube of radius a following a (p,q) torus knot:

    E_EM  = ½∫(E² + B²) dV    — electromagnetic energy
    E_P   = ½∫P² dV           — pivot (rest mass) energy
    E_grad = ½∫(∇P)² dV       — pivot gradient energy (confinement cost)

The minimization: find (a, R, r) that minimize total energy subject to:
    1. Topology is (p,q) torus knot
    2. Total charge = e (normalization)
    3. Resonance condition: ω = nc/L (n wavelengths fit around knot)
    4. Self-consistency: ωP = div(E), ωE = curl(B) - grad(P)

Usage:
    python3 williamson_variational.py --mode energy_landscape --save
    python3 williamson_variational.py --mode minimize --save
    python3 williamson_variational.py --mode spectrum --save
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.special import jn_zeros
import argparse


def torus_knot_length(p, q, R, r):
    """Length of a (p,q) torus knot on a torus with radii R, r.

    The knot curve: x(t) = (R + r cos(qt)) cos(pt), etc.
    Length = ∫₀²π |dr/dt| dt

    For large R/r: L ≈ 2π √(p²R² + q²r²)
    Exact: numerical integration.
    """
    n_pts = 1000
    t = np.linspace(0, 2 * np.pi, n_pts)
    dt = t[1] - t[0]

    # Derivatives
    dx = -(R + r * np.cos(q * t)) * p * np.sin(p * t) - \
         r * q * np.sin(q * t) * np.cos(p * t)
    dy = (R + r * np.cos(q * t)) * p * np.cos(p * t) - \
         r * q * np.sin(q * t) * np.sin(p * t)
    dz = r * q * np.cos(q * t)

    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.sum(ds) * dt


def torus_knot_curvature(p, q, R, r):
    """Average and max curvature of a (p,q) torus knot.

    Higher curvature → stronger confinement needed → more energy.
    """
    n_pts = 1000
    t = np.linspace(0, 2 * np.pi, n_pts)
    dt = t[1] - t[0]

    # Position
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)

    # First derivatives
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    speed = np.sqrt(dx**2 + dy**2 + dz**2)

    # Second derivatives
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    ddz = np.gradient(dz, dt)

    # Curvature: |v × a| / |v|³
    cross_x = dy * ddz - dz * ddy
    cross_y = dz * ddx - dx * ddz
    cross_z = dx * ddy - dy * ddx
    cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

    kappa = cross_mag / (speed**3 + 1e-20)

    return np.mean(kappa), np.max(kappa)


def knot_energy_analytic(p, q, R, r, a, n_wavelengths=None):
    """Compute the energy of a Williamson field configuration
    on a (p,q) torus knot.

    Model: traveling wave confined to a tube of radius a
    around the knot curve.

    E = A cos(phase) r̂   (radial, creates div(E))
    B = A sin(phase) t̂   (tangential, along knot)
    P = (A/ωa) cos(phase) (from div(E)/ω)

    where phase = ωt - k·s (s = arc length along knot)

    Args:
        p, q: torus knot indices
        R: major radius
        r: minor radius (knot lives on torus surface)
        a: field tube radius (cross-section of the field distribution)
        n_wavelengths: number of wavelengths around the knot
                      (default: p for toroidal winding)

    Returns:
        dict with energy components and derived quantities
    """
    if n_wavelengths is None:
        n_wavelengths = p  # p wavelengths around the torus

    # Knot geometry
    L = torus_knot_length(p, q, R, r)
    kappa_avg, kappa_max = torus_knot_curvature(p, q, R, r)

    # Wavelength and frequency
    wavelength = L / n_wavelengths
    omega = 2 * np.pi / wavelength  # angular frequency (c=1)
    k = 2 * np.pi / wavelength      # wavenumber

    # Field amplitude from charge normalization
    # For a radial E field in a tube: ∫div(E) dV = Q = 1 (natural units)
    # div(E) ~ 2A/a for cylindrical geometry
    # ∫div(E) dV ~ (2A/a) × π a² × L = 2πAaL
    # So A = Q / (2πaL) = 1/(2πaL)
    A = 1.0 / (2 * np.pi * a * L)

    # Volume of the field tube
    V_tube = np.pi * a**2 * L

    # EM energy: ½∫(E² + B²) dV
    # E² and B² average to A²/2 each over the oscillation
    # But E is radial (profile ~ ρ/a) and B is uniform-ish
    # For Gaussian tube: ∫E² dV = A² × πa²L × (1/2) [averaged over cross-section]
    E_em = 0.5 * A**2 * V_tube  # ½(E² + B²) averaged

    # Pivot energy: ½∫P² dV
    # P = div(E)/ω = (2A/a)/ω at tube center
    # Average P² over the tube ~ (A/(aω))²
    P_amp = A / (a * omega)
    E_pivot = 0.5 * P_amp**2 * V_tube

    # Pivot gradient energy (confinement cost):
    # ∇P has a component ~ P/a (radial) and ~ kP (longitudinal)
    # The radial gradient costs energy and provides confinement
    grad_P_radial = P_amp / a
    grad_P_long = k * P_amp
    E_grad = 0.5 * (grad_P_radial**2 + grad_P_long**2) * V_tube

    # Curvature energy: bending the field tube costs energy
    # Scale: ~ A² × kappa² × V_tube
    E_curvature = 0.5 * A**2 * kappa_avg**2 * a**2 * V_tube

    # Total energy
    E_total = E_em + E_pivot + E_grad + E_curvature

    # Compton wavelength (if this is an electron)
    # λ_C = h/(mc) = 2π/m, so m = 2π/λ_C
    # But in our units, m = E_total (natural units)

    return {
        'E_em': E_em,
        'E_pivot': E_pivot,
        'E_grad': E_grad,
        'E_curvature': E_curvature,
        'E_total': E_total,
        'L': L,
        'wavelength': wavelength,
        'omega': omega,
        'A': A,
        'P_amp': P_amp,
        'kappa_avg': kappa_avg,
        'kappa_max': kappa_max,
        'V_tube': V_tube,
        'a': a,
    }


def minimize_knot_energy(p, q, R_range=(1.0, 20.0), r_range=(0.1, 5.0),
                          a_range=(0.01, 2.0)):
    """Find the minimum energy configuration for a (p,q) torus knot.

    Minimize E_total over (R, r, a) subject to constraints:
        - R > r (torus doesn't self-intersect)
        - a < r (tube fits within torus tube)
        - a > 0 (finite tube radius)
    """
    def objective(params):
        log_R, log_r, log_a = params
        R = np.exp(log_R)
        r_val = np.exp(log_r)
        a = np.exp(log_a)

        # Constraints
        if R < r_val or a > r_val or a < 0.001 or R > 100 or r_val > 50:
            return 1e20

        try:
            result = knot_energy_analytic(p, q, R, r_val, a)
            return result['E_total']
        except Exception:
            return 1e20

    # Try multiple starting points
    best_result = None
    best_energy = np.inf

    for R0 in [2.0, 5.0, 10.0]:
        for r0 in [0.5, 1.0, 2.0]:
            for a0 in [0.1, 0.3, 0.5]:
                if R0 <= r0 or a0 >= r0:
                    continue
                x0 = [np.log(R0), np.log(r0), np.log(a0)]
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-6})
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_result = res
                except Exception:
                    continue

    if best_result is None:
        return None

    R_opt = np.exp(best_result.x[0])
    r_opt = np.exp(best_result.x[1])
    a_opt = np.exp(best_result.x[2])

    return {
        'R': R_opt, 'r': r_opt, 'a': a_opt,
        'energy': knot_energy_analytic(p, q, R_opt, r_opt, a_opt),
        'success': best_result.success,
    }


def min_strand_distance(p, q, R, r, n_pts=2000):
    """Compute minimum distance between non-adjacent segments of a (p,q) torus knot.

    This is the self-avoidance constraint: the tube radius a must be
    less than half this distance, otherwise the tube self-intersects.

    For more complex knots, strands pass closer together, forcing smaller a.
    This is the key mechanism: smaller a → higher energy density → more mass.
    """
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Knot curve points
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)

    # Arc length between adjacent points (for determining "non-adjacent")
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    dz = np.diff(z, append=z[0])
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    total_length = np.sum(ds)

    # Minimum separation to consider "non-adjacent" (skip nearby points along curve)
    # Use ~10% of total length as the adjacency threshold
    skip = max(n_pts // (2 * max(p, q)), n_pts // 10)

    d_min = np.inf

    # Sample a subset of points to keep computation manageable
    sample_idx = np.arange(0, n_pts, max(1, n_pts // 500))

    for i in sample_idx:
        # Compute distances from point i to all non-adjacent points
        # Non-adjacent means |index difference| > skip (wrapping around)
        j_all = np.arange(n_pts)
        diff = np.minimum(np.abs(j_all - i), n_pts - np.abs(j_all - i))
        non_adj = diff > skip

        if not np.any(non_adj):
            continue

        dist = np.sqrt((x[non_adj] - x[i])**2 +
                       (y[non_adj] - y[i])**2 +
                       (z[non_adj] - z[i])**2)

        d_min_local = np.min(dist)
        if d_min_local < d_min:
            d_min = d_min_local

    return d_min


def coulomb_self_energy(p, q, R, r, a, n_pts=500):
    """Coulomb self-energy of charge distributed along a torus knot.

    E_Coulomb = ∫∫ λ(s)λ(t) / |r(s) - r(t)| ds dt

    where λ = linear charge density = Q/L.

    For knotted curves, crossings bring charge elements close together,
    dramatically increasing self-energy for more complex knots.
    This is the dominant topology-dependent energy term.

    Regularized: exclude self-interaction within tube radius a.
    """
    t_param = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    dt = t_param[1] - t_param[0]

    # Knot curve
    x = (R + r * np.cos(q * t_param)) * np.cos(p * t_param)
    y = (R + r * np.cos(q * t_param)) * np.sin(p * t_param)
    z = r * np.sin(q * t_param)

    # Arc length elements
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    ds = np.sqrt(dx**2 + dy**2 + dz**2) * dt

    L = np.sum(ds)
    lam = 1.0 / L  # linear charge density (total charge = 1)

    # Double sum (vectorized with broadcasting)
    # Compute all pairwise distances
    rx = x[:, None] - x[None, :]
    ry = y[:, None] - y[None, :]
    rz = z[:, None] - z[None, :]
    dist = np.sqrt(rx**2 + ry**2 + rz**2 + a**2)  # regularized by tube radius

    # Arc distance (for excluding nearby points)
    cumlen = np.cumsum(ds)
    arc_dist = np.abs(cumlen[:, None] - cumlen[None, :])
    arc_dist = np.minimum(arc_dist, L - arc_dist)  # periodic

    # Only count non-adjacent interactions (arc distance > 2*a)
    mask = arc_dist > 2 * a
    np.fill_diagonal(mask, False)

    # Coulomb energy: sum 1/r with ds weights
    integrand = lam**2 * mask / dist
    E_coulomb = 0.5 * np.sum(integrand * ds[:, None] * ds[None, :])

    return E_coulomb


def ropelength_ratio(p, q):
    """Approximate ropelength ratio for a (p,q) torus knot
    relative to the unknot.

    Uses the Denne-Sullivan lower bound and numerical estimates.
    For torus knots, ropelength ≈ 2π × (crossing_number + some constant).
    """
    # Known ropelength values (from numerical studies)
    known = {
        (2, 1): 1.0,       # unknot (normalized)
        (1, 2): 1.0,
        (3, 2): 2.61,      # trefoil
        (2, 3): 2.61,
        (5, 2): 4.19,      # cinquefoil
        (2, 5): 4.19,
        (4, 3): 5.54,      # (4,3) torus knot
        (3, 4): 5.54,
        (5, 3): 6.50,      # (5,3) torus knot
        (3, 5): 6.50,
        (7, 2): 5.76,      # (7,2) torus knot
        (2, 7): 5.76,
    }

    key = (p, q)
    if key in known:
        return known[key]

    # Estimate: crossing number × constant
    crossing = min(p, q) * (max(p, q) - 1)
    return 1.0 + 0.45 * crossing  # rough fit to known values


def run_energy_landscape(save=False):
    """Visualize how energy depends on tube radius a for different knots."""
    print("=" * 60)
    print("Energy Landscape: E(a) for different knot topologies")
    print("=" * 60)

    knots = [
        (2, 1, "unknot (e⁻)"),
        (3, 2, "trefoil (p?)"),
        (5, 2, "cinquefoil"),
        (4, 3, "(4,3)"),
        (5, 3, "(5,3)"),
    ]

    R = 5.0
    r = 1.5

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Variational Energy of Torus Knots\n"
        f"R={R}, r={r} — Energy vs tube radius a",
        fontsize=13, fontweight='bold')

    a_values = np.logspace(-2, np.log10(r * 0.9), 100)
    colors = plt.cm.Set1(np.linspace(0, 1, len(knots)))

    all_results = {}

    for i, (p, q, name) in enumerate(knots):
        L = torus_knot_length(p, q, R, r)
        kappa_avg, kappa_max = torus_knot_curvature(p, q, R, r)
        crossing = min(p, q) * (max(p, q) - 1)

        energies = {'total': [], 'em': [], 'pivot': [], 'grad': [], 'curv': []}

        for a in a_values:
            e = knot_energy_analytic(p, q, R, r, a)
            energies['total'].append(e['E_total'])
            energies['em'].append(e['E_em'])
            energies['pivot'].append(e['E_pivot'])
            energies['grad'].append(e['E_grad'])
            energies['curv'].append(e['E_curvature'])

        # Find minimum
        min_idx = np.argmin(energies['total'])
        a_min = a_values[min_idx]
        e_min = energies['total'][min_idx]

        all_results[(p, q)] = {
            'name': name, 'L': L, 'crossing': crossing,
            'kappa_avg': kappa_avg, 'kappa_max': kappa_max,
            'a_min': a_min, 'E_min': e_min,
        }

        print(f"  ({p},{q}) {name}: L={L:.2f}, κ_avg={kappa_avg:.3f}, "
              f"a_min={a_min:.4f}, E_min={e_min:.6e}")

    # Plot 1: Total energy vs a for each knot
    ax = axes[0, 0]
    for i, (p, q, name) in enumerate(knots):
        energies_total = []
        for a in a_values:
            e = knot_energy_analytic(p, q, R, r, a)
            energies_total.append(e['E_total'])
        ax.loglog(a_values, energies_total, color=colors[i], linewidth=2,
                  label=f"({p},{q}) {name}")
    ax.set_xlabel('Tube radius a')
    ax.set_ylabel('Total Energy')
    ax.set_title('E_total vs tube radius')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy components for unknot
    ax = axes[0, 1]
    for comp, style, label in [('em', '-', 'EM'), ('pivot', '--', 'P²'),
                                 ('grad', ':', '∇P'), ('curv', '-.', 'κ²')]:
        vals = []
        for a in a_values:
            e = knot_energy_analytic(2, 1, R, r, a)
            vals.append(e[f'E_{comp}' if comp != 'curv' else 'E_curvature'])
        ax.loglog(a_values, vals, style, linewidth=2, label=label)
    ax.set_xlabel('Tube radius a')
    ax.set_ylabel('Energy')
    ax.set_title('Energy components (unknot)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy components for trefoil
    ax = axes[0, 2]
    for comp, style, label in [('em', '-', 'EM'), ('pivot', '--', 'P²'),
                                 ('grad', ':', '∇P'), ('curv', '-.', 'κ²')]:
        vals = []
        for a in a_values:
            e = knot_energy_analytic(3, 2, R, r, a)
            vals.append(e[f'E_{comp}' if comp != 'curv' else 'E_curvature'])
        ax.loglog(a_values, vals, style, linewidth=2, label=label)
    ax.set_xlabel('Tube radius a')
    ax.set_ylabel('Energy')
    ax.set_title('Energy components (trefoil)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Minimum energy vs crossing number
    ax = axes[1, 0]
    crossings = [all_results[(p, q)]['crossing'] for p, q, _ in knots]
    e_mins = [all_results[(p, q)]['E_min'] for p, q, _ in knots]
    ax.plot(crossings, e_mins, 'ko-', markersize=10, linewidth=2)
    for i, (p, q, name) in enumerate(knots):
        ax.annotate(f"({p},{q})", (crossings[i], e_mins[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Minimum Energy')
    ax.set_title('Minimum Energy vs Knot Complexity')
    ax.grid(True, alpha=0.3)

    # Plot 5: Mass ratios
    ax = axes[1, 1]
    e_ref = all_results[(2, 1)]['E_min']
    ratios = [all_results[(p, q)]['E_min'] / e_ref for p, q, _ in knots]
    labels = [f"({p},{q})\n{name}" for p, q, name in knots]
    bars = ax.bar(range(len(knots)), ratios, color=colors[:len(knots)], alpha=0.8)
    ax.set_xticks(range(len(knots)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mass ratio (m / m_electron)')
    ax.set_title('Mass Ratios from Variational Minimum')
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.01, f'{v:.3f}x', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Knot geometry comparison
    ax = axes[1, 2]
    # Draw the knot curves in 2D projection
    t = np.linspace(0, 2 * np.pi, 500)
    for i, (p, q, name) in enumerate(knots):
        x_knot = (R + r * np.cos(q * t)) * np.cos(p * t)
        y_knot = (R + r * np.cos(q * t)) * np.sin(p * t)
        ax.plot(x_knot / R, y_knot / R, color=colors[i], linewidth=1.5,
                alpha=0.7, label=f"({p},{q})")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Knot curves (XY projection)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/variational_landscape.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def run_minimize(save=False):
    """Find minimum-energy configurations for each knot topology."""
    print("=" * 60)
    print("Variational Minimization: Find optimal (R, r, a) per knot")
    print("=" * 60)

    knots = [
        (2, 1, "unknot (electron)"),
        (3, 2, "trefoil (proton?)"),
        (5, 2, "cinquefoil"),
        (4, 3, "(4,3) knot"),
        (5, 3, "(5,3) knot"),
    ]

    results = []
    for p, q, name in knots:
        print(f"\n--- ({p},{q}) {name} ---")
        opt = minimize_knot_energy(p, q)
        if opt is None:
            print("  Minimization failed")
            continue

        e = opt['energy']
        crossing = min(p, q) * (max(p, q) - 1)
        rl = ropelength_ratio(p, q)

        results.append({
            'p': p, 'q': q, 'name': name,
            'crossing': crossing,
            'R': opt['R'], 'r': opt['r'], 'a': opt['a'],
            'E_total': e['E_total'],
            'E_em': e['E_em'],
            'E_pivot': e['E_pivot'],
            'E_grad': e['E_grad'],
            'L': e['L'],
            'kappa_avg': e['kappa_avg'],
            'ropelength_ratio': rl,
        })

        print(f"  Optimal: R={opt['R']:.3f}, r={opt['r']:.3f}, a={opt['a']:.4f}")
        print(f"  L={e['L']:.3f}, κ={e['kappa_avg']:.3f}")
        print(f"  Energy: total={e['E_total']:.6e}")
        print(f"    EM={e['E_em']:.6e}, P²={e['E_pivot']:.6e}, "
              f"∇P={e['E_grad']:.6e}, κ²={e['E_curvature']:.6e}")

    if not results:
        return

    # Mass ratios
    e_ref = results[0]['E_total']
    print("\n" + "=" * 70)
    print("VARIATIONAL MASS SPECTRUM")
    print("=" * 70)
    print(f"{'Knot':>15s} {'Cross':>6s} {'R':>8s} {'r':>8s} {'a':>8s} "
          f"{'Energy':>12s} {'Ratio':>8s} {'Rope':>6s}")
    print("-" * 70)

    for r_item in results:
        ratio = r_item['E_total'] / e_ref
        print(f"({r_item['p']},{r_item['q']}) {r_item['name']:>8s} "
              f"{r_item['crossing']:>6d} "
              f"{r_item['R']:>8.3f} {r_item['r']:>8.3f} {r_item['a']:>8.4f} "
              f"{r_item['E_total']:>12.6e} {ratio:>8.3f}x "
              f"{r_item['ropelength_ratio']:>6.2f}")

    print("\nKnown mass ratios:")
    print("  muon/electron  = 206.8")
    print("  proton/electron = 1836.2")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Variational Minimum Energy per Knot Topology",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    # Bar chart of mass ratios
    ax = axes[0]
    labels = [f"({r['p']},{r['q']})\n{r['name']}" for r in results]
    ratios = [r['E_total'] / e_ref for r in results]
    bars = ax.bar(range(len(results)), ratios, color=colors[:len(results)])
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mass ratio (m / m_e)')
    ax.set_title('Mass Ratios (variational)')
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.01, f'{v:.3f}x', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Energy vs crossing number
    ax = axes[1]
    crossings = [r['crossing'] for r in results]
    energies = [r['E_total'] for r in results]
    ropelengths = [r['ropelength_ratio'] for r in results]
    ax.plot(crossings, ratios, 'ko-', markersize=10, linewidth=2,
            label='Variational')
    ax.plot(crossings, ropelengths, 'rs--', markersize=8, linewidth=1.5,
            label='Ropelength ratio')
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Mass ratio')
    ax.set_title('Mass ratio vs complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy composition
    ax = axes[2]
    x = np.arange(len(results))
    width = 0.2
    ax.bar(x - 1.5*width, [r['E_em'] / e_ref for r in results],
           width, label='EM', color='blue', alpha=0.7)
    ax.bar(x - 0.5*width, [r['E_pivot'] / e_ref for r in results],
           width, label='P²', color='red', alpha=0.7)
    ax.bar(x + 0.5*width, [r['E_grad'] / e_ref for r in results],
           width, label='∇P', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({r['p']},{r['q']})" for r in results], fontsize=9)
    ax.set_ylabel('Energy / E_electron')
    ax.set_title('Energy composition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/variational_spectrum.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def run_spectrum(save=False):
    """Comprehensive mass spectrum analysis.

    Combine variational energy with ropelength ratios
    and knot-theoretic invariants.
    """
    print("=" * 60)
    print("Comprehensive Knot Mass Spectrum")
    print("=" * 60)

    # Extended knot table
    knots = [
        # (p, q, name, known_mass_ratio)
        (2, 1, "unknot", 1.0),         # electron
        (3, 2, "trefoil", None),        # proton?
        (5, 2, "5₁", None),
        (7, 2, "7₁", None),
        (4, 3, "8₁₈", None),
        (5, 3, "10₁₂₄", None),
        (7, 3, "T(7,3)", None),
        (5, 4, "T(5,4)", None),
    ]

    R = 5.0  # fixed major radius for comparison

    print(f"\nFixed R={R}, optimizing r and a for each knot\n")

    results = []
    for p, q, name, known_ratio in knots:
        crossing = min(p, q) * (max(p, q) - 1)
        rl = ropelength_ratio(p, q)

        # Optimize r and a for fixed R
        best_E = np.inf
        best_params = None

        for r_val in np.logspace(-0.5, np.log10(R * 0.8), 30):
            for a_val in np.logspace(-2, np.log10(r_val * 0.9), 30):
                try:
                    e = knot_energy_analytic(p, q, R, r_val, a_val)
                    if e['E_total'] < best_E:
                        best_E = e['E_total']
                        best_params = (r_val, a_val, e)
                except Exception:
                    continue

        if best_params is None:
            continue

        r_opt, a_opt, e_opt = best_params
        L = torus_knot_length(p, q, R, r_opt)

        results.append({
            'p': p, 'q': q, 'name': name,
            'crossing': crossing,
            'R': R, 'r': r_opt, 'a': a_opt,
            'L': L,
            'E_total': e_opt['E_total'],
            'E_em': e_opt['E_em'],
            'E_pivot': e_opt['E_pivot'],
            'kappa_avg': e_opt['kappa_avg'],
            'ropelength': rl,
            'known_ratio': known_ratio,
        })

    # Compute ratios
    e_ref = results[0]['E_total']

    print(f"{'Knot':>12s} {'Cross':>6s} {'L':>8s} {'κ_avg':>8s} "
          f"{'E_var':>12s} {'Ratio':>8s} {'Rope':>6s} {'L_ratio':>8s}")
    print("-" * 80)

    for r_item in results:
        ratio = r_item['E_total'] / e_ref
        L_ratio = r_item['L'] / results[0]['L']
        print(f"({r_item['p']},{r_item['q']}) {r_item['name']:>6s} "
              f"{r_item['crossing']:>6d} "
              f"{r_item['L']:>8.2f} {r_item['kappa_avg']:>8.4f} "
              f"{r_item['E_total']:>12.6e} {ratio:>8.3f}x "
              f"{r_item['ropelength']:>6.2f} {L_ratio:>8.3f}")

    # What scaling would give proton mass?
    # If mass ~ L^n × κ^m, what n, m reproduce known masses?
    print("\n\nScaling analysis:")
    L_trefoil = results[1]['L']
    L_electron = results[0]['L']
    kappa_trefoil = results[1]['kappa_avg']
    kappa_electron = results[0]['kappa_avg']

    L_ratio = L_trefoil / L_electron
    k_ratio = kappa_trefoil / kappa_electron

    print(f"  L_trefoil/L_electron = {L_ratio:.3f}")
    print(f"  κ_trefoil/κ_electron = {k_ratio:.3f}")
    print(f"  Ropelength ratio = {results[1]['ropelength']:.3f}")

    # If m ~ L^n: 1836 = L_ratio^n → n = log(1836)/log(L_ratio)
    if L_ratio > 1:
        n_L = np.log(1836.2) / np.log(L_ratio)
        print(f"  For m~L^n to give proton: n = {n_L:.2f}")

    # If m ~ κ^n: 1836 = k_ratio^n → n = log(1836)/log(k_ratio)
    if k_ratio > 1:
        n_k = np.log(1836.2) / np.log(k_ratio)
        print(f"  For m~κ^n to give proton: n = {n_k:.2f}")

    # If m ~ (L×κ)^n
    Lk_ratio = L_ratio * k_ratio
    if Lk_ratio > 1:
        n_Lk = np.log(1836.2) / np.log(Lk_ratio)
        print(f"  For m~(Lκ)^n to give proton: n = {n_Lk:.2f}")

    # Ropelength scaling
    rl_ratio = results[1]['ropelength']
    if rl_ratio > 1:
        n_rl = np.log(1836.2) / np.log(rl_ratio)
        print(f"  For m~ropelength^n to give proton: n = {n_rl:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Knot Mass Spectrum — Scaling Analysis",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    # Variational energy ratio
    ax = axes[0]
    crossings = [r['crossing'] for r in results]
    var_ratios = [r['E_total'] / e_ref for r in results]
    ax.semilogy(crossings, var_ratios, 'ko-', markersize=10, linewidth=2,
                label='Variational E')
    # Add ropelength for comparison
    rope_ratios = [r['ropelength'] for r in results]
    ax.semilogy(crossings, rope_ratios, 'rs--', markersize=8, linewidth=1.5,
                label='Ropelength')
    # Add L ratio
    L_ratios = [r['L'] / results[0]['L'] for r in results]
    ax.semilogy(crossings, L_ratios, 'b^:', markersize=8, linewidth=1.5,
                label='Curve length')
    ax.axhline(y=206.8, color='gray', linestyle=':', alpha=0.5)
    ax.text(max(crossings) * 0.7, 206.8, 'muon', fontsize=8, alpha=0.5)
    ax.axhline(y=1836.2, color='gray', linestyle=':', alpha=0.5)
    ax.text(max(crossings) * 0.7, 1836.2, 'proton', fontsize=8, alpha=0.5)
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Ratio (log scale)')
    ax.set_title('Scaling with knot complexity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Curvature vs crossing
    ax = axes[1]
    kappas = [r['kappa_avg'] for r in results]
    ax.plot(crossings, kappas, 'go-', markersize=10, linewidth=2)
    for i, r in enumerate(results):
        ax.annotate(f"({r['p']},{r['q']})", (crossings[i], kappas[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Average curvature')
    ax.set_title('Curvature vs complexity')
    ax.grid(True, alpha=0.3)

    # What exponent n is needed for each observable to reach proton mass?
    ax = axes[2]
    observables = {
        'L (curve length)': L_ratios,
        'κ (curvature)': [r['kappa_avg'] / results[0]['kappa_avg'] for r in results],
        'Ropelength': rope_ratios,
        'Variational E': var_ratios,
    }
    for label, vals in observables.items():
        needed_n = []
        for v in vals:
            if v > 1:
                needed_n.append(np.log(1836.2) / np.log(v))
            else:
                needed_n.append(np.nan)
        ax.plot(crossings, needed_n, 'o-', markersize=8, linewidth=1.5,
                label=label)
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Exponent n needed for m_proton')
    ax.set_title('Required scaling exponent')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/variational_scaling.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def run_self_avoiding(save=False):
    """Mass spectrum with self-avoidance constraint.

    The key insight: for a knotted tube, the maximum tube radius is
    constrained by the minimum distance between non-adjacent strands.
    More complex knots → strands closer together → smaller max tube radius
    → higher energy density → MORE mass.

    This inverts the naive scaling: without self-avoidance, complex knots
    have LESS energy. With it, they have MORE.
    """
    print("=" * 60)
    print("Self-Avoiding Knot Mass Spectrum")
    print("=" * 60)

    knots = [
        (2, 1, "unknot (e⁻)", 1.0),
        (3, 2, "trefoil (p?)", 1836.2),
        (5, 2, "5₁", None),
        (7, 2, "7₁", None),
        (4, 3, "8₁₈", None),
        (5, 3, "10₁₂₄", None),
        (7, 3, "T(7,3)", None),
        (5, 4, "T(5,4)", None),
    ]

    # For each knot, find optimal R, r subject to a < d_min/2
    # Fix R and scan r to find the configuration that minimizes energy
    # while respecting self-avoidance.

    print("\nPhase 1: Computing self-avoidance constraints...\n")

    results = []
    for p, q, name, known_ratio in knots:
        crossing = min(p, q) * (max(p, q) - 1)
        rl = ropelength_ratio(p, q)

        # Scan R and r to find optimal configuration
        best_E = np.inf
        best_params = None

        for R in [3.0, 5.0, 8.0, 12.0]:
            for r_val in np.linspace(0.3, R * 0.6, 20):
                # Compute self-avoidance constraint
                d_min = min_strand_distance(p, q, R, r_val)
                a_max = d_min / 2.0  # tube can't exceed half the min gap

                if a_max < 0.01:  # too tight, skip
                    continue

                # With the constraint, find optimal a ≤ a_max
                # Energy typically decreases with a (spreading charge out)
                # so optimal a = a_max (as large as self-avoidance allows)
                a_opt = a_max * 0.95  # slight margin

                try:
                    e = knot_energy_analytic(p, q, R, r_val, a_opt)
                    if e['E_total'] < best_E:
                        best_E = e['E_total']
                        best_params = {
                            'R': R, 'r': r_val, 'a': a_opt,
                            'd_min': d_min, 'energy': e,
                        }
                except Exception:
                    continue

        if best_params is None:
            print(f"  ({p},{q}) {name}: FAILED to find valid config")
            continue

        e = best_params['energy']
        L = e['L']

        # Compute Coulomb self-energy
        E_coulomb = coulomb_self_energy(
            p, q, best_params['R'], best_params['r'], best_params['a'])

        E_total_with_coulomb = e['E_total'] + E_coulomb

        results.append({
            'p': p, 'q': q, 'name': name,
            'crossing': crossing,
            'R': best_params['R'], 'r': best_params['r'],
            'a': best_params['a'], 'd_min': best_params['d_min'],
            'L': L,
            'E_local': e['E_total'],
            'E_coulomb': E_coulomb,
            'E_total': E_total_with_coulomb,
            'E_em': e['E_em'], 'E_pivot': e['E_pivot'],
            'E_grad': e['E_grad'],
            'kappa_avg': e['kappa_avg'],
            'ropelength': rl,
            'known_ratio': known_ratio,
        })

        print(f"  ({p},{q}) {name:>10s}: R={best_params['R']:.1f}, "
              f"r={best_params['r']:.2f}, a={best_params['a']:.4f}, "
              f"d_min={best_params['d_min']:.4f}, "
              f"E_local={e['E_total']:.4e}, E_coul={E_coulomb:.4e}")

    if not results:
        print("No valid results!")
        return

    # Mass spectrum
    e_ref = results[0]['E_total']

    print(f"\n{'=' * 80}")
    print("SELF-AVOIDING MASS SPECTRUM")
    print(f"{'=' * 80}")
    print(f"{'Knot':>12s} {'Cross':>6s} {'a_max':>8s} "
          f"{'E_local':>10s} {'E_coul':>10s} {'E_total':>10s} "
          f"{'Ratio':>8s} {'Rope':>6s}")
    print("-" * 90)

    for r_item in results:
        ratio = r_item['E_total'] / e_ref
        print(f"({r_item['p']},{r_item['q']}) {r_item['name']:>10s} "
              f"{r_item['crossing']:>6d} "
              f"{r_item['a']:>8.4f} "
              f"{r_item['E_local']:>10.4e} {r_item['E_coulomb']:>10.4e} "
              f"{r_item['E_total']:>10.4e} {ratio:>8.3f}x "
              f"{r_item['ropelength']:>6.2f}")

    print("\nKnown mass ratios:")
    print("  muon/electron  = 206.8")
    print("  proton/electron = 1836.2")

    # Scaling analysis
    if len(results) > 1:
        print("\nScaling analysis:")
        for i, r in enumerate(results[1:], 1):
            ratio = r['E_total'] / e_ref
            a_ratio = results[0]['a'] / r['a']  # how much tighter
            L_ratio = r['L'] / results[0]['L']
            print(f"  ({r['p']},{r['q']}): "
                  f"a_e/a_knot = {a_ratio:.2f}x tighter, "
                  f"L_knot/L_e = {L_ratio:.2f}x longer, "
                  f"mass ratio = {ratio:.3f}x")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Self-Avoiding Knot Mass Spectrum\n"
                 "(tube radius constrained by minimum strand distance)",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    # 1. Mass ratios bar chart
    ax = axes[0, 0]
    labels = [f"({r['p']},{r['q']})\n{r['name']}" for r in results]
    ratios = [r['E_total'] / e_ref for r in results]
    ax.bar(range(len(results)), ratios, color=colors)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Mass ratio (m / m_e)')
    ax.set_title('Mass Spectrum')
    for i, v in enumerate(ratios):
        ax.text(i, v + max(ratios) * 0.02, f'{v:.2f}x', ha='center', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. d_min and a_max vs crossing number
    ax = axes[0, 1]
    crossings = [r['crossing'] for r in results]
    d_mins = [r['d_min'] for r in results]
    a_maxs = [r['a'] for r in results]
    ax.semilogy(crossings, d_mins, 'ro-', markersize=8, label='d_min (strand gap)')
    ax.semilogy(crossings, a_maxs, 'bs-', markersize=8, label='a_max (tube radius)')
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Distance')
    ax.set_title('Self-avoidance constraint')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mass ratio vs crossing (log scale)
    ax = axes[0, 2]
    ax.semilogy(crossings, ratios, 'ko-', markersize=10, linewidth=2,
                label='Self-avoiding')
    rope_ratios = [r['ropelength'] for r in results]
    ax.semilogy(crossings, rope_ratios, 'rs--', markersize=8, label='Ropelength')
    ax.axhline(y=206.8, color='orange', linestyle=':', alpha=0.7)
    ax.text(max(crossings) * 0.7, 206.8 * 1.2, 'muon', fontsize=9, color='orange')
    ax.axhline(y=1836.2, color='red', linestyle=':', alpha=0.7)
    ax.text(max(crossings) * 0.7, 1836.2 * 1.2, 'proton', fontsize=9, color='red')
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Mass ratio (log)')
    ax.set_title('Mass vs complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Energy composition
    ax = axes[1, 0]
    x = np.arange(len(results))
    width = 0.25
    ax.bar(x - width, [r['E_em'] / e_ref for r in results],
           width, label='EM', color='blue', alpha=0.7)
    ax.bar(x, [r['E_pivot'] / e_ref for r in results],
           width, label='P²', color='red', alpha=0.7)
    ax.bar(x + width, [r['E_grad'] / e_ref for r in results],
           width, label='∇P', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({r['p']},{r['q']})" for r in results], fontsize=9)
    ax.set_ylabel('Energy / E_electron')
    ax.set_title('Energy composition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Tube radius vs curve length
    ax = axes[1, 1]
    Ls = [r['L'] for r in results]
    ax.plot(Ls, a_maxs, 'go-', markersize=10)
    for i, r in enumerate(results):
        ax.annotate(f"({r['p']},{r['q']})", (Ls[i], a_maxs[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel('Curve length L')
    ax.set_ylabel('Max tube radius a')
    ax.set_title('Self-avoidance: longer → thinner')
    ax.grid(True, alpha=0.3)

    # 6. Knot projections
    ax = axes[1, 2]
    t = np.linspace(0, 2 * np.pi, 1000)
    knot_colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        p_k, q_k = r['p'], r['q']
        R_k, r_k = r['R'], r['r']
        x_knot = (R_k + r_k * np.cos(q_k * t)) * np.cos(p_k * t)
        y_knot = (R_k + r_k * np.cos(q_k * t)) * np.sin(p_k * t)
        ax.plot(x_knot / R_k, y_knot / R_k, color=knot_colors[i],
                linewidth=1, alpha=0.7, label=f"({p_k},{q_k})")
    ax.set_aspect('equal')
    ax.set_title('Knot curves (XY)')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/variational_self_avoiding.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def run_tight_knots(save=False):
    """Mass spectrum from tight (ropelength-minimized) knot configurations.

    Instead of fixing R and optimizing r, use the tight configuration
    where the knot fills its available volume as efficiently as possible.
    The tube diameter is fixed (= 1 in natural units), and the knot
    geometry is determined by the minimum-length configuration.

    In this approach:
    - All knots have the SAME tube radius (= the Compton wavelength)
    - Different knots have different total lengths
    - The charge density ~ 1/L, so shorter knots have more charge density
    - The key question: does the self-energy scale appropriately?

    Known tight ropelengths (in units of tube diameter):
        unknot:    2π ≈ 6.28
        trefoil:   16.37
        figure-8:  21.04
        5₁ (cinq): 26.29
        5₂:        24.68
    """
    print("=" * 60)
    print("Tight Knot Mass Spectrum")
    print("=" * 60)
    print("\nFixed tube radius a=1 (Compton wavelength)")
    print("Knot geometry from ropelength minimization\n")

    # Known tight ropelength values (tube diameter = 1)
    # From Cantarella, Kusner, Sullivan (2002) and Ashton et al.
    knots = [
        # (p, q, name, ropelength, crossing_number)
        (2, 1, "unknot (e⁻)", 2 * np.pi, 0),   # circle
        (3, 2, "trefoil (p?)", 16.37, 3),
        (5, 2, "5₁", 26.29, 5),
        (7, 2, "7₁", 36.0, 7),     # estimated
        (4, 3, "8₁₈", 32.74, 8),   # estimated
        (5, 3, "10₁₂₄", 39.0, 10), # estimated
    ]

    a = 1.0  # fixed tube radius (Compton wavelength)

    results = []
    for p, q, name, ropelen, cross in knots:
        L = ropelen  # curve length = ropelength (in tube-diameter units)

        # For tight configuration, R and r are determined by ropelength
        # Approximate: for torus knot, R ~ L/(2πp), r ~ a × (some factor)
        # But for tight knots, the actual shape isn't a torus anymore.
        # We use the ropelength directly for L.

        # Charge normalization: total charge = 1
        # Linear charge density: λ = 1/L
        lam = 1.0 / L

        # EM energy of the tube: E_em = Q²/(2C) where C ~ L (capacitance of a line)
        # More precisely, for a uniformly charged tube:
        # E_em ~ λ² × L × ln(L/a) (logarithmic correction for thin tube)
        E_em = 0.5 * lam**2 * L * np.log(max(L / a, 2.0))

        # Pivot field energy: P ~ div(E) ~ λ/a
        # E_pivot ~ P² × V_tube = (λ/a)² × πa²L = πλ²L
        P_amp = lam / a
        V_tube = np.pi * a**2 * L
        E_pivot = 0.5 * P_amp**2 * V_tube

        # Coulomb self-energy: dominant for tight knots
        # For a tight trefoil, crossings bring charge within distance ~a
        # E_coulomb ≈ Σ_crossings λ²/a + background
        # Background: uniform ring ~ λ² L ln(L/a)
        # Crossings: each crossing contributes ~ λ²/(2a) (two strand segments near each other)
        # For n_cross crossings: E_cross ~ n_cross × λ² × (L_near/a)
        # where L_near ~ 2a (length of each strand segment near the crossing)
        E_coulomb_bg = lam**2 * L * np.log(max(L / a, 2.0)) / (4 * np.pi)
        # At each crossing, ~2a of each strand are within distance ~a of each other
        # Interaction energy per crossing ~ λ² × (2a)² / a = 4λ²a
        E_coulomb_crossings = cross * 4.0 * lam**2 * a
        E_coulomb = E_coulomb_bg + E_coulomb_crossings

        E_total = E_em + E_pivot + E_coulomb

        results.append({
            'p': p, 'q': q, 'name': name,
            'crossing': cross,
            'L': L, 'a': a,
            'E_em': E_em, 'E_pivot': E_pivot,
            'E_coulomb_bg': E_coulomb_bg,
            'E_coulomb_cross': E_coulomb_crossings,
            'E_coulomb': E_coulomb,
            'E_total': E_total,
        })

    # Compute ratios
    e_ref = results[0]['E_total']

    print(f"{'Knot':>12s} {'Cross':>5s} {'L':>7s} "
          f"{'E_em':>9s} {'E_P':>9s} {'E_coul':>9s} "
          f"{'E_total':>9s} {'Ratio':>8s}")
    print("-" * 80)

    for r in results:
        ratio = r['E_total'] / e_ref
        print(f"({r['p']},{r['q']}) {r['name']:>10s} "
              f"{r['crossing']:>5d} {r['L']:>7.2f} "
              f"{r['E_em']:>9.4e} {r['E_pivot']:>9.4e} "
              f"{r['E_coulomb']:>9.4e} "
              f"{r['E_total']:>9.4e} {ratio:>8.3f}x")

    print(f"\n  Target: trefoil/unknot = 1836.2")

    # What scaling WOULD work?
    print("\n--- What scaling gives proton mass? ---")
    L_e = results[0]['L']
    L_t = results[1]['L']
    c_e = results[0]['crossing']
    c_t = results[1]['crossing']

    # If mass ~ 1/L^n (shorter = heavier from charge density)
    ratio_L = L_e / L_t  # < 1 since trefoil is longer
    print(f"  L_unknot/L_trefoil = {ratio_L:.3f}")
    print(f"  (Longer knots have LESS charge density → this goes wrong direction)")

    # If mass ~ crossing^n
    if c_t > 0:
        n_cross = np.log(1836.2) / np.log(c_t)
        print(f"  For m ~ crossing^n: 1836 = 3^n → n = {n_cross:.2f}")

    # If mass ~ exp(c × crossing)
    if c_t > 0:
        c_exp = np.log(1836.2) / c_t
        print(f"  For m ~ exp(c × crossing): c = {c_exp:.4f}")
        print(f"    Predictions:")
        for r in results:
            if r['crossing'] > 0:
                m_pred = np.exp(c_exp * r['crossing'])
                print(f"      ({r['p']},{r['q']}) crossing={r['crossing']}: "
                      f"m/m_e = {m_pred:.1f}")

    # If mass ~ ropelength^n
    rl_ratio = L_t / L_e
    if rl_ratio > 1:
        n_rl = np.log(1836.2) / np.log(rl_ratio)
        print(f"  For m ~ ropelength^n: n = {n_rl:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tight Knot Mass Spectrum\n"
                 "Fixed tube radius, variable ropelength",
                 fontsize=13, fontweight='bold')

    # 1. Energy components
    ax = axes[0, 0]
    x = np.arange(len(results))
    width = 0.25
    ax.bar(x - width, [r['E_em'] / e_ref for r in results],
           width, label='EM', color='blue', alpha=0.7)
    ax.bar(x, [r['E_pivot'] / e_ref for r in results],
           width, label='P² (pivot)', color='red', alpha=0.7)
    ax.bar(x + width, [r['E_coulomb'] / e_ref for r in results],
           width, label='Coulomb', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({r['p']},{r['q']})" for r in results])
    ax.set_ylabel('Energy / E_electron')
    ax.set_title('Energy Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mass ratios (actual model)
    ax = axes[0, 1]
    ratios = [r['E_total'] / e_ref for r in results]
    crossings = [r['crossing'] for r in results]
    ax.plot(crossings, ratios, 'ko-', markersize=10, linewidth=2,
            label='Model (local + Coulomb)')
    # Exponential fit
    if c_t > 0:
        c_exp = np.log(1836.2) / c_t
        x_fit = np.linspace(0, max(crossings), 100)
        ax.plot(x_fit, np.exp(c_exp * x_fit), 'r--', linewidth=1.5,
                label=f'exp({c_exp:.2f}×crossing)', alpha=0.7)
    ax.axhline(y=206.8, color='orange', linestyle=':', alpha=0.7)
    ax.text(8, 206.8 * 1.3, 'muon', fontsize=9, color='orange')
    ax.axhline(y=1836.2, color='red', linestyle=':', alpha=0.7)
    ax.text(8, 1836.2 * 1.3, 'proton', fontsize=9, color='red')
    ax.set_xlabel('Crossing Number')
    ax.set_ylabel('Mass ratio (log)')
    ax.set_yscale('log')
    ax.set_title('Mass Spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Crossing contribution analysis
    ax = axes[1, 0]
    bg = [r['E_coulomb_bg'] / e_ref for r in results]
    xing = [r['E_coulomb_cross'] / e_ref for r in results]
    ax.bar(x, bg, 0.4, label='Background Coulomb', color='lightgreen')
    ax.bar(x, xing, 0.4, bottom=bg, label='Crossing Coulomb', color='darkgreen')
    ax.set_xticks(x)
    ax.set_xticklabels([f"({r['p']},{r['q']})\ncross={r['crossing']}"
                        for r in results], fontsize=7)
    ax.set_ylabel('Energy / E_electron')
    ax.set_title('Coulomb: background vs crossings')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        "KEY FINDINGS\n"
        "━━━━━━━━━━━━\n\n"
        "1. Charge normalization (Q=1 for all knots)\n"
        "   → longer knots have LESS charge density\n"
        "   → E_em and E_pivot DECREASE with complexity\n\n"
        "2. Self-avoidance constrains tube radius\n"
        "   → more complex knots are tighter\n"
        "   → but effect is weak (~2x at most)\n\n"
        "3. Coulomb self-energy from crossings\n"
        "   → each crossing brings strands within ~a\n"
        "   → adds ~ 4λ²a per crossing\n"
        "   → but still small vs background\n\n"
        f"4. Actual model ratio (trefoil): {results[1]['E_total']/e_ref:.3f}x\n"
        f"   Needed: 1836x\n\n"
        "5. EXPONENTIAL scaling with crossing #\n"
        "   would match: m ~ exp(2.51 × crossing)\n"
        "   → requires nonlinear self-interaction\n"
        "   → NOT in linearized Maxwell + pivot"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/variational_tight.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def run_particle_map(save=False):
    """Map known particle masses to topological invariants.

    If Williamson's hypothesis is correct, particle masses should
    correlate with some topological quantity. Test:
    - crossing number
    - ropelength
    - braid word length
    - writhe
    - genus

    Known stable/long-lived particles and their masses (in electron masses):
    """
    print("=" * 60)
    print("Particle Mass → Topology Mapping")
    print("=" * 60)

    # Particle data: name, mass (MeV), mass ratio to electron, stability, proposed topology
    m_e = 0.511  # MeV
    particles = [
        ("e⁻", 0.511, 1.0, "stable", "unknot (2,1)"),
        ("μ⁻", 105.66, 206.8, "τ=2.2μs", "excited unknot?"),
        ("τ⁻", 1776.86, 3477.2, "τ=2.9e-13s", "?"),
        ("π⁰", 134.98, 264.1, "τ=8.4e-17s", "link?"),
        ("π±", 139.57, 273.1, "τ=2.6e-8s", "link?"),
        ("K±", 493.68, 966.1, "τ=1.2e-8s", "?"),
        ("p", 938.27, 1836.2, "stable", "trefoil (3,2)?"),
        ("n", 939.57, 1838.7, "τ=880s", "trefoil variant?"),
        ("Λ", 1115.68, 2183.3, "τ=2.6e-10s", "?"),
        ("Σ±", 1189.37, 2327.6, "τ=8.0e-11s", "?"),
        ("Ω⁻", 1672.45, 3272.0, "τ=8.2e-11s", "?"),
    ]

    print(f"\n{'Particle':>8s} {'Mass(MeV)':>10s} {'m/m_e':>10s} {'ln(m/m_e)':>10s} "
          f"{'Stability':>15s} {'Proposed':>20s}")
    print("-" * 80)

    for name, mass, ratio, stability, topology in particles:
        ln_ratio = np.log(ratio)
        print(f"{name:>8s} {mass:>10.2f} {ratio:>10.1f} {ln_ratio:>10.3f} "
              f"{stability:>15s} {topology:>20s}")

    # Analysis: if m = m_e × exp(c × n), what n values are implied?
    c = np.log(1836.2) / 3  # calibrated from proton = trefoil (crossing 3)
    print(f"\n\nIf m = m_e × exp({c:.4f} × n):")
    print(f"{'Particle':>8s} {'m/m_e':>10s} {'Implied n':>10s} {'Nearest integer':>15s}")
    print("-" * 50)

    for name, mass, ratio, stability, topology in particles:
        if ratio > 1:
            n_implied = np.log(ratio) / c
            n_near = round(n_implied)
            m_predicted = np.exp(c * n_near)
            err = abs(ratio - m_predicted) / ratio * 100
            print(f"{name:>8s} {ratio:>10.1f} {n_implied:>10.3f} "
                  f"{n_near:>10d} (pred={m_predicted:.1f}, err={err:.1f}%)")

    # Also try c calibrated from muon mass
    # If muon is n=2: c = ln(206.8)/2 = 2.664
    c2 = np.log(206.8) / 2
    print(f"\n\nAlternative: muon = n=2, c = {c2:.4f}")
    print(f"{'Particle':>8s} {'m/m_e':>10s} {'Implied n':>10s} {'Nearest int':>11s}")
    print("-" * 50)

    for name, mass, ratio, stability, topology in particles:
        if ratio > 1:
            n_implied = np.log(ratio) / c2
            n_near = round(n_implied)
            m_predicted = np.exp(c2 * n_near)
            err = abs(ratio - m_predicted) / ratio * 100
            print(f"{name:>8s} {ratio:>10.1f} {n_implied:>10.3f} "
                  f"{n_near:>10d} (pred={m_predicted:.1f}, err={err:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Particle Masses and Topological Scaling",
                 fontsize=13, fontweight='bold')

    # 1. ln(mass) spectrum
    ax = axes[0]
    masses = [p[2] for p in particles]
    names = [p[0] for p in particles]
    ln_masses = [np.log(m) for m in masses]
    colors_stability = ['green' if 'stable' in p[3] else
                        'orange' if 'μs' in p[3] or 's,' in p[3] else
                        'red' for p in particles]
    ax.barh(range(len(particles)), ln_masses,
            color=colors_stability, alpha=0.7)
    ax.set_yticks(range(len(particles)))
    ax.set_yticklabels(names)
    ax.set_xlabel('ln(m / m_e)')
    ax.set_title('Particle mass spectrum (log scale)')
    # Add grid lines at integer n values
    for n_val in range(1, 10):
        ax.axvline(x=c * n_val, color='gray', linestyle=':', alpha=0.3)
        ax.text(c * n_val, len(particles) - 0.5, f'n={n_val}',
                fontsize=7, alpha=0.5, ha='center')

    # 2. Mass vs implied n (proton calibration)
    ax = axes[1]
    n_implied_vals = [np.log(p[2]) / c for p in particles]
    ax.scatter(n_implied_vals, np.log10(masses), s=100,
               c=colors_stability, edgecolors='black', zorder=5)
    for i, p in enumerate(particles):
        ax.annotate(p[0], (n_implied_vals[i], np.log10(masses[i])),
                    textcoords="offset points", xytext=(8, 3), fontsize=8)
    # Perfect fit line
    n_line = np.linspace(0, 5, 100)
    ax.plot(n_line, np.log10(np.exp(c * n_line)), 'r--', linewidth=1.5,
            label=f'm = exp({c:.2f}n)', alpha=0.7)
    ax.set_xlabel(f'Implied topological number n (c={c:.3f})')
    ax.set_ylabel('log₁₀(m / m_e)')
    ax.set_title('Mass vs topological number\n(proton = n=3 calibration)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Knot table with known crossing numbers
    ax = axes[2]
    ax.axis('off')

    knot_table = (
        "KNOWN TORUS KNOT PROPERTIES\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Knot    Cross  Ropelen  Particle?\n"
        "─────────────────────────────────\n"
        "unknot    0      6.28   electron ✓\n"
        "(2,1)     1*     —      —\n"
        "trefoil   3     16.37   proton? (1836×)\n"
        "fig-8     4     21.04   —\n"
        "5₁        5     26.29   —\n"
        "5₂        5     24.68   —\n"
        "6₁        6     28.0    —\n"
        "7₁        7     36.0    —\n\n"
        "* (2,1) torus 'knot' = unknot\n\n"
        "OPEN QUESTIONS:\n"
        "━━━━━━━━━━━━━━\n"
        "• Muon: excited unknot\n"
        "  or topological link?\n"
        "• Pion: 2-component link\n"
        "  (photon pair)?\n"
        "• Why exp(c×crossing)?\n"
        "  → nonlinear self-interaction\n"
        "  → each crossing = energy barrier"
    )
    ax.text(0.05, 0.95, knot_table, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/variational_particle_map.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Variational knot energy — Williamson model')
    parser.add_argument('--mode',
                        choices=['energy_landscape', 'minimize', 'spectrum',
                                 'self_avoiding', 'tight', 'particle_map'],
                        default='spectrum')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.mode == 'energy_landscape':
        run_energy_landscape(save=args.save)
    elif args.mode == 'minimize':
        run_minimize(save=args.save)
    elif args.mode == 'spectrum':
        run_spectrum(save=args.save)
    elif args.mode == 'self_avoiding':
        run_self_avoiding(save=args.save)
    elif args.mode == 'tight':
        run_tight_knots(save=args.save)
    elif args.mode == 'particle_map':
        run_particle_map(save=args.save)


if __name__ == '__main__':
    main()
