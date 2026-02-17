#!/usr/bin/env python3
"""
Williamson-Benn Charge Calculation with Golden Ratio Correction
================================================================

Computes the effective electric charge of a photon confined to a
helical path on a torus, following Williamson's 1997 model.

Key insight from Arnie Benn: the helical pitch angle introduces a
correction factor related to the golden ratio phi = (1+sqrt(5))/2.

The charge arises because path curvature creates effective div(E),
which by Gauss's law acts as charge density. The integrated charge
depends on:
  1. Aspect ratio R/r (major/minor radius)
  2. Winding ratio w (poloidal/toroidal frequency)
  3. Helical pitch angle

Finding: R/r = phi gives charge factor q ≈ 0.949, remarkably close
to unity. The q=1.0 contour in (w, R/r) parameter space passes
through the golden ratio neighborhood.

Usage:
    python3 williamson_charge_golden.py
    python3 williamson_charge_golden.py --save
    python3 williamson_charge_golden.py --aspect 1.618 --winding 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

PHI = (1 + np.sqrt(5)) / 2


def compute_charge_from_path(R, r, winding_ratio, n_path=2000):
    """Compute effective charge from helical photon path on torus.

    The charge arises from path curvature projecting onto the
    radial direction (toward/away from torus symmetry axis).
    Curvature creates effective div(E) = charge density.

    q_eff = ∮ (κ · r̂) ds

    where κ is the curvature vector and r̂ points toward the axis.

    Args:
        R: Major radius (distance from axis to torus center)
        r: Minor radius (tube radius)
        winding_ratio: poloidal winds per toroidal wind
        n_path: Number of discretization points

    Returns:
        dict with q_integral, q_ratio, path_length, mean_kappa
    """
    s = np.linspace(0, 2 * np.pi, n_path, endpoint=False)
    ds_val = s[1] - s[0]
    theta = winding_ratio * s

    # Path on torus
    px = (R + r * np.cos(theta)) * np.cos(s)
    py = (R + r * np.cos(theta)) * np.sin(s)
    pz = r * np.sin(theta)

    # Tangent vector
    dx = (-(R + r * np.cos(theta)) * np.sin(s)
          - r * np.sin(theta) * winding_ratio * np.cos(s))
    dy = ((R + r * np.cos(theta)) * np.cos(s)
          - r * np.sin(theta) * winding_ratio * np.sin(s))
    dz = r * np.cos(theta) * winding_ratio
    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    tx, ty, tz = dx / speed, dy / speed, dz / speed

    # Curvature: κ = |dT/ds| / |dr/ds|
    dtx = np.gradient(tx, ds_val) / speed
    dty = np.gradient(ty, ds_val) / speed
    dtz = np.gradient(tz, ds_val) / speed
    kappa = np.sqrt(dtx**2 + dty**2 + dtz**2)
    kn_mag = kappa + 1e-15
    kn_x, kn_y, kn_z = dtx / kn_mag, dty / kn_mag, dtz / kn_mag

    # Radial direction (toward torus symmetry axis)
    rho = np.sqrt(px**2 + py**2) + 1e-10
    rad_x, rad_y = -px / rho, -py / rho

    # Radial projection of curvature normal
    kappa_radial = kappa * (kn_x * rad_x + kn_y * rad_y)

    # Integrate charge
    q_integral = np.sum(kappa_radial * speed) * ds_val
    path_length = np.sum(speed) * ds_val

    # Normalize to circular orbit (baseline charge)
    q_circular = 2 * np.pi / R
    q_ratio = abs(q_integral / q_circular)

    return {
        'q_integral': q_integral,
        'q_ratio': q_ratio,
        'path_length': path_length,
        'mean_kappa': np.mean(kappa),
    }


def scan_winding_ratio(R=3.0, r=1.0, n_points=80, n_path=2000):
    """Scan charge factor over winding ratios."""
    winding_ratios = np.linspace(0.5, 4.0, n_points)
    q_ratios = []
    for w in winding_ratios:
        res = compute_charge_from_path(R, r, w, n_path=n_path)
        q_ratios.append(res['q_ratio'])
    return winding_ratios, np.array(q_ratios)


def scan_aspect_ratio(winding=2.0, r=1.0, n_points=80, n_path=2000):
    """Scan charge factor over aspect ratios R/r."""
    aspect_ratios = np.linspace(1.3, 8.0, n_points)
    q_ratios = []
    for alpha in aspect_ratios:
        res = compute_charge_from_path(alpha * r, r, winding, n_path=n_path)
        q_ratios.append(res['q_ratio'])
    return aspect_ratios, np.array(q_ratios)


def compute_charge_map(w_range, a_range, r=1.0, n_path=500):
    """Compute charge factor over 2D parameter space."""
    Q = np.zeros((len(a_range), len(w_range)))
    for i, a in enumerate(a_range):
        for j, w in enumerate(w_range):
            res = compute_charge_from_path(a * r, r, w, n_path=n_path)
            Q[i, j] = res['q_ratio']
    return Q


def main(save=False, aspect=None, winding=None):
    print("=" * 60)
    print("Williamson-Benn Charge Calculation")
    print(f"Golden ratio phi = {PHI:.6f}")
    print("=" * 60)

    if aspect is not None and winding is not None:
        # Single-point calculation
        res = compute_charge_from_path(aspect, 1.0, winding, n_path=20000)
        print(f"\n  R/r = {aspect:.4f}, w = {winding:.4f}")
        print(f"  q_ratio = {res['q_ratio']:.6f}")
        print(f"  path_length = {res['path_length']:.4f}")
        print(f"  mean_kappa = {res['mean_kappa']:.6f}")
        return

    # === Key reference points ===
    print("\n--- Key Configurations ---")
    configs = [
        (3.0, 2.0, "Williamson 1997 (R/r=3, w=2)"),
        (PHI, 2.0, "R/r=phi, w=2"),
        (3.0, PHI, "R/r=3, w=phi"),
        (PHI, PHI, "R/r=phi, w=phi (double golden)"),
        (PHI**2, PHI, "R/r=phi^2, w=phi"),
        (2.0, 2.0, "R/r=2, w=2"),
    ]
    for R, w, label in configs:
        res = compute_charge_from_path(R, 1.0, w, n_path=20000)
        print(f"  {label:40s}: q = {res['q_ratio']:.6f}")

    # === Scans ===
    w_scan, q_w = scan_winding_ratio(R=3.0, r=1.0)
    a_scan, q_a = scan_aspect_ratio(winding=2.0)

    # Find q=1.0 crossings
    for name, x, q in [("winding", w_scan, q_w), ("aspect", a_scan, q_a)]:
        crossings = []
        for i in range(len(q) - 1):
            if (q[i] - 1.0) * (q[i+1] - 1.0) < 0:
                # Linear interpolation
                x_cross = x[i] + (1.0 - q[i]) / (q[i+1] - q[i]) * (x[i+1] - x[i])
                crossings.append(x_cross)
        if crossings:
            print(f"\n  q=1.0 crossings in {name} scan: {crossings}")

    # === 2D Map ===
    w_range = np.linspace(0.8, 3.5, 50)
    a_range = np.linspace(1.3, 6.0, 50)
    Q = compute_charge_map(w_range, a_range)

    # Find nearest to q=1.0
    closest = np.unravel_index(np.argmin(np.abs(Q - 1.0)), Q.shape)
    print(f"\n  Nearest q=1.0 in map: R/r={a_range[closest[0]]:.3f}, "
          f"w={w_range[closest[1]]:.3f}, q={Q[closest]:.4f}")

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Williamson-Benn: Charge Factor from Helical Path Geometry\n"
        r"Does $\varphi$ (golden ratio) correct the charge calculation?",
        fontsize=13, fontweight='bold'
    )

    ax = axes[0]
    ax.plot(w_scan, q_w, 'b-', linewidth=2)
    ax.axvline(x=PHI, color='gold', linewidth=2.5, linestyle='--',
               label=rf'w = $\varphi$ = {PHI:.3f}')
    ax.axvline(x=2.0, color='red', linewidth=1.5, linestyle='--',
               label='w = 2 (Williamson)')
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle=':')
    ax.set_xlabel('Winding ratio (poloidal/toroidal)', fontsize=11)
    ax.set_ylabel(r'Charge factor $q/q_0$', fontsize=11)
    ax.set_title('Charge vs Winding Ratio\n(R/r = 3)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(a_scan, q_a, 'b-', linewidth=2)
    ax.axvline(x=PHI, color='gold', linewidth=2.5, linestyle='--',
               label=rf'R/r = $\varphi$ = {PHI:.3f}')
    ax.axvline(x=3.0, color='red', linewidth=1.5, linestyle='--',
               label='R/r = 3')
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle=':')
    ax.set_xlabel('Aspect ratio R/r', fontsize=11)
    ax.set_ylabel(r'Charge factor $q/q_0$', fontsize=11)
    ax.set_title('Charge vs Aspect Ratio\n(w = 2)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    W, A = np.meshgrid(w_range, a_range)
    levels = np.linspace(Q.min(), Q.max(), 25)
    im = ax.contourf(W, A, Q, levels=levels, cmap='RdYlBu_r')
    cs = ax.contour(W, A, Q, levels=[1.0], colors='white', linewidths=2)
    ax.clabel(cs, fmt='q=%.1f')
    ax.plot(PHI, PHI, '*', color='gold', markersize=18,
            markeredgecolor='black', label=rf'($\varphi$, $\varphi$)')
    ax.plot(2.0, 3.0, 'ro', markersize=10, markeredgecolor='black',
            label='Williamson (2, 3)')
    ax.set_xlabel('Winding ratio', fontsize=11)
    ax.set_ylabel('Aspect ratio R/r', fontsize=11)
    ax.set_title('Charge Factor Parameter Space\n(white line = q = 1.0)')
    ax.legend(fontsize=7, loc='upper left')
    plt.colorbar(im, ax=ax, label=r'$q/q_0$')

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/pivot_3d_charge_golden.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Williamson-Benn charge calculation with golden ratio')
    parser.add_argument('--save', action='store_true',
                        help='Save plot to file')
    parser.add_argument('--aspect', type=float, default=None,
                        help='Aspect ratio R/r for single calculation')
    parser.add_argument('--winding', type=float, default=None,
                        help='Winding ratio for single calculation')
    args = parser.parse_args()
    main(save=args.save, aspect=args.aspect, winding=args.winding)
