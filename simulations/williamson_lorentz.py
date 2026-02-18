#!/usr/bin/env python3
"""
Williamson Torus under Lorentz Boost
======================================

The electron-as-torus resolves the size paradox:
  - Low energy: appears diffuse (size ~ R, the major radius)
  - High energy: Lorentz-contracted → appears pointlike

This script computes:
1. The 3D charge distribution of a toroidal resonance
2. The form factor F(q) at rest and under Lorentz boost
3. How the apparent size depends on probe energy (γ factor)
4. Visualization of the contracted torus at various velocities

The form factor F(q) = ∫ρ(r)e^{iq·r}d³r / Q determines scattering
cross-sections. For a point particle, F=1 at all q. Deviations from
F=1 reveal structure — the q where F starts to drop gives 1/R_apparent.

Usage:
    python3 williamson_lorentz.py --mode form_factor --save
    python3 williamson_lorentz.py --mode contraction --save
    python3 williamson_lorentz.py --mode size_vs_energy --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def torus_charge_density(x, y, z, R=3.45, r=1.0, n_tor=2, n_pol=1):
    """Charge density of a (2,1) toroidal resonance in 3D.

    The charge density comes from ∂P/∂t = -∇·E. For the (2,1) mode,
    this creates a pattern with 2 toroidal oscillations and 1 poloidal.

    In cylindrical coordinates (s, φ, z) where s = √(x²+y²):
      - Distance from torus tube center: d = √((s-R)² + z²)
      - Poloidal angle: θ = atan2(z, s-R)
      - Toroidal angle: φ = atan2(y, x)

    The charge density is concentrated on the torus tube and oscillates
    as cos(n_tor·φ + n_pol·θ) × exp(-d²/(2σ²))
    """
    s = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Distance from tube center and poloidal angle
    ds = s - R
    d_sq = ds**2 + z**2
    d = np.sqrt(d_sq + 1e-30)  # avoid division by zero
    theta = np.arctan2(z, ds)

    # Gaussian tube profile with width ~ r/3
    sigma = r / 3.0
    profile = np.exp(-d_sq / (2 * sigma**2))

    # Oscillating charge pattern from (n_tor, n_pol) mode
    phase = n_tor * phi + n_pol * theta
    rho = np.cos(phase) * profile

    return rho


def lorentz_boost_z(x, y, z, gamma):
    """Apply Lorentz boost along z-axis.

    In the boosted frame, z' = γ(z - vt). At t=0: z' = γz.
    The charge distribution contracts: ρ'(x,y,z') = ρ(x, y, z'/γ) × γ
    (the γ factor preserves total charge).
    """
    return x, y, z * gamma


def compute_form_factor_1d(R=3.45, r=1.0, gamma=1.0, n_points=200,
                            q_max=10.0, n_q=500, direction='z'):
    """Compute the form factor F(q) for the torus.

    F(q) = |∫ρ(r)e^{iq·r}d³r| / Q

    For a spherically averaged form factor, we'd integrate over
    all directions of q. Here we compute F(q) for q along a
    specific axis, which shows the anisotropy.

    Args:
        R, r: Torus major and minor radii
        gamma: Lorentz factor (boost along z)
        n_points: Grid points per axis for 3D integration
        q_max: Maximum momentum transfer
        n_q: Number of q points
        direction: Direction of momentum transfer ('x', 'y', 'z')
    """
    # Set up 3D grid
    L = 1.5 * (R + r)  # box size
    grid = np.linspace(-L, L, n_points)
    dx = grid[1] - grid[0]
    dV = dx**3

    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')

    # Charge density (in rest frame, then boost)
    rho = torus_charge_density(X, Y, Z / gamma, R=R, r=r)
    # Boost: ρ' = γ × ρ(x, y, z/γ) — preserves total charge
    rho *= gamma

    # Total charge (normalization)
    Q = np.sum(rho) * dV

    if abs(Q) < 1e-10:
        # Net charge is ~0 for oscillating pattern; use |ρ| for form factor
        Q = np.sum(np.abs(rho)) * dV

    # Momentum transfer values
    q_vals = np.linspace(0, q_max, n_q)
    F = np.zeros(n_q)

    for i, q in enumerate(q_vals):
        if direction == 'z':
            phase = q * Z
        elif direction == 'x':
            phase = q * X
        else:  # y
            phase = q * Y

        # F(q) = |∫ρ·e^{iqr}dV| / Q
        integrand_real = np.sum(rho * np.cos(phase)) * dV
        integrand_imag = np.sum(rho * np.sin(phase)) * dV
        F[i] = np.sqrt(integrand_real**2 + integrand_imag**2) / abs(Q)

    return q_vals, F


def compute_form_factor_fast(R=3.45, r=1.0, gamma=1.0, n_q=500,
                              q_max=10.0, direction='z'):
    """Fast form factor via analytical integration over torus.

    Instead of 3D grid, integrate analytically over the torus surface.
    For a thin torus (r << R), the charge distribution is:
      ρ(s,φ,z) ~ δ(d-0)·cos(2φ+θ)

    where d = distance from tube axis.

    The form factor for a ring of radius R in the xy-plane is:
      F(q_z) = J_0(0) = 1  (independent of q_z for a thin ring)
      F(q_perp) = J_0(q_perp·R)  (Bessel function)

    With Lorentz boost along z, the ring becomes an ellipse with
    semi-axes R (in x,y) and R/γ (in z).
    """
    q_vals = np.linspace(0, q_max, n_q)

    # Sample points on the torus
    n_phi = 500
    n_theta = 100
    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    PHI, THETA = np.meshgrid(phi_arr, theta_arr, indexing='ij')

    # Torus coordinates in rest frame
    s = R + r * np.cos(THETA)
    X = s * np.cos(PHI)
    Y = s * np.sin(PHI)
    Z_rest = r * np.sin(THETA)

    # Lorentz boost: z → z/γ in the density, meaning physical z' = γz
    Z_boosted = Z_rest / gamma  # contracted position

    # Charge pattern (from (2,1) mode)
    charge = np.cos(2 * PHI + THETA)

    # Area element on torus: dA = r(R + r cos θ) dφ dθ
    # With boost: multiply by γ to preserve total charge
    weight = r * (R + r * np.cos(THETA)) * gamma
    dphi = phi_arr[1] - phi_arr[0]
    dtheta = theta_arr[1] - theta_arr[0]

    rho_weighted = charge * weight * dphi * dtheta
    Q = np.sum(np.abs(rho_weighted))

    F = np.zeros(n_q)
    for i, q in enumerate(q_vals):
        if direction == 'z':
            phase = q * Z_boosted
        elif direction == 'x':
            phase = q * X
        else:
            phase = q * Y

        re = np.sum(rho_weighted * np.cos(phase))
        im = np.sum(rho_weighted * np.sin(phase))
        F[i] = np.sqrt(re**2 + im**2) / Q

    return q_vals, F


def run_form_factor(R=3.45, r=1.0, save=False):
    """Show form factor at different Lorentz boosts."""
    print("=" * 60)
    print("Form Factor F(q) at Different Lorentz Boosts")
    print(f"Torus: R={R}, r={r}, mode=(2,1)")
    print("=" * 60)

    gammas = [1, 2, 5, 10, 50, 200]
    q_max = 15.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Torus Form Factor: R={R}, r={r}, (2,1) mode", fontsize=13)

    # Panel 1: F(q_z) for different γ — probing along boost direction
    ax = axes[0]
    for gamma in gammas:
        q, F = compute_form_factor_fast(R=R, r=r, gamma=gamma,
                                         q_max=q_max, direction='z')
        ax.plot(q, F, label=f'γ={gamma}')

    ax.set_xlabel('q (momentum transfer)')
    ax.set_ylabel('|F(q)|')
    ax.set_title('F(q) along boost axis (z)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    # Panel 2: F(q_x) for different γ — probing perpendicular to boost
    ax = axes[1]
    for gamma in gammas:
        q, F = compute_form_factor_fast(R=R, r=r, gamma=gamma,
                                         q_max=q_max, direction='x')
        ax.plot(q, F, label=f'γ={gamma}')

    ax.set_xlabel('q (momentum transfer)')
    ax.set_ylabel('|F(q)|')
    ax.set_title('F(q) perpendicular to boost (x)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    # Panel 3: Apparent size vs γ
    ax = axes[2]
    gammas_scan = np.logspace(0, 5, 50)
    sizes_z = []
    sizes_x = []

    for gamma in gammas_scan:
        q, F = compute_form_factor_fast(R=R, r=r, gamma=gamma,
                                         q_max=30.0, n_q=300, direction='z')
        # Apparent size = 1/q where F first drops to 0.5
        idx = np.where(F < 0.5)[0]
        q_half_z = q[idx[0]] if len(idx) > 0 else q_max
        sizes_z.append(1.0 / q_half_z if q_half_z > 0 else R)

        q, F = compute_form_factor_fast(R=R, r=r, gamma=gamma,
                                         q_max=30.0, n_q=300, direction='x')
        idx = np.where(F < 0.5)[0]
        q_half_x = q[idx[0]] if len(idx) > 0 else q_max
        sizes_x.append(1.0 / q_half_x if q_half_x > 0 else R)

    ax.loglog(gammas_scan, sizes_z, 'b-', linewidth=2, label='Along boost (z)')
    ax.loglog(gammas_scan, sizes_x, 'r-', linewidth=2, label='Perpendicular (x)')
    ax.loglog(gammas_scan, R / gammas_scan, 'k--', linewidth=1,
              label=f'R/γ (Lorentz contraction)')
    ax.axhline(y=R, color='blue', linestyle=':', alpha=0.3)
    ax.axhline(y=r, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('γ (Lorentz factor)')
    ax.set_ylabel('Apparent size (1/q_half)')
    ax.set_title('Apparent Size vs Probe Energy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = '/home/jim/repos/claude-prime/simulations/torus_form_factor.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def run_contraction(R=3.45, r=1.0, save=False):
    """Visualize the torus at different Lorentz boosts."""
    print("=" * 60)
    print("Lorentz Contraction of the Williamson Torus")
    print("=" * 60)

    gammas = [1, 2, 5, 20, 100]

    fig, axes = plt.subplots(2, len(gammas), figsize=(3.5 * len(gammas), 7))
    fig.suptitle("Williamson Torus under Lorentz Boost (along z-axis)",
                 fontsize=14)

    # Generate torus surface
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 50)
    PHI, THETA = np.meshgrid(phi, theta)

    for col, gamma in enumerate(gammas):
        # Torus surface coordinates
        s = R + r * np.cos(THETA)
        X = s * np.cos(PHI)
        Y = s * np.sin(PHI)
        Z = r * np.sin(THETA) / gamma  # Lorentz contracted

        # Charge pattern for coloring
        charge = np.cos(2 * PHI + THETA)

        # Top row: side view (x-z plane)
        ax = axes[0, col]
        # Project onto x-z plane: plot the torus cross-section
        ax.scatter(X.flatten(), Z.flatten(), c=charge.flatten(),
                   s=0.5, cmap='RdBu', alpha=0.6, vmin=-1, vmax=1)
        lim = R + 2 * r
        ax.set_xlim(-lim, lim)
        z_lim = max(2 * r / gamma, 0.1)
        ax.set_ylim(-z_lim * 1.5, z_lim * 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'γ = {gamma}', fontsize=11)
        if col == 0:
            ax.set_ylabel('z (boost axis)')
        ax.set_xlabel('x')

        # Bottom row: top view (x-y plane)
        ax = axes[1, col]
        ax.scatter(X.flatten(), Y.flatten(), c=charge.flatten(),
                   s=0.5, cmap='RdBu', alpha=0.6, vmin=-1, vmax=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('y')
        ax.set_xlabel('x')

        # Compute contracted dimensions
        z_extent = 2 * r / gamma
        print(f"γ={gamma:6d}: z-extent = {z_extent:.4f} "
              f"(contraction ratio = {1/gamma:.6f})")

    plt.tight_layout()

    if save:
        path = '/home/jim/repos/claude-prime/simulations/torus_contraction.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def run_size_vs_energy(R=3.45, r=1.0, save=False):
    """Map probe energy to apparent electron size.

    Using physical units:
      - R ≈ λ_C/π ≈ 122 fm (if λ_C = 386 fm)
      - r ≈ r_e ≈ 2.82 fm (classical electron radius)
      - Probe energy E determines resolution: Δx ~ ℏc/E

    At different experimental scales:
      - Atomic physics (eV): sees charge, not structure → point-like
      - Nuclear physics (MeV): wavelength ~ fm → might see structure
      - Collider (GeV-TeV): γ ~ 10³-10⁶ → highly contracted
    """
    print("=" * 60)
    print("Electron Apparent Size vs Probe Energy")
    print("=" * 60)
    print()

    # Physical constants (natural units: ℏc = 197.3 MeV·fm)
    hbar_c = 197.3  # MeV·fm
    m_e = 0.511  # MeV
    lambda_C = hbar_c / m_e  # reduced Compton wavelength = 386 fm
    r_e = 2.818  # fm (classical electron radius)
    R_phys = lambda_C / np.pi  # ~ 123 fm (torus major radius from our model)

    print(f"Physical scales:")
    print(f"  Compton wavelength λ_C = {lambda_C:.1f} fm")
    print(f"  Classical electron radius r_e = {r_e:.3f} fm")
    print(f"  Torus major radius R = λ_C/π = {R_phys:.1f} fm")
    print(f"  Torus minor radius r ≈ r_e = {r_e:.3f} fm")
    print(f"  Aspect ratio R/r = {R_phys/r_e:.1f}")
    print()

    # Probe energies from eV to TeV
    energies_mev = np.logspace(-3, 9, 200)  # 1 eV to 1 PeV

    # At each energy, the probe wavelength is λ = ℏc/E
    wavelengths = hbar_c / energies_mev  # fm

    # Lorentz factor of the electron at each energy
    gamma = energies_mev / m_e
    gamma = np.maximum(gamma, 1.0)

    # Apparent sizes
    # Along boost: R_app_z = min(R_phys/γ, wavelength)
    # The probe sees the LARGER of: contracted size or probe resolution
    r_app_z = R_phys / gamma  # Lorentz-contracted torus
    r_app_x = R_phys * np.ones_like(gamma)  # perpendicular — unchanged

    # Tube radius also contracts along z
    r_tube_z = r_e / gamma

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Apparent size vs energy
    ax = axes[0]
    ax.loglog(energies_mev, r_app_z, 'b-', linewidth=2,
              label=f'Torus R (along boost): R/γ')
    ax.loglog(energies_mev, r_app_x, 'r-', linewidth=2,
              label=f'Torus R (perpendicular): constant')
    ax.loglog(energies_mev, r_tube_z, 'b--', linewidth=1,
              label=f'Tube r (along boost): r/γ')
    ax.loglog(energies_mev, wavelengths, 'k:', linewidth=1,
              label='Probe wavelength ℏc/E')

    # Mark key experiments
    experiments = [
        (13.6e-3, 'Hydrogen\n(13.6 eV)', 'green'),
        (1.0, 'Nuclear\n(1 MeV)', 'orange'),
        (91e3, 'LEP\n(91 GeV)', 'purple'),
        (13e6, 'LHC\n(13 TeV)', 'red'),
    ]
    for E, label, color in experiments:
        ax.axvline(x=E, color=color, linestyle='-.', alpha=0.5)
        ax.text(E * 1.3, 1e4, label, fontsize=8, color=color, rotation=90,
                va='top')

    ax.set_xlabel('Probe Energy (MeV)')
    ax.set_ylabel('Size (fm)')
    ax.set_title('Apparent Electron Size vs Probe Energy')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(1e-12, 1e6)
    ax.grid(True, alpha=0.3)

    # Panel 2: Aspect ratio vs energy
    ax = axes[1]
    aspect = r_app_x / np.maximum(r_app_z, 1e-20)
    ax.loglog(energies_mev, aspect, 'g-', linewidth=2)
    ax.set_xlabel('Probe Energy (MeV)')
    ax.set_ylabel('Apparent Aspect Ratio R_perp/R_parallel')
    ax.set_title('Torus Becomes a Pancake')
    ax.grid(True, alpha=0.3)

    for E, label, color in experiments:
        ax.axvline(x=E, color=color, linestyle='-.', alpha=0.5)

    # Annotate
    ax.text(0.05, 0.95, 'γ=1: round torus\nγ>>1: flat disk → point',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save:
        path = '/home/jim/repos/claude-prime/simulations/torus_size_vs_energy.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

    # Print size at each experiment
    print()
    print(f"{'Experiment':>15s} {'Energy':>12s} {'γ':>10s} {'R_z (fm)':>12s} "
          f"{'r_z (fm)':>12s} {'Aspect':>10s}")
    print("-" * 75)
    for E, label, _ in experiments:
        g = max(E / m_e, 1.0)
        rz = R_phys / g
        rtz = r_e / g
        asp = R_phys / rz
        label_clean = label.replace('\n', ' ')
        print(f"{label_clean:>15s} {E:12.1f} {g:10.1f} {rz:12.4e} "
              f"{rtz:12.4e} {asp:10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Williamson Torus Lorentz")
    parser.add_argument('--mode', choices=['form_factor', 'contraction',
                        'size_vs_energy'], default='contraction')
    parser.add_argument('--R', type=float, default=3.45)
    parser.add_argument('--r', type=float, default=1.0)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.mode == 'form_factor':
        run_form_factor(R=args.R, r=args.r, save=args.save)
    elif args.mode == 'contraction':
        run_contraction(R=args.R, r=args.r, save=args.save)
    elif args.mode == 'size_vs_energy':
        run_size_vs_energy(R=args.R, r=args.r, save=args.save)


if __name__ == '__main__':
    main()
