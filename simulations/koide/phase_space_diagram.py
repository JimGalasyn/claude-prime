#!/usr/bin/env python3
"""
Phase Space Diagram for the Null Worldtube Model
=================================================

Generates a four-panel figure illustrating:
  (a) Mass spectrum on the Koide circle: m_i(θ) with forbidden zones
  (b) Phase portrait (θ, dθ/dt) showing stable orbits
  (c) The Koide 2-sphere in √m space (3D projection)
  (d) Effective potential V(θ) on the Koide circle

Usage:
    python3 phase_space_diagram.py              # saves to phase_space.png
    python3 phase_space_diagram.py --output X   # saves to X
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Physical constants ──────────────────────────────────────────────
m_e_MeV = 0.51099895
m_mu_MeV = 105.6583755
m_tau_MeV = 1776.86

S = np.sqrt(m_e_MeV) + np.sqrt(m_mu_MeV) + np.sqrt(m_tau_MeV)
S_over_3 = S / 3

# Koide angle from torus geometry
p_wind, q_wind, N_c = 2, 1, 3
theta_K = (p_wind / N_c) * (np.pi + q_wind / N_c)  # = (6π + 2)/9

# ── Color palette ───────────────────────────────────────────────────
C_ELECTRON = '#1565C0'   # strong blue
C_MUON     = '#E65100'   # strong orange
C_TAU      = '#AD1457'   # strong pink
C_EQUIL    = '#2E7D32'   # strong green
C_ORBIT    = '#5E35B1'   # purple
C_SPHERE   = '#64B5F6'   # light blue
C_BG       = '#FAFAFA'   # off-white
C_DARK     = '#263238'   # near-black


def koide_masses(theta):
    """Compute √m_i at angle theta on the Koide circle."""
    f = np.array([
        1 + np.sqrt(2) * np.cos(theta),
        1 + np.sqrt(2) * np.cos(theta + 2 * np.pi / 3),
        1 + np.sqrt(2) * np.cos(theta + 4 * np.pi / 3),
    ])
    sqrt_m = S_over_3 * f
    return sqrt_m


def find_forbidden_boundaries():
    """Find θ values where any √m_i = 0 (boundaries of forbidden zones)."""
    zeros = []
    for k in range(3):
        for sign in [+1, -1]:
            z = sign * 3 * np.pi / 4 - 2 * np.pi * k / 3
            zeros.append(z % (2 * np.pi))
    return sorted(zeros)


def compute_mass_curves(N=2000, margin=0.0):
    """Compute mass curves over the full Koide circle.
    margin: require all √m > margin * S_over_3 to be counted as valid.
    This avoids numerical issues near the forbidden boundaries where
    one mass is barely positive but others are huge.
    """
    theta = np.linspace(0, 2 * np.pi, N)
    m_vals = np.full((3, N), np.nan)
    sqrt_m_vals = np.full((3, N), np.nan)
    valid = np.zeros(N, dtype=bool)

    threshold = margin * S_over_3
    for i, th in enumerate(theta):
        sqrt_m = koide_masses(th)
        if np.all(sqrt_m > threshold):
            valid[i] = True
            m_vals[:, i] = sqrt_m**2
            sqrt_m_vals[:, i] = sqrt_m

    return theta, m_vals, sqrt_m_vals, valid


# ── Panel (a): Mass spectrum on the Koide circle ───────────────────
def plot_mass_spectrum(ax):
    theta, m_vals, _, valid = compute_mass_curves()
    th_deg = np.degrees(theta)

    # Shade forbidden zones (batch the spans)
    in_forbidden = False
    span_start = 0
    for i in range(len(theta)):
        if not valid[i] and not in_forbidden:
            span_start = th_deg[i]
            in_forbidden = True
        elif valid[i] and in_forbidden:
            ax.axvspan(span_start, th_deg[i], alpha=0.12, color='#EF5350',
                       linewidth=0, zorder=0)
            in_forbidden = False
    if in_forbidden:
        ax.axvspan(span_start, 360, alpha=0.12, color='#EF5350',
                   linewidth=0, zorder=0)

    # Plot masses on log scale
    ax.semilogy(th_deg, m_vals[0], color=C_ELECTRON, linewidth=2.2,
                label=r'$m_e(\theta)$', zorder=3)
    ax.semilogy(th_deg, m_vals[1], color=C_MUON, linewidth=2.2,
                label=r'$m_\mu(\theta)$', zorder=3)
    ax.semilogy(th_deg, m_vals[2], color=C_TAU, linewidth=2.2,
                label=r'$m_\tau(\theta)$', zorder=3)

    # Mark the physical Koide angle
    theta_K_deg = np.degrees(theta_K)
    ax.axvline(theta_K_deg, color=C_EQUIL, linewidth=2.5, linestyle='--',
               alpha=0.9, zorder=4)

    # Physical masses as dots on the vertical line
    ax.plot(theta_K_deg, m_e_MeV, 'o', color=C_ELECTRON, markersize=8,
            zorder=5, markeredgecolor='white', markeredgewidth=1)
    ax.plot(theta_K_deg, m_mu_MeV, 'o', color=C_MUON, markersize=8,
            zorder=5, markeredgecolor='white', markeredgewidth=1)
    ax.plot(theta_K_deg, m_tau_MeV, 'o', color=C_TAU, markersize=8,
            zorder=5, markeredgecolor='white', markeredgewidth=1)

    # Annotation for θ_K
    ax.annotate(r'$\theta_K = \frac{6\pi+2}{9}$',
                xy=(theta_K_deg, 3e4), fontsize=11,
                color=C_EQUIL, fontweight='bold',
                ha='center', va='bottom')

    # Mass labels on dots
    ax.annotate('0.511 MeV', xy=(theta_K_deg, m_e_MeV),
                xytext=(theta_K_deg + 18, m_e_MeV * 0.6),
                fontsize=8, color=C_ELECTRON,
                arrowprops=dict(arrowstyle='-', color=C_ELECTRON,
                                lw=0.8, alpha=0.5))
    ax.annotate('105.7 MeV', xy=(theta_K_deg, m_mu_MeV),
                xytext=(theta_K_deg + 18, m_mu_MeV * 0.6),
                fontsize=8, color=C_MUON,
                arrowprops=dict(arrowstyle='-', color=C_MUON,
                                lw=0.8, alpha=0.5))
    ax.annotate('1777 MeV', xy=(theta_K_deg, m_tau_MeV),
                xytext=(theta_K_deg + 18, m_tau_MeV * 1.5),
                fontsize=8, color=C_TAU,
                arrowprops=dict(arrowstyle='-', color=C_TAU,
                                lw=0.8, alpha=0.5))

    ax.set_xlim(0, 360)
    ax.set_ylim(0.01, 1e5)
    ax.set_xlabel(r'Koide angle $\theta$ (degrees)', fontsize=11)
    ax.set_ylabel('Mass (MeV)', fontsize=11)
    ax.set_title('(a)  Lepton masses on the Koide circle', fontsize=13,
                 fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.15, which='both')


# ── Panel (b): Phase portrait (θ, dθ/dt) ──────────────────────────
def plot_phase_portrait(ax):
    theta_K_deg = np.degrees(theta_K)
    nearest_zero_deg = np.degrees(2.3562)  # 135°

    # Three normal mode frequencies from the stability analysis
    # The modes have very different ω, so plotting all in (θ, θ̇) makes
    # the slow mode look like a flat line. Instead, NORMALIZE each mode's
    # θ̇ by its ω so all orbits are circular and distinguishable.
    # This shows the phase space STRUCTURE rather than raw frequencies.

    mode_colors = [C_TAU, C_MUON, C_ELECTRON]
    mode_labels = [r'Mode 1: $\tau$ oscillation ($\omega_1$)',
                   r'Mode 2: $\mu$ mixing ($\omega_2$)',
                   r'Mode 3: $e$ breathing ($\omega_3$)']

    t = np.linspace(0, 2 * np.pi, 600)

    # Draw nested orbits for each mode, with increasing radii
    for m_idx, (color, label) in enumerate(zip(mode_colors, mode_labels)):
        # Each mode gets its own set of orbits at different amplitudes
        base_amp = 0.15 + m_idx * 0.3  # separate modes radially
        for a_idx, scale in enumerate([0.6, 1.0, 1.4]):
            A = base_amp * scale
            theta_orbit = theta_K_deg + A * np.cos(t)
            theta_dot_norm = -A * np.sin(t)  # normalized: ω factored out

            lw = 2.0 if a_idx == 1 else 1.0
            alph = 0.85 if a_idx == 1 else 0.35
            lbl = label if a_idx == 1 else None
            ax.plot(theta_orbit, theta_dot_norm, color=color,
                    linewidth=lw, alpha=alph, label=lbl)

            # Direction arrows on middle orbit
            if a_idx == 1:
                for frac in [0.12, 0.62]:
                    idx = int(frac * len(t))
                    ax.annotate('',
                                xy=(theta_orbit[idx+3], theta_dot_norm[idx+3]),
                                xytext=(theta_orbit[idx], theta_dot_norm[idx]),
                                arrowprops=dict(arrowstyle='-|>',
                                                color=color, lw=1.8))

    # Mark the equilibrium point
    ax.plot(theta_K_deg, 0, 'o', color=C_EQUIL, markersize=14,
            zorder=10, markeredgecolor='white', markeredgewidth=2.5)
    ax.annotate(r'$\theta_K$', xy=(theta_K_deg, 0),
                xytext=(theta_K_deg + 0.15, 0.25),
                fontsize=12, color=C_EQUIL, fontweight='bold')

    # Mark forbidden zone boundary (shading to the RIGHT of the line)
    ax.axvspan(nearest_zero_deg, nearest_zero_deg + 2,
               alpha=0.12, color='#EF5350', zorder=0)
    ax.axvline(nearest_zero_deg, color='#EF5350', linewidth=2,
               linestyle='--', alpha=0.5)
    ax.text(nearest_zero_deg + 0.3, -0.8,
            r'$m_e \to 0$' + '\nforbidden',
            fontsize=8, color='#EF5350', alpha=0.7, style='italic')

    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=11)
    ax.set_ylabel(r'$\dot{\theta}/\omega$ (normalized)', fontsize=11)
    ax.set_title('(b)  Phase portrait: three normal modes',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(theta_K_deg - 1.3, nearest_zero_deg + 0.8)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.15)
    ax.set_aspect('equal')


# ── Panel (c): Koide 2-sphere in √m space ─────────────────────────
def plot_koide_sphere(ax):
    center = np.array([S_over_3, S_over_3, S_over_3])
    R_sphere = S / np.sqrt(3)

    # Draw sphere wireframe (positive octant only)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = R_sphere * np.outer(np.cos(u), np.sin(v)) + center[0]
    ys = R_sphere * np.outer(np.sin(u), np.sin(v)) + center[1]
    zs = R_sphere * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    # Clip to positive octant
    mask = (xs >= 0) & (ys >= 0) & (zs >= 0)
    xs_c = np.where(mask, xs, np.nan)
    ys_c = np.where(mask, ys, np.nan)
    zs_c = np.where(mask, zs, np.nan)

    ax.plot_surface(xs_c, ys_c, zs_c, alpha=0.10, color=C_SPHERE,
                    shade=True, zorder=1)
    ax.plot_wireframe(xs_c, ys_c, zs_c, alpha=0.06, color='steelblue',
                      linewidth=0.3, zorder=1)

    # Koide circle (allowed region on the sphere)
    theta_circle = np.linspace(0, 2 * np.pi, 800)
    coords = np.full((3, len(theta_circle)), np.nan)
    for i, th in enumerate(theta_circle):
        sqrt_m = koide_masses(th)
        if np.all(sqrt_m > 0):
            coords[:, i] = sqrt_m

    ax.plot(coords[0], coords[1], coords[2], color=C_ORBIT,
            linewidth=3, alpha=0.85, label='Koide circle', zorder=5)

    # Physical point
    x_e = np.sqrt(m_e_MeV)
    x_mu = np.sqrt(m_mu_MeV)
    x_tau = np.sqrt(m_tau_MeV)
    ax.scatter([x_e], [x_mu], [x_tau], color=C_EQUIL, s=180,
               zorder=10, edgecolors='white', linewidth=2.5,
               label=r'Physical state $\theta_K$', depthshade=False)

    # Center of sphere
    ax.scatter([center[0]], [center[1]], [center[2]], color='gray',
               s=60, marker='+', zorder=6, linewidth=2, depthshade=False)

    # Draw line from center to physical point
    ax.plot([center[0], x_e], [center[1], x_mu], [center[2], x_tau],
            '--', color=C_EQUIL, linewidth=1.2, alpha=0.6)

    # Draw coordinate axes from origin
    ax_len = 55
    ax.plot([0, ax_len], [0, 0], [0, 0], '-', color=C_ELECTRON,
            linewidth=0.8, alpha=0.4)
    ax.plot([0, 0], [0, ax_len], [0, 0], '-', color=C_MUON,
            linewidth=0.8, alpha=0.4)
    ax.plot([0, 0], [0, 0], [0, ax_len], '-', color=C_TAU,
            linewidth=0.8, alpha=0.4)

    # Diagonal line (equal mass direction: √m_e = √m_μ = √m_τ)
    ax.plot([0, S_over_3 * 1.3], [0, S_over_3 * 1.3], [0, S_over_3 * 1.3],
            ':', color='gray', linewidth=1, alpha=0.4)

    ax.set_xlabel(r'$\sqrt{m_e}$', fontsize=10, labelpad=1)
    ax.set_ylabel(r'$\sqrt{m_\mu}$', fontsize=10, labelpad=1)
    ax.set_zlabel(r'$\sqrt{m_\tau}$', fontsize=10, labelpad=1)

    ax.set_title(r'(c)  Koide sphere in $\sqrt{m}$ space',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95)

    ax.view_init(elev=18, azim=30)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 50)
    ax.tick_params(labelsize=7, pad=1)

    # Add annotation about sphere
    ax.text2D(0.02, 0.12,
              f'$S/\\sqrt{{3}}$ = {R_sphere:.1f} $\\sqrt{{\\mathrm{{MeV}}}}$',
              transform=ax.transAxes, fontsize=8, color='#555')


# ── Panel (d): Individual masses and conserved total ───────────────
def plot_potential(ax):
    theta, m_vals, sqrt_m_vals, valid = compute_mass_curves(N=3000, margin=0.005)
    th_deg = np.degrees(theta)
    theta_K_deg = np.degrees(theta_K)

    # KEY INSIGHT: Σm_i is EXACTLY CONSTANT on the Koide circle!
    # Proof: Σf_i² = Σ[1 + √2 cos(θ + 2πi/3)]² = 6 (trigonometric identity)
    # So V(θ) = (S/3)² × 6 = 2S²/3 = 1883 MeV for ALL θ.
    # The "potential" is FLAT. θ_K CANNOT be selected by energy.

    # Focus on the allowed region containing θ_K
    theta_K_idx = np.argmin(np.abs(theta - theta_K))

    # Find contiguous allowed region
    in_allowed = False
    plot_start, plot_end = 0, len(theta) - 1
    for i in range(len(valid)):
        if valid[i] and not in_allowed:
            start_i = i
            in_allowed = True
        elif not valid[i] and in_allowed:
            if start_i <= theta_K_idx <= i - 1:
                plot_start = start_i
                plot_end = i - 1
            in_allowed = False

    region = slice(plot_start, plot_end + 1)
    reg_deg = th_deg[region]

    # Plot individual masses on log scale
    ax.semilogy(reg_deg, m_vals[0, region], color=C_ELECTRON,
                linewidth=2, label=r'$m_e(\theta)$', zorder=3)
    ax.semilogy(reg_deg, m_vals[1, region], color=C_MUON,
                linewidth=2, label=r'$m_\mu(\theta)$', zorder=3)
    ax.semilogy(reg_deg, m_vals[2, region], color=C_TAU,
                linewidth=2, label=r'$m_\tau(\theta)$', zorder=3)

    # Plot total mass (constant line!)
    V_total = 2 * S**2 / 3  # exact value
    ax.axhline(V_total, color=C_DARK, linewidth=3, linestyle='-',
               alpha=0.9, zorder=4, label=r'$\Sigma m_i$ = const')

    # Mark θ_K
    ax.axvline(theta_K_deg, color=C_EQUIL, linewidth=2.5, linestyle='--',
               alpha=0.9, zorder=4)

    # Physical masses as dots
    ax.plot(theta_K_deg, m_e_MeV, 'o', color=C_ELECTRON, markersize=8,
            zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(theta_K_deg, m_mu_MeV, 'o', color=C_MUON, markersize=8,
            zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(theta_K_deg, m_tau_MeV, 'o', color=C_TAU, markersize=8,
            zorder=6, markeredgecolor='white', markeredgewidth=1.5)

    # Forbidden boundaries
    boundaries = find_forbidden_boundaries()
    for b in boundaries:
        b_deg = np.degrees(b)
        if reg_deg[0] - 2 < b_deg < reg_deg[-1] + 2:
            ax.axvline(b_deg, color='#EF5350', linewidth=2,
                       linestyle='--', alpha=0.4)

    # Annotations
    ax.annotate(r'$\theta_K$', xy=(theta_K_deg, 3e4),
                fontsize=11, color=C_EQUIL, fontweight='bold',
                ha='center', va='bottom')

    # The key insight annotation
    ax.text(0.03, 0.97,
            r'$\Sigma m_i(\theta)$ = $\frac{2}{3}S^2$'
            f' = {V_total:.0f} MeV\n'
            r'$\mathbf{exactly\ constant\ for\ all\ \theta}$'
            '\n\n'
            r'$\theta_K$ is selected by resonance,'
            '\nnot energy minimization',
            transform=ax.transAxes, fontsize=9, va='top',
            color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                      edgecolor=C_EQUIL, alpha=0.9))

    ax.set_xlim(reg_deg[0] - 1, reg_deg[-1] + 1)
    ax.set_ylim(0.005, 5e4)
    ax.set_xlabel(r'Koide angle $\theta$ (degrees)', fontsize=11)
    ax.set_ylabel('Mass (MeV)', fontsize=11)
    ax.set_title('(d)  Mass exchange at constant total energy',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.15, which='both')


# ── Main figure ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUTPUT_DIR / 'phase_space.png'),
                        help='Output filename')
    args = parser.parse_args()

    fig = plt.figure(figsize=(18, 15), facecolor='white')

    # Use GridSpec for precise control
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax4]:
        ax.set_facecolor(C_BG)

    plot_mass_spectrum(ax1)
    plot_phase_portrait(ax2)
    plot_koide_sphere(ax3)
    plot_potential(ax4)

    # Title
    fig.suptitle('Phase Space of the Null Worldtube Model',
                 fontsize=18, fontweight='bold', y=0.97)
    fig.text(0.5, 0.94,
             'Three charged lepton generations as a dynamical system '
             'on the Koide sphere',
             ha='center', fontsize=12, color='#555', style='italic')

    # Bottom annotation
    fig.text(0.5, 0.01,
             r'Koide angle $\theta_K = (6\pi+2)/9$ from torus geometry '
             r'$(p,q,N_c) = (2,1,3)$   |   '
             r'Koide sphere radius $S/\sqrt{3}$   |   '
             r'All Lyapunov exponents purely imaginary $\Rightarrow$ stable',
             ha='center', fontsize=9, color='#888')

    plt.savefig(args.output, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved to {args.output}")


if __name__ == '__main__':
    main()
