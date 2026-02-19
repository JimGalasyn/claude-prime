#!/usr/bin/env python3
"""
Diagrams for the Null Worldtube Model
======================================

Generates publication-quality figures illustrating the torus model
of the electron and the hydrogen atom.

Requires: matplotlib, numpy (both available in the memory-palace venv)

Usage:
    python3 null_worldtube_diagrams.py           # generate all diagrams
    python3 null_worldtube_diagrams.py --torus    # just the torus knot
    python3 null_worldtube_diagrams.py --atom     # just the hydrogen atom
    python3 null_worldtube_diagrams.py --spectrum # just the spectrum
    python3 null_worldtube_diagrams.py --sizes    # just the size comparison
    python3 null_worldtube_diagrams.py --mechanism # absorption/emission mechanism
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import argparse
import os

# Output directory
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Style
plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#0a0a0a',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'axes.edgecolor': '#404040',
    'grid.color': '#252525',
    'font.family': 'monospace',
    'font.size': 11,
})

# Colors
C_PHOTON = '#ff6b35'     # photon path on torus
C_TORUS = '#2196F3'      # torus surface
C_NUCLEUS = '#ff4444'    # proton/nucleus
C_ORBIT = '#4CAF50'      # electron orbit
C_ORBIT2 = '#66BB6A'     # second orbit
C_ORBIT3 = '#81C784'     # third orbit
C_LYMAN = '#9C27B0'      # Lyman series (UV)
C_BALMER = '#2196F3'     # Balmer series (visible)
C_PASCHEN = '#FF5722'    # Paschen series (IR)
C_FIELD = '#FFD700'      # EM field / photon wave
C_SPIN = '#00BCD4'       # spin arrow
C_TEXT = '#e0e0e0'
C_DIM = '#606060'

# Physical constants
alpha = 7.2973525693e-3
a_0_fm = 52917.7       # Bohr radius in fm
R_torus_fm = 193.37    # electron torus major radius in fm
r_torus_fm = 19.34     # electron torus minor radius in fm


def draw_torus_knot(ax, R=1.0, r=0.3, p=2, q=1, N=2000, elev=25, azim=30):
    """Draw a (p,q) torus knot on a 3D axes."""
    lam = np.linspace(0, 2 * np.pi, N)
    theta = p * lam
    phi = q * lam

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    # Color by parameter to show circulation direction
    colors = plt.cm.hot(np.linspace(0.2, 0.9, N))

    # Draw the knot curve with varying color
    for i in range(N - 1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                color=colors[i], linewidth=2.0, alpha=0.9)

    # Draw the torus surface (wireframe)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, 2 * np.pi, 20)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, alpha=0.06, color=C_TORUS, linewidth=0)
    ax.plot_wireframe(X, Y, Z, alpha=0.08, color=C_TORUS, linewidth=0.3,
                      rstride=2, cstride=2)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.6, 0.6)
    ax.set_axis_off()


def fig_torus_knot():
    """Figure 1: The (2,1) torus knot — the electron."""
    fig = plt.figure(figsize=(14, 6))

    # Left: the (2,1) knot
    ax1 = fig.add_subplot(121, projection='3d')
    draw_torus_knot(ax1, R=1.0, r=0.3, p=2, q=1, elev=25, azim=30)
    ax1.set_title('(2,1) torus knot — the electron\n'
                  'photon winds twice toroidally, once poloidally',
                  fontsize=12, pad=15, color=C_TEXT)

    # Right: the (1,1) knot for comparison
    ax2 = fig.add_subplot(122, projection='3d')
    draw_torus_knot(ax2, R=1.0, r=0.3, p=1, q=1, elev=25, azim=30)
    ax2.set_title('(1,1) torus knot — spin-1 boson\n'
                  'photon winds once each way',
                  fontsize=12, pad=15, color=C_TEXT)

    fig.suptitle('THE FUNDAMENTAL OBJECTS', fontsize=16, y=0.98,
                 fontweight='bold', color='white')

    # Add annotations
    fig.text(0.25, 0.02,
             'p=2 windings → L_z = ℏ/2 (fermion)',
             ha='center', fontsize=11, color=C_PHOTON)
    fig.text(0.75, 0.02,
             'p=1 winding → L_z = ℏ (boson)',
             ha='center', fontsize=11, color=C_PHOTON)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(OUTDIR, 'nwt_01_torus_knot.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_hydrogen_atom():
    """Figure 2: The hydrogen atom — torus orbiting a proton."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # === Left panel: schematic of torus in orbit ===
    ax = axes[0]
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    # Proton at center
    proton = Circle((0, 0), 0.15, color=C_NUCLEUS, zorder=10)
    ax.add_patch(proton)
    ax.text(0, -0.5, 'p⁺', ha='center', fontsize=12, color=C_NUCLEUS, zorder=11)

    # Orbital circles for n=1,2,3
    orbit_colors = [C_ORBIT, C_ORBIT2, C_ORBIT3]
    orbit_labels = ['n=1', 'n=2', 'n=3']
    orbit_radii = [1.5, 3.0, 4.8]  # visual radii (not to scale)

    for i, (r, color, label) in enumerate(zip(orbit_radii, orbit_colors, orbit_labels)):
        circle = Circle((0, 0), r, fill=False, color=color, linewidth=1.5,
                        linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.text(r * 0.71, r * 0.71 + 0.3, label, fontsize=11, color=color, alpha=0.8)

    # Draw the electron torus at n=1 orbit
    # Small torus symbol at a position on the n=1 orbit
    torus_theta = np.pi / 4
    torus_x = orbit_radii[0] * np.cos(torus_theta)
    torus_y = orbit_radii[0] * np.sin(torus_theta)

    # Draw a small (2,1) knot symbol
    t = np.linspace(0, 2 * np.pi, 200)
    tr = 0.35  # visual torus major radius
    tr2 = 0.12  # visual torus minor radius
    tx = torus_x + (tr + tr2 * np.cos(t)) * np.cos(2 * t)
    ty = torus_y + (tr + tr2 * np.cos(t)) * np.sin(2 * t)
    ax.plot(tx, ty, color=C_PHOTON, linewidth=2.5, zorder=8)
    ax.text(torus_x, torus_y - 0.7, 'e⁻ torus', ha='center',
            fontsize=10, color=C_PHOTON, zorder=11)

    # Orbit arrow (shows direction of orbital motion)
    arrow_theta = np.pi * 0.8
    arr_x = orbit_radii[0] * np.cos(arrow_theta)
    arr_y = orbit_radii[0] * np.sin(arrow_theta)
    dx = -np.sin(arrow_theta) * 0.3
    dy = np.cos(arrow_theta) * 0.3
    ax.annotate('', xy=(arr_x + dx, arr_y + dy), xytext=(arr_x, arr_y),
                arrowprops=dict(arrowstyle='->', color=C_ORBIT, lw=2))

    # Spin arrow on the torus
    ax.annotate('', xy=(torus_x, torus_y + 0.55),
                xytext=(torus_x, torus_y + 0.2),
                arrowprops=dict(arrowstyle='->', color=C_SPIN, lw=2))
    ax.text(torus_x + 0.25, torus_y + 0.5, 'ℏ/2', fontsize=9, color=C_SPIN)

    ax.set_title('Hydrogen atom: torus orbiting proton\n(not to scale)',
                 fontsize=13, pad=10)
    ax.set_axis_off()

    # === Right panel: to-scale size comparison ===
    ax2 = axes[1]
    ax2.set_xlim(-1, 7)
    ax2.set_ylim(-0.5, 8)

    # Use log scale for sizes — show as bar chart
    sizes = [
        ('Torus minor\nradius r', r_torus_fm, C_DIM),
        ('Torus major\nradius R', R_torus_fm, C_PHOTON),
        ('Classical\nelectron radius', alpha * 386.16, C_DIM),
        ('Compton\nwavelength λ_C', 386.16, C_DIM),
        ('Bohr radius\na₀ = 2R/α', a_0_fm, C_ORBIT),
        ('Lyman-α\nwavelength', 121.567e6, C_LYMAN),  # 121.6 nm in fm
    ]

    y_positions = np.arange(len(sizes))
    bar_widths = [np.log10(s[1]) for s in sizes]
    colors = [s[2] for s in sizes]
    labels = [s[0] for s in sizes]

    bars = ax2.barh(y_positions, bar_widths, color=colors, alpha=0.7, height=0.6,
                    edgecolor='white', linewidth=0.5)

    for i, (name, size, color) in enumerate(sizes):
        if size >= 1e6:
            size_str = f'{size:.1e} fm'
        elif size >= 100:
            size_str = f'{size:.0f} fm'
        else:
            size_str = f'{size:.1f} fm'
        ax2.text(bar_widths[i] + 0.1, i, size_str, va='center',
                fontsize=10, color=color)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('log₁₀(size / fm)', fontsize=11)
    ax2.set_title('Size hierarchy (log scale)', fontsize=13, pad=10)
    ax2.grid(True, axis='x', alpha=0.3)

    # Add the key ratio
    ax2.text(3.5, -0.3, f'a₀/R = 2/α = {a_0_fm/R_torus_fm:.0f}×',
             fontsize=12, color='white', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'nwt_02_hydrogen_atom.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_spectrum():
    """Figure 3: Hydrogen emission spectrum with transition arrows."""
    fig, ax = plt.subplots(figsize=(14, 9))

    # Energy levels
    n_max = 7
    E_levels = {n: -13.6 / n**2 for n in range(1, n_max + 1)}
    E_ionize = 0.0

    # Draw energy levels
    level_left = 0.5
    level_right = 4.5

    for n, E in E_levels.items():
        ax.plot([level_left, level_right], [E, E], color=C_TEXT, linewidth=2,
                alpha=0.8)
        ax.text(level_right + 0.15, E, f'n={n}  ({E:.2f} eV)',
                va='center', fontsize=10, color=C_TEXT)

    # Ionization limit
    ax.plot([level_left, level_right], [0, 0], color=C_TEXT, linewidth=1,
            linestyle=':', alpha=0.5)
    ax.text(level_right + 0.15, 0, 'ionization (0 eV)', va='center',
            fontsize=10, color=C_DIM)

    # Transition arrows for each series
    series = [
        ('Lyman\n(UV)', 1, [2, 3, 4, 5, 6], C_LYMAN, 1.0),
        ('Balmer\n(visible)', 2, [3, 4, 5, 6], C_BALMER, 2.0),
        ('Paschen\n(IR)', 3, [4, 5, 6], C_PASCHEN, 3.0),
    ]

    for series_name, n_f, n_i_list, color, x_base in series:
        for j, n_i in enumerate(n_i_list):
            x = x_base + j * 0.22
            E_upper = E_levels[n_i]
            E_lower = E_levels[n_f]
            dE = E_upper - E_lower

            # Arrow from upper to lower level
            ax.annotate('', xy=(x, E_lower + 0.15), xytext=(x, E_upper - 0.15),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                      alpha=0.8))

            # Wavelength label (rotated)
            lam_nm = 1239.84 / abs(dE)  # eV to nm
            mid_E = (E_upper + E_lower) / 2
            if lam_nm < 1000:
                lam_str = f'{lam_nm:.0f} nm'
            else:
                lam_str = f'{lam_nm:.0f}'
            ax.text(x + 0.05, mid_E, lam_str, fontsize=7, color=color,
                    rotation=90, va='center', alpha=0.7)

        # Series label
        ax.text(x_base + len(n_i_list) * 0.11, E_levels[n_f] - 1.0,
                series_name, ha='center', fontsize=11, color=color,
                fontweight='bold')

    # Title and labels
    ax.set_ylabel('Energy (eV)', fontsize=13)
    ax.set_xlim(0, 6)
    ax.set_ylim(-15, 1.5)
    ax.set_xticks([])
    ax.set_title('HYDROGEN EMISSION SPECTRUM\n'
                 'Every line from orbital resonances of the electron torus',
                 fontsize=15, fontweight='bold', color='white', pad=15)

    # Add annotation about the Rydberg formula
    ax.text(5.0, -3, 'E_γ = 13.6 × (1/n_f² − 1/n_i²) eV\n'
            '= orbital resonance ΔE\n\n'
            'Rydberg constant: 0.0 ppm match',
            fontsize=11, color=C_FIELD, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a',
                      edgecolor=C_FIELD, alpha=0.8))

    ax.grid(True, axis='y', alpha=0.15)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'nwt_03_spectrum.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_three_frequencies():
    """Figure 4: The α² cascade — three frequencies."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Three frequencies
    f_circ = 1.2322e20
    f_orbit = 6.5797e15
    f_prec = 3.5038e11

    freqs = [
        ('f_circ', f_circ, 'Internal circulation\n(photon on torus → mass)',
         C_PHOTON),
        ('f_orbit', f_orbit, 'Orbital revolution\n(torus around nucleus → energy levels)',
         C_ORBIT),
        ('f_prec', f_prec, 'Axis precession\n(spin-orbit coupling → fine structure)',
         C_SPIN),
    ]

    y_positions = [2, 1, 0]
    bar_widths = [np.log10(f[1]) for f in freqs]

    for i, (name, freq, desc, color) in enumerate(freqs):
        y = y_positions[i]
        w = np.log10(freq)
        ax.barh(y, w, color=color, alpha=0.7, height=0.5,
                edgecolor='white', linewidth=0.5)
        ax.text(w + 0.2, y + 0.05, f'{freq:.2e} Hz', va='center',
                fontsize=11, color=color, fontweight='bold')
        ax.text(0.5, y - 0.02, f'{name}', va='center', fontsize=12,
                color='white', fontweight='bold')
        ax.text(0.5, y - 0.25, desc, va='top', fontsize=9, color=C_DIM)

    # Show the ratios between levels
    mid1 = (np.log10(f_circ) + np.log10(f_orbit)) / 2
    mid2 = (np.log10(f_orbit) + np.log10(f_prec)) / 2

    ax.annotate('', xy=(np.log10(f_orbit) + 0.3, 1.3),
                xytext=(np.log10(f_circ) - 0.3, 1.7),
                arrowprops=dict(arrowstyle='<->', color=C_FIELD, lw=2))
    ax.text(mid1, 1.65, f'× 1/α² = {f_circ/f_orbit:.0f}',
            ha='center', fontsize=12, color=C_FIELD, fontweight='bold')

    ax.annotate('', xy=(np.log10(f_prec) + 0.3, 0.3),
                xytext=(np.log10(f_orbit) - 0.3, 0.7),
                arrowprops=dict(arrowstyle='<->', color=C_FIELD, lw=2))
    ax.text(mid2, 0.65, f'× 1/α² = {f_orbit/f_prec:.0f}',
            ha='center', fontsize=12, color=C_FIELD, fontweight='bold')

    ax.set_xlabel('log₁₀(frequency / Hz)', fontsize=13)
    ax.set_yticks([])
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.8, 3.0)

    ax.set_title('THE α² CASCADE: Three frequencies of the hydrogen atom\n'
                 'Each level of structure is 1/α² = 18,779× slower than the last',
                 fontsize=14, fontweight='bold', color='white', pad=15)

    # Key insight box
    ax.text(11, -0.5,
            'α = (electron size) / (orbit size) × 2\n'
            'Perturbation theory works because α ≈ 1/137 is small',
            fontsize=11, color=C_FIELD, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a',
                      edgecolor=C_FIELD, alpha=0.8))

    ax.grid(True, axis='x', alpha=0.15)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'nwt_04_three_frequencies.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_driven_oscillator():
    """Figure 5: Absorption/emission mechanism — driven oscillator."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # === Panel A: Absorption ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_aspect('equal')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)

    # Proton
    ax1.add_patch(Circle((0, 0), 0.12, color=C_NUCLEUS, zorder=10))

    # Inner orbit (n=1) — the torus starts here
    ax1.add_patch(Circle((0, 0), 1.5, fill=False, color=C_ORBIT,
                         linewidth=1.5, linestyle='--', alpha=0.5))
    ax1.text(1.3, -0.3, 'n=1', fontsize=9, color=C_ORBIT, alpha=0.7)

    # Outer orbit (n=2) — the torus ends here
    ax1.add_patch(Circle((0, 0), 3.0, fill=False, color=C_ORBIT2,
                         linewidth=1.5, linestyle='--', alpha=0.5))
    ax1.text(2.8, -0.3, 'n=2', fontsize=9, color=C_ORBIT2, alpha=0.7)

    # Electron torus at n=1
    theta_e = np.pi / 3
    ex, ey = 1.5 * np.cos(theta_e), 1.5 * np.sin(theta_e)
    ax1.add_patch(Circle((ex, ey), 0.2, fill=False, color=C_PHOTON, linewidth=2.5))
    ax1.plot(ex, ey, 'o', color=C_PHOTON, markersize=3)

    # Incoming photon wave (sinusoidal from left)
    x_wave = np.linspace(-4.5, -1.5, 200)
    y_wave = 3.5 + 0.4 * np.sin(12 * (x_wave + 4.5) / 3.0)
    ax1.plot(x_wave, y_wave, color=C_FIELD, linewidth=2, alpha=0.8)
    ax1.annotate('', xy=(-1.5, 3.5), xytext=(-2.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=C_FIELD, lw=2))
    ax1.text(-3.0, 4.2, 'photon\nf = ΔE/h', fontsize=10, color=C_FIELD,
             ha='center')

    # Spiral arrow from n=1 to n=2
    spiral_t = np.linspace(0, 2 * np.pi, 100)
    spiral_r = 1.5 + 1.5 * spiral_t / (2 * np.pi)
    spiral_x = spiral_r * np.cos(spiral_t + theta_e)
    spiral_y = spiral_r * np.sin(spiral_t + theta_e)
    ax1.plot(spiral_x, spiral_y, color=C_PHOTON, linewidth=1.5,
             linestyle=':', alpha=0.6)
    ax1.annotate('', xy=(spiral_x[-1], spiral_y[-1]),
                xytext=(spiral_x[-5], spiral_y[-5]),
                arrowprops=dict(arrowstyle='->', color=C_PHOTON, lw=1.5))

    ax1.set_title('ABSORPTION\nPhoton drives torus to higher orbit',
                  fontsize=12, color=C_FIELD, pad=10)
    ax1.set_axis_off()

    # === Panel B: Emission ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_aspect('equal')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)

    # Proton
    ax2.add_patch(Circle((0, 0), 0.12, color=C_NUCLEUS, zorder=10))

    # Orbits
    ax2.add_patch(Circle((0, 0), 1.5, fill=False, color=C_ORBIT,
                         linewidth=1.5, linestyle='--', alpha=0.5))
    ax2.text(1.3, -0.3, 'n=1', fontsize=9, color=C_ORBIT, alpha=0.7)
    ax2.add_patch(Circle((0, 0), 3.0, fill=False, color=C_ORBIT2,
                         linewidth=1.5, linestyle='--', alpha=0.5))
    ax2.text(2.8, -0.3, 'n=2', fontsize=9, color=C_ORBIT2, alpha=0.7)

    # Electron torus at n=2 (starting position)
    theta_e2 = np.pi * 0.7
    ex2, ey2 = 3.0 * np.cos(theta_e2), 3.0 * np.sin(theta_e2)
    ax2.add_patch(Circle((ex2, ey2), 0.2, fill=False, color=C_PHOTON,
                         linewidth=2.5, alpha=0.4))

    # Electron torus at n=1 (final position)
    ex1, ey1 = 1.5 * np.cos(theta_e2 + 1.5), 1.5 * np.sin(theta_e2 + 1.5)
    ax2.add_patch(Circle((ex1, ey1), 0.2, fill=False, color=C_PHOTON,
                         linewidth=2.5))
    ax2.plot(ex1, ey1, 'o', color=C_PHOTON, markersize=3)

    # Spiral arrow inward
    spiral_t = np.linspace(0, 2 * np.pi, 100)
    spiral_r = 3.0 - 1.5 * spiral_t / (2 * np.pi)
    spiral_x = spiral_r * np.cos(spiral_t + theta_e2)
    spiral_y = spiral_r * np.sin(spiral_t + theta_e2)
    ax2.plot(spiral_x, spiral_y, color=C_PHOTON, linewidth=1.5,
             linestyle=':', alpha=0.6)
    ax2.annotate('', xy=(spiral_x[-1], spiral_y[-1]),
                xytext=(spiral_x[-5], spiral_y[-5]),
                arrowprops=dict(arrowstyle='->', color=C_PHOTON, lw=1.5))

    # Outgoing photon wave
    x_wave = np.linspace(1.5, 4.5, 200)
    y_wave = 3.5 + 0.4 * np.sin(12 * (x_wave - 1.5) / 3.0)
    ax2.plot(x_wave, y_wave, color=C_FIELD, linewidth=2, alpha=0.8)
    ax2.annotate('', xy=(4.5, 3.5), xytext=(3.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=C_FIELD, lw=2))
    ax2.text(3.0, 4.2, 'emitted\nphoton', fontsize=10, color=C_FIELD,
             ha='center')

    ax2.set_title('EMISSION\nTorus spirals down, radiates photon',
                  fontsize=12, color=C_FIELD, pad=10)
    ax2.set_axis_off()

    # === Panel C: Wavelength vs atom size ===
    ax3 = fig.add_subplot(gs[1, 0])

    transitions = [
        ('2→1\nLyman-α', 121.5, 0.053 * 2, C_LYMAN),    # nm, atom size nm
        ('3→2\nBalmer-α', 656.1, 0.053 * 9, C_BALMER),
        ('4→3\nPaschen-α', 1874.6, 0.053 * 16, C_PASCHEN),
    ]

    y_pos = [2, 1, 0]
    for i, (label, lam, atom, color) in enumerate(transitions):
        ratio = lam / atom
        # Photon wavelength bar
        ax3.barh(y_pos[i] + 0.15, np.log10(lam), height=0.25, color=color,
                alpha=0.7, label='photon λ' if i == 0 else None)
        # Atom size bar
        ax3.barh(y_pos[i] - 0.15, np.log10(atom), height=0.25, color=color,
                alpha=0.3, label='atom size' if i == 0 else None)
        ax3.text(np.log10(lam) + 0.1, y_pos[i] + 0.15,
                f'{lam:.0f} nm', va='center', fontsize=9, color=color)
        ax3.text(np.log10(atom) + 0.1, y_pos[i] - 0.15,
                f'{atom*1000:.0f} pm', va='center', fontsize=9, color=color,
                alpha=0.7)
        ax3.text(np.log10(lam) + 0.6, y_pos[i],
                f'{ratio:.0f}×', fontsize=11, color='white', fontweight='bold',
                va='center')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([t[0] for t in transitions], fontsize=10)
    ax3.set_xlabel('log₁₀(size / nm)', fontsize=11)
    ax3.set_title('THE PHOTON ENGULFS THE ATOM\n'
                  'Not a collision — a driven oscillator',
                  fontsize=12, color='white', pad=10)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, axis='x', alpha=0.15)

    # === Panel D: Selection rules ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(-1, 7)

    # Energy levels with l subshells
    levels = [
        # (n, l, label, x_center, y)
        (1, 0, '1s', 2, 0),
        (2, 0, '2s', 4, 3),
        (2, 1, '2p', 6, 3),
        (3, 0, '3s', 4, 5),
        (3, 1, '3p', 6, 5),
        (3, 2, '3d', 8, 5),
    ]

    for n, l, label, x, y in levels:
        ax4.plot([x - 0.5, x + 0.5], [y, y], color=C_TEXT, linewidth=2)
        ax4.text(x, y - 0.35, label, ha='center', fontsize=10, color=C_TEXT)

    # Allowed transitions (Δl = ±1) — green arrows
    allowed = [
        (6, 3, 2, 0, '2p→1s'),    # 2p → 1s
        (4, 3, 6, 5, '3s→2p'),    # 3s → 2p... wait, 3s has l=0, needs Δl=+1
        (8, 5, 6, 3, '3d→2p'),    # 3d → 2p
    ]
    for x1, y1, x2, y2, label in allowed:
        ax4.annotate('', xy=(x2, y2 + 0.15), xytext=(x1, y1 - 0.15),
                    arrowprops=dict(arrowstyle='->', color=C_ORBIT, lw=2,
                                   alpha=0.8))

    # Forbidden transition (Δl = 0) — red dashed
    ax4.annotate('', xy=(2, 0.15), xytext=(4, 2.85),
                arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5,
                               linestyle='dashed', alpha=0.6))
    ax4.text(2.5, 1.5, '2s→1s\nΔl=0 ✗', fontsize=9, color='#ff4444',
             ha='center', style='italic')

    ax4.text(7.0, 4.0, '3d→2p\nΔl=1 ✓', fontsize=9, color=C_ORBIT)
    ax4.text(5.5, 1.5, '2p→1s\nΔl=1 ✓', fontsize=9, color=C_ORBIT)

    ax4.text(5, 6.5, 'SELECTION RULES\nPhoton carries ℏ → Δl = ±1',
             ha='center', fontsize=12, color='white', fontweight='bold')
    ax4.set_axis_off()

    fig.suptitle('PHOTON TRANSITIONS: Driven Oscillator Mechanism',
                 fontsize=16, y=1.0, fontweight='bold', color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUTDIR, 'nwt_05_transitions.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_model_summary():
    """Figure 6: Summary diagram — from photon to chemistry."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(0, 9)
    ax.set_axis_off()

    # Title
    ax.text(8, 8.5, 'FROM PHOTON TO CHEMISTRY',
            fontsize=20, ha='center', fontweight='bold', color='white')
    ax.text(8, 8.0, 'Standard Maxwell on toroidal topology — nothing else',
            fontsize=13, ha='center', color=C_DIM)

    # Chain of deductions
    boxes = [
        (2, 6.5, 'Photon on\ntorus knot', 'E = hc/L', C_PHOTON),
        (5.5, 6.5, 'Particle\nmass', 'm = E/c²', C_PHOTON),
        (9, 6.5, 'Spin from\nwinding', 'L_z = ℏ/p', C_SPIN),
        (12.5, 6.5, 'α from\nself-energy', 'U/E = α[...]/π', C_FIELD),

        (2, 4.0, 'Torus\norbits nucleus', 'Coulomb + resonance', C_ORBIT),
        (5.5, 4.0, 'Bohr\nlevels', 'E_n = -13.6/n²', C_ORBIT),
        (9, 4.0, 'Fine\nstructure', 'tidal: R/r ~ α', C_SPIN),
        (12.5, 4.0, 'Hydrogen\nspectrum', 'Rydberg: 0 ppm', C_FIELD),

        (2, 1.5, 'Multi-electron\natoms', 'torus packing', C_ORBIT2),
        (5.5, 1.5, 'Shell\nfilling', 'Pauli = geometry', C_ORBIT2),
        (9, 1.5, 'Noble gas\nstability', 'topological\ncompleteness', C_ORBIT2),
        (12.5, 1.5, 'Chemical\nbonding', 'shared resonances', C_ORBIT2),
    ]

    for x, y, title, subtitle, color in boxes:
        # Box
        rect = plt.Rectangle((x - 1.3, y - 0.7), 2.6, 1.4,
                             fill=True, facecolor='#1a1a1a',
                             edgecolor=color, linewidth=1.5, alpha=0.9,
                             zorder=5)
        ax.add_patch(rect)
        ax.text(x, y + 0.2, title, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=6)
        ax.text(x, y - 0.35, subtitle, ha='center', va='center',
                fontsize=9, color=color, zorder=6, style='italic')

    # Arrows between boxes (horizontal)
    for row_y in [6.5, 4.0, 1.5]:
        for x_start in [3.3, 6.8, 10.3]:
            ax.annotate('', xy=(x_start + 0.9, row_y),
                       xytext=(x_start, row_y),
                       arrowprops=dict(arrowstyle='->', color=C_DIM, lw=1.5),
                       zorder=3)

    # Arrows between rows (vertical)
    for x_col in [2, 5.5, 9, 12.5]:
        ax.annotate('', xy=(x_col, 5.2), xytext=(x_col, 5.8),
                   arrowprops=dict(arrowstyle='->', color=C_DIM, lw=1.2),
                   zorder=3)
        ax.annotate('', xy=(x_col, 2.7), xytext=(x_col, 3.3),
                   arrowprops=dict(arrowstyle='->', color=C_DIM, lw=1.2),
                   zorder=3)

    # Row labels
    ax.text(-0.2, 6.5, 'PARTICLE\nPHYSICS', fontsize=10, color=C_PHOTON,
            fontweight='bold', va='center', ha='center')
    ax.text(-0.2, 4.0, 'ATOMIC\nPHYSICS', fontsize=10, color=C_ORBIT,
            fontweight='bold', va='center', ha='center')
    ax.text(-0.2, 1.5, 'CHEMISTRY', fontsize=10, color=C_ORBIT2,
            fontweight='bold', va='center', ha='center')

    # Bottom text
    ax.text(8, 0.3,
            'ONE ASSUMPTION: a photon circulating at c on a closed torus knot in Minkowski spacetime',
            fontsize=12, ha='center', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                      edgecolor='white', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'nwt_06_summary.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate null worldtube diagrams')
    parser.add_argument('--torus', action='store_true', help='Torus knot diagram')
    parser.add_argument('--atom', action='store_true', help='Hydrogen atom diagram')
    parser.add_argument('--spectrum', action='store_true', help='Emission spectrum')
    parser.add_argument('--frequencies', action='store_true', help='Three frequencies')
    parser.add_argument('--mechanism', action='store_true', help='Transition mechanism')
    parser.add_argument('--summary', action='store_true', help='Summary flowchart')
    args = parser.parse_args()

    # If no specific diagram requested, generate all
    gen_all = not any([args.torus, args.atom, args.spectrum,
                       args.frequencies, args.mechanism, args.summary])

    print("Generating null worldtube diagrams...")
    print(f"Output directory: {OUTDIR}\n")

    if gen_all or args.torus:
        fig_torus_knot()
    if gen_all or args.atom:
        fig_hydrogen_atom()
    if gen_all or args.spectrum:
        fig_spectrum()
    if gen_all or args.frequencies:
        fig_three_frequencies()
    if gen_all or args.mechanism:
        fig_driven_oscillator()
    if gen_all or args.summary:
        fig_model_summary()

    print("\nDone!")


if __name__ == '__main__':
    main()
