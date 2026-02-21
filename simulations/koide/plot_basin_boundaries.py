#!/usr/bin/env python3
"""
Fractal basin boundaries of the dissipative Koide oscillator.

For a grid of initial conditions (θ₀, θ̇₀), integrates the autonomous
damped system and determines which of the three Z₃ wells each trajectory
settles into. If the boundaries are fractal, the 'choice' of which
particle to become is fundamentally unpredictable at fine scales.

Optimized: uses simple RK4 stepper for speed.

Usage:
    python plot_basin_boundaries.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Physical parameters ───────────────────────────────────────────────
m_e_MeV = 0.51099895000
m_mu = 105.6583755
m_tau = 1776.86
S = np.sqrt(m_e_MeV) + np.sqrt(m_mu) + np.sqrt(m_tau)
S3 = S / 3
theta_K = (6 * np.pi + 2) / 9

m_ref = np.max(np.array([(S3 * (1 + np.sqrt(2) * np.cos(theta_K + 2 * np.pi * i / 3)))**2
                          for i in range(3)]))

def V(th):
    return 0.5 - (np.sqrt(2) / 2) * np.cos(3 * th)

def dV_arr(th):
    """Vectorized derivative of potential."""
    return (3 * np.sqrt(2) / 2) * np.sin(3 * th)

def Gamma_arr(th, g0):
    """Vectorized damping — use simplified form for speed."""
    # Approximate: Gamma ~ g0 * (1 + cos²(θ)) avoids per-point mass calc
    return g0 * (1 + 0.5 * np.cos(th)**2)

# Wells at θ = 0, 2π/3, 4π/3
wells = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])

def classify_basin(th_final, om_final):
    """Vectorized classification of endpoints to wells."""
    th_mod = th_final % (2 * np.pi)
    result = np.full(th_mod.shape, -1, dtype=int)
    for k in range(3):
        dist = np.abs(th_mod - wells[k])
        dist = np.minimum(dist, 2 * np.pi - dist)
        mask = (dist < np.pi / 4) & (np.abs(om_final) < 0.5)
        result[mask] = k
    return result


def rk4_step(th, om, g0, dt):
    """Vectorized RK4 step for the damped oscillator."""
    def f_th(th, om):
        return om
    def f_om(th, om):
        return -dV_arr(th) - Gamma_arr(th, g0) * om

    k1_th = f_th(th, om)
    k1_om = f_om(th, om)

    k2_th = f_th(th + 0.5*dt*k1_th, om + 0.5*dt*k1_om)
    k2_om = f_om(th + 0.5*dt*k1_th, om + 0.5*dt*k1_om)

    k3_th = f_th(th + 0.5*dt*k2_th, om + 0.5*dt*k2_om)
    k3_om = f_om(th + 0.5*dt*k2_th, om + 0.5*dt*k2_om)

    k4_th = f_th(th + dt*k3_th, om + dt*k3_om)
    k4_om = f_om(th + dt*k3_th, om + dt*k3_om)

    th_new = th + dt/6 * (k1_th + 2*k2_th + 2*k3_th + k4_th)
    om_new = om + dt/6 * (k1_om + 2*k2_om + 2*k3_om + k4_om)
    return th_new, om_new


# ── Basin computation (fully vectorized) ──────────────────────────────
N = 600
th_range = (0, 2 * np.pi)
om_range = (-6, 6)

th_grid = np.linspace(th_range[0], th_range[1], N)
om_grid = np.linspace(om_range[0], om_range[1], N)
TH0, OM0 = np.meshgrid(th_grid, om_grid)

# Flatten for vectorized integration
th_flat = TH0.ravel().copy()
om_flat = OM0.ravel().copy()

dt = 0.05
g0_values = [0.3, 0.05]
t_ends = [50, 80]  # weak damping needs longer
results = []

for g0, t_end in zip(g0_values, t_ends):
    print(f"Computing basin map: g₀={g0}, {N}×{N} = {N*N} points, "
          f"t_end={t_end}...")

    th = TH0.ravel().copy()
    om = OM0.ravel().copy()

    n_steps = int(t_end / dt)
    for step in range(n_steps):
        th, om = rk4_step(th, om, g0, dt)
        if (step + 1) % 5000 == 0:
            print(f"  step {step+1}/{n_steps} "
                  f"(t = {(step+1)*dt:.1f}/{t_end})")

    basin = classify_basin(th, om).reshape(N, N)

    total = N * N
    for k in range(3):
        frac = np.sum(basin == k) / total * 100
        print(f"  Well {k}: {frac:.1f}%")
    unclass = np.sum(basin == -1) / total * 100
    print(f"  Unclassified: {unclass:.1f}%")

    results.append((g0, basin))

# ── Fractal dimension estimate ────────────────────────────────────────
print("\nEstimating fractal dimension of basin boundaries...")
for g0, basin in results:
    boundary = np.zeros_like(basin, dtype=bool)
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        shifted = np.roll(np.roll(basin, di, axis=0), dj, axis=1)
        boundary |= (basin != shifted) & (basin >= 0) & (shifted >= 0)

    n_boundary = np.sum(boundary)
    if n_boundary > 0:
        d_est = np.log(n_boundary) / np.log(N)
        print(f"  g₀={g0}: boundary pixels = {n_boundary}, "
              f"d_box ≈ {d_est:.3f}")

# ── Figure ────────────────────────────────────────────────────────────
print("\nRendering...")
bg = '#060612'
text_color = '#d0d0d0'
grid_color = '#1a1a3a'

fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=bg,
                          gridspec_kw={'wspace': 0.22,
                                       'left': 0.06, 'right': 0.97,
                                       'top': 0.88, 'bottom': 0.10})

cmap_basin = ListedColormap(['#0e0e20',    # -1: unclassified
                              '#2980b9',    # 0: well 0
                              '#27ae60',    # 1: well 2π/3
                              '#c0392b'])   # 2: well 4π/3

well_colors = ['#3498db', '#2ecc71', '#e74c3c']

for idx, (g0, basin) in enumerate(results):
    ax = axes[idx]
    ax.set_facecolor(bg)

    basin_shifted = basin + 1  # shift -1→0 for colormap

    ax.imshow(basin_shifted, extent=[th_range[0], th_range[1],
                                      om_range[0], om_range[1]],
              origin='lower', aspect='auto', cmap=cmap_basin,
              vmin=0, vmax=3, interpolation='nearest')

    # Mark wells
    for k in range(3):
        ax.plot(wells[k], 0, 'o', color='white', markersize=10,
                markeredgecolor='black', markeredgewidth=2, zorder=10)
        ax.text(wells[k], -0.8, f'{k*120}°', color='white', fontsize=9,
                ha='center', va='top', fontweight='bold')

    # Saddle points
    for k in range(3):
        saddle_th = (2 * k + 1) * np.pi / 3
        ax.plot(saddle_th, 0, 'X', color='#f1c40f', markersize=10,
                markeredgecolor='black', markeredgewidth=1, zorder=10)

    # Koide angle
    ax.axvline(theta_K, color='#f1c40f', linewidth=2, linestyle='--',
               alpha=0.7)
    ax.text(theta_K + 0.08, om_range[1] - 0.5, f'θ_K',
            color='#f1c40f', fontsize=11, fontweight='bold')

    # Separatrix (conservative energy contour through saddle)
    V_saddle = V(np.pi / 3)
    th_sep = np.linspace(0, 2 * np.pi, 500)
    for th_s in th_sep:
        E_sep = V_saddle - V(th_s)
        if E_sep >= 0:
            om_sep = np.sqrt(2 * E_sep)
            ax.plot(th_s, om_sep, '.', color='white', markersize=0.3, alpha=0.4)
            ax.plot(th_s, -om_sep, '.', color='white', markersize=0.3, alpha=0.4)

    damping_label = 'Moderate' if g0 > 0.1 else 'Weak'

    # Boundary overlay
    boundary = np.zeros_like(basin, dtype=bool)
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        shifted = np.roll(np.roll(basin, di, axis=0), dj, axis=1)
        boundary |= (basin != shifted) & (basin >= 0) & (shifted >= 0)
    n_bnd = np.sum(boundary)
    d_est = np.log(n_bnd) / np.log(N) if n_bnd > 0 else 0

    ax.set_xlabel('θ₀ (initial angle)', color=text_color, fontsize=12)
    if idx == 0:
        ax.set_ylabel('θ̇₀ (initial velocity)', color=text_color, fontsize=12)
    ax.set_title(f'{damping_label} Damping (Γ₀ = {g0})\n'
                 f'd_box ≈ {d_est:.2f}  ({n_bnd} boundary pixels)',
                 color='white', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
    ax.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=9)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2980b9', label='→ Well 0° (e-sector)'),
    Patch(facecolor='#27ae60', label='→ Well 120° (μ-sector)'),
    Patch(facecolor='#c0392b', label='→ Well 240° (τ-sector)'),
    Patch(facecolor='#0e0e20', edgecolor='#4a4a6a', label='Unclassified'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markersize=8, label='Stable well', linewidth=0),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#f1c40f',
           markersize=8, label='Saddle point', linewidth=0),
    Line2D([0], [0], color='#f1c40f', linestyle='--', linewidth=2,
           label=f'θ_K = {np.degrees(theta_K):.1f}°'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=10, facecolor=bg, edgecolor=grid_color,
           labelcolor=text_color, bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Basin of Attraction Boundaries — Autonomous Koide Oscillator\n'
             'Which Well Does Each Initial Condition Fall Into?',
             color='white', fontsize=16, fontweight='bold', y=0.97)

outpath = OUTPUT_DIR / 'basin_boundaries.png'
fig.savefig(outpath, dpi=200, facecolor=bg, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.close()
