#!/usr/bin/env python3
"""
Bifurcation diagram for the driven dissipative Koide oscillator.

    θ̈ + Γ(θ)θ̇ + (3√2/2)sin(3θ) = F·sin(Ωτ)

Stroboscopic Poincaré section at the driving period T = 2π/Ω,
swept over driving amplitude F from laminar to fully chaotic regime.

Usage:
    python plot_bifurcation.py
"""

from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def masses_at(th):
    return np.array([(S3 * (1 + np.sqrt(2) * np.cos(th + 2 * np.pi * i / 3)))**2
                     for i in range(3)])

def dV(th):
    return (3 * np.sqrt(2) / 2) * np.sin(3 * th)

omega_0 = np.sqrt(9 * np.sqrt(2) / 2)
Om = omega_0 * 2.0 / 3  # subharmonic driving
T = 2 * np.pi / Om       # driving period

def Gamma(th, g0):
    mmax = np.max(masses_at(th))
    return g0 * (1 + (mmax / m_ref)**2) / 2

def rhs(t, y, g0, F):
    th, om = y
    return [om, -dV(th) - Gamma(th, g0) * om + F * np.sin(Om * t)]

# ── Bifurcation sweep ────────────────────────────────────────────────
g0 = 0.3
N_trans = 250       # transient periods to discard
N_rec = 120         # periods to record

# Two-resolution sweep: coarse overview + fine cascade region
F_coarse = np.linspace(0.01, 12.0, 400)
F_fine = np.linspace(0.05, 2.5, 400)
F_all = np.unique(np.sort(np.concatenate([F_coarse, F_fine])))

print(f"Computing bifurcation diagram: {len(F_all)} values of F")
print(f"  Transient: {N_trans} periods, Record: {N_rec} periods")
print(f"  Driving: Ω = (2/3)ω₀ = {Om:.4f}, T = {T:.4f}")

# Two initial conditions to capture coexisting attractors
ic_list = [
    [theta_K, 0.0],
    [0.1, 0.5],
]

bif_F = []
bif_theta = []
bif_omega = []

for ic_idx, y0_base in enumerate(ic_list):
    print(f"  IC {ic_idx+1}/{len(ic_list)}: θ₀={y0_base[0]:.2f}")
    for idx, F in enumerate(F_all):
        if idx % 200 == 0:
            print(f"    F = {F:.2f}  ({idx}/{len(F_all)})")

        y0 = list(y0_base)

        # Transient
        try:
            sol_t = solve_ivp(lambda t, y: rhs(t, y, g0, F),
                              (0, N_trans * T), y0,
                              rtol=1e-9, atol=1e-11, max_step=T / 10)
            if not sol_t.success or sol_t.y.shape[1] == 0:
                continue
        except Exception:
            continue

        y_after = sol_t.y[:, -1]

        # Record stroboscopic points
        t_rec = np.arange(N_rec) * T
        try:
            sol_r = solve_ivp(lambda t, y: rhs(t, y, g0, F),
                              (0, N_rec * T), y_after, t_eval=t_rec,
                              rtol=1e-9, atol=1e-11, max_step=T / 10)
            if not sol_r.success or sol_r.y.shape[1] < 10:
                continue
        except Exception:
            continue

        th_pts = sol_r.y[0] % (2 * np.pi)
        om_pts = sol_r.y[1]
        bif_F.extend([F] * len(th_pts))
        bif_theta.extend(th_pts.tolist())
        bif_omega.extend(om_pts.tolist())

bif_F = np.array(bif_F)
bif_theta = np.array(bif_theta)
bif_omega = np.array(bif_omega)
print(f"  Total points: {len(bif_F)}")

# ── Figure ────────────────────────────────────────────────────────────
print("\nRendering...")
bg = '#060612'
text_color = '#d0d0d0'
grid_color = '#1a1a3a'

fig, (ax_main, ax_zoom) = plt.subplots(2, 1, figsize=(18, 14), facecolor=bg,
                                         gridspec_kw={'height_ratios': [1.2, 1],
                                                      'hspace': 0.28})

# ── Panel A: Full bifurcation diagram ─────────────────────────────────
ax_main.set_facecolor(bg)

# Direct white-hot scatter — let density build up naturally
ax_main.scatter(bif_F, bif_theta, s=0.05, c='#ff6633', alpha=0.15,
                edgecolors='none', rasterized=True)

# Brighter overlay via 2D histogram rendered as image
from scipy.ndimage import gaussian_filter
hbins_x, hbins_y = 800, 400
h2d, xed, yed = np.histogram2d(bif_F, bif_theta, bins=[hbins_x, hbins_y],
                                range=[[0, 12], [0, 2*np.pi]])
h2d = gaussian_filter(h2d.T, sigma=[0.5, 0.5])  # transpose for imshow
# Log scale to reveal faint structure
h2d_log = np.log1p(h2d)
h2d_log = h2d_log / (h2d_log.max() + 1e-12)

ax_main.imshow(h2d_log, extent=[0, 12, 0, 2*np.pi], origin='lower',
               aspect='auto', cmap='hot', alpha=0.9, vmin=0, vmax=0.6,
               interpolation='bilinear')

# Mark θ_K
ax_main.axhline(theta_K, color='#f1c40f', linewidth=1.5, linestyle='--',
                alpha=0.7, label=f'θ_K = {np.degrees(theta_K):.1f}°')

# Mark Z₃ wells
for k in range(3):
    well_th = k * 2 * np.pi / 3
    ax_main.axhline(well_th, color='#2ecc71', linewidth=0.8,
                    linestyle=':', alpha=0.4)

ax_main.set_xlabel('Driving amplitude F', color=text_color, fontsize=13)
ax_main.set_ylabel('θ (mod 2π)', color=text_color, fontsize=13)
ax_main.set_title('A.  Bifurcation Diagram: Period-Doubling Route to Chaos\n'
                   'θ̈ + Γ(θ)θ̇ + V′(θ) = F·sin(Ωτ),  Ω = (2/3)ω₀',
                   color='white', fontsize=15, fontweight='bold', pad=12)
ax_main.set_xlim(0, 12)
ax_main.set_ylim(0, 2 * np.pi)
ax_main.set_yticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_main.set_yticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=10)
ax_main.tick_params(colors=text_color)
ax_main.legend(fontsize=10, facecolor=bg, edgecolor=grid_color,
               labelcolor=text_color, loc='upper left')
for spine in ax_main.spines.values():
    spine.set_color(grid_color)

# ── Panel B: Zoom into cascade region ─────────────────────────────────
ax_zoom.set_facecolor(bg)

F_zoom_lo, F_zoom_hi = 0.0, 3.0
mask_zoom = (bif_F >= F_zoom_lo) & (bif_F <= F_zoom_hi)
bif_F_z = bif_F[mask_zoom]
bif_theta_z = bif_theta[mask_zoom]

# Same technique: histogram image
h2d_z, xed_z, yed_z = np.histogram2d(bif_F_z, bif_theta_z,
                                       bins=[600, 400],
                                       range=[[F_zoom_lo, F_zoom_hi], [0, 2*np.pi]])
h2d_z = gaussian_filter(h2d_z.T, sigma=[0.4, 0.4])
h2d_z_log = np.log1p(h2d_z)
h2d_z_log = h2d_z_log / (h2d_z_log.max() + 1e-12)

ax_zoom.imshow(h2d_z_log, extent=[F_zoom_lo, F_zoom_hi, 0, 2*np.pi],
               origin='lower', aspect='auto', cmap='hot', alpha=0.95,
               vmin=0, vmax=0.5, interpolation='bilinear')

# Also scatter for crispness at branches
ax_zoom.scatter(bif_F_z, bif_theta_z, s=0.08, c='white', alpha=0.1,
                edgecolors='none', rasterized=True)

# Koide angle
ax_zoom.axhline(theta_K, color='#f1c40f', linewidth=1.5, linestyle='--',
                alpha=0.7)
for k in range(3):
    ax_zoom.axhline(k * 2 * np.pi / 3, color='#2ecc71', linewidth=0.8,
                    linestyle=':', alpha=0.4)

ax_zoom.set_xlabel('Driving amplitude F', color=text_color, fontsize=13)
ax_zoom.set_ylabel('θ (mod 2π)', color=text_color, fontsize=13)
ax_zoom.set_title('B.  Zoom: Period-Doubling Cascade (Feigenbaum Route)',
                   color='white', fontsize=15, fontweight='bold', pad=12)
ax_zoom.set_xlim(F_zoom_lo, F_zoom_hi)
ax_zoom.set_ylim(0, 2 * np.pi)
ax_zoom.set_yticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_zoom.set_yticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=10)
ax_zoom.tick_params(colors=text_color)
for spine in ax_zoom.spines.values():
    spine.set_color(grid_color)

# ── Supertitle ────────────────────────────────────────────────────────
fig.suptitle('Dissipative Koide Oscillator — Bifurcation Diagram\n'
             'Period-Doubling Cascade to Strange Attractor',
             color='white', fontsize=17, fontweight='bold', y=0.99)

outpath = OUTPUT_DIR / 'bifurcation_diagram.png'
fig.savefig(outpath, dpi=200, facecolor=bg, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
