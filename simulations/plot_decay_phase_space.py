#!/usr/bin/env python3
"""
Phase-space diagram of dissipative decay dynamics on the Koide circle.

Shows the cos(3θ) potential wells, saddle points, autonomous decay
trajectories, the driven strange attractor (Poincaré section), and
the attractor's band structure aligned with the physical Koide angle.

Usage:
    python plot_decay_phase_space.py              # show plot
    python plot_decay_phase_space.py --save       # save to PNG
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ── Physical parameters ────────────────────────────────────────────────
alpha = 1 / 137.035999084
m_e_MeV = 0.51099895000
m_mu = 105.6583755
m_tau = 1776.86

S = np.sqrt(m_e_MeV) + np.sqrt(m_mu) + np.sqrt(m_tau)
S3 = S / 3
theta_K = (6 * np.pi + 2) / 9  # physical Koide angle

def masses_at(th):
    f = np.array([1 + np.sqrt(2) * np.cos(th + 2 * np.pi * i / 3)
                   for i in range(3)])
    return (S3 * f)**2

m_ref = np.max(masses_at(theta_K))

# ── Potential and dynamics ─────────────────────────────────────────────
def V(th):
    return 0.5 - (np.sqrt(2) / 2) * np.cos(3 * th)

def dV(th):
    return (3 * np.sqrt(2) / 2) * np.sin(3 * th)

omega_0 = np.sqrt(9 * np.sqrt(2) / 2)

def Gamma(th, g0):
    mmax = np.max(masses_at(th))
    return g0 * (1 + (mmax / m_ref)**2) / 2

def rhs_auto(t, y, g0):
    th, om = y
    return [om, -dV(th) - Gamma(th, g0) * om]

def rhs_driven(t, y, g0, F, Om):
    th, om = y
    g = Gamma(th, g0)
    return [om, -dV(th) - g * om + F * np.sin(Om * t)]

def koide_lorenz(t, y, kappa, gamma_kl, delta, rho):
    th, p, sigma = y
    g = gamma_kl * (1 + 0.5 * np.cos(3 * th))
    d_th = p
    d_p = -dV(th) * (1 - kappa * sigma) - g * p
    d_sigma = -delta * sigma + rho - sigma * p**2
    return [d_th, d_p, d_sigma]


# ── Compute data ───────────────────────────────────────────────────────
print("Computing autonomous trajectories...")
g0 = 0.3
t_end = 60
t_eval = np.linspace(0, t_end, 8000)

trajectories = []
ics = [
    ([0.1, 4.0],   '#e74c3c', 'High energy (τ-like)'),
    ([0.1, 2.0],   '#3498db', 'Moderate energy'),
    ([0.1, 0.5],   '#2ecc71', 'Trapped in well'),
    ([np.pi/3 + 0.05, 0.1], '#f39c12', 'Near saddle'),
    ([4*np.pi/3 + 0.1, 5.0], '#9b59b6', 'Cross-well cascade'),
]

for y0, color, label in ics:
    sol = solve_ivp(lambda t, y: rhs_auto(t, y, g0),
                    (0, t_end), y0, t_eval=t_eval,
                    rtol=1e-10, atol=1e-12, max_step=0.03)
    if sol.success:
        trajectories.append((sol.y[0], sol.y[1], color, label))

# ── Poincaré section (driven chaotic attractor) ───────────────────────
print("Computing Poincaré section (driven attractor)...")
g0_bif = 0.3
F_poin = 8.0
Om_bif = omega_0 * 2.0 / 3
T_bif = 2 * np.pi / Om_bif

N_trans = 500
N_rec = 3000

y0_p = [theta_K, 0]
sol_tp = solve_ivp(lambda t, y: rhs_driven(t, y, g0_bif, F_poin, Om_bif),
                   (0, N_trans * T_bif), y0_p,
                   rtol=1e-10, atol=1e-12, max_step=T_bif / 15)

poincare_theta = None
poincare_omega = None
if sol_tp.success and sol_tp.y.shape[1] > 0:
    yp = sol_tp.y[:, -1]
    t_rec = np.arange(N_rec) * T_bif
    sol_rp = solve_ivp(lambda t, y: rhs_driven(t, y, g0_bif, F_poin, Om_bif),
                       (0, N_rec * T_bif), yp, t_eval=t_rec,
                       rtol=1e-10, atol=1e-12, max_step=T_bif / 15)
    if sol_rp.success and sol_rp.y.shape[1] > 100:
        poincare_theta = sol_rp.y[0] % (2 * np.pi)
        poincare_omega = sol_rp.y[1]

# ── Koide-Lorenz autonomous attractor ─────────────────────────────────
print("Computing Koide-Lorenz attractor...")
kap, gam, delt, rh = 1.5, 0.3, 1.0, 1.5
y0_kl = [theta_K + 0.1, 0.3, rh / delt + 0.1]
sol_kl = solve_ivp(
    lambda t, y: koide_lorenz(t, y, kap, gam, delt, rh),
    (0, 500), y0_kl, t_eval=np.linspace(80, 500, 40000),
    rtol=1e-10, atol=1e-12, max_step=0.015)

kl_theta = None
kl_p = None
kl_sigma = None
if sol_kl.success and sol_kl.y.shape[1] > 1000:
    kl_theta = sol_kl.y[0] % (2 * np.pi)
    kl_p = sol_kl.y[1]
    kl_sigma = sol_kl.y[2]

# ── Build figure ───────────────────────────────────────────────────────
print("Rendering figure...")

fig = plt.figure(figsize=(18, 14), facecolor='#0a0a1a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30,
                       left=0.06, right=0.97, top=0.94, bottom=0.05)

# Color scheme
bg_color = '#0a0a1a'
text_color = '#e0e0e0'
grid_color = '#1a1a3a'
well_color = '#1a3a2a'
saddle_color = '#3a1a1a'

# ── Panel A: Potential landscape ───────────────────────────────────────
ax_pot = fig.add_subplot(gs[0, 0])
ax_pot.set_facecolor(bg_color)

theta_range = np.linspace(0, 2 * np.pi, 500)
V_vals = V(theta_range)

ax_pot.fill_between(theta_range, V_vals, V_vals.min() - 0.1,
                     alpha=0.3, color='#2980b9')
ax_pot.plot(theta_range, V_vals, color='#3498db', linewidth=2.5)

# Mark wells and saddles
for k in range(3):
    well_th = k * 2 * np.pi / 3
    ax_pot.plot(well_th, V(well_th), 'o', color='#2ecc71', markersize=10,
                zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    saddle_th = (2 * k + 1) * np.pi / 3
    ax_pot.plot(saddle_th, V(saddle_th), 'X', color='#e74c3c', markersize=11,
                zorder=5, markeredgecolor='white', markeredgewidth=1.5)

# Mark physical Koide angle
ax_pot.axvline(theta_K, color='#f1c40f', linewidth=1.5, linestyle='--',
               alpha=0.8, label=f'θ_K = {theta_K:.2f}')

ax_pot.set_xlabel('θ (rad)', color=text_color, fontsize=11)
ax_pot.set_ylabel('V(θ)', color=text_color, fontsize=11)
ax_pot.set_title('A.  cos(3θ) Potential Landscape', color='white',
                  fontsize=13, fontweight='bold', pad=10)
ax_pot.legend(fontsize=9, facecolor=bg_color, edgecolor=grid_color,
              labelcolor=text_color)
ax_pot.set_xlim(0, 2 * np.pi)
ax_pot.tick_params(colors=text_color)
ax_pot.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_pot.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'],
                        fontsize=9)
for spine in ax_pot.spines.values():
    spine.set_color(grid_color)

# ── Panel B: Phase portrait with trajectories ──────────────────────────
ax_phase = fig.add_subplot(gs[0, 1:])
ax_phase.set_facecolor(bg_color)

# Background: potential as color map
th_bg = np.linspace(0, 2 * np.pi, 200)
om_bg = np.linspace(-6, 6, 200)
TH, OM = np.meshgrid(th_bg, om_bg)
E_bg = 0.5 * OM**2 + V(TH)

# Custom colormap: dark blue (low E) to dark red (high E)
cmap_energy = LinearSegmentedColormap.from_list('energy',
    ['#0a0a2e', '#0a1a3e', '#1a2a4e', '#2a3a5e', '#3a2a4e',
     '#4a2a3e', '#5a1a2e'], N=256)

im = ax_phase.pcolormesh(TH, OM, E_bg, cmap=cmap_energy, shading='gouraud',
                          vmin=-0.5, vmax=20, alpha=0.6)

# Separatrices (energy = V at saddle)
V_saddle = V(np.pi / 3)
for th0 in np.linspace(0, 2 * np.pi, 300):
    E_sep = V_saddle - V(th0)
    if E_sep >= 0:
        om_sep = np.sqrt(2 * E_sep)
        ax_phase.plot(th0, om_sep, '.', color='#7f8c8d', markersize=0.3, alpha=0.4)
        ax_phase.plot(th0, -om_sep, '.', color='#7f8c8d', markersize=0.3, alpha=0.4)

# Trajectories
for th_traj, om_traj, color, label in trajectories:
    th_mod = th_traj % (2 * np.pi)
    # Plot segments (break at wrapping points)
    breaks = np.where(np.abs(np.diff(th_mod)) > np.pi)[0]
    segments = np.split(np.arange(len(th_mod)), breaks + 1)
    for seg in segments:
        if len(seg) > 1:
            ax_phase.plot(th_mod[seg], om_traj[seg], color=color,
                          linewidth=0.8, alpha=0.85)
    # Starting point
    ax_phase.plot(th_mod[0], om_traj[0], 'o', color=color, markersize=5,
                  markeredgecolor='white', markeredgewidth=0.5, zorder=6)
    # Label at start
    ax_phase.annotate(label, (th_mod[0], om_traj[0]),
                       xytext=(8, 5), textcoords='offset points',
                       fontsize=7, color=color, alpha=0.9)

# Fixed points
for k in range(3):
    well_th = k * 2 * np.pi / 3
    ax_phase.plot(well_th, 0, 'o', color='#2ecc71', markersize=9,
                  zorder=7, markeredgecolor='white', markeredgewidth=1.5)
    saddle_th = (2 * k + 1) * np.pi / 3
    ax_phase.plot(saddle_th, 0, 'X', color='#e74c3c', markersize=10,
                  zorder=7, markeredgecolor='white', markeredgewidth=1.5)

# Koide angle
ax_phase.axvline(theta_K, color='#f1c40f', linewidth=1.5, linestyle='--',
                  alpha=0.7)

ax_phase.set_xlabel('θ (rad)', color=text_color, fontsize=11)
ax_phase.set_ylabel('θ̇', color=text_color, fontsize=11)
ax_phase.set_title('B.  Phase Portrait: Autonomous Decay Cascades',
                    color='white', fontsize=13, fontweight='bold', pad=10)
ax_phase.set_xlim(0, 2 * np.pi)
ax_phase.set_ylim(-6, 6)
ax_phase.tick_params(colors=text_color)
ax_phase.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_phase.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'],
                          fontsize=9)
for spine in ax_phase.spines.values():
    spine.set_color(grid_color)

# Legend for fixed points
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
           markersize=8, label='Stable spiral (well)', linewidth=0),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c',
           markersize=8, label='Saddle point', linewidth=0),
    Line2D([0], [0], color='#f1c40f', linestyle='--', linewidth=1.5,
           label=f'θ_K = (6π+2)/9'),
]
ax_phase.legend(handles=legend_elements, fontsize=8, facecolor=bg_color,
                edgecolor=grid_color, labelcolor=text_color, loc='upper right')

# ── Panel C: Poincaré section (driven attractor) ──────────────────────
ax_poin = fig.add_subplot(gs[1, 0:2])
ax_poin.set_facecolor(bg_color)

if poincare_theta is not None:
    # Density-based coloring to reveal fractal structure
    from scipy.ndimage import gaussian_filter
    _hbin = 200
    _h2d, _xed, _yed = np.histogram2d(poincare_theta, poincare_omega,
                                        bins=_hbin)
    _h2d = gaussian_filter(_h2d, sigma=1.2)
    # Map each point to its density
    _xi = np.clip(np.searchsorted(_xed, poincare_theta) - 1, 0, _hbin - 1)
    _yi = np.clip(np.searchsorted(_yed, poincare_omega) - 1, 0, _hbin - 1)
    _dens = _h2d[_xi, _yi]
    _dens = _dens / (_dens.max() + 1e-12)  # normalize 0-1

    ax_poin.scatter(poincare_theta, poincare_omega, s=1.8, c=_dens,
                     cmap='hot', alpha=0.95, edgecolors='none',
                     vmin=0.0, vmax=0.7)
    # Mark Z₃ wells
    for k in range(3):
        well_th = k * 2 * np.pi / 3
        ax_poin.axvline(well_th, color='#2ecc71', linewidth=0.8,
                         linestyle=':', alpha=0.5)
    ax_poin.axvline(theta_K, color='#f1c40f', linewidth=1.5,
                     linestyle='--', alpha=0.8,
                     label=f'θ_K (Koide angle)')

ax_poin.set_xlabel('θ (rad)', color=text_color, fontsize=11)
ax_poin.set_ylabel('θ̇', color=text_color, fontsize=11)
ax_poin.set_title(f'C.  Poincaré Section — Driven Attractor (F = {F_poin}, '
                   f'd ≈ 1.5)',
                   color='white', fontsize=13, fontweight='bold', pad=10)
ax_poin.set_xlim(0, 2 * np.pi)
ax_poin.tick_params(colors=text_color)
ax_poin.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_poin.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'],
                          fontsize=9)
ax_poin.legend(fontsize=9, facecolor=bg_color, edgecolor=grid_color,
               labelcolor=text_color)
for spine in ax_poin.spines.values():
    spine.set_color(grid_color)

# ── Panel D: Koide-Lorenz attractor (θ, p projection) ─────────────────
ax_kl = fig.add_subplot(gs[1, 2])
ax_kl.set_facecolor(bg_color)

if kl_theta is not None:
    # Color by σ value
    sc = ax_kl.scatter(kl_theta, kl_p, s=0.15, c=kl_sigma,
                        cmap='plasma', alpha=0.5, edgecolors='none',
                        vmin=np.percentile(kl_sigma, 5),
                        vmax=np.percentile(kl_sigma, 95))
    cb = plt.colorbar(sc, ax=ax_kl, shrink=0.7, pad=0.02)
    cb.set_label('σ (breathing)', color=text_color, fontsize=9)
    cb.ax.tick_params(colors=text_color, labelsize=8)

    # Mark wells and Koide angle
    for k in range(3):
        ax_kl.axvline(k * 2 * np.pi / 3, color='#2ecc71', linewidth=0.8,
                       linestyle=':', alpha=0.5)
    ax_kl.axvline(theta_K, color='#f1c40f', linewidth=1.5,
                   linestyle='--', alpha=0.8)

ax_kl.set_xlabel('θ (rad)', color=text_color, fontsize=11)
ax_kl.set_ylabel('p (momentum)', color=text_color, fontsize=11)
ax_kl.set_title('D.  Koide-Lorenz Attractor (autonomous)',
                 color='white', fontsize=13, fontweight='bold', pad=10)
ax_kl.set_xlim(0, 2 * np.pi)
ax_kl.tick_params(colors=text_color)
ax_kl.set_xticks([0, 2*np.pi/3, 4*np.pi/3, 2*np.pi])
ax_kl.set_xticklabels(['0', '2π/3', '4π/3', '2π'], fontsize=9)
for spine in ax_kl.spines.values():
    spine.set_color(grid_color)

# ── Panel E: Band structure histogram ──────────────────────────────────
ax_band = fig.add_subplot(gs[2, 0:2])
ax_band.set_facecolor(bg_color)

if kl_theta is not None:
    N_bins = 120
    hist, edges = np.histogram(kl_theta, bins=N_bins, range=(0, 2 * np.pi))
    centers = (edges[:-1] + edges[1:]) / 2
    # Smooth
    kernel = np.ones(3) / 3
    hist_s = np.convolve(hist, kernel, mode='same')

    # Color bars by proximity to wells
    bar_colors = []
    for c in centers:
        dists = [min(abs(c - k * 2 * np.pi / 3),
                      2 * np.pi - abs(c - k * 2 * np.pi / 3))
                 for k in range(3)]
        min_dist = min(dists)
        if min_dist < np.pi / 6:
            bar_colors.append('#2ecc71')
        elif min(abs(c - theta_K), 2 * np.pi - abs(c - theta_K)) < 0.15:
            bar_colors.append('#f1c40f')
        else:
            bar_colors.append('#3498db')

    ax_band.bar(centers, hist_s, width=2 * np.pi / N_bins * 0.9,
                 color=bar_colors, alpha=0.7, edgecolor='none')

    # Mark physical angles
    ax_band.axvline(theta_K, color='#f1c40f', linewidth=2.5,
                     linestyle='--', alpha=0.9,
                     label=f'θ_K = {theta_K:.3f} ({np.degrees(theta_K):.1f}°)')
    for k in range(3):
        well_th = k * 2 * np.pi / 3
        ax_band.axvline(well_th, color='#2ecc71', linewidth=1.5,
                         linestyle=':', alpha=0.6)

    # Mean line
    ax_band.axhline(np.mean(hist_s), color='#7f8c8d', linewidth=1,
                     linestyle=':', alpha=0.5, label='Mean density')

    # Annotate the bands
    mean_h = np.mean(hist_s)
    for i in range(1, N_bins - 1):
        if (hist_s[i] > hist_s[i-1] and hist_s[i] > hist_s[i+1]
                and hist_s[i] > mean_h * 1.5):
            m_b = masses_at(centers[i])
            m_sort = sorted(m_b, reverse=True)
            ax_band.annotate(
                f'{np.degrees(centers[i]):.0f}°\n'
                f'{m_sort[0]:.0f}, {m_sort[1]:.0f}, {m_sort[2]:.1f} MeV',
                xy=(centers[i], hist_s[i]),
                xytext=(0, 15), textcoords='offset points',
                fontsize=7.5, color='white', ha='center',
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a4a',
                          edgecolor='#4a4a6a', alpha=0.9))

ax_band.set_xlabel('θ (rad)', color=text_color, fontsize=11)
ax_band.set_ylabel('Density (points/bin)', color=text_color, fontsize=11)
ax_band.set_title('E.  Attractor Band Structure → Mass Spectrum',
                   color='white', fontsize=13, fontweight='bold', pad=10)
ax_band.set_xlim(0, 2 * np.pi)
ax_band.tick_params(colors=text_color)
ax_band.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_band.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'],
                          fontsize=9)
ax_band.legend(fontsize=9, facecolor=bg_color, edgecolor=grid_color,
               labelcolor=text_color, loc='upper right')
for spine in ax_band.spines.values():
    spine.set_color(grid_color)

# ── Panel F: Koide-Lorenz 3D view ─────────────────────────────────────
ax_3d = fig.add_subplot(gs[2, 2], projection='3d')
ax_3d.set_facecolor(bg_color)

if kl_theta is not None and kl_sigma is not None:
    # Subsample for clarity
    step = max(1, len(kl_theta) // 5000)
    th_sub = kl_theta[::step]
    p_sub = kl_p[::step]
    sig_sub = kl_sigma[::step]

    ax_3d.scatter(th_sub, p_sub, sig_sub, s=1.5, c=sig_sub,
                   cmap='hot', alpha=0.9, edgecolors='none',
                   vmin=np.percentile(kl_sigma, 5),
                   vmax=np.percentile(kl_sigma, 95))

    ax_3d.set_xlabel('θ', color=text_color, fontsize=9, labelpad=2)
    ax_3d.set_ylabel('p', color=text_color, fontsize=9, labelpad=2)
    ax_3d.set_zlabel('σ', color=text_color, fontsize=9, labelpad=2)
    ax_3d.tick_params(colors=text_color, labelsize=7)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor(grid_color)
    ax_3d.yaxis.pane.set_edgecolor(grid_color)
    ax_3d.zaxis.pane.set_edgecolor(grid_color)
    ax_3d.view_init(elev=25, azim=45)

ax_3d.set_title('F.  3D Koide-Lorenz\nStrange Attractor',
                 color='white', fontsize=12, fontweight='bold', pad=5)

# ── Title ──────────────────────────────────────────────────────────────
fig.suptitle('Dissipative Decay Dynamics in Koide Phase Space\n'
             'Null Worldtube Model — Strange Attractors and Band Structure',
             color='white', fontsize=16, fontweight='bold', y=0.99)

# ── Save / show ────────────────────────────────────────────────────────
outpath = '/home/jim/repos/claude-prime/simulations/decay_phase_space.png'
fig.savefig(outpath, dpi=200, facecolor=bg_color, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
