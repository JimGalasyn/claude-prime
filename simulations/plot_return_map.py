#!/usr/bin/env python3
"""
Poincaré return map for the driven Koide oscillator.

Extracts θ_n from the stroboscopic Poincaré section and plots
θ_{n+1} vs θ_n to reveal the underlying 1D map structure of the
strange attractor.

Usage:
    python plot_return_map.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
Om = omega_0 * 2.0 / 3
T = 2 * np.pi / Om

def Gamma(th, g0):
    mmax = np.max(masses_at(th))
    return g0 * (1 + (mmax / m_ref)**2) / 2

def rhs_driven(t, y, g0, F):
    th, om = y
    return [om, -dV(th) - Gamma(th, g0) * om + F * np.sin(Om * t)]

# ── Compute Poincaré section ─────────────────────────────────────────
g0 = 0.3
F = 8.0
N_trans = 500
N_rec = 8000  # many points for good return map

print(f"Computing Poincaré section: F={F}, {N_rec} stroboscopic points...")

y0 = [theta_K, 0.0]
sol_t = solve_ivp(lambda t, y: rhs_driven(t, y, g0, F),
                  (0, N_trans * T), y0,
                  rtol=1e-10, atol=1e-12, max_step=T / 15)

y_after = sol_t.y[:, -1]
t_rec = np.arange(N_rec) * T
sol_r = solve_ivp(lambda t, y: rhs_driven(t, y, g0, F),
                  (0, N_rec * T), y_after, t_eval=t_rec,
                  rtol=1e-10, atol=1e-12, max_step=T / 15)

th_n = sol_r.y[0] % (2 * np.pi)
om_n = sol_r.y[1]
print(f"  Got {len(th_n)} points")

# Return map pairs
th_now = th_n[:-1]
th_next = th_n[1:]
om_now = om_n[:-1]
om_next = om_n[1:]

# ── Figure ────────────────────────────────────────────────────────────
print("Rendering...")
bg = '#060612'
text_color = '#d0d0d0'
grid_color = '#1a1a3a'

fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor=bg,
                          gridspec_kw={'hspace': 0.30, 'wspace': 0.25})

# ── Panel A: θ return map (θ_{n+1} vs θ_n) ───────────────────────────
ax = axes[0, 0]
ax.set_facecolor(bg)

# Density coloring
hbin = 250
h2d, xed, yed = np.histogram2d(th_now, th_next, bins=hbin,
                                range=[[0, 2*np.pi], [0, 2*np.pi]])
h2d = gaussian_filter(h2d, sigma=1.0)
xi = np.clip(np.searchsorted(xed, th_now) - 1, 0, hbin - 1)
yi = np.clip(np.searchsorted(yed, th_next) - 1, 0, hbin - 1)
dens = h2d[xi, yi]
dens = dens / (dens.max() + 1e-12)

ax.scatter(th_now, th_next, s=0.8, c=dens, cmap='hot', alpha=0.9,
           edgecolors='none', vmin=0, vmax=0.5)

# Identity line
ax.plot([0, 2*np.pi], [0, 2*np.pi], color='#3498db', linewidth=1,
        linestyle='--', alpha=0.5, label='θ_{n+1} = θ_n')

# Koide angle crosshairs
ax.axvline(theta_K, color='#f1c40f', linewidth=1, linestyle=':', alpha=0.5)
ax.axhline(theta_K, color='#f1c40f', linewidth=1, linestyle=':', alpha=0.5)

ax.set_xlabel('θ_n', color=text_color, fontsize=12)
ax.set_ylabel('θ_{n+1}', color=text_color, fontsize=12)
ax.set_title('A.  Poincaré Return Map: θ_{n+1} vs θ_n',
             color='white', fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], fontsize=9)
ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'], fontsize=9)
ax.set_aspect('equal')
ax.tick_params(colors=text_color)
ax.legend(fontsize=9, facecolor=bg, edgecolor=grid_color, labelcolor=text_color)
for spine in ax.spines.values():
    spine.set_color(grid_color)

# ── Panel B: ω return map (ω_{n+1} vs ω_n) ──────────────────────────
ax = axes[0, 1]
ax.set_facecolor(bg)

h2d_o, xed_o, yed_o = np.histogram2d(om_now, om_next, bins=hbin)
h2d_o = gaussian_filter(h2d_o, sigma=1.0)
xi_o = np.clip(np.searchsorted(xed_o, om_now) - 1, 0, hbin - 1)
yi_o = np.clip(np.searchsorted(yed_o, om_next) - 1, 0, hbin - 1)
dens_o = h2d_o[xi_o, yi_o]
dens_o = dens_o / (dens_o.max() + 1e-12)

ax.scatter(om_now, om_next, s=0.8, c=dens_o, cmap='hot', alpha=0.9,
           edgecolors='none', vmin=0, vmax=0.5)

# Identity line
om_range = [om_n.min(), om_n.max()]
ax.plot(om_range, om_range, color='#3498db', linewidth=1,
        linestyle='--', alpha=0.5)

ax.set_xlabel('θ̇_n', color=text_color, fontsize=12)
ax.set_ylabel('θ̇_{n+1}', color=text_color, fontsize=12)
ax.set_title('B.  Momentum Return Map: θ̇_{n+1} vs θ̇_n',
             color='white', fontsize=13, fontweight='bold', pad=10)
ax.set_aspect('equal')
ax.tick_params(colors=text_color)
for spine in ax.spines.values():
    spine.set_color(grid_color)

# ── Panel C: Time series showing chaos ────────────────────────────────
ax = axes[1, 0]
ax.set_facecolor(bg)

# Show a stretch of the stroboscopic θ series
n_show = min(500, len(th_n))
n_idx = np.arange(n_show)
ax.plot(n_idx, th_n[:n_show], color='#e74c3c', linewidth=0.6, alpha=0.8)
ax.scatter(n_idx, th_n[:n_show], s=1.5, c='#ff9944', alpha=0.9,
           edgecolors='none', zorder=5)

# Koide angle
ax.axhline(theta_K, color='#f1c40f', linewidth=1.5, linestyle='--',
           alpha=0.6, label=f'θ_K = {np.degrees(theta_K):.1f}°')

# Well positions
for k in range(3):
    ax.axhline(k * 2 * np.pi / 3, color='#2ecc71', linewidth=0.8,
               linestyle=':', alpha=0.4)

ax.set_xlabel('Strobe index n', color=text_color, fontsize=12)
ax.set_ylabel('θ_n (mod 2π)', color=text_color, fontsize=12)
ax.set_title('C.  Stroboscopic Time Series (Chaotic)',
             color='white', fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0, n_show)
ax.set_ylim(0, 2 * np.pi)
ax.set_yticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax.set_yticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=9)
ax.tick_params(colors=text_color)
ax.legend(fontsize=9, facecolor=bg, edgecolor=grid_color, labelcolor=text_color)
for spine in ax.spines.values():
    spine.set_color(grid_color)

# ── Panel D: Histogram of return map residuals ────────────────────────
ax = axes[1, 1]
ax.set_facecolor(bg)

# Δθ = θ_{n+1} - θ_n (mod 2π, centered)
delta_th = th_next - th_now
delta_th = (delta_th + np.pi) % (2 * np.pi) - np.pi  # center on 0

ax.hist(delta_th, bins=150, color='#e74c3c', alpha=0.7, edgecolor='none',
        density=True)

ax.axvline(0, color='#f1c40f', linewidth=1.5, linestyle='--', alpha=0.7,
           label='Δθ = 0 (fixed point)')

# Statistics
mean_d = np.mean(delta_th)
std_d = np.std(delta_th)
ax.text(0.95, 0.95,
        f'⟨Δθ⟩ = {mean_d:.4f}\nσ(Δθ) = {std_d:.3f} rad\n'
        f'    = {np.degrees(std_d):.1f}°',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, color='white',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a3a',
                  edgecolor='#4a4a6a', alpha=0.9))

ax.set_xlabel('Δθ = θ_{n+1} − θ_n', color=text_color, fontsize=12)
ax.set_ylabel('Probability density', color=text_color, fontsize=12)
ax.set_title('D.  Return Map Step Distribution',
             color='white', fontsize=13, fontweight='bold', pad=10)
ax.tick_params(colors=text_color)
ax.legend(fontsize=9, facecolor=bg, edgecolor=grid_color, labelcolor=text_color)
for spine in ax.spines.values():
    spine.set_color(grid_color)

# ── Supertitle ────────────────────────────────────────────────────────
fig.suptitle(f'Poincaré Return Maps — Driven Koide Oscillator (F = {F})\n'
             f'Stroboscopic Sampling at T = 2π/Ω,  {N_rec} iterates',
             color='white', fontsize=16, fontweight='bold', y=0.99)

outpath = '/home/jim/repos/claude-prime/simulations/return_map.png'
fig.savefig(outpath, dpi=200, facecolor=bg, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
