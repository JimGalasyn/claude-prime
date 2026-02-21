#!/usr/bin/env python3
"""
Standalone visualization of the Koide-Lorenz strange attractor.

The autonomous 3D system on the Koide circle:
    θ̇ = p
    ṗ = -V'(θ)(1 - κσ) - Γ(θ)p
    σ̇ = -δσ + ρ - σp²

where V(θ) = ½ - (√2/2)cos(3θ) is the Borromean coupling potential.

Usage:
    python plot_koide_lorenz_attractor.py
"""

from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────
m_e_MeV = 0.51099895000
m_mu = 105.6583755
m_tau = 1776.86
S = np.sqrt(m_e_MeV) + np.sqrt(m_mu) + np.sqrt(m_tau)
theta_K = (6 * np.pi + 2) / 9

# ── Dynamics ──────────────────────────────────────────────────────────
def dV(th):
    return (3 * np.sqrt(2) / 2) * np.sin(3 * th)

def koide_lorenz(t, y, kappa, gamma_kl, delta, rho):
    th, p, sigma = y
    g = gamma_kl * (1 + 0.5 * np.cos(3 * th))
    d_th = p
    d_p = -dV(th) * (1 - kappa * sigma) - g * p
    d_sigma = -delta * sigma + rho - sigma * p**2
    return [d_th, d_p, d_sigma]

# ── Integrate ─────────────────────────────────────────────────────────
print("Integrating Koide-Lorenz system...")
kap, gam, delt, rh = 1.5, 0.3, 1.0, 1.5
y0 = [theta_K + 0.1, 0.3, rh / delt + 0.1]

# Long integration for dense coverage
sol = solve_ivp(
    lambda t, y: koide_lorenz(t, y, kap, gam, delt, rh),
    (0, 800), y0, t_eval=np.linspace(80, 800, 120000),
    rtol=1e-11, atol=1e-13, max_step=0.012)

th = sol.y[0] % (2 * np.pi)
p = sol.y[1]
sigma = sol.y[2]

print(f"  {len(th)} points on attractor")

# ── Figure ────────────────────────────────────────────────────────────
print("Rendering...")
bg = '#060612'
fig = plt.figure(figsize=(16, 14), facecolor=bg)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(bg)

# Build line segments colored by σ for smooth trails
points = np.array([th, p, sigma]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Remove segments that wrap in θ
dth = np.abs(np.diff(th))
mask = dth < np.pi
segments = segments[mask]
sig_mid = ((sigma[:-1] + sigma[1:]) / 2)[mask]

# Normalize σ for colormap
sig_norm = (sig_mid - np.percentile(sigma, 3)) / (
    np.percentile(sigma, 97) - np.percentile(sigma, 3) + 1e-12)
sig_norm = np.clip(sig_norm, 0, 1)

# Use inferno for a vivid dark-to-bright palette
cmap = plt.cm.inferno
colors = cmap(sig_norm)
colors[:, 3] = 0.35  # base alpha

# Boost alpha for high-σ regions (bright breathing mode)
colors[:, 3] = np.where(sig_norm > 0.6, 0.65, colors[:, 3])

lc = Line3DCollection(segments, colors=colors, linewidths=0.4)
ax.add_collection3d(lc)

# Scatter overlay for bright structure highlights
step = 3
sc = ax.scatter(th[::step], p[::step], sigma[::step],
                s=0.6, c=sigma[::step], cmap='hot',
                alpha=0.7, edgecolors='none',
                vmin=np.percentile(sigma, 5),
                vmax=np.percentile(sigma, 95))

# Mark the three Z₃ wells as vertical ghost lines
for k in range(3):
    well_th = k * 2 * np.pi / 3
    z_lo, z_hi = sigma.min() - 0.1, sigma.max() + 0.1
    ax.plot([well_th, well_th], [0, 0], [z_lo, z_hi],
            color='#2ecc71', linewidth=0.8, linestyle=':', alpha=0.3)

# Mark Koide angle
z_lo, z_hi = sigma.min() - 0.1, sigma.max() + 0.1
ax.plot([theta_K, theta_K], [0, 0], [z_lo, z_hi],
        color='#f1c40f', linewidth=1.2, linestyle='--', alpha=0.5)

# Axis styling
ax.set_xlabel('θ  (Koide angle)', color='#b0b0b0', fontsize=13, labelpad=10)
ax.set_ylabel('p  (momentum)', color='#b0b0b0', fontsize=13, labelpad=10)
ax.set_zlabel('σ  (breathing mode)', color='#b0b0b0', fontsize=13, labelpad=10)
ax.tick_params(colors='#808080', labelsize=9)

ax.set_xlim(0, 2 * np.pi)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], fontsize=9)

# Pane styling
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#1a1a3a')

# Viewing angle — show the three-lobed structure
ax.view_init(elev=22, azim=135)

# Colorbar
cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, aspect=25)
cb.set_label('σ (breathing mode amplitude)', color='#b0b0b0', fontsize=11)
cb.ax.tick_params(colors='#808080', labelsize=9)

# Title
ax.set_title(
    'Koide-Lorenz Strange Attractor\n'
    'Autonomous Chaos on the Koide Circle',
    color='white', fontsize=18, fontweight='bold', pad=25)

# Subtitle annotation
fig.text(0.5, 0.02,
         f'θ̇ = p,   ṗ = −V′(θ)(1−κσ) − Γ(θ)p,   σ̇ = −δσ + ρ − σp²'
         f'      |      κ={kap}, γ={gam}, δ={delt}, ρ={rh}'
         f'      |      λ₁ ≈ +0.159 (chaotic)      d_corr ≈ 1.5 (fractal)',
         ha='center', color='#707090', fontsize=10, family='monospace')

outpath = OUTPUT_DIR / 'koide_lorenz_attractor.png'
fig.savefig(outpath, dpi=250, facecolor=bg, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
