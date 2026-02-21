#!/usr/bin/env python3
"""
Attractor band structure → particle identification.

Maps the density bands of the Koide-Lorenz strange attractor to the
physical mass spectrum. Each band corresponds to a region in θ-space
where the trajectory lingers, and the Koide formula m_i(θ) tells us
which particle dominates at each angle.

Usage:
    python plot_attractor_bands.py
"""

from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────
m_e_MeV = 0.51099895000
m_mu = 105.6583755
m_tau = 1776.86
S = np.sqrt(m_e_MeV) + np.sqrt(m_mu) + np.sqrt(m_tau)
S3 = S / 3
theta_K = (6 * np.pi + 2) / 9

particle_names = ['e', 'μ', 'τ']
particle_colors = ['#3498db', '#2ecc71', '#e74c3c']  # blue, green, red

def masses_at(th):
    """Mass triplet at Koide angle θ."""
    return np.array([(S3 * (1 + np.sqrt(2) * np.cos(th + 2 * np.pi * i / 3)))**2
                     for i in range(3)])

def dominant_particle(th):
    """Which particle has the largest mass at this θ?"""
    m = masses_at(th)
    return np.argmax(m)

def mass_label(th):
    """Formatted mass triplet string."""
    m = masses_at(th)
    idx = np.argsort(m)[::-1]  # heaviest first
    parts = []
    for i in idx:
        name = particle_names[i]
        if m[i] > 100:
            parts.append(f'{name}: {m[i]:.0f}')
        elif m[i] > 1:
            parts.append(f'{name}: {m[i]:.1f}')
        else:
            parts.append(f'{name}: {m[i]:.3f}')
    return '\n'.join(parts) + ' MeV'

# ── Dynamics ──────────────────────────────────────────────────────────
def dV(th):
    return (3 * np.sqrt(2) / 2) * np.sin(3 * th)

def koide_lorenz(t, y, kappa, gamma_kl, delta, rho):
    th, p, sigma = y
    g = gamma_kl * (1 + 0.5 * np.cos(3 * th))
    return [p,
            -dV(th) * (1 - kappa * sigma) - g * p,
            -delta * sigma + rho - sigma * p**2]

# ── Integrate ─────────────────────────────────────────────────────────
print("Integrating Koide-Lorenz system...")
kap, gam, delt, rh = 1.5, 0.3, 1.0, 1.5
y0 = [theta_K + 0.1, 0.3, rh / delt + 0.1]

sol = solve_ivp(
    lambda t, y: koide_lorenz(t, y, kap, gam, delt, rh),
    (0, 800), y0, t_eval=np.linspace(80, 800, 120000),
    rtol=1e-11, atol=1e-13, max_step=0.012)

th_att = sol.y[0] % (2 * np.pi)
p_att = sol.y[1]
sigma_att = sol.y[2]
print(f"  {len(th_att)} points on attractor")

# ── Band analysis ─────────────────────────────────────────────────────
print("Analyzing band structure...")
N_bins = 300
hist, edges = np.histogram(th_att, bins=N_bins, range=(0, 2 * np.pi))
centers = (edges[:-1] + edges[1:]) / 2
hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)

# Find peaks (bands)
mean_h = np.mean(hist_smooth)
peaks, props = find_peaks(hist_smooth, height=mean_h * 1.3,
                          distance=N_bins // 15, prominence=mean_h * 0.3)

print(f"  Found {len(peaks)} bands:")
for pk in peaks:
    th_pk = centers[pk]
    m = masses_at(th_pk)
    dom = dominant_particle(th_pk)
    print(f"    θ = {np.degrees(th_pk):6.1f}° | dominant: {particle_names[dom]} | "
          f"masses: {m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f} MeV")

# ── Compute mass curves over θ ────────────────────────────────────────
th_range = np.linspace(0, 2 * np.pi, 1000)
m_curves = np.array([masses_at(th) for th in th_range])  # shape (1000, 3)

# ── Figure ────────────────────────────────────────────────────────────
print("Rendering...")
bg = '#060612'
text_color = '#d0d0d0'
grid_color = '#1a1a3a'

fig = plt.figure(figsize=(18, 16), facecolor=bg)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.28,
                       height_ratios=[1.2, 1, 1.2],
                       left=0.08, right=0.95, top=0.93, bottom=0.06)

# ── Panel A: Mass spectrum as function of θ ───────────────────────────
ax_mass = fig.add_subplot(gs[0])
ax_mass.set_facecolor(bg)

for i in range(3):
    ax_mass.semilogy(th_range, m_curves[:, i], color=particle_colors[i],
                     linewidth=2.5, label=f'{particle_names[i]} mass',
                     alpha=0.9)

# Shade the bands
for pk in peaks:
    th_pk = centers[pk]
    # Find band width at half prominence
    half_h = (hist_smooth[pk] + mean_h) / 2
    left = pk
    while left > 0 and hist_smooth[left] > half_h:
        left -= 1
    right = pk
    while right < N_bins - 1 and hist_smooth[right] > half_h:
        right += 1
    th_lo = centers[max(left, 0)]
    th_hi = centers[min(right, N_bins - 1)]

    dom = dominant_particle(th_pk)
    ax_mass.axvspan(th_lo, th_hi, alpha=0.15,
                     color=particle_colors[dom])
    ax_mass.axvline(th_pk, color=particle_colors[dom],
                     linewidth=1.5, linestyle=':', alpha=0.7)

# Physical Koide angle
ax_mass.axvline(theta_K, color='#f1c40f', linewidth=2, linestyle='--',
                alpha=0.9, label=f'θ_K = {np.degrees(theta_K):.1f}°')

# Physical masses as horizontal lines
for mval, name, col in [(m_e_MeV, 'e', particle_colors[0]),
                         (m_mu, 'μ', particle_colors[1]),
                         (m_tau, 'τ', particle_colors[2])]:
    ax_mass.axhline(mval, color=col, linewidth=1, linestyle='--', alpha=0.4)
    ax_mass.text(0.05, mval * 1.3, f'{name} = {mval:.2f} MeV',
                 color=col, fontsize=9, alpha=0.7)

ax_mass.set_ylabel('Mass (MeV)', color=text_color, fontsize=12)
ax_mass.set_title('A.  Koide Mass Spectrum: m_i(θ) = (S/3)²(1 + √2 cos(θ + 2πi/3))²',
                   color='white', fontsize=14, fontweight='bold', pad=12)
ax_mass.legend(fontsize=10, facecolor=bg, edgecolor=grid_color,
               labelcolor=text_color, loc='upper right')
ax_mass.set_xlim(0, 2 * np.pi)
ax_mass.set_ylim(1e-2, 5000)
ax_mass.tick_params(colors=text_color)
ax_mass.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_mass.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=10)
for spine in ax_mass.spines.values():
    spine.set_color(grid_color)

# ── Panel B: Attractor density histogram with particle ID ─────────────
ax_band = fig.add_subplot(gs[1])
ax_band.set_facecolor(bg)

# Color each bar by dominant particle
bar_colors = [particle_colors[dominant_particle(c)] for c in centers]
ax_band.bar(centers, hist_smooth, width=2 * np.pi / N_bins * 0.9,
            color=bar_colors, alpha=0.7, edgecolor='none')

ax_band.axhline(mean_h, color='#707070', linewidth=1, linestyle=':',
                alpha=0.5, label='Mean density')

# Annotate peaks with particle identification
for pk in peaks:
    th_pk = centers[pk]
    dom = dominant_particle(th_pk)
    m = masses_at(th_pk)
    m_sort_idx = np.argsort(m)[::-1]

    label_lines = [f'θ = {np.degrees(th_pk):.1f}°']
    for idx in m_sort_idx:
        name = particle_names[idx]
        mval = m[idx]
        if mval > 100:
            label_lines.append(f'{name}: {mval:.0f} MeV')
        elif mval > 1:
            label_lines.append(f'{name}: {mval:.1f} MeV')
        else:
            label_lines.append(f'{name}: {mval:.3f} MeV')

    ax_band.annotate(
        '\n'.join(label_lines),
        xy=(th_pk, hist_smooth[pk]),
        xytext=(0, 25), textcoords='offset points',
        fontsize=8.5, color='white', ha='center',
        arrowprops=dict(arrowstyle='->', color=particle_colors[dom], lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=particle_colors[dom],
                  edgecolor='white', alpha=0.35))

# Koide angle
ax_band.axvline(theta_K, color='#f1c40f', linewidth=2, linestyle='--',
                alpha=0.9, label=f'θ_K = {np.degrees(theta_K):.1f}°')

ax_band.set_ylabel('Density (points/bin)', color=text_color, fontsize=12)
ax_band.set_title('B.  Attractor Band Structure → Particle Identification',
                   color='white', fontsize=14, fontweight='bold', pad=12)
ax_band.legend(fontsize=10, facecolor=bg, edgecolor=grid_color,
               labelcolor=text_color, loc='upper right')
ax_band.set_xlim(0, 2 * np.pi)
ax_band.tick_params(colors=text_color)
ax_band.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_band.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=10)
for spine in ax_band.spines.values():
    spine.set_color(grid_color)

# ── Panel C: Attractor colored by dominant particle ───────────────────
ax_att = fig.add_subplot(gs[2])
ax_att.set_facecolor(bg)

# Color each point by which particle dominates at that θ
dom_colors = np.array([particle_colors[dominant_particle(th)] for th in th_att])

# Plot in layers: first all, then bright overlay for dense regions
ax_att.scatter(th_att, p_att, s=0.3, c=dom_colors, alpha=0.4, edgecolors='none')

# Density-weighted overlay
from scipy.ndimage import gaussian_filter
_hbin = 200
_h2d, _xed, _yed = np.histogram2d(th_att, p_att, bins=_hbin)
_h2d = gaussian_filter(_h2d, sigma=1.2)
_xi = np.clip(np.searchsorted(_xed, th_att) - 1, 0, _hbin - 1)
_yi = np.clip(np.searchsorted(_yed, p_att) - 1, 0, _hbin - 1)
_dens = _h2d[_xi, _yi]
_dens_norm = _dens / (_dens.max() + 1e-12)

# High-density points get brighter
bright_mask = _dens_norm > 0.3
if np.any(bright_mask):
    ax_att.scatter(th_att[bright_mask], p_att[bright_mask],
                   s=1.2, c=dom_colors[bright_mask], alpha=0.8,
                   edgecolors='none')

# Mark wells and Koide angle
for k in range(3):
    ax_att.axvline(k * 2 * np.pi / 3, color='#ffffff', linewidth=0.8,
                   linestyle=':', alpha=0.3)
ax_att.axvline(theta_K, color='#f1c40f', linewidth=2, linestyle='--',
               alpha=0.8)

# Legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=particle_colors[0],
           markersize=10, label=f'e-dominated (m_e = {m_e_MeV:.3f} MeV)', linewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=particle_colors[1],
           markersize=10, label=f'μ-dominated (m_μ = {m_mu:.1f} MeV)', linewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=particle_colors[2],
           markersize=10, label=f'τ-dominated (m_τ = {m_tau:.0f} MeV)', linewidth=0),
    Line2D([0], [0], color='#f1c40f', linestyle='--', linewidth=2,
           label=f'Physical θ_K = {np.degrees(theta_K):.1f}°'),
]
ax_att.legend(handles=legend_els, fontsize=9.5, facecolor=bg, edgecolor=grid_color,
              labelcolor=text_color, loc='upper right')

ax_att.set_xlabel('θ (Koide angle)', color=text_color, fontsize=12)
ax_att.set_ylabel('p (momentum)', color=text_color, fontsize=12)
ax_att.set_title('C.  Koide-Lorenz Attractor Colored by Dominant Particle',
                  color='white', fontsize=14, fontweight='bold', pad=12)
ax_att.set_xlim(0, 2 * np.pi)
ax_att.tick_params(colors=text_color)
ax_att.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_att.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=10)
for spine in ax_att.spines.values():
    spine.set_color(grid_color)

# ── Supertitle ────────────────────────────────────────────────────────
fig.suptitle('Strange Attractor Band Structure → Particle Mass Identification\n'
             'Koide-Lorenz Autonomous System on the Null Worldtube',
             color='white', fontsize=17, fontweight='bold', y=0.98)

outpath = OUTPUT_DIR / 'attractor_bands.png'
fig.savefig(outpath, dpi=200, facecolor=bg, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.close()
