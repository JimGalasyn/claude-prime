#!/usr/bin/env python3
"""
Animation: particle decay cascade through Koide phase space.

Shows a trajectory spiraling through the cos(3θ) potential landscape,
crossing saddle points and settling into a well — the dynamical process
by which a particle 'chooses' its mass.

Outputs GIF via pillow.
"""

from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Physics ───────────────────────────────────────────────────────────
m_e_MeV = 0.51099895000
m_mu = 105.6583755
m_tau = 1776.86
S = np.sqrt(m_e_MeV) + np.sqrt(m_mu) + np.sqrt(m_tau)
S3 = S / 3
theta_K = (6 * np.pi + 2) / 9

m_ref = np.max(np.array([(S3 * (1 + np.sqrt(2) * np.cos(theta_K + 2*np.pi*i/3)))**2
                          for i in range(3)]))

def masses_at(th):
    return np.array([(S3 * (1 + np.sqrt(2) * np.cos(th + 2*np.pi*i/3)))**2
                     for i in range(3)])

def V(th):
    return 0.5 - (np.sqrt(2)/2) * np.cos(3*th)

def dV(th):
    return (3*np.sqrt(2)/2) * np.sin(3*th)

def Gamma(th, g0):
    mmax = np.max(masses_at(th))
    return g0 * (1 + (mmax/m_ref)**2) / 2

def rhs(t, y, g0):
    th, om = y
    return [om, -dV(th) - Gamma(th, g0)*om]

# ── Integrate trajectory ──────────────────────────────────────────────
g0 = 0.15  # moderate damping — slow enough to see the dynamics
y0 = [theta_K + 0.3, 4.5]  # high energy start near Koide angle
t_end = 45
N_pts = 3000
t_eval = np.linspace(0, t_end, N_pts)

print("Integrating decay trajectory...")
sol = solve_ivp(lambda t, y: rhs(t, y, g0), (0, t_end), y0,
                t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=0.02)
th = sol.y[0]
om = sol.y[1]

# Wrap θ for display
th_mod = th % (2*np.pi)

# ── Set up figure ─────────────────────────────────────────────────────
print("Setting up animation...")
bg = '#060612'
fig = plt.figure(figsize=(12, 6), facecolor=bg)

# Two panels: potential landscape (left) and phase portrait (right)
ax_pot = fig.add_axes([0.05, 0.12, 0.42, 0.78])
ax_phase = fig.add_axes([0.55, 0.12, 0.42, 0.78])

for ax in [ax_pot, ax_phase]:
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color('#1a1a3a')
    ax.tick_params(colors='#a0a0a0')

# ── Potential panel ───────────────────────────────────────────────────
th_range = np.linspace(0, 2*np.pi, 500)
V_vals = V(th_range)

ax_pot.fill_between(th_range, V_vals, -0.15, alpha=0.25, color='#2980b9')
ax_pot.plot(th_range, V_vals, color='#3498db', linewidth=2)

# Wells and saddles
for k in range(3):
    ax_pot.plot(k*2*np.pi/3, V(k*2*np.pi/3), 'o', color='#2ecc71',
                markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    saddle = (2*k+1)*np.pi/3
    ax_pot.plot(saddle, V(saddle), 'X', color='#e74c3c', markersize=10,
                zorder=5, markeredgecolor='white', markeredgewidth=1.5)

ax_pot.axvline(theta_K, color='#f1c40f', linewidth=1, linestyle='--', alpha=0.5)
ax_pot.set_xlim(0, 2*np.pi)
ax_pot.set_ylim(-0.15, 1.4)
ax_pot.set_xlabel('θ (Koide angle)', color='#a0a0a0', fontsize=11)
ax_pot.set_ylabel('V(θ) / Energy', color='#a0a0a0', fontsize=11)
ax_pot.set_title('Potential Landscape', color='white', fontsize=13,
                 fontweight='bold')
ax_pot.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_pot.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=8)

# Ball on potential
ball_pot, = ax_pot.plot([], [], 'o', color='#ff6633', markersize=14,
                         zorder=10, markeredgecolor='white', markeredgewidth=2)
# Energy line
energy_line, = ax_pot.plot([], [], '-', color='#ff6633', linewidth=1,
                            alpha=0.4)
# Trail on potential
trail_pot, = ax_pot.plot([], [], '-', color='#ff6633', linewidth=1.5,
                          alpha=0.3)

# ── Phase portrait panel ─────────────────────────────────────────────
# Background: separatrix
V_saddle = V(np.pi/3)
for th0 in np.linspace(0, 2*np.pi, 300):
    E_sep = V_saddle - V(th0)
    if E_sep >= 0:
        om_sep = np.sqrt(2*E_sep)
        ax_phase.plot(th0, om_sep, '.', color='#7f8c8d', markersize=0.3, alpha=0.3)
        ax_phase.plot(th0, -om_sep, '.', color='#7f8c8d', markersize=0.3, alpha=0.3)

# Fixed points
for k in range(3):
    ax_phase.plot(k*2*np.pi/3, 0, 'o', color='#2ecc71', markersize=8,
                  zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    saddle = (2*k+1)*np.pi/3
    ax_phase.plot(saddle, 0, 'X', color='#e74c3c', markersize=8,
                  zorder=5, markeredgecolor='white', markeredgewidth=1.5)

ax_phase.axvline(theta_K, color='#f1c40f', linewidth=1, linestyle='--', alpha=0.5)
ax_phase.set_xlim(0, 2*np.pi)
ax_phase.set_ylim(-6, 6)
ax_phase.set_xlabel('θ (Koide angle)', color='#a0a0a0', fontsize=11)
ax_phase.set_ylabel('θ̇ (angular velocity)', color='#a0a0a0', fontsize=11)
ax_phase.set_title('Phase Portrait', color='white', fontsize=13,
                    fontweight='bold')
ax_phase.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
ax_phase.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'], fontsize=8)

# Trajectory trail (colored by time)
trail_phase = ax_phase.scatter([], [], s=2, c=[], cmap='hot',
                                vmin=0, vmax=N_pts, edgecolors='none')
# Current point
ball_phase, = ax_phase.plot([], [], 'o', color='#ff6633', markersize=12,
                             zorder=10, markeredgecolor='white', markeredgewidth=2)

# Mass label
mass_text = ax_phase.text(0.02, 0.95, '', transform=ax_phase.transAxes,
                           color='white', fontsize=10, va='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='#1a1a3a', alpha=0.9))

# Title
fig.suptitle('Particle Decay: Trajectory Through Koide Phase Space',
             color='white', fontsize=15, fontweight='bold', y=0.98)

# Time label
time_text = fig.text(0.5, 0.02, '', ha='center', color='#a0a0a0', fontsize=11)

# ── Animation ─────────────────────────────────────────────────────────
# Step size for animation frames
step = 5  # every 5th point
n_frames = N_pts // step

# Trail history length
trail_len = 200  # points in trail

def init():
    ball_pot.set_data([], [])
    energy_line.set_data([], [])
    trail_pot.set_data([], [])
    ball_phase.set_data([], [])
    trail_phase.set_offsets(np.empty((0, 2)))
    mass_text.set_text('')
    time_text.set_text('')
    return ball_pot, energy_line, trail_pot, ball_phase, trail_phase, mass_text, time_text

def update(frame):
    i = frame * step
    if i >= N_pts:
        i = N_pts - 1

    th_i = th_mod[i]
    om_i = om[i]
    E_i = 0.5*om_i**2 + V(th[i])

    # Ball on potential
    ball_pot.set_data([th_i], [V(th_i)])

    # Energy level line
    energy_line.set_data([0, 2*np.pi], [E_i, E_i])

    # Trail on potential (recent history)
    i_start = max(0, i - trail_len * step)
    trail_pot.set_data(th_mod[i_start:i+1], V(th[i_start:i+1]))

    # Phase portrait
    ball_phase.set_data([th_i], [om_i])

    # Trail in phase space
    i_trail_start = max(0, i - trail_len * step)
    trail_idx = np.arange(i_trail_start, i+1, max(1, step//2))
    if len(trail_idx) > 0:
        offsets = np.column_stack([th_mod[trail_idx], om[trail_idx]])
        trail_phase.set_offsets(offsets)
        trail_phase.set_array(trail_idx.astype(float))

    # Mass readout
    m = masses_at(th[i])
    m_sort = sorted(enumerate(m), key=lambda x: -x[1])
    names = ['e', 'μ', 'τ']
    lines = [f'E = {E_i:.2f}  θ = {np.degrees(th_i):.0f}°']
    for idx_m, val in m_sort:
        if val > 100:
            lines.append(f'  {names[idx_m]}: {val:.0f} MeV')
        elif val > 1:
            lines.append(f'  {names[idx_m]}: {val:.1f} MeV')
        else:
            lines.append(f'  {names[idx_m]}: {val:.3f} MeV')
    mass_text.set_text('\n'.join(lines))

    # Time
    time_text.set_text(f't = {t_eval[i]:.1f} / {t_end}')

    return ball_pot, energy_line, trail_pot, ball_phase, trail_phase, mass_text, time_text

print(f"Rendering {n_frames} frames...")
anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                     blit=False, interval=33)  # ~30 fps

outpath = OUTPUT_DIR / 'anim_phase_decay.gif'
anim.save(outpath, writer=PillowWriter(fps=30), dpi=120)
print(f"Saved: {outpath}")
plt.close()
