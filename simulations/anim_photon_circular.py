#!/usr/bin/env python3
"""
Animation: circularly polarized photon wave packet.

The E field vector rotates around the propagation axis, tracing a helix.
B rotates 90 deg behind, always perpendicular to E and to propagation.
The helix radius is modulated by a Gaussian envelope.

This is the more fundamental polarization state for individual photons:
each photon carries angular momentum +/-hbar, corresponding to
right/left circular polarization. Plane polarization is a superposition
of L + R circular.

Connection to the torus model: the photon's helical field structure
IS the winding that, when closed on itself (trapped on a torus),
becomes a particle. Pair production = helix closing into a knot.

Outputs GIF via pillow.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


# ── Wave packet parameters ───────────────────────────────────────────

wavelength = 1.0
k = 2 * np.pi / wavelength
n_cycles = 10
sigma = n_cycles * wavelength / 4.0
amp_E = 1.6
amp_B = 1.1

# Spatial
x_half = 12.0
speed = 0.12

# Frames
N_FRAMES = 280
x_start = -x_half - 3 * sigma

# Colors
BG      = '#060612'
CLR_E   = '#4aa3df'
CLR_E2  = '#2980b9'
CLR_B   = '#52be80'
CLR_B2  = '#27ae60'
CLR_ENV = '#f1c40f'
CLR_AX  = '#1a1a3a'
CLR_TXT = '#a0a0a0'
CLR_HEL = '#85c1e9'   # helix trail


# ── Helpers ──────────────────────────────────────────────────────────

def envelope(x, center):
    return np.exp(-(x - center)**2 / (2 * sigma**2))


# ── Figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 8), facecolor=BG)
ax = fig.add_subplot(111, projection='3d')

fig.suptitle(
    'Circularly Polarized Photon  \u2014  Right-Handed (spin +\u0127)\n'
    'E (blue) rotates around propagation axis, tracing a helix;  '
    'B (green) follows, always 90\u00b0 behind',
    color='white', fontsize=13, fontweight='bold', y=0.96)


# ── Per-frame ────────────────────────────────────────────────────────

def draw_frame(frame):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55 + frame * 0.08)

    xc = x_start + speed * frame

    # ── Propagation axis ─────────────────────────────────────────
    ax.plot([-x_half, x_half], [0, 0], [0, 0],
            '-', color=CLR_AX, linewidth=0.8, alpha=0.6)
    ax.quiver(x_half - 1.5, 0, 0, 1.2, 0, 0,
              color=CLR_TXT, arrow_length_ratio=0.25,
              linewidth=1.5, alpha=0.5)
    ax.text(x_half - 0.2, 0.4, 0.1, 'c', color=CLR_TXT,
            fontsize=12, alpha=0.6, fontstyle='italic')

    # ── Field computation ────────────────────────────────────────
    # Right circular polarization:
    #   E_y =  E0 cos(k(x-xc))  * env
    #   E_z =  E0 sin(k(x-xc))  * env
    #   B_y = -B0 sin(k(x-xc))  * env   (90 deg behind E)
    #   B_z =  B0 cos(k(x-xc))  * env
    xs = np.linspace(-x_half, x_half, 800)
    env = envelope(xs, xc)
    phase = k * (xs - xc)

    Ey = amp_E * np.cos(phase) * env
    Ez = amp_E * np.sin(phase) * env
    By = -amp_B * np.sin(phase) * env
    Bz = amp_B * np.cos(phase) * env

    # ── E helix (main visual) ────────────────────────────────────
    ax.plot(xs, Ey, Ez, '-', color=CLR_E, linewidth=2.5, alpha=0.85,
            zorder=5)

    # ── B helix ──────────────────────────────────────────────────
    ax.plot(xs, By, Bz, '-', color=CLR_B, linewidth=2.0, alpha=0.7,
            zorder=4)

    # ── Envelope (circular cross-section envelope) ───────────────
    # Show envelope as circles at a few x positions
    theta_circ = np.linspace(0, 2 * np.pi, 60)
    for x_ring in np.arange(xc - 2 * sigma, xc + 2 * sigma + 0.01,
                            sigma * 0.7):
        if -x_half < x_ring < x_half:
            r_env = amp_E * envelope(np.array([x_ring]), xc)[0]
            if r_env > 0.05:
                cy = r_env * np.cos(theta_circ)
                cz = r_env * np.sin(theta_circ)
                cx = np.full_like(theta_circ, x_ring)
                ax.plot(cx, cy, cz, '-', color=CLR_ENV,
                        linewidth=0.6, alpha=0.2)

    # ── Field arrows ─────────────────────────────────────────────
    n_arr = 32
    xa = np.linspace(-x_half, x_half, n_arr)
    env_a = envelope(xa, xc)
    phase_a = k * (xa - xc)

    Ey_a = amp_E * np.cos(phase_a) * env_a
    Ez_a = amp_E * np.sin(phase_a) * env_a
    By_a = -amp_B * np.sin(phase_a) * env_a
    Bz_a = amp_B * np.cos(phase_a) * env_a

    mask = env_a > 0.04
    if np.any(mask):
        xm = xa[mask]
        zm = np.zeros(mask.sum())

        # E arrows (from axis to helix)
        ax.quiver(xm, zm, zm, zm, Ey_a[mask], Ez_a[mask],
                  color=CLR_E2, arrow_length_ratio=0.10,
                  linewidth=0.9, alpha=0.5)

        # B arrows (from axis to helix)
        ax.quiver(xm, zm, zm, zm, By_a[mask], Bz_a[mask],
                  color=CLR_B2, arrow_length_ratio=0.10,
                  linewidth=0.9, alpha=0.4)

    # ── Instantaneous cross-section at packet center ─────────────
    if -x_half < xc < x_half:
        e0y = amp_E * 1.0   # cos(0) = 1 at center
        e0z = amp_E * 0.0   # sin(0) = 0
        b0y = -amp_B * 0.0  # -sin(0) = 0
        b0z = amp_B * 1.0   # cos(0) = 1

        # Rotation indicator: small arc showing E rotates
        arc_th = np.linspace(0, np.pi / 2, 30)
        arc_r = amp_E * 0.4
        arc_y = arc_r * np.cos(arc_th)
        arc_z = arc_r * np.sin(arc_th)
        arc_x = np.full_like(arc_th, xc)
        ax.plot(arc_x, arc_y, arc_z, '-', color=CLR_ENV,
                linewidth=1.5, alpha=0.5)
        # Arrowhead direction
        ax.quiver(xc, arc_y[-1], arc_z[-1], 0, -0.1, 0.1,
                  color=CLR_ENV, arrow_length_ratio=0.5,
                  linewidth=1.5, alpha=0.5)

        # Labels
        ax.text(xc + 0.3, e0y + 0.2, e0z, 'E', color=CLR_E,
                fontsize=14, fontweight='bold', zorder=10)
        ax.text(xc + 0.3, b0y, b0z + 0.2, 'B', color=CLR_B,
                fontsize=14, fontweight='bold', zorder=10)

    # ── Wavelength bracket ───────────────────────────────────────
    if -x_half + 2 < xc < x_half - 2:
        bkt = -amp_E - 0.8
        ax.plot([xc - wavelength/2, xc + wavelength/2],
                [0, 0], [bkt, bkt],
                '-', color=CLR_ENV, linewidth=1.5, alpha=0.5)
        ax.plot([xc - wavelength/2, xc - wavelength/2],
                [0, 0], [bkt - 0.15, bkt + 0.15],
                '-', color=CLR_ENV, linewidth=1.5, alpha=0.5)
        ax.plot([xc + wavelength/2, xc + wavelength/2],
                [0, 0], [bkt - 0.15, bkt + 0.15],
                '-', color=CLR_ENV, linewidth=1.5, alpha=0.5)
        ax.text(xc, 0.1, bkt - 0.35, '\u03bb', color=CLR_ENV,
                fontsize=12, ha='center')

    # ── Info panel ───────────────────────────────────────────────
    ax.text2D(0.02, 0.14,
              'E tip traces a RIGHT-HANDED HELIX',
              transform=ax.transAxes, color=CLR_E, fontsize=10,
              fontfamily='monospace', fontweight='bold')
    ax.text2D(0.02, 0.10,
              'B helix follows, 90\u00b0 rotated',
              transform=ax.transAxes, color=CLR_B, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.06,
              'angular momentum: L = +\u0127 (spin-1 boson)',
              transform=ax.transAxes, color=CLR_ENV, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.02,
              'plane polarization = R + L superposition',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              fontfamily='monospace')

    # Right side
    ax.text2D(0.98, 0.14,
              'helix pitch = \u03bb',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.10,
              'one full rotation per wavelength',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.06,
              'left-handed = mirror image (spin -\u0127)',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.02,
              'torus model: this helix IS the winding',
              transform=ax.transAxes, color='#e07050', fontsize=10,
              ha='right', fontfamily='monospace')

    return []


# ── Render ───────────────────────────────────────────────────────────

print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                     blit=False, interval=50)

outpath = '/home/jim/repos/claude-prime/simulations/anim_photon_circular.gif'
anim.save(outpath, writer=PillowWriter(fps=20), dpi=100)
print(f"Saved: {outpath}")
plt.close()
