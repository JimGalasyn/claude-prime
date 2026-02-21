#!/usr/bin/env python3
"""
Animation: plane-polarized photon wave packet propagating in free space.

Shows the actual 3D structure of a photon: oscillating E and B vector
fields, perpendicular to each other and to propagation, modulated by
a Gaussian envelope. E and B are IN PHASE (not 90° offset — a common
textbook error).

For a photon in vacuum: no dispersion, so the wave pattern moves
rigidly with the envelope at c. Phase velocity = group velocity = c.

Plane polarization shown first (E oscillates in one fixed plane).
Circular polarization is the natural next step — E rotates, tracing
a helix, corresponding to definite photon spin ±hbar.

Outputs GIF via pillow.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Wave packet parameters ───────────────────────────────────────────

wavelength = 1.0          # arbitrary visual units
k = 2 * np.pi / wavelength
n_cycles = 10             # oscillation cycles within the envelope
sigma = n_cycles * wavelength / 4.0   # Gaussian envelope width
amp_E = 1.8               # E field visual amplitude
amp_B = 1.2               # B field visual amplitude (would be E/c; scaled for visibility)

# Spatial
x_half = 12.0             # half-width of visible window
speed = 0.12              # packet displacement per frame

# Frames
N_FRAMES = 280
x_start = -x_half - 3 * sigma   # start off-screen left

# Colors
BG      = '#060612'
CLR_E   = '#4aa3df'     # E field — sky blue
CLR_E2  = '#2980b9'     # E arrows — slightly darker
CLR_B   = '#52be80'     # B field — green
CLR_B2  = '#27ae60'     # B arrows
CLR_ENV = '#f1c40f'     # envelope
CLR_AX  = '#1a1a3a'     # axis line
CLR_TXT = '#a0a0a0'     # info text


# ── Helpers ──────────────────────────────────────────────────────────

def envelope(x, center):
    """Gaussian envelope centered at 'center'."""
    return np.exp(-(x - center)**2 / (2 * sigma**2))


def field(x, center):
    """
    Carrier × envelope.  No dispersion: carrier and envelope move together.
    """
    env = envelope(x, center)
    carrier = np.cos(k * (x - center))
    return carrier * env, env


# ── Figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 8), facecolor=BG)
ax = fig.add_subplot(111, projection='3d')

fig.suptitle(
    'Plane-Polarized Photon  \u2014  Free-Streaming Wave Packet\n'
    'E (blue, vertical) and B (green, horizontal) oscillate in phase,  '
    'perpendicular to each other and to propagation',
    color='white', fontsize=13, fontweight='bold', y=0.96)


# ── Per-frame ────────────────────────────────────────────────────────

def draw_frame(frame):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_axis_off()

    # Slow rotation for depth cue
    ax.view_init(elev=20, azim=-62 + frame * 0.06)

    # Packet center
    xc = x_start + speed * frame

    # ── Propagation axis ─────────────────────────────────────────
    ax.plot([-x_half, x_half], [0, 0], [0, 0],
            '-', color=CLR_AX, linewidth=0.8, alpha=0.6)

    # Propagation arrow
    ax.quiver(x_half - 1.5, 0, 0, 1.2, 0, 0,
              color=CLR_TXT, arrow_length_ratio=0.25,
              linewidth=1.5, alpha=0.5)
    ax.text(x_half - 0.2, 0.4, 0.1, 'c', color=CLR_TXT,
            fontsize=12, alpha=0.6, fontstyle='italic')

    # ── Smooth field curves ──────────────────────────────────────
    xs = np.linspace(-x_half, x_half, 800)
    wave, env = field(xs, xc)
    Ez = amp_E * wave        # E in z direction
    By = amp_B * wave        # B in y direction
    env_E = amp_E * env
    env_B = amp_B * env

    # E field curve (xz plane) — the "sine wave" tracing arrow tips
    ax.plot(xs, np.zeros_like(xs), Ez, '-', color=CLR_E,
            linewidth=2.2, alpha=0.85, zorder=5)

    # B field curve (xy plane)
    ax.plot(xs, By, np.zeros_like(xs), '-', color=CLR_B,
            linewidth=2.2, alpha=0.85, zorder=5)

    # ── Envelope curves (dashed) ─────────────────────────────────
    ax.plot(xs, np.zeros_like(xs), env_E, '--', color=CLR_ENV,
            linewidth=0.8, alpha=0.35)
    ax.plot(xs, np.zeros_like(xs), -env_E, '--', color=CLR_ENV,
            linewidth=0.8, alpha=0.35)

    # ── Field vectors (arrows) ───────────────────────────────────
    # Fewer arrows than curve points, for clarity
    n_arr = 40
    xa = np.linspace(-x_half, x_half, n_arr)
    wa, ea = field(xa, xc)
    Ea = amp_E * wa
    Ba = amp_B * wa

    # Only draw where envelope is visible
    mask = ea > 0.03
    if np.any(mask):
        xm = xa[mask]
        zm = np.zeros(mask.sum())

        # E arrows (vertical)
        ax.quiver(xm, zm, zm, zm, zm, Ea[mask],
                  color=CLR_E2, arrow_length_ratio=0.12,
                  linewidth=1.0, alpha=0.55)

        # B arrows (horizontal)
        ax.quiver(xm, zm, zm, zm, Ba[mask], zm,
                  color=CLR_B2, arrow_length_ratio=0.12,
                  linewidth=1.0, alpha=0.55)

    # ── "Cross-section" at packet center ─────────────────────────
    # A faint cross showing E vertical + B horizontal at one point,
    # reinforcing the perpendicularity.
    if -x_half < xc < x_half:
        w0, _ = field(np.array([xc]), xc)
        e0 = float(amp_E * w0[0])
        b0 = float(amp_B * w0[0])

        # Vertical E line through the axis
        ax.plot([xc, xc], [0, 0], [-abs(amp_E) * 0.3, abs(amp_E) * 0.3],
                '-', color=CLR_E, linewidth=0.5, alpha=0.2)
        # Horizontal B line through the axis
        ax.plot([xc, xc], [-abs(amp_B) * 0.3, abs(amp_B) * 0.3], [0, 0],
                '-', color=CLR_B, linewidth=0.5, alpha=0.2)

        # Labels near packet center
        label_x = xc + 0.4
        ax.text(label_x, 0, amp_E * 0.75, 'E', color=CLR_E,
                fontsize=14, fontweight='bold', zorder=10)
        ax.text(label_x, amp_B * 0.75, 0, 'B', color=CLR_B,
                fontsize=14, fontweight='bold', zorder=10)

    # ── Wavelength bracket ───────────────────────────────────────
    # Show one wavelength near the center of the packet
    if -x_half + 2 < xc < x_half - 2:
        bkt_z = -amp_E - 0.5
        ax.plot([xc - wavelength/2, xc + wavelength/2],
                [0, 0], [bkt_z, bkt_z],
                '-', color=CLR_ENV, linewidth=1.5, alpha=0.5)
        ax.plot([xc - wavelength/2, xc - wavelength/2],
                [0, 0], [bkt_z - 0.15, bkt_z + 0.15],
                '-', color=CLR_ENV, linewidth=1.5, alpha=0.5)
        ax.plot([xc + wavelength/2, xc + wavelength/2],
                [0, 0], [bkt_z - 0.15, bkt_z + 0.15],
                '-', color=CLR_ENV, linewidth=1.5, alpha=0.5)
        ax.text(xc, 0.1, bkt_z - 0.3, '\u03bb', color=CLR_ENV,
                fontsize=12, ha='center')

    # ── Info panel ───────────────────────────────────────────────
    ax.text2D(0.02, 0.12, 'E oscillates vertically (z)',
              transform=ax.transAxes, color=CLR_E, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.08, 'B oscillates horizontally (y)',
              transform=ax.transAxes, color=CLR_B, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.04, 'E and B in phase, same envelope',
              transform=ax.transAxes, color=CLR_ENV, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.00, 'no dispersion: pattern moves rigidly at c',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              fontfamily='monospace')

    # Right side
    ax.text2D(0.98, 0.08, f'{n_cycles} cycles in Gaussian envelope',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.04, 'S = E \u00d7 B \u2192 energy flows along x',
              transform=ax.transAxes, color='#e07050', fontsize=10,
              ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.00, 'plane polarization: E stays in one plane',
              transform=ax.transAxes, color=CLR_TXT, fontsize=10,
              ha='right', fontfamily='monospace')

    return []


# ── Render ───────────────────────────────────────────────────────────

print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                     blit=False, interval=50)

outpath = str(OUTPUT_DIR / 'anim_photon_wave_packet.gif')
anim.save(outpath, writer=PillowWriter(fps=20), dpi=100)
print(f"Saved: {outpath}")
plt.close()
