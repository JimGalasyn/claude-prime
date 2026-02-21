#!/usr/bin/env python3
"""
Animation: electron-positron annihilation with Lorentz contraction.

Two (2,1) torus knots — one electron, one positron — approach each other
with increasing velocity. As they accelerate, Lorentz contraction visibly
flattens both tori along the direction of motion. On collision, the
topology unwinds: (2,+1) + (2,-1) -> (0,0), and two photons emerge.

Each torus has a circulating photon (glowing dot) tracing the knot path,
consistent with the null worldtube model where particles ARE photons
trapped on torus knots.

Outputs GIF via pillow.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Torus geometry with Lorentz contraction ──────────────────────────

def torus_knot(p, q, R, r, gamma=1.0, offset_x=0.0, chirality=1, N=600):
    """
    (p,q) torus knot, Lorentz-contracted along x-axis.

    gamma: Lorentz factor (1=rest, >1=contracted along x)
    chirality: +1 electron, -1 positron (reverses toroidal winding)
    """
    t = np.linspace(0, 2 * np.pi, N)
    x = (R + r * np.cos(q * t)) * np.cos(p * t * chirality)
    y = (R + r * np.cos(q * t)) * np.sin(p * t * chirality)
    z = r * np.sin(q * t)
    # Lorentz contraction along x
    x = x / gamma + offset_x
    return x, y, z, t


def torus_surface(R, r, gamma=1.0, offset_x=0.0, n_u=32, n_v=16):
    """Torus surface mesh, Lorentz-contracted along x."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    X = X / gamma + offset_x
    return X, Y, Z


def photon_on_knot(p, q, R, r, gamma, offset_x, chirality, phase):
    """Position of circulating photon at given phase on the knot."""
    t = phase % (2 * np.pi)
    x = (R + r * np.cos(q * t)) * np.cos(p * t * chirality)
    y = (R + r * np.cos(q * t)) * np.sin(p * t * chirality)
    z = r * np.sin(q * t)
    x = x / gamma + offset_x
    return x, y, z


def wave_packet(center_x, direction, phase, amplitude=0.5, width=1.2, freq=12):
    """Helical photon wave packet moving along x."""
    s = np.linspace(-width, width, 120)
    envelope = np.exp(-s**2 * 2.5)
    x = center_x + s * direction
    y = envelope * amplitude * np.sin(freq * s + phase)
    z = envelope * amplitude * np.cos(freq * s + phase)
    return x, y, z


# ── Parameters ───────────────────────────────────────────────────────

R, r = 1.5, 0.45     # torus major/minor radii
p, q = 2, 1          # (2,1) torus knot = fermion (spin-1/2)
x_start = 8.0        # starting separation
x_contact = 0.3      # gap at "contact" before annihilation

# Phase durations (frames)
N_APPROACH = 140
N_COLLIDE  = 35
N_PHOTON   = 70
N_HOLD     = 25
N_FRAMES   = N_APPROACH + N_COLLIDE + N_PHOTON + N_HOLD

# Colors
CLR_BG       = '#060612'
CLR_ELECTRON = '#3498db'
CLR_POSITRON = '#e74c3c'
CLR_E_GLOW   = '#85c1e9'
CLR_P_GLOW   = '#f1948a'
CLR_PHOTON   = '#f1c40f'
CLR_FLASH    = '#ffffaa'
CLR_INFO     = '#a0a0a0'

# ── Figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 7), facecolor=CLR_BG)
ax = fig.add_subplot(111, projection='3d')

fig.suptitle(
    'Electron\u2013Positron Annihilation  \u2014  Null Worldtube Model\n'
    'Lorentz contraction flattens torus knots along direction of motion',
    color='white', fontsize=13, fontweight='bold', y=0.96)


# ── Per-frame draw ───────────────────────────────────────────────────

def draw_frame(frame):
    ax.cla()
    ax.set_facecolor(CLR_BG)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-3.5, 3.5)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-8 + frame * 0.25)

    # ── PHASE 1: approach ────────────────────────────────────────────
    if frame < N_APPROACH:
        t = frame / N_APPROACH                       # 0 -> 1
        pos = x_start - (x_start - x_contact) * t    # linear approach
        v = 0.97 * np.sin(t * np.pi / 2) ** 2        # velocity ramps up
        gamma = 1.0 / np.sqrt(max(1e-8, 1 - v**2))

        # Photon circulation phase (spins faster at higher gamma)
        circ_phase = frame * 0.35

        for sign, chir, clr, glow, label in [
            (+1, +1, CLR_ELECTRON, CLR_E_GLOW, 'e\u207b'),
            (-1, -1, CLR_POSITRON, CLR_P_GLOW, 'e\u207a'),
        ]:
            ox = sign * pos
            # Surface
            SX, SY, SZ = torus_surface(R, r, gamma, ox)
            ax.plot_surface(SX, SY, SZ, alpha=0.15, color=clr,
                            edgecolor=clr, linewidth=0.2, antialiased=True)
            # Knot path
            kx, ky, kz, _ = torus_knot(p, q, R, r, gamma, ox, chir)
            ax.plot(kx, ky, kz, color=clr, linewidth=1.2, alpha=0.7)
            # Circulating photon
            px, py, pz = photon_on_knot(p, q, R, r, gamma, ox, chir, circ_phase)
            ax.plot([px], [py], [pz], 'o', color='white', markersize=8,
                    markeredgecolor=glow, markeredgewidth=2, zorder=10)
            # Label
            ax.text(ox, 0, 2.5, label, color=clr, fontsize=14,
                    ha='center', fontweight='bold', zorder=11)

        # Info bar
        rest_len = 2 * (R + r)
        contracted_len = rest_len / gamma
        ax.text2D(0.5, 0.06, f'v/c = {v:.3f}    \u03b3 = {gamma:.2f}',
                  transform=ax.transAxes, color=CLR_PHOTON, fontsize=12,
                  ha='center', fontfamily='monospace')
        ax.text2D(0.5, 0.02,
                  f'rest diameter {rest_len:.1f}  \u2192  '
                  f'contracted {contracted_len:.2f}  '
                  f'({1/gamma:.0%} of rest)',
                  transform=ax.transAxes, color=CLR_INFO, fontsize=10,
                  ha='center', fontfamily='monospace')

        # Topology labels (appear in second half)
        if t > 0.3:
            alpha_t = min(1.0, (t - 0.3) / 0.3)
            ax.text2D(0.18, 0.94, '(2, +1)', transform=ax.transAxes,
                      color=CLR_ELECTRON, fontsize=11, ha='center',
                      alpha=alpha_t, fontfamily='monospace')
            ax.text2D(0.82, 0.94, '(2, \u22121)', transform=ax.transAxes,
                      color=CLR_POSITRON, fontsize=11, ha='center',
                      alpha=alpha_t, fontfamily='monospace')

    # ── PHASE 2: collision / annihilation ────────────────────────────
    elif frame < N_APPROACH + N_COLLIDE:
        cf = frame - N_APPROACH
        t = cf / N_COLLIDE  # 0 -> 1

        # Tori merge at center, expand, fade
        fade = max(0, 1.0 - t ** 0.6)
        expand = 1.0 + t * 2.5
        r_shrink = r * max(0.05, 1.0 - t * 0.9)
        gamma_fade = max(1.0, 3.0 * (1 - t))

        if fade > 0.02:
            for chir, clr in [(+1, CLR_ELECTRON), (-1, CLR_POSITRON)]:
                kx, ky, kz, _ = torus_knot(p, q, R * expand, r_shrink,
                                            gamma_fade, 0, chir)
                ax.plot(kx, ky, kz, color=clr, linewidth=1.5 * fade,
                        alpha=fade * 0.8)

        # Flash sphere
        flash_r = 0.5 + t * 3.0
        flash_alpha = 0.4 * (1 - t ** 2)
        u_f = np.linspace(0, 2 * np.pi, 20)
        v_f = np.linspace(0, np.pi, 10)
        FX = flash_r * np.outer(np.cos(u_f), np.sin(v_f))
        FY = flash_r * np.outer(np.sin(u_f), np.sin(v_f))
        FZ = flash_r * np.outer(np.ones_like(u_f), np.cos(v_f))
        ax.plot_surface(FX, FY, FZ, alpha=max(0, flash_alpha),
                        color=CLR_FLASH, edgecolor='none')

        # Nascent photons emerge in second half
        if t > 0.4:
            pt = (t - 0.4) / 0.6
            ph_pos = pt * 3.0
            phase = frame * 0.5
            for sign in [+1, -1]:
                wx, wy, wz = wave_packet(sign * ph_pos, sign, phase,
                                          amplitude=0.3 * pt)
                ax.plot(wx, wy, wz, color=CLR_PHOTON,
                        linewidth=2 * pt, alpha=pt * 0.9)

        ax.text2D(0.5, 0.06,
                  'Annihilation:  (2,+1) + (2,\u22121) \u2192 (0,0)',
                  transform=ax.transAxes, color=CLR_PHOTON, fontsize=12,
                  ha='center', fontfamily='monospace', fontweight='bold')
        ax.text2D(0.5, 0.02, 'topology unwinds \u2014 charge cancels',
                  transform=ax.transAxes, color=CLR_INFO, fontsize=10,
                  ha='center', fontfamily='monospace')

    # ── PHASE 3: photon emission ─────────────────────────────────────
    elif frame < N_APPROACH + N_COLLIDE + N_PHOTON:
        pf = frame - N_APPROACH - N_COLLIDE
        t = pf / N_PHOTON  # 0 -> 1

        ph_pos = 1.5 + t * 7.5
        phase = frame * 0.5

        # Fading flash
        if t < 0.4:
            flash_alpha = 0.2 * (1 - t / 0.4)
            flash_r = 3 * (1 - t / 0.4) + 0.5
            u_f = np.linspace(0, 2 * np.pi, 16)
            v_f = np.linspace(0, np.pi, 8)
            FX = flash_r * np.outer(np.cos(u_f), np.sin(v_f))
            FY = flash_r * np.outer(np.sin(u_f), np.sin(v_f))
            FZ = flash_r * np.outer(np.ones_like(u_f), np.cos(v_f))
            ax.plot_surface(FX, FY, FZ, alpha=flash_alpha,
                            color=CLR_FLASH, edgecolor='none')

        # Two photon wave packets
        for sign in [+1, -1]:
            wx, wy, wz = wave_packet(sign * ph_pos, sign, phase)
            ax.plot(wx, wy, wz, color=CLR_PHOTON, linewidth=2.5, alpha=0.9)
            ax.text(sign * ph_pos, 0, 1.8, '\u03b3', color=CLR_PHOTON,
                    fontsize=15, ha='center', fontweight='bold')

        ax.text2D(0.5, 0.06,
                  'e\u207b + e\u207a \u2192 2\u03b3    '
                  'E = 2 \u00d7 0.511 MeV = 1.022 MeV',
                  transform=ax.transAxes, color=CLR_PHOTON, fontsize=12,
                  ha='center', fontfamily='monospace')
        ax.text2D(0.5, 0.02,
                  'the photon doesn\u2019t create the pair \u2014 '
                  'it BECOMES them  (and vice versa)',
                  transform=ax.transAxes, color=CLR_INFO, fontsize=10,
                  ha='center', fontfamily='monospace')

    # ── PHASE 4: hold ────────────────────────────────────────────────
    else:
        phase = frame * 0.5
        for sign in [+1, -1]:
            wx, wy, wz = wave_packet(sign * 8.5, sign, phase)
            ax.plot(wx, wy, wz, color=CLR_PHOTON, linewidth=2.5, alpha=0.9)
            ax.text(sign * 8.5, 0, 1.8, '\u03b3', color=CLR_PHOTON,
                    fontsize=15, ha='center', fontweight='bold')

        ax.text2D(0.5, 0.06,
                  'e\u207b + e\u207a \u2192 2\u03b3    '
                  'E = 2 \u00d7 0.511 MeV = 1.022 MeV',
                  transform=ax.transAxes, color=CLR_PHOTON, fontsize=12,
                  ha='center', fontfamily='monospace')
        ax.text2D(0.5, 0.02,
                  'same substrate, different topology \u2014 '
                  'winding \u2192 unwinding',
                  transform=ax.transAxes, color=CLR_INFO, fontsize=10,
                  ha='center', fontfamily='monospace')

    return []


# ── Render ───────────────────────────────────────────────────────────

print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                     blit=False, interval=50)

outpath = OUTPUT_DIR / 'anim_annihilation.gif'
anim.save(outpath, writer=PillowWriter(fps=20), dpi=100)
print(f"Saved: {outpath}")
plt.close()
