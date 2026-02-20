#!/usr/bin/env python3
"""
Animation: electron torus precessing in an external magnetic field.

The (2,1) torus knot — with its circulating photon — acts as a gyroscope
with angular momentum L = hbar/2. An external B field applies torque via
the magnetic moment, causing the spin axis to precess (Larmor precession)
rather than flip.

Shows:
- Torus surface tilted at angle theta to B field
- Spin axis tracing a cone (precession)
- Photon dot circulating on the knot path
- B field direction and precession cone

Outputs GIF via pillow.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


# ── Rotation matrices ────────────────────────────────────────────────

def rot_x(theta):
    """Rotation matrix around x-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_z(phi):
    """Rotation matrix around z-axis."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def apply_rotation(x, y, z, R):
    """Apply 3x3 rotation matrix to coordinate arrays."""
    shape = x.shape
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()])  # (3, N)
    rotated = R @ pts
    return (rotated[0].reshape(shape),
            rotated[1].reshape(shape),
            rotated[2].reshape(shape))


# ── Torus geometry ───────────────────────────────────────────────────

def torus_surface(R, r, n_u=36, n_v=18):
    """Torus surface mesh in standard orientation (axis along z)."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    return X, Y, Z


def torus_knot(p, q, R, r, N=800):
    """(p,q) torus knot in standard orientation."""
    t = np.linspace(0, 2 * np.pi, N)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return x, y, z, t


def point_on_knot(p, q, R, r, phase):
    """Single point on torus knot at given phase."""
    t = phase % (2 * np.pi)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return np.array([x, y, z])


# ── Parameters ───────────────────────────────────────────────────────

R, r = 1.8, 0.55       # torus radii
p, q = 2, 1            # (2,1) = fermion

tilt = np.radians(35)  # angle between spin axis and B field
precession_rate = 0.04 # radians per frame (Larmor precession speed)
circ_rate = 0.4        # photon circulation speed (much faster)

N_FRAMES = 300         # ~15 seconds at 20 fps

# Colors
CLR_BG    = '#060612'
CLR_TORUS = '#3498db'
CLR_GLOW  = '#85c1e9'
CLR_AXIS  = '#f1c40f'
CLR_BFLD  = '#2ecc71'
CLR_CONE  = '#f39c12'
CLR_INFO  = '#a0a0a0'


# ── Figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 10), facecolor=CLR_BG)
ax = fig.add_subplot(111, projection='3d')

fig.suptitle(
    'Larmor Precession of Electron Torus\n'
    'Spin axis precesses around external B field',
    color='white', fontsize=14, fontweight='bold', y=0.95)


# ── Precession cone (static reference) ──────────────────────────────

def draw_cone(ax, tilt, n_pts=80):
    """Draw the precession cone that the spin axis traces."""
    phi = np.linspace(0, 2 * np.pi, n_pts)
    cone_len = 3.5
    # The spin axis tip traces a circle at height cone_len * cos(tilt)
    cx = cone_len * np.sin(tilt) * np.cos(phi)
    cy = cone_len * np.sin(tilt) * np.sin(phi)
    cz = np.full_like(phi, cone_len * np.cos(tilt))
    ax.plot(cx, cy, cz, '--', color=CLR_CONE, linewidth=1.0, alpha=0.4)
    # Cone surface (faint lines from origin to circle)
    for i in range(0, n_pts, n_pts // 8):
        ax.plot([0, cx[i]], [0, cy[i]], [0, cz[i]],
                '-', color=CLR_CONE, linewidth=0.5, alpha=0.15)


# ── Per-frame draw ───────────────────────────────────────────────────

# Pre-compute torus geometry (standard orientation)
SX, SY, SZ = torus_surface(R, r)
KX, KY, KZ, KT = torus_knot(p, q, R, r)

# Store spin axis trail
axis_trail_x = []
axis_trail_y = []
axis_trail_z = []

def draw_frame(frame):
    ax.cla()
    ax.set_facecolor(CLR_BG)
    lim = 4.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=30 + frame * 0.15)

    # Precession angle at this frame
    phi = precession_rate * frame
    # Photon circulation phase (much faster)
    circ_phase = circ_rate * frame

    # Combined rotation: tilt around x, then precess around z
    R_prec = rot_z(phi) @ rot_x(tilt)

    # ── B field (vertical, along z) ──────────────────────────────
    b_len = 4.2
    ax.quiver(0, 0, -b_len * 0.3, 0, 0, b_len,
              color=CLR_BFLD, arrow_length_ratio=0.06,
              linewidth=2.5, alpha=0.8)
    ax.text(0.15, 0, b_len * 0.75, 'B', color=CLR_BFLD, fontsize=16,
            fontweight='bold')

    # ── Precession cone ──────────────────────────────────────────
    draw_cone(ax, tilt)

    # ── Spin axis (current direction) ────────────────────────────
    axis_tip = R_prec @ np.array([0, 0, 3.5])
    ax.quiver(0, 0, 0, axis_tip[0], axis_tip[1], axis_tip[2],
              color=CLR_AXIS, arrow_length_ratio=0.06,
              linewidth=2.5, alpha=0.9)
    ax.text(axis_tip[0] * 1.15, axis_tip[1] * 1.15, axis_tip[2] * 1.15,
            'S', color=CLR_AXIS, fontsize=14, fontweight='bold')

    # Spin axis trail
    axis_trail_x.append(axis_tip[0])
    axis_trail_y.append(axis_tip[1])
    axis_trail_z.append(axis_tip[2])
    trail_len = min(len(axis_trail_x), 80)
    if trail_len > 2:
        ax.plot(axis_trail_x[-trail_len:],
                axis_trail_y[-trail_len:],
                axis_trail_z[-trail_len:],
                '-', color=CLR_AXIS, linewidth=1.5, alpha=0.3)

    # ── Rotated torus surface ────────────────────────────────────
    rx, ry, rz = apply_rotation(SX, SY, SZ, R_prec)
    ax.plot_surface(rx, ry, rz, alpha=0.15, color=CLR_TORUS,
                    edgecolor=CLR_TORUS, linewidth=0.2, antialiased=True)

    # ── Rotated knot path ────────────────────────────────────────
    rkx, rky, rkz = apply_rotation(KX, KY, KZ, R_prec)
    ax.plot(rkx, rky, rkz, color=CLR_TORUS, linewidth=1.0, alpha=0.5)

    # ── Circulating photon ───────────────────────────────────────
    ph_pt = point_on_knot(p, q, R, r, circ_phase)
    ph_rot = R_prec @ ph_pt
    ax.plot([ph_rot[0]], [ph_rot[1]], [ph_rot[2]], 'o',
            color='white', markersize=9,
            markeredgecolor=CLR_GLOW, markeredgewidth=2, zorder=10)

    # Short photon trail
    trail_n = 30
    trail_phases = [circ_phase - i * 0.04 for i in range(trail_n)]
    trail_pts = np.array([R_prec @ point_on_knot(p, q, R, r, ph)
                          for ph in trail_phases])
    alphas = np.linspace(0.6, 0.0, trail_n)
    for i in range(trail_n - 1):
        ax.plot(trail_pts[i:i+2, 0], trail_pts[i:i+2, 1],
                trail_pts[i:i+2, 2], '-', color=CLR_GLOW,
                linewidth=2.0, alpha=alphas[i])

    # ── Magnetic moment arrow (opposite to spin for electron) ────
    mu_tip = R_prec @ np.array([0, 0, -2.5])
    ax.quiver(0, 0, 0, mu_tip[0], mu_tip[1], mu_tip[2],
              color='#e74c3c', arrow_length_ratio=0.08,
              linewidth=1.8, alpha=0.6)
    ax.text(mu_tip[0] * 1.15, mu_tip[1] * 1.15, mu_tip[2] * 1.15,
            '\u03bc', color='#e74c3c', fontsize=13, fontweight='bold',
            alpha=0.8)

    # ── Info text ────────────────────────────────────────────────
    deg = np.degrees(tilt)
    ax.text2D(0.02, 0.12, f'tilt \u03b8 = {deg:.0f}\u00b0',
              transform=ax.transAxes, color=CLR_INFO, fontsize=11,
              fontfamily='monospace')
    ax.text2D(0.02, 0.08,
              f'precession: \u03c9_L = eB/2m\u2091',
              transform=ax.transAxes, color=CLR_INFO, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.04,
              f'circulation: f_circ = c/\u03bbC = 1.24\u00d710\u00b2\u2070 Hz',
              transform=ax.transAxes, color=CLR_INFO, fontsize=10,
              fontfamily='monospace')
    ax.text2D(0.02, 0.00,
              'L = \u0127/2 along S  (gyroscope)',
              transform=ax.transAxes, color=CLR_AXIS, fontsize=10,
              fontfamily='monospace')

    # Phase labels
    phase_deg = np.degrees(phi % (2 * np.pi))
    ax.text2D(0.98, 0.04,
              f'precession phase: {phase_deg:.0f}\u00b0',
              transform=ax.transAxes, color=CLR_CONE, fontsize=10,
              fontfamily='monospace', ha='right')

    # Legend
    ax.text2D(0.98, 0.16, 'S = spin axis', transform=ax.transAxes,
              color=CLR_AXIS, fontsize=10, ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.12, 'B = magnetic field', transform=ax.transAxes,
              color=CLR_BFLD, fontsize=10, ha='right', fontfamily='monospace')
    ax.text2D(0.98, 0.08, '\u03bc = magnetic moment', transform=ax.transAxes,
              color='#e74c3c', fontsize=10, ha='right', fontfamily='monospace')

    return []


# ── Render ───────────────────────────────────────────────────────────

print(f"Rendering {N_FRAMES} frames...")
anim = FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                     blit=False, interval=50)

outpath = '/home/jim/repos/claude-prime/simulations/anim_precession.gif'
anim.save(outpath, writer=PillowWriter(fps=20), dpi=100)
print(f"Saved: {outpath}")
plt.close()
