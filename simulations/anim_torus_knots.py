#!/usr/bin/env python3
"""
Animation: photons circulating on torus knots for e, μ, τ.

Shows a luminous point traveling along (p,q) torus knot paths,
comparing the three charged leptons side by side. The electron
has the simplest winding, the tau the most complex.

The torus surface FLEXES as the photon passes — a local bulge in
the minor radius representing the gravitational distortion from
the photon's energy. This shows EM and gravity unified: the
circulating photon IS the source of spacetime curvature.

In the null worldtube model, particles ARE photons trapped on
torus knots — the winding numbers (p,q) determine the mass.

Outputs GIF via pillow.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# ── Torus knot geometry ───────────────────────────────────────────────
def torus_surface_flexed(R, r, photon_u, photon_v, flex_amp=0.25,
                          flex_sigma=0.6, n_major=36, n_minor=18):
    """
    Generate torus surface mesh with a local bulge near the photon.

    The minor radius r increases near (photon_u, photon_v) by flex_amp,
    with Gaussian falloff controlled by flex_sigma. This represents the
    gravitational distortion from the photon's trapped energy.

    photon_u: major angle (toroidal direction)
    photon_v: minor angle (poloidal direction)
    """
    u = np.linspace(0, 2*np.pi, n_major)
    v = np.linspace(0, 2*np.pi, n_minor)
    U, V = np.meshgrid(u, v)

    # Angular distance on torus (periodic)
    du = np.arctan2(np.sin(U - photon_u), np.cos(U - photon_u))
    dv = np.arctan2(np.sin(V - photon_v), np.cos(V - photon_v))
    dist2 = du**2 + dv**2

    # Local bulge: r increases near photon
    r_local = r * (1 + flex_amp * np.exp(-dist2 / (2*flex_sigma**2)))

    X = (R + r_local*np.cos(V)) * np.cos(U)
    Y = (R + r_local*np.cos(V)) * np.sin(U)
    Z = r_local * np.sin(V)
    return X, Y, Z


def torus_knot(p, q, R, r, N=1000):
    """Generate (p,q) torus knot curve."""
    t = np.linspace(0, 2*np.pi, N)
    x = (R + r*np.cos(q*t)) * np.cos(p*t)
    y = (R + r*np.cos(q*t)) * np.sin(p*t)
    z = r * np.sin(q*t)
    return x, y, z, t

# ── Particle configurations ──────────────────────────────────────────
# Torus size scaled by winding number (p+q) to show mass hierarchy.
R_max, r_max = 2.5, 0.8
pq_max = 12

particles = [
    {'name': 'Electron (e)', 'p': 2, 'q': 3, 'mass': 0.511,
     'color': '#3498db', 'glow': '#85c1e9'},
    {'name': 'Muon (μ)', 'p': 3, 'q': 5, 'mass': 105.7,
     'color': '#2ecc71', 'glow': '#82e0aa'},
    {'name': 'Tau (τ)', 'p': 5, 'q': 7, 'mass': 1776.9,
     'color': '#e74c3c', 'glow': '#f1948a'},
]
for part in particles:
    scale = (part['p'] + part['q']) / pq_max
    part['R'] = R_max * scale
    part['r'] = r_max * scale

# ── Set up figure ─────────────────────────────────────────────────────
bg = '#060612'
fig = plt.figure(figsize=(18, 6.5), facecolor=bg)

axes = []
knot_data = []
torus_surf_artists = []

for idx, part in enumerate(particles):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    ax.set_facecolor(bg)
    axes.append(ax)

    R, r = part['R'], part['r']
    p, q = part['p'], part['q']

    # Initial torus surface (will be redrawn each frame)
    X, Y, Z = torus_surface_flexed(R, r, 0, 0, flex_amp=0,
                                    n_major=36, n_minor=18)
    surf = ax.plot_surface(X, Y, Z, alpha=0.18, color=part['color'],
                           edgecolor=part['color'], linewidth=0.3,
                           antialiased=True)
    torus_surf_artists.append(surf)

    # Full knot path (static background)
    kx, ky, kz, kt = torus_knot(p, q, R, r, N=2000)
    ax.plot(kx, ky, kz, color=part['color'], linewidth=0.5, alpha=0.3)
    knot_data.append((kx, ky, kz, kt))

    # Styling
    lim_xy = R_max + r_max + 0.5
    lim_z = r_max + 0.3
    ax.set_xlim(-lim_xy, lim_xy)
    ax.set_ylim(-lim_xy, lim_xy)
    ax.set_zlim(-lim_z, lim_z)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=0)

    ax.set_title(f'{part["name"]}\n({p},{q}) torus knot\n'
                 f'm = {part["mass"]} MeV',
                 color='white', fontsize=12, fontweight='bold', pad=-5)

fig.suptitle('Photons on Torus Knots — Null Worldtube Model\n'
             'Torus flexes as photon passes (gravitational distortion from trapped energy)',
             color='white', fontsize=14, fontweight='bold', y=0.98)

# ── Photon markers ────────────────────────────────────────────────────
photon_balls = []
photon_trails = []
trail_length = 150

for idx, part in enumerate(particles):
    ax = axes[idx]
    ball, = ax.plot([], [], [], 'o', color='white', markersize=10,
                    zorder=10, markeredgecolor=part['glow'], markeredgewidth=2)
    photon_balls.append(ball)

    trail, = ax.plot([], [], [], '-', color=part['glow'], linewidth=2.5,
                     alpha=0.6, zorder=9)
    photon_trails.append(trail)

# ── Animation ─────────────────────────────────────────────────────────
N_knot = 2000
n_anim_frames = 200
n_hold_frames = 40
n_frames = n_anim_frames + n_hold_frames
steps_per_frame = N_knot // n_anim_frames

def update(frame):
    anim_frame = min(frame, n_anim_frames - 1)

    for idx, part in enumerate(particles):
        kx, ky, kz, kt = knot_data[idx]
        R, r = part['R'], part['r']
        p, q = part['p'], part['q']

        speed = 1.0 + idx * 0.3
        pos = int(anim_frame * steps_per_frame * speed) % N_knot

        # Photon ball
        photon_balls[idx].set_data_3d([kx[pos]], [ky[pos]], [kz[pos]])

        # Trail
        if frame >= n_anim_frames:
            photon_trails[idx].set_data_3d(kx, ky, kz)
        elif pos >= trail_length:
            photon_trails[idx].set_data_3d(
                kx[pos-trail_length:pos+1],
                ky[pos-trail_length:pos+1],
                kz[pos-trail_length:pos+1])
        else:
            wrap_start = N_knot + pos - trail_length
            wrap_x = np.concatenate([kx[wrap_start:], kx[:pos+1]])
            wrap_y = np.concatenate([ky[wrap_start:], ky[:pos+1]])
            wrap_z = np.concatenate([kz[wrap_start:], kz[:pos+1]])
            photon_trails[idx].set_data_3d(wrap_x, wrap_y, wrap_z)

        # ── Flex the torus ───────────────────────────────────────────
        # Photon's position on the torus surface in (u, v) coordinates:
        # On a (p,q) torus knot at parameter t, major angle = p*t, minor angle = q*t
        t_param = kt[pos]  # knot parameter at current position
        photon_u = (p * t_param) % (2*np.pi)
        photon_v = (q * t_param) % (2*np.pi)

        # Flex amplitude scales with mass (heavier = more energy = more flex)
        flex = 0.15 + 0.15 * idx  # e: 0.15, μ: 0.30, τ: 0.45

        # Remove old surface and draw flexed one
        torus_surf_artists[idx].remove()
        X, Y, Z = torus_surface_flexed(R, r, photon_u, photon_v,
                                        flex_amp=flex, flex_sigma=0.6,
                                        n_major=36, n_minor=18)
        torus_surf_artists[idx] = axes[idx].plot_surface(
            X, Y, Z, alpha=0.18, color=part['color'],
            edgecolor=part['color'], linewidth=0.3, antialiased=True)

        # Rotate view
        axes[idx].view_init(elev=20, azim=frame * 1.8)

    return photon_balls + photon_trails

print(f"Rendering {n_frames} frames (with torus flexing)...")
anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=50)

outpath = '/home/jim/repos/claude-prime/simulations/anim_torus_knots.gif'
anim.save(outpath, writer=PillowWriter(fps=20), dpi=100)
print(f"Saved: {outpath}")
plt.close()
