"""
Tripolar Vortex from Shielded Vortex Instability
=================================================
The m=2 instability that produces tripolar vortices requires a SHIELDED
vortex: positive core surrounded by negative vorticity annulus (net
circulation ≈ 0). This is the Carton & McWilliams (1989) result.

Profile: ω(r) = ω₀ (1 - ½α(r/R₀)^α) exp(-(r/R₀)^α)

For α=2: ω = ω₀(1 - r²/R₀²)exp(-r²/R₀²) — strongly unstable to m=2
For α=4: even more unstable

This is physically relevant to dusty plasma because:
- The Coulomb crystal cloud has finite extent
- At the boundary, charge balance creates a negative vorticity sheath
- Dharodi's GH model further destabilizes the boundary

Uses 2D pseudo-spectral Navier-Stokes with very low dissipation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
N = 512
L = 2 * np.pi
dx = L / N

# Shielded vortex profile
alpha = 2            # Profile steepness (2 = Gaussian derivative)
R0 = 0.7             # Vortex radius
omega0 = 8.0         # Peak vorticity

# Dissipation
nu = 2e-5            # Very weak viscosity
nu_h = 0.0           # No hyperviscosity (rely on dealiasing)

# Time
dt = 0.002
N_steps = 30000      # t_final = 60
save_every = 300

print("Shielded Vortex → Tripolar Formation")
print("="*50)
print(f"  N={N}, α={alpha}, R₀={R0}, ω₀={omega0}")
print(f"  ν={nu}, dt={dt}, steps={N_steps}")
print(f"  Turnover time ≈ {2*np.pi/omega0:.2f}")
print(f"  Re = ω₀R₀²/ν = {omega0*R0**2/nu:.0f}")

# ============================================================
# Grid
# ============================================================
x = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x)
xc, yc = L/2, L/2

kx = np.fft.fftfreq(N, d=dx) * 2*np.pi
KX, KY = np.meshgrid(kx, kx)
K2 = KX**2 + KY**2
K2_safe = K2.copy(); K2_safe[0,0] = 1.0

# 2/3 dealiasing
dealias = np.ones((N, N))
dealias[np.abs(KX) > (N//3)*2*np.pi/L] = 0
dealias[np.abs(KY) > (N//3)*2*np.pi/L] = 0

# Integrating factor for diffusion
E_dt = np.exp(-nu * K2 * dt)
E_dt2 = np.exp(-nu * K2 * dt/2)

# ============================================================
# Initial condition: shielded vortex + m=2 perturbation
# ============================================================
r = np.sqrt((X - xc)**2 + (Y - yc)**2)
theta = np.arctan2(Y - yc, X - xc)

# Shielded vortex: positive core, negative annulus
# ω(r) = ω₀ · (1 - α/2 · (r/R₀)^α) · exp(-(r/R₀)^α)
rn = r / R0
omega_init = omega0 * (1.0 - 0.5 * alpha * rn**alpha) * np.exp(-rn**alpha)

# Small m=2 perturbation
omega_init += 0.01 * omega0 * np.cos(2*theta) * rn * np.exp(-rn**2)

# Verify it's shielded (net circulation ≈ 0)
net_circ = np.mean(omega_init) * L**2
print(f"  Net circulation: {net_circ:.4f} (should be ≈ 0)")
print(f"  Initial max|ω| = {np.max(np.abs(omega_init)):.3f}")
print(f"  Min ω = {np.min(omega_init):.3f} (negative annulus)")

# ============================================================
# Time integration: RK4 with integrating factor
# ============================================================
def nonlinear(omega_hat):
    """N(ω) = -J(ψ,ω)"""
    oh = omega_hat * dealias
    psi_hat = -oh / K2_safe; psi_hat[0,0] = 0
    u = np.real(np.fft.ifft2(-1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(1j * KX * psi_hat))
    ox = np.real(np.fft.ifft2(1j * KX * oh))
    oy = np.real(np.fft.ifft2(1j * KY * oh))
    return np.fft.fft2(-(u*ox + v*oy)) * dealias

snapshots = []; times = []
diag = {'t':[], 'Z':[], 'max_o':[], 'min_o':[], 'asym':[], 'E':[]}

def record(omega_hat, t):
    omega = np.real(np.fft.ifft2(omega_hat))
    snapshots.append(omega.copy())
    times.append(t)

    Z = 0.5 * np.mean(omega**2)
    max_o = np.max(omega)
    min_o = np.min(omega)

    # Asymmetry in annular region
    mask = (r > 0.3*R0) & (r < 2.5*R0)
    m0 = np.abs(np.mean(omega[mask]))
    m2 = 2 * np.abs(np.mean(omega[mask] * np.exp(-2j*theta[mask])))
    asym = m2 / max(m0, 1e-10)

    psi_hat = -omega_hat / K2_safe; psi_hat[0,0] = 0
    u = np.real(np.fft.ifft2(-1j*KY*psi_hat))
    v = np.real(np.fft.ifft2(1j*KX*psi_hat))
    E = 0.5 * np.mean(u**2 + v**2)

    diag['t'].append(t); diag['Z'].append(Z)
    diag['max_o'].append(max_o); diag['min_o'].append(min_o)
    diag['asym'].append(asym); diag['E'].append(E)
    return Z, max_o, asym

print("\nRunning RK4 + integrating factor...")
t0 = time.time()

omega_hat = np.fft.fft2(omega_init) * dealias
record(omega_hat, 0.0)

for step in range(1, N_steps + 1):
    t = step * dt

    # RK4 with integrating factor for diffusion
    N1 = nonlinear(omega_hat) * dt
    N2 = nonlinear(E_dt2 * omega_hat + 0.5 * E_dt2 * N1) * dt
    N3 = nonlinear(E_dt2 * omega_hat + 0.5 * N2) * dt
    N4 = nonlinear(E_dt * omega_hat + E_dt2 * N3) * dt

    omega_hat = E_dt * omega_hat + (1.0/6.0) * (
        E_dt * N1 + 2*E_dt2*N2 + 2*E_dt2*N3 + N4)
    omega_hat *= dealias

    if step % save_every == 0:
        Z, max_o, asym = record(omega_hat, t)
        if step % (save_every * 5) == 0:
            elapsed = time.time() - t0
            print(f"  t={t:5.1f} | max ω={max_o:.3f} | asym={asym:.3f} | {elapsed:.0f}s")

    if step % 5000 == 0 and np.any(np.isnan(omega_hat)):
        print(f"  NaN at step {step}!")
        break

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")

asym_arr = np.array(diag['asym'])
time_arr = np.array(diag['t'])
max_o_arr = np.array(diag['max_o'])
min_o_arr = np.array(diag['min_o'])

print(f"\nAsymmetry: {asym_arr[0]:.3f} → {asym_arr[-1]:.3f}")
print(f"  Peak: {asym_arr.max():.3f} at t={time_arr[np.argmax(asym_arr)]:.1f}")
print(f"Max ω retained: {100*max_o_arr[-1]/max_o_arr[0]:.0f}%")

# ============================================================
# Find key frames
# ============================================================
idx_peak = np.argmax(asym_arr)
idx_half = np.searchsorted(asym_arr[:idx_peak+1], asym_arr[idx_peak]*0.5) if idx_peak > 0 else len(asym_arr)//4

# 4 key frames for publication
pub4 = sorted(set([
    0,                              # Initial
    max(1, idx_half),              # Growing
    min(idx_peak, len(snapshots)-1), # Peak tripole
    len(snapshots)-1               # Final
]))[:4]
while len(pub4) < 4:
    mid = (pub4[-2] + pub4[-1]) // 2
    pub4.insert(-1, mid)
    pub4 = sorted(set(pub4))[:4]

print(f"Publication frames: {pub4}")
for idx in pub4:
    print(f"  t={time_arr[idx]:.1f}, asym={asym_arr[idx]:.3f}, max_ω={max_o_arr[idx]:.3f}")

# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

def get_vel(omega):
    oh = np.fft.fft2(omega)
    ph = -oh / K2_safe; ph[0,0] = 0
    return (np.real(np.fft.ifft2(-1j*KY*ph)),
            np.real(np.fft.ifft2(1j*KX*ph)))

pub_labels = ['1. Initial shielded\n   vortex',
              '2. m=2 instability\n   growing',
              '3. Tripolar\n   structure',
              '4. Mature state']

# --- Fig 1: 6-panel evolution ---
n_frames = 6
frame_idx = np.linspace(0, len(snapshots)-1, n_frames).astype(int)

fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8.5))
fig.suptitle('Shielded Vortex → Tripolar Structure\n'
             f'2D Pseudo-Spectral Simulation (N={N}, α={alpha}, Re={omega0*R0**2/nu:.0f})',
             fontsize=14, fontweight='bold')

for i, idx in enumerate(frame_idx):
    snap = snapshots[idx]
    t = time_arr[idx]

    ax = axes[0, i]
    vmax = max(np.abs(snap).max() * 0.95, 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(X, Y, snap, cmap='RdBu_r', norm=norm, shading='auto')
    ax.contour(X, Y, snap, levels=12, colors='k', linewidths=0.3, alpha=0.4)
    ax.set_aspect('equal')
    ax.set_title(f't={t:.1f}, asym={asym_arr[idx]:.2f}', fontsize=9)
    ax.set_xlim(xc-2.2*R0, xc+2.2*R0); ax.set_ylim(yc-2.2*R0, yc+2.2*R0)
    if i == 0: ax.set_ylabel('Vorticity ω', fontsize=11)

    ax2 = axes[1, i]
    lum = gaussian_filter(np.abs(snap)**1.5, sigma=3)
    lum = np.power(lum / max(lum.max(), 1e-20), 0.35)
    ax2.pcolormesh(X, Y, lum, cmap='hot', vmin=0, vmax=1, shading='auto')
    ax2.set_aspect('equal')
    ax2.set_xlim(xc-2.2*R0, xc+2.2*R0); ax2.set_ylim(yc-2.2*R0, yc+2.2*R0)
    ax2.set_facecolor('black')
    if i == 0: ax2.set_ylabel('Camera view', fontsize=11, color='white')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shielded_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Fig 2: 4-panel streamlines ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
fig.suptitle('Circular → Tripolar Transition via Shielded Vortex Instability\n'
             'Mechanism for equilateral triangle formations in dusty plasma (Dharodi et al. 2014)',
             fontsize=12, fontweight='bold')

for i, pidx in enumerate(pub4):
    ax = axes[i]
    snap = snapshots[pidx]
    t = time_arr[pidx]
    u, v = get_vel(snap)

    vmax = max(np.abs(snap).max() * 0.95, 0.1)
    levels = np.linspace(-vmax, vmax, 30)
    cf = ax.contourf(X, Y, snap, levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(X, Y, snap, levels=[0], colors='black', linewidths=1.5)  # Zero contour

    speed = np.sqrt(u**2 + v**2)
    maxspd = max(speed.max(), 1e-10)
    lw = 1.5 * speed / maxspd
    try:
        ax.streamplot(x, x, u, v, color='gray', linewidth=lw, density=1.2, arrowsize=0.6)
    except:
        pass

    ax.set_aspect('equal')
    ax.set_title(f'{pub_labels[i]}\nt = {t:.1f}', fontsize=10)
    ax.set_xlim(xc-2.2*R0, xc+2.2*R0); ax.set_ylim(yc-2.2*R0, yc+2.2*R0)
    plt.colorbar(cf, ax=ax, shrink=0.7, label='ω')
    ax.set_xlabel('x / λ_D')
    if i == 0: ax.set_ylabel('y / λ_D')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shielded_streamlines.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'shielded_streamlines.pdf', bbox_inches='tight')
plt.close()

# --- Fig 3: Camera view ---
fig = plt.figure(figsize=(20, 6), facecolor='black')
fig.suptitle('Simulated Night-Sky View: Plasma Cloud → Triangle Formation\n'
             'Dusty plasma vortex instability',
             fontsize=13, fontweight='bold', color='white')

for i, pidx in enumerate(pub4):
    ax = fig.add_subplot(1, 4, i+1)
    snap = snapshots[pidx]
    lum = gaussian_filter(np.abs(snap)**1.5, sigma=4)
    lum = np.power(lum / max(lum.max(), 1e-20), 0.35)
    ax.pcolormesh(X, Y, lum, cmap='hot', vmin=0, vmax=1, shading='auto')
    ax.set_aspect('equal')
    ax.set_xlim(xc-2.2*R0, xc+2.2*R0); ax.set_ylim(yc-2.2*R0, yc+2.2*R0)
    ax.set_facecolor('black')
    ax.set_title(pub_labels[i], fontsize=10, color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('white')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shielded_camera.png', dpi=200, bbox_inches='tight', facecolor='black')
plt.close()

# --- Fig 4: Diagnostics ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Tripolar Formation Diagnostics (Shielded Vortex)', fontsize=13, fontweight='bold')

axes[0,0].plot(diag['t'], diag['Z'], 'C0-')
axes[0,0].set_ylabel('Enstrophy'); axes[0,0].set_title('A. Enstrophy')

axes[0,1].plot(diag['t'], diag['E'], 'C1-')
axes[0,1].set_ylabel('KE'); axes[0,1].set_title('B. Kinetic Energy')

axes[1,0].plot(diag['t'], diag['max_o'], 'C2-', label='max ω')
axes[1,0].plot(diag['t'], diag['min_o'], 'C3-', label='min ω')
axes[1,0].axhline(y=0, color='gray', ls='--', lw=0.5)
axes[1,0].legend()
axes[1,0].set_ylabel('ω'); axes[1,0].set_title('C. Peak Vorticity')

axes[1,1].plot(diag['t'], diag['asym'], 'C3-', lw=2)
axes[1,1].axhline(y=0.3, color='red', ls='--', alpha=0.5, label='Onset')
axes[1,1].legend()
axes[1,1].set_ylabel('m=2/m=0'); axes[1,1].set_title('D. Tripolar Asymmetry')

for ax in axes.flat: ax.set_xlabel('Time')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shielded_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Fig 5: Radial + angular profiles ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

r_bins = np.linspace(0, 2.5*R0, 60)
r_centers = 0.5*(r_bins[:-1] + r_bins[1:])

for idx, label, color in [(0,'Initial','C0'), (idx_peak,'Peak','C3'), (-1,'Final','C2')]:
    snap = snapshots[idx]
    profile = np.zeros(len(r_centers))
    for j in range(len(r_centers)):
        mask = (r >= r_bins[j]) & (r < r_bins[j+1])
        if mask.sum() > 0:
            profile[j] = np.mean(snap[mask])
    axes[0].plot(r_centers/R0, profile/omega0, f'{color}-', lw=2, label=f'{label} (t={time_arr[idx]:.0f})')

axes[0].axhline(y=0, color='gray', ls='--', lw=0.5)
axes[0].set_xlabel('r / R₀'); axes[0].set_ylabel('⟨ω⟩_θ / ω₀')
axes[0].set_title('A. Radial Profile'); axes[0].legend(fontsize=9)

# Angular at r≈R0
if idx_peak < len(snapshots):
    snap_peak = snapshots[idx_peak]
    theta_bins = np.linspace(-np.pi, np.pi, 120)
    mask_r = (r > 0.6*R0) & (r < 1.4*R0)
    ang = np.zeros(len(theta_bins)-1)
    for j in range(len(ang)):
        mask_t = (theta >= theta_bins[j]) & (theta < theta_bins[j+1]) & mask_r
        if mask_t.sum() > 0: ang[j] = np.mean(snap_peak[mask_t])
    tc = 0.5*(theta_bins[:-1] + theta_bins[1:])
    axes[1].plot(np.degrees(tc), ang, 'C3-', lw=2)
    axes[1].axhline(y=0, color='gray', ls='--', lw=0.5)

axes[1].set_xlabel('θ (degrees)'); axes[1].set_ylabel('ω at r≈R₀')
axes[1].set_title('B. Angular Profile (tripolar = 2 peaks)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shielded_profiles.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}")
print(f"\n{'='*60}")
if asym_arr.max() > 0.3:
    print(f"SUCCESS: Tripolar deformation detected!")
else:
    print(f"Asymmetry peak = {asym_arr.max():.3f} — may need longer run or steeper profile")
print(f"  Asymmetry: {asym_arr[0]:.3f} → {asym_arr.max():.3f} (peak at t={time_arr[idx_peak]:.1f})")
print(f"  Amplification: {asym_arr.max()/max(asym_arr[0],1e-6):.0f}x")
print(f"{'='*60}")
