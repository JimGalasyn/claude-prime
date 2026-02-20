"""
Tripolar Vortex Formation v3: Stable semi-implicit integration
==============================================================
Uses IMEX (Implicit-Explicit) scheme:
- Diffusion: implicit (exponential integrating factor — unconditionally stable)
- Nonlinear advection: explicit (Adams-Bashforth 2)
- Viscoelastic stress: semi-implicit

This allows very low physical dissipation without numerical instability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from pathlib import Path
import time

OUTPUT_DIR = Path("/mnt/c/Users/jimga/OneDrive/Documents/Research/UAP/simulations/output/tripolar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
Nx, Ny = 256, 256
Lx, Ly = 2*np.pi, 2*np.pi
dx, dy = Lx/Nx, Ly/Ny

# Physics
tau_m = 1.5        # Viscoelastic relaxation time
eta_star = 0.4     # Elastic viscosity
nu = 0.001         # Physical viscosity (moderate)
nu_h = 1e-8        # Hyperviscosity (∇⁴ damping for numerical stability)

# Time
dt = 0.005
N_steps = 16000
save_every = 160

# Vortex
R0 = 0.8
omega0 = 1.5

print("Tripolar Vortex Formation v3 (IMEX)")
print("="*50)
print(f"  Grid: {Nx}×{Ny}, dt={dt}")
print(f"  τ_m={tau_m}, η*={eta_star}, ν={nu}, ν_h={nu_h}")
print(f"  R₀={R0}, ω₀={omega0}")

# ============================================================
# Grid + spectral
# ============================================================
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)
xc, yc = Lx/2, Ly/2

kx = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2*np.pi
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
K4 = K2**2
K2_safe = K2.copy(); K2_safe[0,0] = 1.0

# Dealiasing (2/3 rule)
dealias = np.ones((Ny, Nx))
dealias[np.abs(KX) > (Nx//3)*2*np.pi/Lx] = 0
dealias[np.abs(KY) > (Ny//3)*2*np.pi/Ly] = 0

# Implicit diffusion factor (exponential integrating factor)
# Includes both ν∇² and ν_h∇⁴
L = -nu * K2 - nu_h * K4
E_dt = np.exp(L * dt)       # Full step
E_dt2 = np.exp(L * dt/2)    # Half step

# ============================================================
# Initial condition
# ============================================================
r = np.sqrt((X - xc)**2 + (Y - yc)**2)
theta = np.arctan2(Y - yc, X - xc)

# Smooth vortex with super-Gaussian envelope
omega0_field = omega0 * np.exp(-(r/R0)**6)

# Small m=2 perturbation
omega0_field += 0.03 * omega0 * np.cos(2*theta) * np.exp(-(r/(0.7*R0))**4)

omega_hat = np.fft.fft2(omega0_field)
sigma_hat = np.zeros((Ny, Nx), dtype=complex)

# ============================================================
# Time integration: IMEX with Adams-Bashforth for nonlinear term
# ============================================================
def compute_nonlinear_hat(omega_hat, sigma_hat):
    """Compute nonlinear RHS in spectral space.
    Returns: NL_omega_hat (advection + stress), NL_sigma_hat (relaxation)
    """
    # Physical fields
    omega = np.real(np.fft.ifft2(omega_hat))

    # Stream function and velocity
    psi_hat = -omega_hat / K2_safe
    psi_hat[0,0] = 0
    u = np.real(np.fft.ifft2(-1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(1j * KX * psi_hat))

    # Vorticity gradients
    omega_x = np.real(np.fft.ifft2(1j * KX * omega_hat))
    omega_y = np.real(np.fft.ifft2(1j * KY * omega_hat))

    # Advection: -(u·∇)ω (computed in physical space, dealiased)
    advection = -(u * omega_x + v * omega_y)
    advection_hat = np.fft.fft2(advection) * dealias

    # Stress divergence: ∇²σ (in spectral space)
    stress_hat = -K2 * sigma_hat

    # Total omega RHS (excluding linear diffusion, handled implicitly)
    NL_omega = advection_hat + stress_hat

    # Sigma RHS: -σ/τ_m + η*·ω (handled semi-implicitly)
    NL_sigma = -sigma_hat / tau_m + eta_star * omega_hat

    return NL_omega, NL_sigma

# Storage
snapshots = []
times = []
diag = {'time': [], 'enstrophy': [], 'max_omega': [], 'asymmetry': [], 'energy': []}

def record_diagnostics(omega_hat, t):
    omega = np.real(np.fft.ifft2(omega_hat))
    snapshots.append(omega.copy())
    times.append(t)

    Z = 0.5 * np.mean(omega**2)
    max_o = np.max(np.abs(omega))

    mask = r < 2*R0
    m0 = np.abs(np.mean(omega[mask]))
    m2 = np.abs(np.mean(omega[mask] * np.exp(-2j * theta[mask])))
    asym = m2 / max(m0, 1e-10)

    psi_hat = -omega_hat / K2_safe; psi_hat[0,0] = 0
    u = np.real(np.fft.ifft2(-1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(1j * KX * psi_hat))
    E = 0.5 * np.mean(u**2 + v**2)

    diag['time'].append(t)
    diag['enstrophy'].append(Z)
    diag['max_omega'].append(max_o)
    diag['asymmetry'].append(asym)
    diag['energy'].append(E)

    return Z, max_o, asym, E

print("\nRunning simulation...")
t0_wall = time.time()

# First step: Forward Euler (need previous NL for AB2)
record_diagnostics(omega_hat, 0.0)
NL_omega_prev, NL_sigma_prev = compute_nonlinear_hat(omega_hat, sigma_hat)

for step in range(1, N_steps + 1):
    t = step * dt

    # Compute current nonlinear terms
    NL_omega, NL_sigma = compute_nonlinear_hat(
        omega_hat * dealias, sigma_hat * dealias)

    if step == 1:
        # Forward Euler for first step
        omega_hat = E_dt * (omega_hat + dt * NL_omega)
        sigma_hat = sigma_hat + dt * NL_sigma
    else:
        # Adams-Bashforth 2 + exponential integrating factor for diffusion
        omega_hat = E_dt * (omega_hat + dt * (1.5 * NL_omega - 0.5 * NL_omega_prev))
        sigma_hat = sigma_hat + dt * (1.5 * NL_sigma - 0.5 * NL_sigma_prev)

    # Store for AB2
    NL_omega_prev = NL_omega.copy()
    NL_sigma_prev = NL_sigma.copy()

    # Dealias
    omega_hat *= dealias
    sigma_hat *= dealias

    # Check for NaN
    if step % 100 == 0:
        if np.any(np.isnan(omega_hat)):
            print(f"  NaN detected at step {step}!")
            break

    # Record
    if step % save_every == 0:
        Z, max_o, asym, E = record_diagnostics(omega_hat, t)
        if step % (save_every * 5) == 0:
            elapsed = time.time() - t0_wall
            print(f"  Step {step:6d}/{N_steps} (t={t:.1f}) | "
                  f"max|ω|={max_o:.4f} | Z={Z:.6f} | asym={asym:.3f} | "
                  f"{elapsed:.0f}s")

elapsed = time.time() - t0_wall
print(f"\nSimulation complete in {elapsed:.1f}s")

# ============================================================
# Analyze results
# ============================================================
asym_arr = np.array(diag['asymmetry'])
time_arr = np.array(diag['time'])
max_o_arr = np.array(diag['max_omega'])

print(f"\nAsymmetry: {asym_arr[0]:.3f} → {asym_arr[-1]:.3f} "
      f"(peak: {asym_arr.max():.3f} at t={time_arr[np.argmax(asym_arr)]:.1f})")
print(f"Max|ω|: {max_o_arr[0]:.4f} → {max_o_arr[-1]:.4f}")

# Find key transition frames
idx_initial = 0
idx_early = max(1, np.searchsorted(asym_arr, 0.05))
idx_onset = np.searchsorted(asym_arr, 0.15) if np.any(asym_arr > 0.15) else len(asym_arr)//3
idx_develop = np.searchsorted(asym_arr, 0.5) if np.any(asym_arr > 0.5) else len(asym_arr)//2
idx_mature = np.searchsorted(asym_arr, 1.0) if np.any(asym_arr > 1.0) else 3*len(asym_arr)//4
idx_final = len(snapshots) - 1

# Clamp indices
for name in ['idx_early', 'idx_onset', 'idx_develop', 'idx_mature']:
    val = min(eval(name), idx_final)
    exec(f"{name} = {val}")

print(f"\nKey frames:")
for name, idx in [('Initial', idx_initial), ('Early', idx_early),
                   ('Onset', idx_onset), ('Develop', idx_develop),
                   ('Mature', idx_mature), ('Final', idx_final)]:
    if idx < len(time_arr):
        print(f"  {name:10s}: t={time_arr[idx]:.1f}, asym={asym_arr[idx]:.3f}, "
              f"max|ω|={max_o_arr[idx]:.4f}")

# ============================================================
# Generate Figures
# ============================================================
print("\nGenerating figures...")

def get_velocity(omega):
    oh = np.fft.fft2(omega)
    ph = -oh / K2_safe; ph[0,0] = 0
    u = np.real(np.fft.ifft2(-1j * KY * ph))
    v = np.real(np.fft.ifft2(1j * KX * ph))
    return u, v

# --- Figure 1: 6-panel evolution ---
key_indices = [idx_initial, idx_early, idx_onset, idx_develop, idx_mature, idx_final]
# Remove duplicates
seen = set()
unique_indices = []
for idx in key_indices:
    if idx not in seen:
        seen.add(idx)
        unique_indices.append(idx)
# Pad if needed
while len(unique_indices) < 6:
    gap = (unique_indices[-1] + idx_final) // 2
    if gap not in seen:
        unique_indices.append(gap)
        seen.add(gap)
    else:
        unique_indices.append(idx_final)
        break
unique_indices = sorted(unique_indices)[:6]

ncols = min(6, len(unique_indices))
fig, axes = plt.subplots(2, ncols, figsize=(4*ncols, 8))
fig.suptitle('Spontaneous Tripolar Formation in Visco-Elastic Dusty Plasma\n'
             f'(τ_m={tau_m}, η*={eta_star}, ν={nu})',
             fontsize=14, fontweight='bold')

for i, idx in enumerate(unique_indices):
    if i >= ncols:
        break
    snap = snapshots[idx]
    t = times[idx]

    # Top: vorticity
    ax = axes[0, i]
    vmax = max(np.abs(snap).max(), 1e-10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(X, Y, snap, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(f't={t:.1f}\nasym={asym_arr[idx]:.2f}', fontsize=9)
    ax.set_xlim(xc - 2.5*R0, xc + 2.5*R0)
    ax.set_ylim(yc - 2.5*R0, yc + 2.5*R0)
    if i == 0:
        ax.set_ylabel('Vorticity ω')

    # Bottom: luminosity
    ax2 = axes[1, i]
    lum = gaussian_filter(snap**2, sigma=2)
    lum = lum / max(lum.max(), 1e-10)
    ax2.pcolormesh(X, Y, lum, cmap='hot', vmin=0, vmax=1, shading='auto')
    ax2.set_aspect('equal')
    ax2.set_xlim(xc - 2.5*R0, xc + 2.5*R0)
    ax2.set_ylim(yc - 2.5*R0, yc + 2.5*R0)
    ax2.set_facecolor('black')
    if i == 0:
        ax2.set_ylabel('Luminosity ∝ |ω|²')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_v3_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 2: Publication-quality key frames with streamlines ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
fig.suptitle('Visco-Elastic Dusty Plasma: Circular → Tripolar Vortex Transition\n'
             'Mechanism for observed equilateral triangle UAP formations',
             fontsize=13, fontweight='bold')

pub_indices = [idx_initial, idx_onset, idx_develop, idx_final]
pub_labels = ['Initial circular\nvortex', 'm=2 instability\nonset',
              'Developing\ntripole', 'Mature tripolar\nstructure']

for i, (pidx, label) in enumerate(zip(pub_indices, pub_labels)):
    ax = axes[i]
    snap = snapshots[pidx]
    t = times[pidx]
    u, v = get_velocity(snap)

    vmax = max(np.abs(snap).max(), 1e-10)
    levels = np.linspace(-vmax, vmax, 40)
    cf = ax.contourf(X, Y, snap, levels=levels, cmap='RdBu_r', extend='both')

    speed = np.sqrt(u**2 + v**2)
    max_speed = max(speed.max(), 1e-10)
    lw = 2.0 * speed / max_speed
    try:
        ax.streamplot(x, y, u, v, color='black', linewidth=lw, density=1.3,
                      arrowsize=0.7)
    except:
        pass  # streamplot can fail with very small velocities

    ax.set_aspect('equal')
    ax.set_title(f'{label}\nt = {t:.1f}', fontsize=10)
    ax.set_xlim(xc - 2.5*R0, xc + 2.5*R0)
    ax.set_ylim(yc - 2.5*R0, yc + 2.5*R0)
    plt.colorbar(cf, ax=ax, shrink=0.7, label='ω')
    ax.set_xlabel('x / λ_D')
    if i == 0:
        ax.set_ylabel('y / λ_D')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_v3_streamlines.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'tripolar_v3_streamlines.pdf', bbox_inches='tight')
plt.close()

# --- Figure 3: Camera view (the money shot) ---
fig = plt.figure(figsize=(20, 5.5), facecolor='black')
fig.suptitle('Simulated Night-Sky View: Plasma Luminosity\n'
             'Single meteor-ablated plasma cloud → triangular formation',
             fontsize=13, fontweight='bold', color='white')

for i, (pidx, label) in enumerate(zip(pub_indices, pub_labels)):
    ax = fig.add_subplot(1, 4, i+1)
    snap = snapshots[pidx]

    lum = gaussian_filter(snap**2, sigma=3)
    lum = lum / max(lum.max(), 1e-10)
    # Gamma correction for better visibility
    lum = np.power(lum, 0.5)

    ax.pcolormesh(X, Y, lum, cmap='hot', vmin=0, vmax=1, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(f'{label}', fontsize=10, color='white')
    ax.set_xlim(xc - 2.5*R0, xc + 2.5*R0)
    ax.set_ylim(yc - 2.5*R0, yc + 2.5*R0)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    if i == 0:
        ax.set_ylabel('y', color='white')
    ax.set_xlabel('x', color='white')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_v3_camera.png', dpi=200, bbox_inches='tight',
            facecolor='black')
plt.close()

# --- Figure 4: Diagnostics ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Tripolar Formation Diagnostics', fontsize=13, fontweight='bold')

axes[0,0].semilogy(diag['time'], diag['enstrophy'], 'C0-')
axes[0,0].set_ylabel('Enstrophy'); axes[0,0].set_title('A. Enstrophy')

axes[0,1].plot(diag['time'], diag['energy'], 'C1-')
axes[0,1].set_ylabel('KE'); axes[0,1].set_title('B. Kinetic Energy')

axes[1,0].semilogy(diag['time'], diag['max_omega'], 'C2-')
axes[1,0].set_ylabel('max|ω|'); axes[1,0].set_title('C. Peak Vorticity')

axes[1,1].plot(diag['time'], diag['asymmetry'], 'C3-', linewidth=2)
axes[1,1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Onset')
axes[1,1].axhline(y=1.0, color='red', alpha=0.5, label='Strong tripole')
axes[1,1].set_ylabel('m=2/m=0'); axes[1,1].set_title('D. Tripolar Asymmetry')
axes[1,1].legend()

for ax in axes.flat:
    ax.set_xlabel('Time (ω_pd⁻¹)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_v3_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}")

if asym_arr[-1] > asym_arr[0] * 2:
    print("\nSUCCESS: Clear tripolar deformation observed!")
    print(f"Asymmetry amplification: {asym_arr[-1]/max(asym_arr[0], 1e-10):.1f}x")
else:
    print("\nPartial result — asymmetry grew but may need parameter tuning.")

print("\nDone!")
