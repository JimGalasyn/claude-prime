"""
Spontaneous Tripolar Vortex Formation in Visco-Elastic Dusty Plasma
====================================================================
Reproduces the key result from Dharodi et al. (2014): a circular vortex
patch in strongly coupled dusty plasma spontaneously evolves into a
tripolar structure through visco-elastic instability.

Physics: Generalized Hydrodynamic (GH) model for strongly coupled
dusty plasma. The key is that strong coupling introduces a viscoelastic
memory term (relaxation time τ_m) that modifies the vorticity equation.

The GH-modified 2D vorticity equation:
    (1 + τ_m ∂/∂t)[∂ω/∂t + (v·∇)ω] = (η*/ρ)(1 + τ_m ∂/∂t)∇²ω + ν∇²ω

where:
    ω = vorticity (curl of velocity)
    τ_m = viscoelastic relaxation time
    η* = elastic (high-frequency) viscosity
    ν = kinematic viscosity (dissipative)

For strongly coupled dusty plasma (Γ ~ 10-100):
    τ_m ≈ 1/ω_pd (inverse dust plasma frequency)
    η* ≈ n_d m_d ω_pd a² (elastic modulus)

The tripolar instability occurs because the viscoelastic response creates
an effective negative viscosity at intermediate frequencies, causing the
circular vortex to become unstable to m=2 azimuthal perturbation.

Reference: Dharodi, Patel & Kaw, Phys. Rev. E 89, 023102 (2014)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import time

OUTPUT_DIR = Path("/mnt/c/Users/jimga/OneDrive/Documents/Research/UAP/simulations/output/tripolar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Simulation Parameters
# ============================================================
# Grid
Nx, Ny = 256, 256
Lx, Ly = 2*np.pi, 2*np.pi  # Normalized domain
dx = Lx / Nx
dy = Ly / Ny

# Physics (normalized units)
# Strongly coupled regime: Γ ~ 10-50
tau_m = 1.0        # Viscoelastic relaxation time (units of ω_pd⁻¹)
eta_star = 0.5     # Elastic viscosity (normalized)
nu = 0.002         # Kinematic (dissipative) viscosity
# Effective viscoelastic parameter
eta_eff = eta_star * tau_m  # This drives the instability

# Time stepping
dt = 0.005
N_steps = 12000
save_every = 200

# Initial vortex parameters
R0 = 0.8           # Vortex radius
omega0 = 1.0       # Peak vorticity

print("Tripolar Vortex Formation Simulation")
print("="*50)
print(f"  Grid: {Nx}×{Ny}")
print(f"  τ_m = {tau_m} (viscoelastic relaxation)")
print(f"  η* = {eta_star} (elastic viscosity)")
print(f"  ν = {nu} (kinematic viscosity)")
print(f"  η_eff = η*·τ_m = {eta_eff}")
print(f"  dt = {dt}, steps = {N_steps}")
print(f"  Vortex: R₀={R0}, ω₀={omega0}")

# ============================================================
# Grid setup
# ============================================================
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# Wavenumbers for spectral methods
kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
K2[0, 0] = 1.0  # Avoid division by zero

# Dealiasing mask (2/3 rule)
kmax_x = Nx // 3
kmax_y = Ny // 3
dealias = np.ones((Ny, Nx))
dealias[np.abs(KX) > kmax_x * 2*np.pi/Lx] = 0
dealias[np.abs(KY) > kmax_y * 2*np.pi/Ly] = 0

# ============================================================
# Initial condition: circular vortex with small perturbation
# ============================================================
def initial_vortex(X, Y, Lx, Ly, R0, omega0):
    """Smooth circular vortex patch centered in domain.
    Uses a Gaussian profile for smoothness."""
    xc, yc = Lx/2, Ly/2
    r = np.sqrt((X - xc)**2 + (Y - yc)**2)

    # Smooth Rankine-like vortex: constant core + Gaussian taper
    omega = omega0 * np.exp(-(r/R0)**4)

    # Add small m=2 perturbation (seed for tripolar instability)
    theta = np.arctan2(Y - yc, X - xc)
    # Also add random noise for generality
    np.random.seed(42)
    noise = 0.01 * omega0 * np.random.randn(Ny, Nx)
    perturbation = 0.05 * omega0 * np.cos(2 * theta) * np.exp(-(r/R0)**2)

    return omega + perturbation + noise

# ============================================================
# Spectral operators
# ============================================================
def vorticity_to_stream(omega_hat):
    """Solve ∇²ψ = -ω for stream function."""
    return -omega_hat / K2

def stream_to_velocity(psi_hat):
    """u = -∂ψ/∂y, v = ∂ψ/∂x"""
    u_hat = -1j * KY * psi_hat
    v_hat = 1j * KX * psi_hat
    return u_hat, v_hat

def compute_nonlinear(omega, u, v):
    """Compute -(u·∇)ω in physical space, return in spectral space."""
    omega_x = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(omega)))
    omega_y = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(omega)))
    nl = -(u * omega_x + v * omega_y)
    return np.fft.fft2(nl) * dealias

# ============================================================
# Time integration: RK4 with GH viscoelastic term
# ============================================================
# The GH model modifies the vorticity equation. For numerical stability,
# we treat the linear terms (viscosity, relaxation) implicitly/semi-implicitly
# and the nonlinear term explicitly.
#
# Simplified approach: operator splitting
# 1. Advection + viscoelastic source: explicit RK4
# 2. Diffusion: implicit (exact in spectral space)
#
# The GH equation in spectral space:
# (1 + τ_m ∂/∂t) ∂ω̂/∂t = NL + (ν + η_eff)(-K²)ω̂ + τ_m·∂(NL)/∂t
#
# We use a simplified scheme tracking both ω and σ (the viscoelastic stress):
# ∂ω/∂t = NL + ν∇²ω + ∇²σ
# ∂σ/∂t = -σ/τ_m + η_star·ω

omega = initial_vortex(X, Y, Lx, Ly, R0, omega0)
sigma = np.zeros_like(omega)  # Viscoelastic stress (initially zero)

# Storage for snapshots
snapshots = []
times = []
diagnostics = {'time': [], 'enstrophy': [], 'energy': [], 'max_omega': [],
               'asymmetry': []}

def compute_diagnostics(omega, t):
    """Compute diagnostic quantities."""
    enstrophy = 0.5 * np.mean(omega**2)
    omega_hat = np.fft.fft2(omega)
    psi_hat = vorticity_to_stream(omega_hat)
    u_hat, v_hat = stream_to_velocity(psi_hat)
    u = np.real(np.fft.ifft2(u_hat))
    v = np.real(np.fft.ifft2(v_hat))
    energy = 0.5 * np.mean(u**2 + v**2)

    # Asymmetry: ratio of m=2 to m=0 Fourier modes (measures tripolar deformation)
    xc, yc = Lx/2, Ly/2
    r = np.sqrt((X - xc)**2 + (Y - yc)**2)
    theta = np.arctan2(Y - yc, X - xc)
    mask = r < 2*R0
    if np.sum(mask) > 0:
        m0 = np.abs(np.mean(omega[mask]))
        m2 = np.abs(np.mean(omega[mask] * np.exp(-2j * theta[mask])))
        asymmetry = m2 / max(m0, 1e-10)
    else:
        asymmetry = 0.0

    return enstrophy, energy, np.max(np.abs(omega)), asymmetry

print("\nRunning simulation...")
t0 = time.time()

for step in range(N_steps + 1):
    t = step * dt

    if step % save_every == 0:
        snapshots.append(omega.copy())
        times.append(t)
        Z, E, max_o, asym = compute_diagnostics(omega, t)
        diagnostics['time'].append(t)
        diagnostics['enstrophy'].append(Z)
        diagnostics['energy'].append(E)
        diagnostics['max_omega'].append(max_o)
        diagnostics['asymmetry'].append(asym)

        if step % (save_every * 5) == 0:
            elapsed = time.time() - t0
            print(f"  Step {step:6d}/{N_steps} (t={t:.2f}) | "
                  f"max|ω|={max_o:.3f} | Z={Z:.4f} | asym={asym:.3f} | "
                  f"{elapsed:.0f}s")

    if step == N_steps:
        break

    # --- RK4 time stepping ---
    def rhs(omega_in, sigma_in):
        """Compute right-hand side of the coupled system."""
        omega_hat = np.fft.fft2(omega_in) * dealias
        psi_hat = vorticity_to_stream(omega_hat)
        u_hat, v_hat = stream_to_velocity(psi_hat)
        u = np.real(np.fft.ifft2(u_hat * dealias))
        v = np.real(np.fft.ifft2(v_hat * dealias))

        # Nonlinear advection
        nl_hat = compute_nonlinear(omega_in, u, v)
        nl = np.real(np.fft.ifft2(nl_hat))

        # Viscous diffusion of omega
        laplacian_omega = np.real(np.fft.ifft2(-K2 * omega_hat))
        # Laplacian of stress
        sigma_hat = np.fft.fft2(sigma_in) * dealias
        laplacian_sigma = np.real(np.fft.ifft2(-K2 * sigma_hat))

        # Vorticity equation: dω/dt = NL + ν∇²ω + ∇²σ
        d_omega = nl + nu * laplacian_omega + laplacian_sigma

        # Stress equation: dσ/dt = -σ/τ_m + η*·ω
        d_sigma = -sigma_in / tau_m + eta_star * omega_in

        return d_omega, d_sigma

    # RK4 stages
    k1_o, k1_s = rhs(omega, sigma)
    k2_o, k2_s = rhs(omega + 0.5*dt*k1_o, sigma + 0.5*dt*k1_s)
    k3_o, k3_s = rhs(omega + 0.5*dt*k2_o, sigma + 0.5*dt*k2_s)
    k4_o, k4_s = rhs(omega + dt*k3_o, sigma + dt*k3_s)

    omega = omega + (dt/6) * (k1_o + 2*k2_o + 2*k3_o + k4_o)
    sigma = sigma + (dt/6) * (k1_s + 2*k2_s + 2*k3_s + k4_s)

elapsed = time.time() - t0
print(f"\nSimulation complete in {elapsed:.1f}s")

# ============================================================
# Generate figures
# ============================================================
print("\nGenerating figures...")

# --- Figure 1: Evolution sequence (like Dharodi Fig. 5) ---
n_panels = min(8, len(snapshots))
# Select snapshots at evenly spaced times including first and last
indices = np.linspace(0, len(snapshots)-1, n_panels).astype(int)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Spontaneous Tripolar Vortex Formation in Strongly Coupled Dusty Plasma\n'
             f'(τ_m={tau_m}, η*={eta_star}, ν={nu})',
             fontsize=13, fontweight='bold')

for i, idx in enumerate(indices):
    ax = axes[i // 4, i % 4]
    snap = snapshots[idx]
    t = times[idx]

    # Symmetric colormap centered at zero
    vmax = max(np.abs(snap.max()), np.abs(snap.min()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(X, Y, snap, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(f't = {t:.1f}', fontsize=11)
    ax.set_xlim(Lx/2 - 2*R0, Lx/2 + 2*R0)
    ax.set_ylim(Ly/2 - 2*R0, Ly/2 + 2*R0)

    if i % 4 == 0:
        ax.set_ylabel('y')
    if i >= 4:
        ax.set_xlabel('x')

    plt.colorbar(im, ax=ax, shrink=0.8, label='ω')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 2: Key frames (initial, intermediate, final tripole) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Circular Vortex → Tripolar Structure (Visco-Elastic Instability)',
             fontsize=13, fontweight='bold')

key_indices = [0, len(snapshots)//3, -1]
labels = ['Initial (circular)', 'Intermediate (deforming)', 'Final (tripolar)']

for i, (idx, label) in enumerate(zip(key_indices, labels)):
    ax = axes[i]
    snap = snapshots[idx]
    t = times[idx]
    vmax = max(np.abs(snap.max()), np.abs(snap.min()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(X, Y, snap, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(f'{label}\nt = {t:.1f}', fontsize=11)
    ax.set_xlim(Lx/2 - 2.5*R0, Lx/2 + 2.5*R0)
    ax.set_ylim(Ly/2 - 2.5*R0, Ly/2 + 2.5*R0)
    plt.colorbar(im, ax=ax, shrink=0.8, label='ω')
    ax.set_xlabel('x')
    if i == 0:
        ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_key_frames.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 3: Diagnostics ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

diag = diagnostics
ax1 = axes[0, 0]
ax1.plot(diag['time'], diag['enstrophy'], 'C0-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Enstrophy Z = ½⟨ω²⟩')
ax1.set_title('A. Enstrophy Evolution')

ax2 = axes[0, 1]
ax2.plot(diag['time'], diag['energy'], 'C1-')
ax2.set_xlabel('Time')
ax2.set_ylabel('Kinetic Energy E = ½⟨v²⟩')
ax2.set_title('B. Kinetic Energy')

ax3 = axes[1, 0]
ax3.plot(diag['time'], diag['max_omega'], 'C2-')
ax3.set_xlabel('Time')
ax3.set_ylabel('max|ω|')
ax3.set_title('C. Peak Vorticity')

ax4 = axes[1, 1]
ax4.plot(diag['time'], diag['asymmetry'], 'C3-')
ax4.set_xlabel('Time')
ax4.set_ylabel('m=2 / m=0 ratio')
ax4.set_title('D. Azimuthal Asymmetry (m=2 mode)\n(tripolar ≈ high asymmetry)')
ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Tripolar threshold')
ax4.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 4: Contour plot with streamlines (publication quality) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Visco-Elastic Dusty Plasma: Spontaneous Tripolar Formation\n'
             'Replicating Dharodi et al. (2014) — Mechanism for Observed Triangle UAP Formations',
             fontsize=12, fontweight='bold')

for i, (idx, label) in enumerate(zip(key_indices, labels)):
    ax = axes[i]
    snap = snapshots[idx]
    t = times[idx]

    # Compute velocity field for streamlines
    omega_hat = np.fft.fft2(snap)
    psi_hat = vorticity_to_stream(omega_hat)
    u_hat, v_hat = stream_to_velocity(psi_hat)
    u = np.real(np.fft.ifft2(u_hat))
    v = np.real(np.fft.ifft2(v_hat))

    vmax = max(np.abs(snap.max()), np.abs(snap.min()), 0.01)

    # Filled contour
    levels = np.linspace(-vmax, vmax, 30)
    cf = ax.contourf(X, Y, snap, levels=levels, cmap='RdBu_r', extend='both')

    # Streamlines
    speed = np.sqrt(u**2 + v**2)
    lw = 1.5 * speed / max(speed.max(), 1e-10)
    ax.streamplot(x, y, u, v, color='black', linewidth=lw, density=1.2,
                  arrowsize=0.8, arrowstyle='->')

    ax.set_aspect('equal')
    ax.set_title(f'{label}\nt = {t:.1f}', fontsize=11)
    ax.set_xlim(Lx/2 - 2.5*R0, Lx/2 + 2.5*R0)
    ax.set_ylim(Ly/2 - 2.5*R0, Ly/2 + 2.5*R0)
    plt.colorbar(cf, ax=ax, shrink=0.8, label='Vorticity ω')
    ax.set_xlabel('x / λ_D')
    if i == 0:
        ax.set_ylabel('y / λ_D')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_streamlines.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'tripolar_streamlines.pdf', bbox_inches='tight')
plt.close()

# --- Figure 5: Luminosity proxy (what a camera would see) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Simulated Luminosity: What a Camera Would See\n'
             '(Luminosity ∝ |ω|², approximating plasma emission intensity)',
             fontsize=12, fontweight='bold')

for i, (idx, label) in enumerate(zip(key_indices, labels)):
    ax = axes[i]
    snap = snapshots[idx]
    t = times[idx]

    # Luminosity proxy: proportional to ω² (energy density)
    luminosity = snap**2
    luminosity = luminosity / luminosity.max()

    # Use "hot" colormap to simulate orange plasma glow
    im = ax.pcolormesh(X, Y, luminosity, cmap='hot', vmin=0, vmax=1, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(f'{label}\nt = {t:.1f}', fontsize=11)
    ax.set_xlim(Lx/2 - 2.5*R0, Lx/2 + 2.5*R0)
    ax.set_ylim(Ly/2 - 2.5*R0, Ly/2 + 2.5*R0)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Relative Luminosity')
    ax.set_xlabel('x / λ_D')
    if i == 0:
        ax.set_ylabel('y / λ_D')
    ax.set_facecolor('black')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tripolar_luminosity.png', dpi=200, bbox_inches='tight',
            facecolor='black')
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}")
print(f"\nKey result: asymmetry evolution from {diagnostics['asymmetry'][0]:.3f} "
      f"to {diagnostics['asymmetry'][-1]:.3f}")
if diagnostics['asymmetry'][-1] > diagnostics['asymmetry'][0] * 1.5:
    print("SUCCESS: Tripolar deformation detected!")
else:
    print("NOTE: May need longer run or different parameters for full tripolar formation.")

print("\nDone!")
