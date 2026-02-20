"""
Molecular Dynamics Simulation of Coulomb Crystallization
in Meteor-Ablated Nanoparticle Clouds

Models charged nanoparticles (Fe-Ni, silicate, MgO) from meteor ablation
interacting via screened Coulomb (Yukawa) potential in atmospheric conditions.
Demonstrates spontaneous Coulomb crystal formation at coupling parameter Γ > 170.

Uses overdamped Brownian dynamics (appropriate for atmospheric pressure where
Epstein drag >> dust plasma frequency). The equation of motion is:

  dx/dt = F/(m*β) + √(2D) * η(t)

where D = k_B T / (m*β) is the diffusion coefficient and η(t) is white noise.
This is exact in the limit β >> ω_pd, which holds at atmospheric pressure.

References:
  - Dharodi, Tiwari, Das (2014) arXiv:1406.5637 - GHD model, Yukawa MD
  - Schwabe et al. (2011) PRL 106, 215004 - Pattern formation
  - Thomas et al. (2020) PPCF 62, 014006 - MDPX experiments
  - Emelin et al. (2006) physics/0604115 - Dust-gas fireballs
  - Kostrov (2020) PPR 46, 443 - Cosmic dust parameters

Authors: Jim Galasyn & Claude (Anthropic)
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
E_CHARGE = 1.602e-19        # elementary charge [C]
EPSILON_0 = 8.854e-12       # vacuum permittivity [F/m]
K_BOLTZMANN = 1.381e-23     # Boltzmann constant [J/K]
COULOMB_K = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
class SimParams:
    """All tunable simulation parameters in one place."""

    # Particle properties (realistic meteor ablation products)
    N_particles = 2000           # number of dust grains
    grain_radius = 200e-9        # 200 nm radius [m] (Kostrov 2020: ~400nm)
    grain_density = 4000.0       # Fe-Ni/silicate mix [kg/m³]
    Z_dust = 600                 # charges per grain [e]
    # Z=600 on 200nm grain: surface potential ~4.3V, reasonable for
    # electron temperature ~1 eV in meteor ablation plasma

    # Plasma environment
    T_gas = 300.0                # gas temperature [K]
    debye_length = 80e-6         # Debye screening length [m]
    gas_density = 1.2            # air density [kg/m³]

    # Simulation domain
    box_size = 1.5e-3            # 1.5 mm cube [m]

    # Overdamped Brownian dynamics
    dt = 5.0e-5                  # timestep [s] (50 μs — safe for Brownian dynamics)
    n_steps = 20000              # total steps (= 1 second of physical time)
    output_interval = 200        # save diagnostics every N steps

    @property
    def grain_mass(self):
        return (4/3) * np.pi * self.grain_radius**3 * self.grain_density

    @property
    def grain_charge(self):
        return self.Z_dust * E_CHARGE

    @property
    def epstein_beta(self):
        """Epstein drag coefficient β [1/s]."""
        m_gas = 29 * 1.66e-27
        v_th = np.sqrt(8 * K_BOLTZMANN * self.T_gas / (np.pi * m_gas))
        return (4/3) * np.pi * self.grain_radius**2 * self.gas_density * v_th / self.grain_mass

    @property
    def mobility(self):
        """Particle mobility μ = 1/(m*β) [m/N/s]."""
        return 1.0 / (self.grain_mass * self.epstein_beta)

    @property
    def diffusion_coeff(self):
        """Brownian diffusion coefficient D = k_B T / (m*β) [m²/s]."""
        return K_BOLTZMANN * self.T_gas / (self.grain_mass * self.epstein_beta)

    @property
    def wigner_seitz_radius(self):
        n_d = self.N_particles / self.box_size**3
        return (3 / (4 * np.pi * n_d))**(1/3)

    @property
    def coupling_parameter(self):
        """Coulomb coupling Γ = (Z_d e)² / (4πε₀ a k_B T)."""
        a = self.wigner_seitz_radius
        return COULOMB_K * self.grain_charge**2 / (a * K_BOLTZMANN * self.T_gas)

    @property
    def kappa(self):
        """Screening parameter κ = a / λ_D."""
        return self.wigner_seitz_radius / self.debye_length

    @property
    def dust_plasma_freq(self):
        n_d = self.N_particles / self.box_size**3
        return np.sqrt(n_d * self.grain_charge**2 / (EPSILON_0 * self.grain_mass))


# ---------------------------------------------------------------------------
# Force computation
# ---------------------------------------------------------------------------
def compute_yukawa_forces(positions, params, cutoff_factor=8.0):
    """
    Compute Yukawa (screened Coulomb) forces using KD-tree neighbor search.
    F_ij = K q² exp(-r/λ) (1/r² + 1/(rλ)) r̂  (repulsive, same-sign charges)
    """
    N = len(positions)
    forces = np.zeros_like(positions)
    q = params.grain_charge
    lam = params.debye_length
    cutoff = cutoff_factor * lam

    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    if len(pairs) == 0:
        return forces

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    dr = positions[j_idx] - positions[i_idx]
    r = np.linalg.norm(dr, axis=1)
    mask = r > 1e-12
    dr, r = dr[mask], r[mask]
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    r_hat = dr / r[:, np.newaxis]
    exp_factor = np.exp(-r / lam)
    force_mag = COULOMB_K * q**2 * exp_factor * (1.0/r**2 + 1.0/(r * lam))
    f_vec = force_mag[:, np.newaxis] * r_hat

    np.add.at(forces, i_idx, -f_vec)  # repulsion
    np.add.at(forces, j_idx, f_vec)

    return forces


def compute_confinement_force(positions, params):
    """Soft parabolic confinement modeling the plasma sheath boundary."""
    R_cloud = params.box_size * 0.4
    k_conf = COULOMB_K * params.grain_charge**2 / params.debye_length**3

    center = np.array([params.box_size/2] * 3)
    dr = positions - center
    r = np.linalg.norm(dr, axis=1)

    forces = np.zeros_like(positions)
    outside = r > R_cloud
    if np.any(outside):
        r_hat = dr[outside] / r[outside, np.newaxis]
        overshoot = r[outside] - R_cloud
        forces[outside] = -k_conf * overshoot[:, np.newaxis] * r_hat

    return forces


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def pair_correlation(positions, params, n_bins=150, r_max=None):
    """Radial pair correlation function g(r). Peaks indicate ordering."""
    N = len(positions)
    a_ws = params.wigner_seitz_radius
    if r_max is None:
        r_max = 5 * a_ws

    tree = cKDTree(positions)
    pairs = tree.query_pairs(r_max, output_type='ndarray')
    if len(pairs) == 0:
        return np.linspace(0, r_max, n_bins), np.zeros(n_bins)

    dr = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    dists = np.linalg.norm(dr, axis=1)

    bins = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(dists, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dr_bin = bins[1] - bins[0]
    n_d = N / params.box_size**3
    shell_volume = 4 * np.pi * bin_centers**2 * dr_bin
    g_r = 2 * hist / (N * n_d * shell_volume + 1e-30)

    return bin_centers, g_r


def compute_coupling_live(positions, params):
    """Compute Γ from actual mean nearest-neighbor distance."""
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)
    mean_nn = np.mean(dists[:, 1])
    gamma = COULOMB_K * params.grain_charge**2 / (mean_nn * K_BOLTZMANN * params.T_gas)
    return gamma, mean_nn


def compute_psi6(positions):
    """Bond-orientational order parameter ψ₆ (2D projection)."""
    pos_2d = positions[:, :2]
    tree = cKDTree(pos_2d)
    N = len(positions)

    psi6_vals = []
    for i in range(min(N, 500)):  # sample for speed
        dists, indices = tree.query(pos_2d[i], k=7)
        neighbors = indices[1:]
        valid = dists[1:] > 1e-15
        if np.sum(valid) < 3:
            continue
        dr = pos_2d[neighbors[valid]] - pos_2d[i]
        angles = np.arctan2(dr[:, 1], dr[:, 0])
        psi6_vals.append(np.abs(np.mean(np.exp(6j * angles))))

    return np.mean(psi6_vals) if psi6_vals else 0.0


# ---------------------------------------------------------------------------
# Main simulation (overdamped Brownian dynamics)
# ---------------------------------------------------------------------------
def run_simulation(params=None, output_dir=None):
    if params is None:
        params = SimParams()

    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "coulomb_crystal"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = params.N_particles
    L = params.box_size
    dt = params.dt
    mu = params.mobility         # 1/(m*β)
    D = params.diffusion_coeff   # k_B T / (m*β)
    noise_amp = np.sqrt(2 * D * dt)

    print("=" * 65)
    print("COULOMB CRYSTAL BROWNIAN DYNAMICS SIMULATION")
    print("=" * 65)
    print(f"  Particles:          {N}")
    print(f"  Grain radius:       {params.grain_radius*1e9:.0f} nm")
    print(f"  Grain charge:       {params.Z_dust} e ({params.grain_charge:.2e} C)")
    print(f"  Grain mass:         {params.grain_mass:.2e} kg")
    print(f"  Debye length:       {params.debye_length*1e6:.0f} μm")
    print(f"  Box size:           {L*1e3:.2f} mm")
    print(f"  W-S radius:         {params.wigner_seitz_radius*1e6:.1f} μm")
    print(f"  Temperature:        {params.T_gas:.0f} K")
    print(f"  Epstein β:          {params.epstein_beta:.2e} s⁻¹")
    print(f"  Mobility:           {mu:.2e} m/(N·s)")
    print(f"  Diffusion coeff:    {D:.2e} m²/s")
    print(f"  Timestep:           {dt*1e6:.1f} μs")
    print(f"  Physical time:      {params.n_steps * dt * 1e3:.1f} ms")
    print(f"  Estimated Γ:        {params.coupling_parameter:.0f}")
    print(f"  Estimated κ:        {params.kappa:.2f}")
    print(f"  ω_pd:               {params.dust_plasma_freq:.0f} rad/s")
    print(f"  Regime:             {'CRYSTAL' if params.coupling_parameter > 170 else 'LIQUID' if params.coupling_parameter > 1 else 'GAS'}")
    print("=" * 65)

    # Initialize: uniform random in sphere
    center = np.array([L/2, L/2, L/2])
    R_init = L * 0.35
    positions = np.zeros((N, 3))
    count = 0
    while count < N:
        cand = center + (np.random.rand(3) - 0.5) * 2 * R_init
        if np.linalg.norm(cand - center) < R_init:
            positions[count] = cand
            count += 1

    # Tracking
    times, gammas, psi6_values = [], [], []
    snapshots = []

    print("\nRunning Brownian dynamics...")
    t_start = time.time()

    for step in range(params.n_steps):
        # Compute deterministic forces
        f_yukawa = compute_yukawa_forces(positions, params)
        f_confine = compute_confinement_force(positions, params)
        f_total = f_yukawa + f_confine

        # Overdamped update: dx = μ*F*dt + √(2D dt) * η
        positions += mu * f_total * dt + noise_amp * np.random.randn(N, 3)

        # Diagnostics
        if step % params.output_interval == 0:
            gamma, a_nn = compute_coupling_live(positions, params)
            psi6 = compute_psi6(positions)

            times.append(step * dt)
            gammas.append(gamma)
            psi6_values.append(psi6)

            elapsed = time.time() - t_start
            pct = (step + 1) / params.n_steps * 100
            regime = "CRYSTAL" if gamma > 170 else "LIQUID" if gamma > 1 else "GAS"
            print(f"  Step {step:6d}/{params.n_steps} ({pct:5.1f}%) | "
                  f"Γ={gamma:7.1f} [{regime:7s}] | ψ₆={psi6:.3f} | "
                  f"a_nn={a_nn*1e6:.1f} μm | {elapsed:.0f}s")

            if step % (params.output_interval * 5) == 0:
                snapshots.append((step * dt, positions.copy()))

    total_time = time.time() - t_start
    print(f"\nComplete in {total_time:.1f}s")

    # --- Save and plot ---
    np.save(output_dir / 'positions_final.npy', positions)

    results = {
        'N': N, 'Z_dust': params.Z_dust,
        'grain_radius_nm': params.grain_radius * 1e9,
        'T_gas': params.T_gas,
        'final_gamma': gammas[-1],
        'final_psi6': psi6_values[-1],
        'physical_time_ms': params.n_steps * dt * 1e3,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nGenerating figures...")

    # --- Figure 1: 3D crystal structure ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    pos_um = (positions - center) * 1e6
    ax.scatter(pos_um[:, 0], pos_um[:, 1], pos_um[:, 2],
               s=2, alpha=0.4, c='darkorange', edgecolors='none')
    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')
    ax.set_zlabel('z [μm]')
    ax.set_title(f'Coulomb Crystal: N={N}, Z={params.Z_dust}e, '
                 f'Γ={gammas[-1]:.0f}, ψ₆={psi6_values[-1]:.3f}')
    fig.savefig(output_dir / 'crystal_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: Diagnostics panel ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2D slice (XY)
    ax = axes[0, 0]
    z_center = L / 2
    slice_w = L * 0.03
    mask_z = np.abs(positions[:, 2] - z_center) < slice_w
    pos_s = (positions[mask_z] - center) * 1e6
    ax.scatter(pos_s[:, 0], pos_s[:, 1], s=8, c='darkorange',
               edgecolors='k', linewidths=0.3, alpha=0.8)
    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')
    ax.set_title(f'XY Slice ({np.sum(mask_z)} particles, Δz={2*slice_w*1e6:.0f} μm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # g(r)
    ax = axes[0, 1]
    r_bins, g_r = pair_correlation(positions, params)
    a_ws = params.wigner_seitz_radius
    ax.plot(r_bins / a_ws, g_r, 'b-', lw=1.5)
    ax.set_xlabel('r / a_WS')
    ax.set_ylabel('g(r)')
    ax.set_title('Pair Correlation Function')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', ls='--', alpha=0.5)
    # Mark expected BCC/FCC peak positions
    for peak_r, label in [(1.0, '1st'), (1.73, '2nd'), (2.0, '3rd')]:
        ax.axvline(x=peak_r, color='red', ls=':', alpha=0.3)

    # Γ evolution
    ax = axes[1, 0]
    ax.plot(np.array(times) * 1e3, gammas, 'r-', lw=2)
    ax.axhline(y=170, color='gray', ls='--', lw=1.5, label='Γ_melt = 170')
    ax.axhline(y=1, color='lightgray', ls=':', label='Γ = 1 (ideal gas)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Coupling Parameter Γ')
    ax.set_title('Crystallization Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ψ₆ evolution
    ax = axes[1, 1]
    ax.plot(np.array(times) * 1e3, psi6_values, 'g-', lw=2)
    ax.axhline(y=0.45, color='gray', ls='--', alpha=0.7, label='Crystal threshold')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('ψ₆ (hexagonal order)')
    ax.set_title('Order Parameter')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Coulomb Crystallization of Meteor-Ablated Nanoparticles',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 3: Snapshots at different times ---
    if len(snapshots) >= 4:
        n_snap = min(4, len(snapshots))
        indices = np.linspace(0, len(snapshots)-1, n_snap, dtype=int)
        fig, axes = plt.subplots(1, n_snap, figsize=(5*n_snap, 5))
        if n_snap == 1:
            axes = [axes]

        for k, idx in enumerate(indices):
            t_snap, pos_snap = snapshots[idx]
            ax = axes[k]
            mask = np.abs(pos_snap[:, 2] - L/2) < L * 0.03
            ps = (pos_snap[mask] - center) * 1e6
            ax.scatter(ps[:, 0], ps[:, 1], s=8, c='darkorange',
                       edgecolors='k', linewidths=0.3)
            ax.set_title(f't = {t_snap*1e3:.1f} ms')
            ax.set_xlabel('x [μm]')
            if k == 0:
                ax.set_ylabel('y [μm]')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        fig.suptitle('Crystal Formation Over Time (XY slices)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(output_dir / 'snapshots.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nFigures saved to {output_dir}")
    print(f"Final: Γ = {gammas[-1]:.0f}, ψ₆ = {psi6_values[-1]:.3f}")

    return positions, results


if __name__ == '__main__':
    run_simulation()
