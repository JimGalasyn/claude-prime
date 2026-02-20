"""
Multi-Ball Plasma Formation Dynamics

Simulates the interaction of multiple plasma fireballs (each modeled as a
charged macro-particle) embedded in a shared lower-density dusty plasma
"dark field." Demonstrates:

  1. Spontaneous arrangement into equilateral triangle (3 balls) or
     hexagonal lattice (6-7 balls) — the 2D Wigner crystal ground state
  2. Energy sharing through the dark field coupling
  3. Node dimming and revival dynamics (coupled oscillator behavior)

The dark field is modeled as a coupling medium with finite conductivity
that allows energy (charge, thermal) to flow between nodes.

Coupled oscillator equations:
  dE_i/dt = G_i(E_i) - L_i(E_i) + Σ_j κ_ij (E_j - E_i)

where E_i = energy in node i, G_i = chemical generation (oxidation),
L_i = radiation loss, κ_ij = coupling through dark field.

References:
  - Vranjes (1999) Phys. Lett. A 258, 317 — Tripolar vortices in dusty plasma
  - Chen et al. (2003) Planet. Space Sci. 51, 81 — Vortex triplets
  - Dharodi et al. (2014) arXiv:1406.5637 — Visco-elastic coherent structures

Authors: Jim Galasyn & Claude (Anthropic)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Physical constants and parameters
# ---------------------------------------------------------------------------
E_CHARGE = 1.602e-19
EPSILON_0 = 8.854e-12
K_BOLTZMANN = 1.381e-23
COULOMB_K = 1 / (4 * np.pi * EPSILON_0)


class FormationParams:
    """Parameters for the multi-ball formation simulation."""

    n_balls = 3                 # number of plasma balls (3=triangle, 7=hex)

    # Ball properties (each ball is a Coulomb crystal ~1m diameter)
    ball_mass = 0.5             # effective mass [kg] (air + dust)
    ball_charge = 1e-4          # effective net charge [C]
    ball_radius = 0.5           # radius of each ball [m]

    # Confinement (atmospheric stratification)
    confinement_k = 1e-4        # harmonic confinement strength [N/m]
    # Models atmospheric stratification trapping balls at a specific altitude

    # Drag (air resistance)
    drag_coeff = 0.5            # linear drag coefficient [kg/s]

    # Energy dynamics
    E_chem_max = 50.0           # max chemical energy per ball [J]
    oxidation_rate = 0.5        # chemical power generation [W] (at full energy)
    radiation_rate = 0.3        # radiation loss rate [W] (at luminosity threshold)
    luminosity_threshold = 5.0  # energy below which ball is "dark" [J]

    # Dark field coupling
    coupling_strength = 0.05    # energy coupling rate through dark field [1/s]
    coupling_range = 50.0       # coupling range [m] (dark field extent)

    # Simulation
    t_max = 300.0               # simulation duration [s]
    dt_output = 0.1             # output interval [s]

    # Perturbation (for dimming demonstration)
    perturb_ball = 0            # which ball to perturb
    perturb_time = 100.0        # when to perturb [s]
    perturb_amount = 0.8        # fraction of energy removed


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------
def formation_rhs(t, state, params):
    """
    Right-hand side of the coupled ODE system.

    State vector: [x0, y0, vx0, vy0, E0, x1, y1, vx1, vy1, E1, ...]
    where (xi, yi) = position, (vxi, vyi) = velocity, Ei = internal energy
    of ball i.
    """
    n = params.n_balls
    dstate = np.zeros_like(state)

    # Unpack state
    x = state[0::5]     # x positions
    y = state[1::5]     # y positions
    vx = state[2::5]    # x velocities
    vy = state[3::5]    # y velocities
    E = state[4::5]     # internal energies

    # Forces on each ball
    fx = np.zeros(n)
    fy = np.zeros(n)

    for i in range(n):
        # 1. Confinement force (harmonic, toward center)
        fx[i] -= params.confinement_k * x[i]
        fy[i] -= params.confinement_k * y[i]

        # 2. Inter-ball Coulomb repulsion (screened)
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)
            r = max(r, params.ball_radius)  # prevent singularity

            # Screened Coulomb: F = k q² / r² * exp(-r/λ)
            screening = np.exp(-r / params.coupling_range)
            f_mag = COULOMB_K * params.ball_charge**2 / r**2 * screening
            fx[i] += f_mag * dx / r
            fy[i] += f_mag * dy / r

        # 3. Drag force
        fx[i] -= params.drag_coeff * vx[i]
        fy[i] -= params.drag_coeff * vy[i]

    # Energy dynamics for each ball
    dE = np.zeros(n)
    for i in range(n):
        # Chemical generation (oxidation): saturates at E_chem_max
        G_i = params.oxidation_rate * (1.0 - E[i] / params.E_chem_max)
        G_i = max(G_i, 0)

        # Radiation loss: proportional to E above threshold
        if E[i] > params.luminosity_threshold:
            L_i = params.radiation_rate * (E[i] / params.E_chem_max)
        else:
            L_i = params.radiation_rate * 0.1 * (E[i] / params.E_chem_max)

        # Dark field coupling: energy flows from high-E to low-E nodes
        coupling = 0.0
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)
            # Coupling decreases with distance
            kappa = params.coupling_strength * np.exp(-r / params.coupling_range)
            coupling += kappa * (E[j] - E[i])

        dE[i] = G_i - L_i + coupling

    # Pack derivatives
    dstate[0::5] = vx
    dstate[1::5] = vy
    dstate[2::5] = fx / params.ball_mass
    dstate[3::5] = fy / params.ball_mass
    dstate[4::5] = dE

    return dstate


def apply_perturbation(t, state, params):
    """Energy perturbation event: drain energy from one ball."""
    idx = params.perturb_ball
    state[idx * 5 + 4] *= (1.0 - params.perturb_amount)
    return state


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def initialize_state(params):
    """Initialize ball positions, velocities, and energies."""
    n = params.n_balls
    state = np.zeros(n * 5)

    if n == 3:
        # Start near equilateral triangle but slightly perturbed
        R = 15.0  # initial radius [m]
        for i in range(3):
            angle = 2 * np.pi * i / 3 + np.random.randn() * 0.3
            r_i = R * (1 + np.random.randn() * 0.2)
            state[i*5 + 0] = r_i * np.cos(angle)
            state[i*5 + 1] = r_i * np.sin(angle)
    elif n == 7:
        # Start with 1 center + 6 in a ring, slightly perturbed
        # Center ball
        state[0] = np.random.randn() * 2
        state[1] = np.random.randn() * 2
        R = 20.0
        for i in range(1, 7):
            angle = 2 * np.pi * (i-1) / 6 + np.random.randn() * 0.2
            r_i = R * (1 + np.random.randn() * 0.15)
            state[i*5 + 0] = r_i * np.cos(angle)
            state[i*5 + 1] = r_i * np.sin(angle)
    else:
        # Generic: random placement
        for i in range(n):
            state[i*5 + 0] = np.random.randn() * 20
            state[i*5 + 1] = np.random.randn() * 20

    # Small random velocities
    for i in range(n):
        state[i*5 + 2] = np.random.randn() * 0.1
        state[i*5 + 3] = np.random.randn() * 0.1

    # Initial energies: near equilibrium
    for i in range(n):
        state[i*5 + 4] = params.E_chem_max * (0.7 + np.random.rand() * 0.2)

    return state


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation(params=None, output_dir=None):
    """Run the multi-ball formation simulation."""

    if params is None:
        params = FormationParams()

    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "multi_ball"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-BALL PLASMA FORMATION DYNAMICS")
    print("=" * 60)
    print(f"  Number of balls:     {params.n_balls}")
    print(f"  Ball charge:         {params.ball_charge*1e6:.0f} μC")
    print(f"  Confinement:         {params.confinement_k:.1e} N/m")
    print(f"  Coupling strength:   {params.coupling_strength:.3f} s⁻¹")
    print(f"  Perturbation at:     t={params.perturb_time:.0f} s")
    print(f"  Simulation time:     {params.t_max:.0f} s")
    print("=" * 60)

    # Initialize
    state0 = initialize_state(params)

    # Integration timepoints
    t_eval = np.arange(0, params.t_max, params.dt_output)

    # Phase 1: integrate up to perturbation
    print("\nPhase 1: Self-organization...")
    sol1 = solve_ivp(
        formation_rhs, [0, params.perturb_time], state0,
        args=(params,), method='RK45', t_eval=t_eval[t_eval <= params.perturb_time],
        rtol=1e-8, atol=1e-10, max_step=0.5
    )

    # Apply perturbation
    state_perturbed = sol1.y[:, -1].copy()
    state_perturbed = apply_perturbation(params.perturb_time, state_perturbed, params)

    print(f"  Perturbation applied: ball {params.perturb_ball} energy "
          f"{sol1.y[params.perturb_ball*5+4, -1]:.1f} → "
          f"{state_perturbed[params.perturb_ball*5+4]:.1f} J")

    # Phase 2: integrate after perturbation
    print("Phase 2: Recovery dynamics...")
    sol2 = solve_ivp(
        formation_rhs, [params.perturb_time, params.t_max], state_perturbed,
        args=(params,), method='RK45',
        t_eval=t_eval[t_eval > params.perturb_time],
        rtol=1e-8, atol=1e-10, max_step=0.5
    )

    # Combine solutions
    t_all = np.concatenate([sol1.t, sol2.t])
    y_all = np.concatenate([sol1.y, sol2.y], axis=1)

    print(f"\nSimulation complete: {len(t_all)} timepoints")

    # --- Extract trajectories ---
    n = params.n_balls
    trajectories = {}
    for i in range(n):
        trajectories[i] = {
            'x': y_all[i*5 + 0, :],
            'y': y_all[i*5 + 1, :],
            'vx': y_all[i*5 + 2, :],
            'vy': y_all[i*5 + 3, :],
            'E': y_all[i*5 + 4, :],
        }

    # --- Generate figures ---
    print("\nGenerating figures...")

    colors = ['#FF6B00', '#FF8C00', '#FFA500', '#FFB347',
              '#FF4500', '#FF7F50', '#FFD700']  # orange shades

    # Figure 1: Trajectory plot showing self-organization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Early state
    ax = axes[0]
    t_early = min(5.0, params.perturb_time * 0.1)
    idx_early = np.argmin(np.abs(t_all - t_early))
    for i in range(n):
        ax.plot(trajectories[i]['x'][:idx_early],
                trajectories[i]['y'][:idx_early],
                '-', color=colors[i % len(colors)], alpha=0.5, lw=0.5)
        ax.scatter(trajectories[i]['x'][idx_early],
                   trajectories[i]['y'][idx_early],
                   s=200, c=colors[i % len(colors)], edgecolors='k',
                   zorder=5)
    ax.set_title(f'Early: t = {t_early:.0f} s')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Equilibrium (just before perturbation)
    ax = axes[1]
    idx_eq = np.argmin(np.abs(t_all - params.perturb_time * 0.95))
    for i in range(n):
        ax.plot(trajectories[i]['x'][:idx_eq],
                trajectories[i]['y'][:idx_eq],
                '-', color=colors[i % len(colors)], alpha=0.3, lw=0.5)
        E_i = trajectories[i]['E'][idx_eq]
        alpha = min(1.0, E_i / params.luminosity_threshold)
        size = 200 + 300 * (E_i / params.E_chem_max)
        ax.scatter(trajectories[i]['x'][idx_eq],
                   trajectories[i]['y'][idx_eq],
                   s=size, c=colors[i % len(colors)], edgecolors='k',
                   alpha=alpha, zorder=5)
    ax.set_title(f'Equilibrium: t = {params.perturb_time*0.95:.0f} s')
    ax.set_xlabel('x [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw triangle/polygon connecting balls
    eq_x = [trajectories[i]['x'][idx_eq] for i in range(n)]
    eq_y = [trajectories[i]['y'][idx_eq] for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            ax.plot([eq_x[i], eq_x[j]], [eq_y[i], eq_y[j]],
                    'k--', alpha=0.2, lw=1)

    # Final state (after recovery)
    ax = axes[2]
    idx_final = -1
    for i in range(n):
        ax.plot(trajectories[i]['x'][idx_eq:],
                trajectories[i]['y'][idx_eq:],
                '-', color=colors[i % len(colors)], alpha=0.3, lw=0.5)
        E_i = trajectories[i]['E'][idx_final]
        alpha = min(1.0, E_i / params.luminosity_threshold)
        size = 200 + 300 * (E_i / params.E_chem_max)
        ax.scatter(trajectories[i]['x'][idx_final],
                   trajectories[i]['y'][idx_final],
                   s=size, c=colors[i % len(colors)], edgecolors='k',
                   alpha=alpha, zorder=5)
    ax.set_title(f'After Recovery: t = {t_all[-1]:.0f} s')
    ax.set_xlabel('x [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Plasma Ball Formation: {n}-Ball Wigner Crystal',
                 fontsize=14)
    fig.savefig(output_dir / 'formation_trajectory.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Energy dynamics (the key figure for dimming/revival)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(n):
        ax1.plot(t_all, trajectories[i]['E'], '-', color=colors[i % len(colors)],
                 lw=2, label=f'Ball {i}')
    ax1.axhline(y=params.luminosity_threshold, color='gray', ls='--',
                alpha=0.5, label='Luminosity threshold')
    ax1.axvline(x=params.perturb_time, color='red', ls=':', alpha=0.5,
                label='Perturbation')
    ax1.set_ylabel('Internal Energy [J]')
    ax1.set_title('Energy Dynamics: Dimming and Revival through Dark Field Coupling')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Luminosity (visual brightness proportional to E above threshold)
    for i in range(n):
        luminosity = np.maximum(0, trajectories[i]['E'] - params.luminosity_threshold)
        luminosity /= (params.E_chem_max - params.luminosity_threshold)
        ax2.plot(t_all, luminosity, '-', color=colors[i % len(colors)],
                 lw=2, label=f'Ball {i}')
    ax2.axvline(x=params.perturb_time, color='red', ls=':', alpha=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Relative Luminosity')
    ax2.set_title('Observable Brightness')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)

    fig.savefig(output_dir / 'energy_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 3: Inter-ball distances over time
    fig, ax = plt.subplots(figsize=(12, 5))
    pair_labels = []
    for i in range(n):
        for j in range(i+1, n):
            dx = trajectories[i]['x'] - trajectories[j]['x']
            dy = trajectories[i]['y'] - trajectories[j]['y']
            dist = np.sqrt(dx**2 + dy**2)
            ax.plot(t_all, dist, '-', lw=1.5, label=f'd({i},{j})')
            pair_labels.append(f'{i}-{j}')

    ax.axvline(x=params.perturb_time, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Inter-ball Distance [m]')
    ax.set_title('Formation Geometry: Convergence to Equilateral Configuration')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'distances.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: Snapshots showing dimming sequence (filmstrip)
    n_frames = 8
    t_frames = np.linspace(params.perturb_time - 10,
                           min(params.perturb_time + 60, params.t_max - 1),
                           n_frames)

    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))

    for k, t_frame in enumerate(t_frames):
        ax = axes[k]
        idx = np.argmin(np.abs(t_all - t_frame))

        ax.set_facecolor('black')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

        for i in range(n):
            xi = trajectories[i]['x'][idx]
            yi = trajectories[i]['y'][idx]
            Ei = trajectories[i]['E'][idx]

            # Brightness based on energy
            brightness = max(0, min(1, (Ei - params.luminosity_threshold * 0.5) /
                                    (params.E_chem_max * 0.5)))

            if brightness > 0.05:
                # Glow effect: concentric circles with decreasing alpha
                for r_mult, alpha_mult in [(3.0, 0.1), (2.0, 0.2), (1.0, 0.6)]:
                    circle = Circle((xi, yi),
                                    params.ball_radius * r_mult,
                                    facecolor=(1, 0.5, 0, brightness * alpha_mult),
                                    edgecolor='none')
                    ax.add_patch(circle)

                # Core
                circle = Circle((xi, yi), params.ball_radius * 0.5,
                                facecolor=(1, 0.8, 0.3, brightness),
                                edgecolor='none')
                ax.add_patch(circle)

        # Dark field lines (faint connections)
        for i in range(n):
            for j in range(i+1, n):
                xi = trajectories[i]['x'][idx]
                yi = trajectories[i]['y'][idx]
                xj = trajectories[j]['x'][idx]
                yj = trajectories[j]['y'][idx]
                ax.plot([xi, xj], [yi, yj], '-', color=(0.3, 0.2, 0.1, 0.15), lw=1)

        ax.set_aspect('equal')
        ax.set_title(f't={t_frame:.0f}s', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('white')

    fig.patch.set_facecolor('black')
    fig.suptitle('Dimming and Revival Sequence', color='white', fontsize=14)
    fig.savefig(output_dir / 'dimming_sequence.png', dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.close(fig)

    print(f"Figures saved to {output_dir}")

    # Compute final geometry metrics
    final_dists = []
    for i in range(n):
        for j in range(i+1, n):
            dx = trajectories[i]['x'][-1] - trajectories[j]['x'][-1]
            dy = trajectories[i]['y'][-1] - trajectories[j]['y'][-1]
            final_dists.append(np.sqrt(dx**2 + dy**2))

    print(f"\nFinal inter-ball distances: {[f'{d:.2f} m' for d in final_dists]}")
    if n == 3 and len(final_dists) == 3:
        std_ratio = np.std(final_dists) / np.mean(final_dists)
        print(f"Distance uniformity: σ/μ = {std_ratio:.4f} "
              f"({'equilateral' if std_ratio < 0.05 else 'not equilateral'})")

    results = {
        'n_balls': n,
        'final_distances': [float(d) for d in final_dists],
        'final_energies': [float(trajectories[i]['E'][-1]) for i in range(n)],
        'distance_uniformity': float(np.std(final_dists) / np.mean(final_dists))
                               if final_dists else 0,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return trajectories, results


# ---------------------------------------------------------------------------
# Run both 3-ball and 7-ball configurations
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 3-ball: equilateral triangle
    print("\n" + "=" * 60)
    print("CONFIGURATION 1: THREE BALLS (TRIANGLE)")
    print("=" * 60)
    params3 = FormationParams()
    params3.n_balls = 3
    run_simulation(params3)

    # 7-ball: hexagonal
    print("\n" + "=" * 60)
    print("CONFIGURATION 2: SEVEN BALLS (HEXAGONAL)")
    print("=" * 60)
    params7 = FormationParams()
    params7.n_balls = 7
    params7.perturb_ball = 3  # perturb an outer ball
    run_simulation(params7,
                   output_dir=Path(__file__).parent / "output" / "multi_ball_hex")
