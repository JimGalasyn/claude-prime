#!/usr/bin/env python3
"""
Williamson Nonlinear Pivot Field Simulation
=============================================

Extends the linear pivot equations with nonlinear self-interaction terms
to test whether nonlinearity enables true self-confinement (solitons).

Linear pivot equations (from williamson_pivot.py):
    ∂B/∂t = -∇×E
    ∂E/∂t = ∇×B - ∇P
    ∂P/∂t = -∇·E

Problem: linear equations can't self-confine. Energy always disperses.

Nonlinear models tested here:

  Model A - "Saturating pivot":
    ∂P/∂t = -∇·E / (1 + α|P|²)
    Motivation: P growth self-limits. Large P resists further change.

  Model B - "Born-Infeld style":
    ∂E/∂t = ∇×B - ∇P / (1 + β(E²+B²+P²)/E_max²)
    Motivation: field strength has a maximum (like Born-Infeld electrodynamics).

  Model C - "Nonlinear wave equation" (φ⁴ model):
    ∂P/∂t = -∇·E - λP(P² - v²)
    Motivation: Mexican hat potential for P creates stable finite-amplitude states.
    v² > 0 gives spontaneous symmetry breaking → topological solitons.

  Model D - "Cubic self-coupling":
    ∂P/∂t = -∇·E - γP³
    Motivation: simplest nonlinearity that prevents runaway. P is attracted to zero
    but can sustain oscillations in a potential well.

  Model E - "Nonlinear polarization" (Kerr effect):
    ∂E/∂t = ∇×B - ∇P - χ|E|²E
    Motivation: intensity-dependent refractive index. In optics, this creates
    spatial solitons (self-focusing balances diffraction).

2D TE mode: Ex, Ey, Bz, P on Yee grid.

Usage:
    python3 williamson_nonlinear.py --mode sweep --steps 500 --save
    python3 williamson_nonlinear.py --mode scan --model C --steps 1000 --save
    python3 williamson_nonlinear.py --mode animate --model C --steps 500
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


class NonlinearPivotSim:
    """2D FDTD with nonlinear pivot field extensions."""

    def __init__(self, nx=200, ny=200, dx=1.0, courant=0.5,
                 model='linear', params=None):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.dt = courant * dx
        self.courant = courant
        self.model = model
        self.params = params or {}

        # Fields: TE mode (Ex, Ey, Bz, P)
        self.Ex = np.zeros((nx, ny))
        self.Ey = np.zeros((nx, ny))
        self.Bz = np.zeros((nx, ny))
        self.P = np.zeros((nx, ny))

        self.time = 0.0
        self.n_step = 0
        self.energy_history = []
        self.confined_history = []
        self.p_energy_history = []

    def init_vortex(self, cx=None, cy=None, radius=15.0, amplitude=1.0,
                    winding=1):
        """Initialize a vortex-like EM configuration.

        Creates a circulating E field with B through the center,
        resembling the 2D cross-section of a toroidal photon.
        Winding number = topological charge.
        """
        if cx is None:
            cx = self.nx // 2
        if cy is None:
            cy = self.ny // 2

        x = np.arange(self.nx) - cx
        y = np.arange(self.ny) - cy
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2) + 1e-10
        theta = np.arctan2(Y, X)

        # Radial profile: Gaussian ring at r=radius
        profile = np.exp(-0.5 * ((R - radius) / (radius * 0.3))**2)

        # Circulating E field (tangential)
        self.Ex = -amplitude * np.sin(winding * theta) * profile
        self.Ey = amplitude * np.cos(winding * theta) * profile

        # B field through center (axial, proportional to curl of E)
        self.Bz = amplitude * profile * np.exp(-0.5 * (R / radius)**2)

        # Seed P from div(E) for models that need it
        if self.model in ('C', 'phi4'):
            v = self.params.get('v', 0.5)
            # Seed P at the vacuum value with a localized perturbation
            self.P = v * np.tanh(profile * 2)

    def init_gaussian_pulse(self, cx=None, cy=None, width=10.0, amplitude=1.0):
        """Simple Gaussian pulse for comparison."""
        if cx is None:
            cx = self.nx // 2
        if cy is None:
            cy = self.ny // 2

        x = np.arange(self.nx) - cx
        y = np.arange(self.ny) - cy
        X, Y = np.meshgrid(x, y, indexing='ij')
        R2 = X**2 + Y**2

        self.Ex = amplitude * np.exp(-R2 / (2 * width**2))
        self.Ey = np.zeros_like(self.Ex)
        self.Bz = np.zeros_like(self.Ex)
        self.P = np.zeros_like(self.Ex)

    def _update_B(self):
        """Faraday: ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)"""
        dEy_dx = (np.roll(self.Ey, -1, axis=0) - self.Ey) / self.dx
        dEx_dy = (np.roll(self.Ex, -1, axis=1) - self.Ex) / self.dy
        self.Bz -= self.dt * (dEy_dx - dEx_dy)

    def _update_E_linear(self):
        """Ampere + pivot: ∂E/∂t = ∇×B - ∇P"""
        dBz_dy = (self.Bz - np.roll(self.Bz, 1, axis=1)) / self.dy
        dBz_dx = (self.Bz - np.roll(self.Bz, 1, axis=0)) / self.dx
        dP_dx = (np.roll(self.P, -1, axis=0) - self.P) / self.dx
        dP_dy = (np.roll(self.P, -1, axis=1) - self.P) / self.dy

        self.Ex += self.dt * (dBz_dy - dP_dx)
        self.Ey += self.dt * (-dBz_dx - dP_dy)

    def _compute_div_E(self):
        """Compute ∇·E"""
        dEx_dx = (self.Ex - np.roll(self.Ex, 1, axis=0)) / self.dx
        dEy_dy = (self.Ey - np.roll(self.Ey, 1, axis=1)) / self.dy
        return dEx_dx + dEy_dy

    def _update_P_linear(self):
        """∂P/∂t = -∇·E"""
        div_E = self._compute_div_E()
        self.P -= self.dt * div_E

    def _step_model_A(self):
        """Saturating pivot: ∂P/∂t = -∇·E / (1 + α|P|²)"""
        alpha = self.params.get('alpha', 1.0)

        self._update_B()
        self._update_E_linear()

        div_E = self._compute_div_E()
        saturation = 1.0 / (1.0 + alpha * self.P**2)
        self.P -= self.dt * div_E * saturation

    def _step_model_B(self):
        """Born-Infeld: field strength maximum."""
        beta = self.params.get('beta', 1.0)
        E_max = self.params.get('E_max', 2.0)

        self._update_B()

        # Modified Ampere with Born-Infeld denominator
        E2 = self.Ex**2 + self.Ey**2
        B2 = self.Bz**2
        P2 = self.P**2
        bi_factor = 1.0 / (1.0 + beta * (E2 + B2 + P2) / E_max**2)

        dBz_dy = (self.Bz - np.roll(self.Bz, 1, axis=1)) / self.dy
        dBz_dx = (self.Bz - np.roll(self.Bz, 1, axis=0)) / self.dx
        dP_dx = (np.roll(self.P, -1, axis=0) - self.P) / self.dx
        dP_dy = (np.roll(self.P, -1, axis=1) - self.P) / self.dy

        self.Ex += self.dt * (dBz_dy - dP_dx) * bi_factor
        self.Ey += self.dt * (-dBz_dx - dP_dy) * bi_factor

        self._update_P_linear()

    def _step_model_C(self):
        """φ⁴ model: ∂P/∂t = -∇·E - λP(P² - v²)"""
        lam = self.params.get('lambda', 1.0)
        v = self.params.get('v', 0.5)

        self._update_B()
        self._update_E_linear()

        div_E = self._compute_div_E()
        self.P -= self.dt * (div_E + lam * self.P * (self.P**2 - v**2))

    def _step_model_D(self):
        """Cubic damping: ∂P/∂t = -∇·E - γP³"""
        gamma = self.params.get('gamma', 0.5)

        self._update_B()
        self._update_E_linear()

        div_E = self._compute_div_E()
        self.P -= self.dt * (div_E + gamma * self.P**3)

    def _step_model_E(self):
        """Kerr nonlinearity: ∂E/∂t = ∇×B - ∇P - χ|E|²E"""
        chi = self.params.get('chi', 0.1)

        self._update_B()

        dBz_dy = (self.Bz - np.roll(self.Bz, 1, axis=1)) / self.dy
        dBz_dx = (self.Bz - np.roll(self.Bz, 1, axis=0)) / self.dx
        dP_dx = (np.roll(self.P, -1, axis=0) - self.P) / self.dx
        dP_dy = (np.roll(self.P, -1, axis=1) - self.P) / self.dy

        E2 = self.Ex**2 + self.Ey**2
        self.Ex += self.dt * (dBz_dy - dP_dx - chi * E2 * self.Ex)
        self.Ey += self.dt * (-dBz_dx - dP_dy - chi * E2 * self.Ey)

        self._update_P_linear()

    def _step_model_F(self):
        """φ⁴ + Kerr combined: topological + self-focusing."""
        lam = self.params.get('lambda', 1.0)
        v = self.params.get('v', 0.5)
        chi = self.params.get('chi', 0.1)

        self._update_B()

        # Kerr-modified E update
        dBz_dy = (self.Bz - np.roll(self.Bz, 1, axis=1)) / self.dy
        dBz_dx = (self.Bz - np.roll(self.Bz, 1, axis=0)) / self.dx
        dP_dx = (np.roll(self.P, -1, axis=0) - self.P) / self.dx
        dP_dy = (np.roll(self.P, -1, axis=1) - self.P) / self.dy

        E2 = self.Ex**2 + self.Ey**2
        self.Ex += self.dt * (dBz_dy - dP_dx - chi * E2 * self.Ex)
        self.Ey += self.dt * (-dBz_dx - dP_dy - chi * E2 * self.Ey)

        # φ⁴ P update
        div_E = self._compute_div_E()
        self.P -= self.dt * (div_E + lam * self.P * (self.P**2 - v**2))

    def step(self):
        """Advance one timestep using the selected model."""
        dispatch = {
            'linear': self._step_linear,
            'A': self._step_model_A,
            'B': self._step_model_B,
            'C': self._step_model_C,
            'D': self._step_model_D,
            'E': self._step_model_E,
            'F': self._step_model_F,
        }
        dispatch[self.model]()
        self.time += self.dt
        self.n_step += 1

    def _step_linear(self):
        self._update_B()
        self._update_E_linear()
        self._update_P_linear()

    def compute_energy_density(self):
        """½(E² + B² + P²)"""
        return 0.5 * (self.Ex**2 + self.Ey**2 + self.Bz**2 + self.P**2)

    def compute_diagnostics(self, conf_radius=None):
        """Compute total energy, confined energy fraction, P energy."""
        u = self.compute_energy_density()
        total = np.sum(u) * self.dx * self.dy

        # P contribution
        p_energy = 0.5 * np.sum(self.P**2) * self.dx * self.dy

        # Confined fraction within radius of center
        if conf_radius is None:
            conf_radius = min(self.nx, self.ny) * 0.25
        cx, cy = self.nx // 2, self.ny // 2
        x = np.arange(self.nx) - cx
        y = np.arange(self.ny) - cy
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        mask = R <= conf_radius
        confined = np.sum(u[mask]) * self.dx * self.dy

        frac = confined / total if total > 1e-15 else 0.0

        self.energy_history.append(total)
        self.confined_history.append(frac)
        self.p_energy_history.append(p_energy)

        return total, frac, p_energy

    def apply_absorbing_bc(self, width=15):
        """Damping layer at boundaries."""
        for field in [self.Ex, self.Ey, self.Bz, self.P]:
            for d in range(2):
                for side in [0, 1]:
                    slices = [slice(None)] * 2
                    if side == 0:
                        slices[d] = slice(0, width)
                    else:
                        slices[d] = slice(-width, None)
                    region = field[tuple(slices)]
                    n = region.shape[d]
                    ramp = np.linspace(0.8, 1.0, n)
                    shape = [1, 1]
                    shape[d] = n
                    ramp = ramp.reshape(shape)
                    if side == 1:
                        ramp = ramp.flip(d) if hasattr(ramp, 'flip') else np.flip(ramp, d)
                    field[tuple(slices)] *= ramp

    def run(self, n_steps, diag_interval=5, absorbing=True):
        """Run simulation for n_steps."""
        for i in range(n_steps):
            self.step()
            if absorbing:
                self.apply_absorbing_bc()
            if i % diag_interval == 0:
                self.compute_diagnostics()
        return self


def run_model_sweep(n_steps=500, save=False):
    """Compare all nonlinear models head-to-head."""
    models = {
        'Linear': ('linear', {}),
        'A: Saturating P\n(α=1.0)': ('A', {'alpha': 1.0}),
        'B: Born-Infeld\n(β=1, E_max=2)': ('B', {'beta': 1.0, 'E_max': 2.0}),
        'C: φ⁴ potential\n(λ=1, v=0.5)': ('C', {'lambda': 1.0, 'v': 0.5}),
        'D: Cubic P³\n(γ=0.5)': ('D', {'gamma': 0.5}),
        'E: Kerr |E|²E\n(χ=0.1)': ('E', {'chi': 0.1}),
    }

    results = {}
    for label, (model, params) in models.items():
        print(f"  Running {label.split(chr(10))[0]}...", end=" ", flush=True)
        sim = NonlinearPivotSim(nx=200, ny=200, model=model, params=params)
        sim.init_vortex(radius=20.0, amplitude=1.0, winding=1)
        sim.run(n_steps, diag_interval=5)
        results[label] = sim
        final_conf = sim.confined_history[-1] if sim.confined_history else 0
        print(f"confined={final_conf:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Nonlinear Pivot Models: Confinement Comparison ({n_steps} steps)\n"
        "Vortex initial condition, absorbing BCs",
        fontsize=14, fontweight='bold'
    )

    for idx, (label, sim) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]
        t = np.arange(len(sim.confined_history)) * 5 * sim.dt

        ax.plot(t, sim.confined_history, 'b-', linewidth=2, label='Confined fraction')

        # Also plot P energy fraction
        if sim.energy_history and sim.energy_history[0] > 0:
            p_frac = [p / e if e > 0 else 0
                      for p, e in zip(sim.p_energy_history, sim.energy_history)]
            ax.plot(t, p_frac, 'r--', linewidth=1.5, label='P²/total')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Annotate final value
        final = sim.confined_history[-1] if sim.confined_history else 0
        ax.annotate(f'{final:.3f}', xy=(t[-1], final),
                    fontsize=11, fontweight='bold', color='blue',
                    ha='right', va='bottom')

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/pivot_nonlinear_sweep.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    else:
        plt.show()
    plt.close()

    return results


def run_parameter_scan(model='C', n_steps=1000, save=False):
    """Scan parameter space for the chosen model."""
    if model == 'C':
        # Scan λ and v for the φ⁴ model
        lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]
        vs = [0.1, 0.3, 0.5, 1.0]
        configs = []
        for lam in lambdas:
            for v in vs:
                configs.append((f'λ={lam}, v={v}', {'lambda': lam, 'v': v}))
    elif model == 'A':
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        configs = [(f'α={a}', {'alpha': a}) for a in alphas]
    elif model == 'D':
        gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        configs = [(f'γ={g}', {'gamma': g}) for g in gammas]
    elif model == 'E':
        chis = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        configs = [(f'χ={c}', {'chi': c}) for c in chis]
    elif model == 'B':
        betas = [0.1, 0.5, 1.0, 2.0]
        emaxs = [1.0, 2.0, 5.0]
        configs = []
        for b in betas:
            for em in emaxs:
                configs.append((f'β={b}, E_max={em}', {'beta': b, 'E_max': em}))
    else:
        print(f"Unknown model: {model}")
        return

    print(f"Scanning {len(configs)} configurations for model {model}...")

    results = []
    for label, params in configs:
        print(f"  {label}...", end=" ", flush=True)
        sim = NonlinearPivotSim(nx=200, ny=200, model=model, params=params)
        sim.init_vortex(radius=20.0, amplitude=1.0, winding=1)
        try:
            sim.run(n_steps, diag_interval=10)
            final_conf = sim.confined_history[-1] if sim.confined_history else 0
            final_energy = sim.energy_history[-1] if sim.energy_history else 0
            stable = not (np.isnan(final_energy) or final_energy > 1e6)
            print(f"conf={final_conf:.3f}, E={final_energy:.1f}, {'OK' if stable else 'UNSTABLE'}")
        except (FloatingPointError, ValueError):
            final_conf = 0
            stable = False
            print("BLOWUP")

        results.append({
            'label': label,
            'params': params,
            'confined': final_conf,
            'stable': stable,
            'sim': sim if stable else None,
        })

    # Find best
    stable_results = [r for r in results if r['stable']]
    if stable_results:
        best = max(stable_results, key=lambda r: r['confined'])
        print(f"\nBest: {best['label']} → confined={best['confined']:.4f}")
    else:
        print("\nNo stable configurations found!")
        return results

    # Plot top configs
    top = sorted(stable_results, key=lambda r: r['confined'], reverse=True)[:6]

    n_plots = min(len(top), 6)
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Model {model} Parameter Scan — Top {n_plots} Configurations\n"
        f"({n_steps} steps, vortex init, absorbing BCs)",
        fontsize=13, fontweight='bold'
    )

    for idx, r in enumerate(top):
        ax = axes[idx // ncols, idx % ncols]
        sim = r['sim']
        if sim is None:
            continue
        t = np.arange(len(sim.confined_history)) * 10 * sim.dt
        ax.plot(t, sim.confined_history, 'b-', linewidth=2)
        if sim.energy_history and sim.energy_history[0] > 0:
            p_frac = [p / e if e > 0 else 0
                      for p, e in zip(sim.p_energy_history, sim.energy_history)]
            ax.plot(t, p_frac, 'r--', linewidth=1.5, label='P²/total')
        ax.set_title(f"{r['label']}\nfinal conf={r['confined']:.4f}", fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Confined fraction')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    if save:
        path = f'/home/jim/repos/claude-prime/simulations/pivot_nonlinear_scan_{model}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close()

    return results


def run_snapshot_comparison(model='C', params=None, n_steps=500, save=False):
    """Show field snapshots at multiple timesteps for one model."""
    if params is None:
        params = {'lambda': 1.0, 'v': 0.5}

    sim = NonlinearPivotSim(nx=200, ny=200, model=model, params=params)
    sim.init_vortex(radius=20.0, amplitude=1.0, winding=1)

    # Also run linear for comparison
    sim_lin = NonlinearPivotSim(nx=200, ny=200, model='linear')
    sim_lin.init_vortex(radius=20.0, amplitude=1.0, winding=1)

    snap_times = [0, n_steps // 4, n_steps // 2, n_steps]
    snapshots_nl = []
    snapshots_lin = []

    # Initial snapshot
    snapshots_nl.append(sim.compute_energy_density().copy())
    snapshots_lin.append(sim_lin.compute_energy_density().copy())

    for target in snap_times[1:]:
        current = sim.n_step
        for _ in range(target - current):
            sim.step()
            sim.apply_absorbing_bc()
            sim_lin.step()
            sim_lin.apply_absorbing_bc()
            if (sim.n_step % 10) == 0:
                sim.compute_diagnostics()
                sim_lin.compute_diagnostics()
        snapshots_nl.append(sim.compute_energy_density().copy())
        snapshots_lin.append(sim_lin.compute_energy_density().copy())

    # Plot
    fig, axes = plt.subplots(2, len(snap_times), figsize=(5 * len(snap_times), 9))
    fig.suptitle(
        f"Field Evolution: Model {model} vs Linear\n"
        f"Params: {params}",
        fontsize=13, fontweight='bold'
    )

    vmax = max(s.max() for s in snapshots_nl + snapshots_lin) * 0.5

    for i, (t, snl, slin) in enumerate(zip(snap_times, snapshots_nl, snapshots_lin)):
        axes[0, i].imshow(snl.T, origin='lower', cmap='inferno',
                          vmin=0, vmax=vmax, aspect='equal')
        axes[0, i].set_title(f't = {t}')
        if i == 0:
            axes[0, i].set_ylabel(f'Model {model}', fontsize=12)

        axes[1, i].imshow(slin.T, origin='lower', cmap='inferno',
                          vmin=0, vmax=vmax, aspect='equal')
        axes[1, i].set_title(f't = {t}')
        if i == 0:
            axes[1, i].set_ylabel('Linear', fontsize=12)

    plt.tight_layout()
    if save:
        path = f'/home/jim/repos/claude-prime/simulations/pivot_nonlinear_snapshots_{model}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Williamson nonlinear pivot simulation')
    parser.add_argument('--mode', choices=['sweep', 'scan', 'snapshot'],
                        default='sweep',
                        help='sweep=compare all models, scan=parameter scan, snapshot=field evolution')
    parser.add_argument('--model', default='C',
                        help='Model for scan/snapshot mode (A/B/C/D/E)')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.mode == 'sweep':
        run_model_sweep(n_steps=args.steps, save=args.save)
    elif args.mode == 'scan':
        run_parameter_scan(model=args.model, n_steps=args.steps, save=args.save)
    elif args.mode == 'snapshot':
        run_snapshot_comparison(model=args.model, n_steps=args.steps, save=args.save)


if __name__ == '__main__':
    main()
