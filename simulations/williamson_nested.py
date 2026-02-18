#!/usr/bin/env python3
"""
Nested Concentric Tori — Multi-Shell Williamson Model
======================================================

Instead of changing topology (knots), stack multiple toroidal shells
at different radii. Each shell carries its own circulating photon.
Coupling between shells occurs through overlapping evanescent fields.

Physics:
    Each shell i has fields (E_θ, E_φ, B, P) on its own torus surface
    with minor radius r_i and shared major radius R.

    Shells couple through their E fields:
        Shell i feels a coupling force from shell j proportional to
        the overlap integral of their fields, decaying exponentially
        with separation |r_i - r_j|.

    This is analogous to coupled waveguides or molecular orbital theory:
    N coupled resonators → N eigenfrequencies → mass spectrum.

Usage:
    python3 williamson_nested.py --mode single --steps 2000 --save
    python3 williamson_nested.py --mode nested --shells 2 --steps 3000 --save
    python3 williamson_nested.py --mode spectrum --max-shells 5 --steps 3000 --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time


class NestedTorusSim:
    """Multiple concentric torus shells with electromagnetic coupling."""

    def __init__(self, R=3.0, radii=None, n_theta=64, n_phi=128,
                 coupling_strength=0.1, coupling_decay=2.0):
        """
        Args:
            R: Major radius (shared by all shells)
            radii: List of minor radii [r_1, r_2, ...] for each shell
            n_theta: Grid points in poloidal direction (per shell)
            n_phi: Grid points in toroidal direction (per shell)
            coupling_strength: Base coupling between adjacent shells
            coupling_decay: Exponential decay rate with separation
        """
        if radii is None:
            radii = [1.0]
        self.R = R
        self.radii = np.array(radii, dtype=float)
        self.n_shells = len(radii)
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.coupling_strength = coupling_strength
        self.coupling_decay = coupling_decay

        self.dtheta = 2 * np.pi / n_theta
        self.dphi = 2 * np.pi / n_phi

        # Coordinate arrays (shared θ, φ grid)
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.theta_grid, self.phi_grid = np.meshgrid(theta, phi, indexing='ij')

        # Per-shell scale factors and fields
        self.shells = []
        dt_min = np.inf
        for i, r in enumerate(self.radii):
            h_phi = R + r * np.cos(self.theta_grid)
            sqrt_g = r * h_phi

            # CFL for this shell
            min_h_phi = R - r
            if min_h_phi <= 0:
                raise ValueError(f"Shell {i}: r={r} >= R={R}, torus self-intersects")
            ds_min = min(r * self.dtheta, min_h_phi * self.dphi)
            dt_shell = 0.4 * ds_min / np.sqrt(2)
            dt_min = min(dt_min, dt_shell)

            shell = {
                'r': r,
                'h_theta': r,
                'h_phi': h_phi,
                'sqrt_g': sqrt_g,
                'E_theta': np.zeros((n_theta, n_phi)),
                'E_phi': np.zeros((n_theta, n_phi)),
                'B': np.zeros((n_theta, n_phi)),
                'P': np.zeros((n_theta, n_phi)),
            }
            self.shells.append(shell)

        self.dt = dt_min  # use most restrictive CFL
        self.time = 0.0
        self.n_step = 0
        self.history = []

        # Precompute coupling matrix (exponential decay with separation)
        self.coupling = np.zeros((self.n_shells, self.n_shells))
        for i in range(self.n_shells):
            for j in range(self.n_shells):
                if i != j:
                    sep = abs(self.radii[i] - self.radii[j])
                    self.coupling[i, j] = coupling_strength * np.exp(
                        -coupling_decay * sep)

        radii_str = ", ".join(f"{r:.3f}" for r in self.radii)
        print(f"Nested torus: R={R}, shells={self.n_shells}, "
              f"radii=[{radii_str}]")
        print(f"  Grid: {n_theta}×{n_phi}, dt={self.dt:.6f}")
        if self.n_shells > 1:
            print(f"  Coupling: κ={coupling_strength}, "
                  f"decay={coupling_decay}")
            for i in range(self.n_shells):
                for j in range(i+1, self.n_shells):
                    print(f"    Shell {i}↔{j}: "
                          f"κ_eff={self.coupling[i,j]:.4f}")

    def _roll_theta(self, f, shift):
        return np.roll(f, shift, axis=0)

    def _roll_phi(self, f, shift):
        return np.roll(f, shift, axis=1)

    def update(self):
        """One FDTD step for all shells + inter-shell coupling."""
        dt = self.dt
        dtheta = self.dtheta
        dphi = self.dphi

        # === Phase 1: Independent shell updates (standard torus FDTD) ===
        for s in self.shells:
            h_t = s['h_theta']
            h_p = s['h_phi']
            sg = s['sqrt_g']
            Et = s['E_theta']
            Ep = s['E_phi']
            B = s['B']
            P = s['P']

            # Update B
            d_hpEp_dtheta = (self._roll_theta(h_p * Ep, -1) - h_p * Ep) / dtheta
            d_htEt_dphi = (self._roll_phi(h_t * Et, -1) - h_t * Et) / dphi
            B -= dt / sg * (d_hpEp_dtheta - d_htEt_dphi)

            # Update E_theta
            dB_dphi = (B - self._roll_phi(B, 1)) / dphi
            Et += dt * dB_dphi / h_p

            dP_dtheta = (self._roll_theta(P, -1) - P) / dtheta
            Et -= dt * dP_dtheta / h_t

            # Update E_phi
            dB_dtheta = (B - self._roll_theta(B, 1)) / dtheta
            Ep -= dt * dB_dtheta / h_t

            dP_dphi = (self._roll_phi(P, -1) - P) / dphi
            Ep -= dt * dP_dphi / h_p

            # Update P
            d_hpEt_dtheta = (h_p * Et - self._roll_theta(h_p * Et, 1)) / dtheta
            d_htEp_dphi = (h_t * Ep - self._roll_phi(h_t * Ep, 1)) / dphi
            P -= dt / sg * (d_hpEt_dtheta + d_htEp_dphi)

            s['E_theta'] = Et
            s['E_phi'] = Ep
            s['B'] = B
            s['P'] = P

        # === Phase 2: Inter-shell coupling (energy-conserving) ===
        # Spring-like coupling: force ~ (E_i - E_j), so energy transfers
        # between shells but total is conserved. Like coupled pendulums.
        if self.n_shells > 1:
            Et_all = [s['E_theta'].copy() for s in self.shells]
            Ep_all = [s['E_phi'].copy() for s in self.shells]

            for i in range(self.n_shells):
                for j in range(i + 1, self.n_shells):
                    if self.coupling[i, j] > 0:
                        kappa = self.coupling[i, j]
                        # Exchange: shell i gains what shell j loses
                        dEt = dt * kappa * (Et_all[j] - Et_all[i])
                        dEp = dt * kappa * (Ep_all[j] - Ep_all[i])
                        self.shells[i]['E_theta'] += dEt
                        self.shells[i]['E_phi'] += dEp
                        self.shells[j]['E_theta'] -= dEt
                        self.shells[j]['E_phi'] -= dEp

        self.time += dt
        self.n_step += 1

    def init_traveling_wave(self, n_toroidal=2, n_poloidal=1, amplitude=1.0,
                            shell_idx=None):
        """Initialize traveling wave on specified shells.

        Args:
            shell_idx: Which shells to excite. None = all shells.
                       Can be int or list of ints.
        """
        if shell_idx is None:
            shell_idx = list(range(self.n_shells))
        elif isinstance(shell_idx, int):
            shell_idx = [shell_idx]

        theta = self.theta_grid
        phi = self.phi_grid
        phase = n_toroidal * phi + n_poloidal * theta

        for i in shell_idx:
            self.shells[i]['E_theta'] = amplitude * np.cos(phase)
            self.shells[i]['B'] = amplitude * np.sin(phase)
            if n_poloidal > 0:
                self.shells[i]['E_phi'] = amplitude * 0.3 * np.sin(phase)

        idx_str = ", ".join(str(i) for i in shell_idx)
        print(f"Traveling wave on shells [{idx_str}]: "
              f"n_tor={n_toroidal}, n_pol={n_poloidal}, A={amplitude}")

    def compute_energy(self, shell_idx=None):
        """Compute energy for specified shells or all shells."""
        if shell_idx is None:
            shell_idx = list(range(self.n_shells))
        elif isinstance(shell_idx, int):
            shell_idx = [shell_idx]

        total_em = 0.0
        total_pivot = 0.0
        per_shell = []

        for i in shell_idx:
            s = self.shells[i]
            dA = s['sqrt_g'] * self.dtheta * self.dphi

            e2 = s['E_theta']**2 + s['E_phi']**2
            b2 = s['B']**2
            p2 = s['P']**2

            em = 0.5 * np.sum((e2 + b2) * dA)
            pivot = 0.5 * np.sum(p2 * dA)
            total_em += em
            total_pivot += pivot
            per_shell.append({'em': em, 'pivot': pivot, 'total': em + pivot})

        return {
            'em': total_em,
            'pivot': total_pivot,
            'total': total_em + total_pivot,
            'per_shell': per_shell,
        }

    def record_diagnostics(self):
        e = self.compute_energy()
        self.history.append({
            'time': self.time,
            'step': self.n_step,
            'em': e['em'],
            'pivot': e['pivot'],
            'total': e['total'],
            'per_shell': e['per_shell'],
        })


def run_single(R=3.0, r=1.0, steps=2000, save=False):
    """Single shell baseline — reproduces original torus results."""
    print("=" * 60)
    print("Single Shell Baseline")
    print("=" * 60)

    sim = NestedTorusSim(R=R, radii=[r])
    sim.init_traveling_wave(n_toroidal=2, n_poloidal=1)
    sim.record_diagnostics()

    t_start = time.time()
    for step_i in range(steps):
        sim.update()
        if step_i % max(1, steps // 100) == 0:
            sim.record_diagnostics()
        if step_i % max(1, steps // 5) == 0:
            e = sim.compute_energy()
            print(f"  Step {step_i:5d}: E_em={e['em']:.4f} "
                  f"P²={e['pivot']:.4f} total={e['total']:.4f}")

    elapsed = time.time() - t_start
    print(f"Done: {elapsed:.1f}s")

    e0 = sim.history[0]['total']
    ef = sim.history[-1]['total']
    print(f"\nEnergy: {e0:.4f} → {ef:.4f} "
          f"(conservation: {abs(ef-e0)/e0*100:.2f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Single Shell Baseline (R={R}, r={r})", fontsize=13)

    h = sim.history
    times = [d['time'] for d in h]
    axes[0].plot(times, [d['em'] for d in h], 'b-', label='EM')
    axes[0].plot(times, [d['pivot'] for d in h], 'r-', label='P²')
    axes[0].plot(times, [d['total'] for d in h], 'k--', label='Total')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Energy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # B field
    vmax = np.percentile(np.abs(sim.shells[0]['B']), 99)
    axes[1].imshow(sim.shells[0]['B'].T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto',
                   extent=[0, 360, 0, 360])
    axes[1].set_xlabel('θ (poloidal) [deg]')
    axes[1].set_ylabel('φ (toroidal) [deg]')
    axes[1].set_title('B field (final)')

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/nested_single.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def run_nested(R=3.0, n_shells=2, steps=3000, coupling=0.1,
               decay=2.0, save=False):
    """Run N concentric shells and observe energy exchange."""
    print("=" * 60)
    print(f"Nested Concentric Tori: {n_shells} shells")
    print("=" * 60)

    # Space shells evenly between r=0.5 and r=R*0.8
    r_min = 0.4
    r_max = R * 0.7
    if n_shells == 1:
        radii = [1.0]
    else:
        radii = np.linspace(r_min, r_max, n_shells).tolist()

    sim = NestedTorusSim(R=R, radii=radii, coupling_strength=coupling,
                         coupling_decay=decay)

    # Excite only the innermost shell — see how energy propagates outward
    sim.init_traveling_wave(n_toroidal=2, n_poloidal=1, shell_idx=0)
    sim.record_diagnostics()

    t_start = time.time()
    for step_i in range(steps):
        sim.update()
        if step_i % max(1, steps // 100) == 0:
            sim.record_diagnostics()
        if step_i % max(1, steps // 5) == 0:
            e = sim.compute_energy()
            shell_es = " | ".join(
                f"S{i}:{s['total']:.3f}" for i, s in enumerate(e['per_shell']))
            print(f"  Step {step_i:5d}: total={e['total']:.4f} [{shell_es}]")

    elapsed = time.time() - t_start
    print(f"Done: {elapsed:.1f}s")

    h = sim.history
    e0 = h[0]['total']
    ef = h[-1]['total']
    print(f"\nTotal energy: {e0:.4f} → {ef:.4f}")

    # Plot
    n_cols = min(n_shells + 1, 4)
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 9))
    fig.suptitle(
        f"Nested Concentric Tori: {n_shells} shells, "
        f"R={R}, κ={coupling}\n"
        f"Radii: [{', '.join(f'{r:.2f}' for r in radii)}]  |  "
        f"Excited: shell 0 only",
        fontsize=12, fontweight='bold')

    times = [d['time'] for d in h]
    colors = plt.cm.viridis(np.linspace(0, 1, n_shells))

    # Top-left: total energy per shell over time
    ax = axes[0, 0]
    for i in range(n_shells):
        shell_e = [d['per_shell'][i]['total'] for d in h]
        ax.plot(times, shell_e, color=colors[i], linewidth=2,
                label=f'Shell {i} (r={radii[i]:.2f})')
    ax.plot(times, [d['total'] for d in h], 'k--', linewidth=1.5,
            label='Total')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy per shell')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-left: EM vs P per shell (final)
    ax = axes[1, 0]
    shell_labels = [f'S{i}\nr={radii[i]:.2f}' for i in range(n_shells)]
    em_vals = [h[-1]['per_shell'][i]['em'] for i in range(n_shells)]
    pv_vals = [h[-1]['per_shell'][i]['pivot'] for i in range(n_shells)]
    x = np.arange(n_shells)
    ax.bar(x - 0.15, em_vals, 0.3, color='blue', alpha=0.7, label='EM')
    ax.bar(x + 0.15, pv_vals, 0.3, color='red', alpha=0.7, label='P²')
    ax.set_xticks(x)
    ax.set_xticklabels(shell_labels, fontsize=8)
    ax.set_ylabel('Energy')
    ax.set_title('Final energy breakdown')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Remaining columns: B field snapshots for each shell
    for col in range(1, n_cols):
        shell_i = col - 1
        if shell_i >= n_shells:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue

        # B field
        ax = axes[0, col]
        B = sim.shells[shell_i]['B']
        vmax = np.percentile(np.abs(B), 99)
        if vmax < 1e-10:
            vmax = 1.0
        ax.imshow(B.T, origin='lower', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, aspect='auto',
                  extent=[0, 360, 0, 360])
        ax.set_xlabel('θ [deg]')
        ax.set_ylabel('φ [deg]')
        ax.set_title(f'Shell {shell_i} B field\n(r={radii[shell_i]:.2f})')

        # P field
        ax = axes[1, col]
        P = sim.shells[shell_i]['P']
        vmax_p = np.percentile(np.abs(P), 99)
        if vmax_p < 1e-10:
            vmax_p = 1.0
        ax.imshow(P.T, origin='lower', cmap='RdBu_r',
                  vmin=-vmax_p, vmax=vmax_p, aspect='auto',
                  extent=[0, 360, 0, 360])
        ax.set_xlabel('θ [deg]')
        ax.set_ylabel('φ [deg]')
        ax.set_title(f'Shell {shell_i} P field')

    plt.tight_layout()
    if save:
        path = f'/home/jim/repos/claude-prime/simulations/nested_{n_shells}shells.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def run_spectrum(R=3.0, max_shells=5, steps=3000, coupling=0.1,
                 decay=2.0, save=False):
    """Compute total energy vs number of shells → mass spectrum.

    Key question: does the mass ratio between N-shell and 1-shell
    configurations match known particle mass ratios?

    electron:  1 shell  → m_e = 0.511 MeV
    muon:     ?shells  → m_μ = 105.7 MeV  (ratio: 206.8)
    pion:     ?shells  → m_π = 139.6 MeV  (ratio: 273.1)
    proton:   ?shells  → m_p = 938.3 MeV  (ratio: 1836.2)
    """
    print("=" * 60)
    print(f"Mass Spectrum: 1 to {max_shells} shells")
    print("=" * 60)

    # Known mass ratios (relative to electron)
    known = {
        'electron': 1.0,
        'muon': 206.768,
        'pion±': 273.132,
        'kaon±': 966.120,
        'proton': 1836.152,
    }

    results = []
    r_min = 0.4
    r_max = R * 0.7

    for n_shells in range(1, max_shells + 1):
        print(f"\n--- {n_shells} shell(s) ---")

        if n_shells == 1:
            radii = [1.0]
        else:
            radii = np.linspace(r_min, r_max, n_shells).tolist()

        sim = NestedTorusSim(R=R, radii=radii,
                             coupling_strength=coupling,
                             coupling_decay=decay)

        # Excite ALL shells with same amplitude
        sim.init_traveling_wave(n_toroidal=2, n_poloidal=1)
        sim.record_diagnostics()

        for step_i in range(steps):
            sim.update()
            if step_i % max(1, steps // 20) == 0:
                sim.record_diagnostics()

        h = sim.history
        # Use time-averaged energy (last quarter of run)
        quarter = len(h) // 4
        avg_total = np.mean([d['total'] for d in h[-quarter:]])
        avg_em = np.mean([d['em'] for d in h[-quarter:]])
        avg_pivot = np.mean([d['pivot'] for d in h[-quarter:]])

        results.append({
            'n_shells': n_shells,
            'radii': radii,
            'total': avg_total,
            'em': avg_em,
            'pivot': avg_pivot,
        })

        print(f"  Avg energy: total={avg_total:.4f} "
              f"(EM={avg_em:.4f}, P²={avg_pivot:.4f})")

    # Compute mass ratios relative to 1-shell
    e1 = results[0]['total']
    print("\n" + "=" * 60)
    print("MASS SPECTRUM")
    print("=" * 60)
    print(f"{'Shells':>8s} {'Energy':>10s} {'Ratio':>10s} "
          f"{'Nearest':>12s} {'Known':>10s}")
    print("-" * 60)

    for r in results:
        ratio = r['total'] / e1
        # Find nearest known particle
        nearest = min(known.keys(),
                      key=lambda k: abs(known[k] - ratio))
        print(f"{r['n_shells']:>8d} {r['total']:>10.4f} {ratio:>10.4f} "
              f"{nearest:>12s} {known[nearest]:>10.3f}")

    # Also try: different mode numbers on nested shells
    print("\n\n--- Heterogeneous modes on 3 shells ---")
    mode_configs = [
        [(2, 1), (2, 1), (2, 1)],  # all same
        [(2, 1), (3, 1), (4, 1)],  # increasing toroidal
        [(2, 1), (2, 2), (2, 3)],  # increasing poloidal
        [(1, 1), (2, 1), (3, 1)],  # 1,2,3 toroidal
    ]

    hetero_results = []
    radii_3 = np.linspace(r_min, r_max, 3).tolist()

    for modes in mode_configs:
        mode_str = "+".join(f"({t},{p})" for t, p in modes)
        print(f"\n  Modes: {mode_str}")

        sim = NestedTorusSim(R=R, radii=radii_3,
                             coupling_strength=coupling,
                             coupling_decay=decay)

        # Initialize each shell with its own mode
        theta = sim.theta_grid
        phi = sim.phi_grid
        for i, (n_tor, n_pol) in enumerate(modes):
            phase = n_tor * phi + n_pol * theta
            sim.shells[i]['E_theta'] = np.cos(phase)
            sim.shells[i]['B'] = np.sin(phase)
            sim.shells[i]['E_phi'] = 0.3 * np.sin(phase)

        sim.record_diagnostics()
        for step_i in range(steps):
            sim.update()
            if step_i % max(1, steps // 20) == 0:
                sim.record_diagnostics()

        h = sim.history
        quarter = len(h) // 4
        avg_total = np.mean([d['total'] for d in h[-quarter:]])
        ratio = avg_total / e1

        hetero_results.append({
            'modes': mode_str,
            'total': avg_total,
            'ratio': ratio,
        })
        print(f"    Energy: {avg_total:.4f}, ratio: {ratio:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Mass Spectrum from Nested Concentric Tori\n"
        f"R={R}, κ={coupling}, {steps} steps",
        fontsize=13, fontweight='bold')

    # 1. Energy vs N shells
    ax = axes[0]
    ns = [r['n_shells'] for r in results]
    totals = [r['total'] for r in results]
    ems = [r['em'] for r in results]
    pivots = [r['pivot'] for r in results]
    ax.bar(np.array(ns) - 0.15, ems, 0.3, color='blue', alpha=0.7, label='EM')
    ax.bar(np.array(ns) + 0.15, pivots, 0.3, color='red', alpha=0.7, label='P²')
    ax.plot(ns, totals, 'ko-', linewidth=2, markersize=8, label='Total')
    ax.set_xlabel('Number of shells')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs Shell Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mass ratio vs N shells
    ax = axes[1]
    ratios = [r['total'] / e1 for r in results]
    ax.plot(ns, ratios, 'ko-', linewidth=2, markersize=10)
    for i, r in enumerate(ratios):
        ax.annotate(f'{r:.2f}×', (ns[i], r), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    # Mark known particles
    for name, mass_ratio in known.items():
        if mass_ratio <= max(ratios) * 1.5:
            ax.axhline(y=mass_ratio, color='gray', linestyle=':', alpha=0.5)
            ax.text(max_shells + 0.1, mass_ratio, f'{name}\n({mass_ratio:.1f})',
                    fontsize=7, va='center')

    ax.set_xlabel('Number of shells')
    ax.set_ylabel('Mass ratio (m / m_electron)')
    ax.set_title('Mass Ratio vs Shell Count')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Heterogeneous modes
    ax = axes[2]
    mode_labels = [r['modes'] for r in hetero_results]
    mode_ratios = [r['ratio'] for r in hetero_results]
    bars = ax.barh(range(len(hetero_results)), mode_ratios, color='teal',
                   alpha=0.7)
    ax.set_yticks(range(len(hetero_results)))
    ax.set_yticklabels(mode_labels, fontsize=8)
    ax.set_xlabel('Mass ratio (m / m_electron)')
    ax.set_title('3-Shell Heterogeneous Modes')
    for i, v in enumerate(mode_ratios):
        ax.text(v + 0.05, i, f'{v:.2f}×', va='center', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = '/home/jim/repos/claude-prime/simulations/nested_spectrum.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Nested concentric tori — multi-shell Williamson model')
    parser.add_argument('--mode', choices=['single', 'nested', 'spectrum'],
                        default='spectrum')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--shells', type=int, default=2,
                        help='Number of shells (for nested mode)')
    parser.add_argument('--max-shells', type=int, default=5,
                        help='Max shells to test (for spectrum mode)')
    parser.add_argument('--coupling', type=float, default=0.1,
                        help='Inter-shell coupling strength')
    parser.add_argument('--decay', type=float, default=2.0,
                        help='Coupling exponential decay rate')
    parser.add_argument('--R', type=float, default=3.0, help='Major radius')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.mode == 'single':
        run_single(R=args.R, steps=args.steps, save=args.save)
    elif args.mode == 'nested':
        run_nested(R=args.R, n_shells=args.shells, steps=args.steps,
                   coupling=args.coupling, decay=args.decay, save=args.save)
    elif args.mode == 'spectrum':
        run_spectrum(R=args.R, max_shells=args.max_shells, steps=args.steps,
                     coupling=args.coupling, decay=args.decay, save=args.save)


if __name__ == '__main__':
    main()
