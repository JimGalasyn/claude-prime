#!/usr/bin/env python3
"""
Williamson Pivot Field on Toroidal Coordinates
================================================

The Copernicus move: instead of embedding a torus in a 3D box
(and fighting boundary artifacts), solve directly ON the torus surface.

Coordinates: (θ, φ) where
    θ = poloidal angle (around tube cross-section), 0 to 2π
    φ = toroidal angle (around the big ring), 0 to 2π

Metric: ds² = r²dθ² + (R + r·cos θ)²dφ²
    h_θ = r                    (constant)
    h_φ = R + r·cos θ          (varies with θ)

Fields on the surface:
    E_θ, E_φ:  tangent electric field (physical components)
    B:          normal magnetic field (scalar in 2D)
    P:          pivot field (scalar)

Extended Maxwell on curved surface:

    ∂B/∂t = -(1/(h_θ h_φ))[∂(h_φ E_φ)/∂θ - ∂(h_θ E_θ)/∂φ]

    ∂E_θ/∂t = (1/h_φ)∂B/∂φ - (1/h_θ)∂P/∂θ

    ∂E_φ/∂t = -(1/h_θ)∂B/∂θ - (1/h_φ)∂P/∂φ

    ∂P/∂t = -(1/(h_θ h_φ))[∂(h_φ E_θ)/∂θ + ∂(h_θ E_φ)/∂φ]

Boundary conditions: periodic in BOTH θ and φ (the torus has no edges).
This is the key advantage — no absorbing BCs, no boundary artifacts.
The topology is built into the coordinate system.

Usage:
    python3 williamson_torus.py --mode compare --steps 2000 --save
    python3 williamson_torus.py --mode modes --steps 5000 --save
    python3 williamson_torus.py --mode golden --steps 3000 --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2  # golden ratio


class TorusSim:
    """Extended Maxwell equations on a torus surface."""

    def __init__(self, R=3.0, r=1.0, n_theta=128, n_phi=256,
                 pivot_enabled=True, nonlinear=None):
        """
        Args:
            R: Major radius (torus center to tube center)
            r: Minor radius (tube radius)
            n_theta: Grid points in poloidal direction
            n_phi: Grid points in toroidal direction
            pivot_enabled: Include pivot field P
            nonlinear: None or {'model': 'phi4', 'lambda': λ, 'v': v}
        """
        self.R = R
        self.r = r
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.pivot_enabled = pivot_enabled
        self.nonlinear = nonlinear or {}

        self.dtheta = 2 * np.pi / n_theta
        self.dphi = 2 * np.pi / n_phi

        # Coordinate arrays
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.theta_grid, self.phi_grid = np.meshgrid(theta, phi, indexing='ij')

        # Scale factors
        self.h_theta = r  # constant
        self.h_phi = R + r * np.cos(self.theta_grid)  # varies with θ
        self.sqrt_g = self.h_theta * self.h_phi  # = r(R + r cos θ)

        # CFL stability: dt ≤ min(h_θ·dθ, h_φ·dφ) / √2
        min_h_phi = R - r  # minimum of h_φ (at θ=π, inside of torus)
        ds_min = min(r * self.dtheta, min_h_phi * self.dphi)
        self.dt = 0.4 * ds_min / np.sqrt(2)

        # Fields
        self.E_theta = np.zeros((n_theta, n_phi))
        self.E_phi = np.zeros((n_theta, n_phi))
        self.B = np.zeros((n_theta, n_phi))
        self.P = np.zeros((n_theta, n_phi))

        self.time = 0.0
        self.n_step = 0
        self.history = []

        aspect = R / r
        nl_str = ""
        if self.nonlinear.get('model'):
            nl = self.nonlinear
            nl_str = f", NL={nl['model']}(λ={nl.get('lambda', 0.5)},v={nl.get('v', 0.3)})"
        print(f"Torus: R={R}, r={r}, R/r={aspect:.3f}, "
              f"{n_theta}×{n_phi} grid, dt={self.dt:.5f}, "
              f"pivot={'ON' if pivot_enabled else 'OFF'}{nl_str}")

    def _roll_theta(self, f, shift):
        """Roll in θ direction (periodic)."""
        return np.roll(f, shift, axis=0)

    def _roll_phi(self, f, shift):
        """Roll in φ direction (periodic)."""
        return np.roll(f, shift, axis=1)

    def update(self):
        """One FDTD step on the torus surface."""
        dt = self.dt
        dtheta = self.dtheta
        dphi = self.dphi
        h_t = self.h_theta  # scalar
        h_p = self.h_phi    # 2D array
        sg = self.sqrt_g    # 2D array

        Et = self.E_theta
        Ep = self.E_phi
        B = self.B
        P = self.P

        # === 1. Update B: ∂B/∂t = -(1/√g)[∂(h_φ E_φ)/∂θ - ∂(h_θ E_θ)/∂φ] ===
        # ∂(h_φ E_φ)/∂θ: forward difference in θ
        d_hpEp_dtheta = (self._roll_theta(h_p * Ep, -1) - h_p * Ep) / dtheta
        # ∂(h_θ E_θ)/∂φ: forward difference in φ
        d_htEt_dphi = (self._roll_phi(h_t * Et, -1) - h_t * Et) / dphi

        B -= dt / sg * (d_hpEp_dtheta - d_htEt_dphi)

        # === 2. Update E_θ: ∂E_θ/∂t = (1/h_φ)∂B/∂φ - (1/h_θ)∂P/∂θ ===
        dB_dphi = (B - self._roll_phi(B, 1)) / dphi
        Et += dt * dB_dphi / h_p

        if self.pivot_enabled:
            dP_dtheta = (self._roll_theta(P, -1) - P) / dtheta
            Et -= dt * dP_dtheta / h_t

        # === 3. Update E_φ: ∂E_φ/∂t = -(1/h_θ)∂B/∂θ - (1/h_φ)∂P/∂φ ===
        dB_dtheta = (B - self._roll_theta(B, 1)) / dtheta
        Ep -= dt * dB_dtheta / h_t

        if self.pivot_enabled:
            dP_dphi = (self._roll_phi(P, -1) - P) / dphi
            Ep -= dt * dP_dphi / h_p

        # === 4. Update P: ∂P/∂t = -div(E) ===
        if self.pivot_enabled:
            # div(E) = (1/√g)[∂(h_φ E_θ)/∂θ + ∂(h_θ E_φ)/∂φ]
            d_hpEt_dtheta = (h_p * Et - self._roll_theta(h_p * Et, 1)) / dtheta
            d_htEp_dphi = (h_t * Ep - self._roll_phi(h_t * Ep, 1)) / dphi

            P -= dt / sg * (d_hpEt_dtheta + d_htEp_dphi)

            # φ⁴ nonlinearity
            if self.nonlinear.get('model') == 'phi4':
                lam = self.nonlinear.get('lambda', 0.5)
                v = self.nonlinear.get('v', 0.3)
                P -= dt * lam * P * (P**2 - v**2)

        self.time += dt
        self.n_step += 1

    def compute_energy(self):
        """Compute field energies integrated over the torus surface.

        For φ⁴ models, includes the potential energy V(P) = λ/4·(P²-v²)²
        so that total energy is properly conserved.
        """
        dA = self.sqrt_g * self.dtheta * self.dphi  # area element

        e2 = self.E_theta**2 + self.E_phi**2
        b2 = self.B**2
        p2 = self.P**2

        em = 0.5 * np.sum((e2 + b2) * dA)
        pivot = 0.5 * np.sum(p2 * dA)

        # φ⁴ potential energy: V(P) = λ/4·(P² - v²)²
        phi4_potential = 0.0
        if self.nonlinear.get('model') == 'phi4':
            lam = self.nonlinear.get('lambda', 0.5)
            v = self.nonlinear.get('v', 0.3)
            phi4_potential = (lam / 4.0) * np.sum((p2 - v**2)**2 * dA)

        return {'em': em, 'pivot': pivot, 'phi4': phi4_potential,
                'total': em + pivot + phi4_potential}

    def compute_mode_spectrum(self):
        """FFT to find dominant (m, n) modes.

        m = poloidal mode number, n = toroidal mode number.
        The key Williamson mode is (m, n) = (1, 2) — one poloidal
        oscillation per two toroidal circuits.
        """
        # FFT of B field
        B_fft = np.fft.fft2(self.B)
        spectrum = np.abs(B_fft)**2
        return spectrum

    def init_traveling_wave(self, n_toroidal=2, n_poloidal=1, amplitude=1.0):
        """Initialize a traveling wave circulating toroidally.

        This is Williamson's electron: a photon making n_toroidal
        complete oscillations per toroidal circuit, with n_poloidal
        poloidal windings.

        For the electron: n_toroidal=2 (double loop, 4π phase).
        """
        theta = self.theta_grid
        phi = self.phi_grid

        # Phase: travels in +φ direction
        phase = n_toroidal * phi + n_poloidal * theta

        # E_θ and B form a traveling wave in φ direction
        self.E_theta = amplitude * np.cos(phase)
        self.B = amplitude * np.sin(phase)

        # E_φ component for poloidal winding
        if n_poloidal > 0:
            self.E_phi = amplitude * 0.3 * np.sin(phase)

        print(f"Traveling wave: n_tor={n_toroidal}, n_pol={n_poloidal}, "
              f"A={amplitude}")

    def init_standing_wave(self, n_toroidal=2, n_poloidal=1, amplitude=1.0):
        """Standing wave — superposition of clockwise and counter-clockwise."""
        theta = self.theta_grid
        phi = self.phi_grid

        self.E_theta = amplitude * np.cos(n_toroidal * phi) * np.cos(n_poloidal * theta)
        self.B = amplitude * np.sin(n_toroidal * phi) * np.sin(n_poloidal * theta)

        print(f"Standing wave: n_tor={n_toroidal}, n_pol={n_poloidal}")

    def init_localized_pulse(self, phi0=0.0, theta0=0.0, width_phi=0.3,
                              width_theta=0.5, amplitude=1.0):
        """Localized pulse on the torus — should disperse if no confinement."""
        theta = self.theta_grid
        phi = self.phi_grid

        # Gaussian in both directions (periodic-aware)
        d_phi = np.minimum(np.abs(phi - phi0), 2*np.pi - np.abs(phi - phi0))
        d_theta = np.minimum(np.abs(theta - theta0), 2*np.pi - np.abs(theta - theta0))

        envelope = amplitude * np.exp(-d_phi**2 / (2 * width_phi**2)
                                       - d_theta**2 / (2 * width_theta**2))

        self.E_theta = envelope
        self.B = envelope * 0.5

        print(f"Localized pulse at (θ={theta0:.2f}, φ={phi0:.2f}), "
              f"widths=({width_theta:.2f}, {width_phi:.2f})")

    def record_diagnostics(self):
        """Record energy and mode info."""
        e = self.compute_energy()
        spec = self.compute_mode_spectrum()

        # Find dominant mode
        spec_shifted = np.fft.fftshift(spec)
        max_idx = np.unravel_index(np.argmax(spec_shifted), spec_shifted.shape)
        m_dom = max_idx[0] - self.n_theta // 2
        n_dom = max_idx[1] - self.n_phi // 2

        self.history.append({
            'time': self.time,
            'step': self.n_step,
            'em': e['em'],
            'pivot': e['pivot'],
            'total': e['total'],
            'dominant_mode': (m_dom, n_dom),
            'peak_B': np.max(np.abs(self.B)),
            'peak_P': np.max(np.abs(self.P)),
        })


def run_comparison(R=3.0, r=1.0, steps=2000, save=False):
    """Compare pivot vs Maxwell on the torus."""
    print("=" * 60)
    print("Torus Surface: Pivot vs Maxwell Comparison")
    print("=" * 60)

    configs = [
        ("Pivot ON", True, {}),
        ("Maxwell (no P)", False, {}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Extended Maxwell on Torus Surface (R={R}, r={r}, R/r={R/r:.2f})\n"
        f"Traveling wave n_tor=2, {steps} steps — Periodic BCs (no boundaries)",
        fontsize=13, fontweight='bold'
    )

    for idx, (label, pivot_on, nl_params) in enumerate(configs):
        print(f"\n--- {label} ---")
        sim = TorusSim(R=R, r=r, pivot_enabled=pivot_on, nonlinear=nl_params)
        sim.init_traveling_wave(n_toroidal=2, n_poloidal=1)
        sim.record_diagnostics()

        t_start = time.time()
        for step_i in range(steps):
            sim.update()
            if step_i % max(1, steps // 200) == 0:
                sim.record_diagnostics()
            if step_i % max(1, steps // 5) == 0:
                e = sim.compute_energy()
                elapsed = time.time() - t_start
                rate = (step_i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Step {step_i:5d}/{steps}: "
                      f"E_em={e['em']:.4f} P²={e['pivot']:.4f} "
                      f"peak_B={np.max(np.abs(sim.B)):.4f} "
                      f"[{rate:.0f} steps/s]")

        elapsed = time.time() - t_start
        print(f"  Done: {elapsed:.1f}s ({steps/elapsed:.0f} steps/s)")

        h = sim.history
        col = idx  # 0 or 1

        # Top row: field snapshots
        ax = axes[0, col]
        vmax = np.percentile(np.abs(sim.B), 99)
        if vmax < 1e-10:
            vmax = 1.0
        im = ax.imshow(sim.B.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='auto',
                       extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_xlabel('θ (poloidal)')
        ax.set_ylabel('φ (toroidal)')
        ax.set_title(f'{label}\nB field (final)')
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Middle: energy vs time
        ax = axes[1, col]
        times = [d['time'] for d in h]
        ax.plot(times, [d['em'] for d in h], 'b-', linewidth=2, label='EM')
        ax.plot(times, [d['total'] for d in h], 'k--', linewidth=1.5, label='Total')
        if pivot_on:
            ax.plot(times, [d['pivot'] for d in h], 'r-', linewidth=1.5, label='P²')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Conservation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Right column: P field and spectrum
    sim_p = TorusSim(R=R, r=r, pivot_enabled=True)
    sim_p.init_traveling_wave(n_toroidal=2, n_poloidal=1)
    for _ in range(steps):
        sim_p.update()

    ax = axes[0, 2]
    vmax_p = np.percentile(np.abs(sim_p.P), 99)
    if vmax_p < 1e-10:
        vmax_p = 1.0
    im = ax.imshow(sim_p.P.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax_p, vmax=vmax_p, aspect='auto',
                   extent=[0, 2*np.pi, 0, 2*np.pi])
    ax.set_xlabel('θ (poloidal)')
    ax.set_ylabel('φ (toroidal)')
    ax.set_title('Pivot field P (final)')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Mode spectrum
    ax = axes[1, 2]
    spec = sim_p.compute_mode_spectrum()
    spec_log = np.log10(np.fft.fftshift(spec) + 1e-20)
    n_t, n_p = sim_p.n_theta, sim_p.n_phi
    extent = [-n_t//2, n_t//2, -n_p//2, n_p//2]
    im = ax.imshow(spec_log[:16, n_p//2-8:n_p//2+8].T, origin='lower',
                   cmap='hot', aspect='auto',
                   extent=[-8, 8, -8, 8])
    ax.set_xlabel('m (poloidal mode)')
    ax.set_ylabel('n (toroidal mode)')
    ax.set_title('Mode spectrum log₁₀|B̂(m,n)|²')
    plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    if save:
        path = str(OUTPUT_DIR / 'torus_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    else:
        plt.show()
    plt.close()


def run_modes(R=3.0, r=1.0, steps=5000, save=False):
    """Test which (m, n) modes are stable on the torus.

    Key test: does the pivot field create stable modes that
    don't exist in standard Maxwell? If so, which (m, n)?
    """
    print("=" * 60)
    print("Torus Mode Stability Analysis")
    print("=" * 60)

    mode_configs = [
        (1, 1, "w=1 (single loop)"),
        (2, 1, "w=2 (Williamson double loop)"),
        (3, 1, "w=3 (triple)"),
        (1, 2, "w=1, 2 poloidal"),
        (2, 2, "w=2, 2 poloidal"),
    ]

    fig, axes = plt.subplots(2, len(mode_configs), figsize=(5*len(mode_configs), 9))
    fig.suptitle(
        f"Torus Mode Analysis: Which modes persist with pivot?\n"
        f"R={R}, r={r}, R/r={R/r:.2f}, {steps} steps",
        fontsize=13, fontweight='bold'
    )

    for col, (n_tor, n_pol, label) in enumerate(mode_configs):
        # Run with pivot
        sim_p = TorusSim(R=R, r=r, pivot_enabled=True)
        sim_p.init_traveling_wave(n_toroidal=n_tor, n_poloidal=n_pol)
        sim_p.record_diagnostics()

        # Run without pivot
        sim_m = TorusSim(R=R, r=r, pivot_enabled=False)
        sim_m.init_traveling_wave(n_toroidal=n_tor, n_poloidal=n_pol)
        sim_m.record_diagnostics()

        print(f"\n--- ({n_tor}, {n_pol}): {label} ---")

        for step_i in range(steps):
            sim_p.update()
            sim_m.update()
            if step_i % max(1, steps // 100) == 0:
                sim_p.record_diagnostics()
                sim_m.record_diagnostics()

        hp = sim_p.history
        hm = sim_m.history

        # Top: energy ratio (should stay at 1.0 if mode is stable)
        ax = axes[0, col]
        e0_p = hp[0]['em'] if hp[0]['em'] > 0 else 1
        e0_m = hm[0]['em'] if hm[0]['em'] > 0 else 1
        ax.plot([d['time'] for d in hp], [d['em']/e0_p for d in hp],
                'b-', linewidth=2, label='Pivot')
        ax.plot([d['time'] for d in hm], [d['em']/e0_m for d in hm],
                'gray', linewidth=2, label='Maxwell')
        ax.set_xlabel('Time')
        ax.set_ylabel('EM energy / initial')
        ax.set_title(f'({n_tor},{n_pol}): {label}')
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.3)

        # Bottom: peak |B| (structure persistence)
        ax = axes[1, col]
        ax.plot([d['time'] for d in hp], [d['peak_B'] for d in hp],
                'b-', linewidth=2, label='Pivot')
        ax.plot([d['time'] for d in hm], [d['peak_B'] for d in hm],
                'gray', linewidth=2, label='Maxwell')
        if sim_p.pivot_enabled:
            ax.plot([d['time'] for d in hp], [d['peak_P'] for d in hp],
                    'r--', linewidth=1.5, label='|P|')
        ax.set_xlabel('Time')
        ax.set_ylabel('Peak amplitude')
        ax.set_title(f'Peak |B| and |P|')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        print(f"  Pivot: E_em {hp[0]['em']:.4f} → {hp[-1]['em']:.4f} "
              f"(ratio {hp[-1]['em']/e0_p:.4f})")
        print(f"  Maxwell: E_em {hm[0]['em']:.4f} → {hm[-1]['em']:.4f} "
              f"(ratio {hm[-1]['em']/e0_m:.4f})")

    plt.tight_layout()
    if save:
        path = str(OUTPUT_DIR / 'torus_modes.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    else:
        plt.show()
    plt.close()


def run_golden(steps=3000, save=False):
    """Test the golden ratio prediction: does R/r = φ give special behavior?

    Scan R/r from 1.5 to 4.0 and look for:
    - Energy conservation quality
    - Mode stability
    - P² (rest mass) generation
    """
    print("=" * 60)
    print("Golden Ratio Scan: R/r vs Mode Stability")
    print("=" * 60)

    aspect_ratios = np.linspace(1.3, 4.0, 15)
    r = 1.0

    results_pivot = []
    results_maxwell = []

    for alpha in aspect_ratios:
        R = alpha * r
        print(f"  R/r = {alpha:.2f}...", end=" ", flush=True)

        sim_p = TorusSim(R=R, r=r, n_theta=64, n_phi=128, pivot_enabled=True)
        sim_p.init_traveling_wave(n_toroidal=2, n_poloidal=1)
        e0 = sim_p.compute_energy()

        sim_m = TorusSim(R=R, r=r, n_theta=64, n_phi=128, pivot_enabled=False)
        sim_m.init_traveling_wave(n_toroidal=2, n_poloidal=1)

        for _ in range(steps):
            sim_p.update()
            sim_m.update()

        ef_p = sim_p.compute_energy()
        ef_m = sim_m.compute_energy()

        ratio_p = ef_p['em'] / (e0['em'] + 1e-15)
        ratio_m = ef_m['em'] / (e0['em'] + 1e-15)

        results_pivot.append({
            'aspect': alpha,
            'em_ratio': ratio_p,
            'p2_final': ef_p['pivot'],
            'peak_B': np.max(np.abs(sim_p.B)),
            'peak_P': np.max(np.abs(sim_p.P)),
        })
        results_maxwell.append({
            'aspect': alpha,
            'em_ratio': ratio_m,
            'peak_B': np.max(np.abs(sim_m.B)),
        })

        print(f"pivot_ratio={ratio_p:.4f}, maxwell_ratio={ratio_m:.4f}, "
              f"P²={ef_p['pivot']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"Golden Ratio Test: R/r vs Torus Mode Stability\n"
        f"Traveling wave (2,1), {steps} steps, periodic BCs",
        fontsize=13, fontweight='bold'
    )

    aspects = [r['aspect'] for r in results_pivot]

    ax = axes[0]
    ax.plot(aspects, [r['em_ratio'] for r in results_pivot],
            'bo-', linewidth=2, markersize=6, label='Pivot')
    ax.plot(aspects, [r['em_ratio'] for r in results_maxwell],
            'gs-', linewidth=2, markersize=6, label='Maxwell')
    ax.axvline(x=PHI, color='gold', linewidth=2.5, linestyle='--',
               label=f'R/r = φ = {PHI:.3f}')
    ax.axhline(y=1.0, color='gray', linewidth=1, linestyle=':')
    ax.set_xlabel('Aspect ratio R/r')
    ax.set_ylabel('EM energy / initial')
    ax.set_title('Energy Conservation vs R/r')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(aspects, [r['p2_final'] for r in results_pivot],
            'ro-', linewidth=2, markersize=6)
    ax.axvline(x=PHI, color='gold', linewidth=2.5, linestyle='--',
               label=f'φ = {PHI:.3f}')
    ax.set_xlabel('Aspect ratio R/r')
    ax.set_ylabel('P² energy (rest mass)')
    ax.set_title('Rest Mass Generation vs R/r')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(aspects, [r['peak_B'] for r in results_pivot],
            'bo-', linewidth=2, markersize=6, label='Pivot |B|')
    ax.plot(aspects, [r['peak_P'] for r in results_pivot],
            'r^-', linewidth=2, markersize=6, label='|P|')
    ax.axvline(x=PHI, color='gold', linewidth=2.5, linestyle='--',
               label=f'φ = {PHI:.3f}')
    ax.set_xlabel('Aspect ratio R/r')
    ax.set_ylabel('Peak amplitude')
    ax.set_title('Field Persistence vs R/r')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = str(OUTPUT_DIR / 'torus_golden.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Williamson pivot field on torus surface')
    parser.add_argument('--mode', choices=['compare', 'modes', 'golden'],
                        default='compare')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--R', type=float, default=3.0, help='Major radius')
    parser.add_argument('--r', type=float, default=1.0, help='Minor radius')
    args = parser.parse_args()

    if args.mode == 'compare':
        run_comparison(R=args.R, r=args.r, steps=args.steps, save=args.save)
    elif args.mode == 'modes':
        run_modes(R=args.R, r=args.r, steps=args.steps, save=args.save)
    elif args.mode == 'golden':
        run_golden(steps=args.steps, save=args.save)


if __name__ == '__main__':
    main()
