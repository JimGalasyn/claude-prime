#!/usr/bin/env python3
"""
Braided Torus — Multi-Strand Williamson Model
===============================================

Instead of nested concentric tori (which don't couple well),
model multiple photon strands braided within a single torus tube.

Physical picture:
    The torus tube has radius r. Inside it, N sub-tubes (strands)
    are arranged symmetrically around the tube center. As you travel
    around the torus (in φ), the strands rotate around each other
    in the poloidal (θ) direction, creating a braid.

    N=1, no braid: electron (unknot)
    N=2, braid: meson (quark-antiquark link)
    N=3, braid: baryon (trefoil if 3 strands make one full twist)

The braid number (how many times strands wind around each other
per toroidal circuit) determines the knot type:
    - 0 twists: N unlinked loops (unstable, should separate)
    - 1 twist per circuit: linked/knotted (stable)

Each strand has its own (E_θ, E_φ, B, P) fields on its own
sub-surface within the torus tube. Strands couple through
their overlapping fields.

Usage:
    python3 williamson_braided.py --mode single --steps 3000 --save
    python3 williamson_braided.py --mode braided --strands 3 --steps 3000 --save
    python3 williamson_braided.py --mode spectrum --steps 3000 --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class BraidedTorusSim:
    """Multiple braided strands within a single torus tube.

    Each strand is a sub-torus at the same major radius R but offset
    in the poloidal direction. The strand centers trace helical paths
    within the tube cross-section.

    Strand i has its center at poloidal angle:
        θ_i(φ) = 2πi/N + braid_number × φ

    where braid_number is how many times the strands twist around
    each other per toroidal circuit.
    """

    def __init__(self, R=3.0, r_tube=1.0, n_strands=1,
                 r_strand=None, braid_number=0,
                 n_theta=64, n_phi=128,
                 coupling_strength=0.0):
        """
        Args:
            R: Major radius of the torus
            r_tube: Radius of the main tube
            n_strands: Number of photon strands (1=electron, 2=meson, 3=baryon)
            r_strand: Radius of each sub-strand (default: r_tube / (n_strands + 1))
            braid_number: Number of full twists per toroidal circuit
                         (0=unlinked, 1=linked/knotted)
            n_theta: Grid points in poloidal direction per strand
            n_phi: Grid points in toroidal direction
            coupling_strength: EM coupling between strands
        """
        self.R = R
        self.r_tube = r_tube
        self.n_strands = n_strands
        self.braid_number = braid_number
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.coupling_strength = coupling_strength

        # Strand geometry
        if r_strand is None:
            if n_strands == 1:
                r_strand = r_tube
            else:
                # Strands packed within the tube
                # Distance from tube center to strand center
                self.r_offset = r_tube * 0.4
                r_strand = r_tube * 0.25
        else:
            self.r_offset = r_tube * 0.4
        self.r_strand = r_strand

        if n_strands == 1:
            self.r_offset = 0.0

        self.dtheta = 2 * np.pi / n_theta
        self.dphi = 2 * np.pi / n_phi

        # Coordinate arrays (shared)
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.theta_grid, self.phi_grid = np.meshgrid(theta, phi, indexing='ij')

        # Each strand has its own effective torus surface
        # For strand i, the effective surface is offset in θ:
        #   θ_eff = θ - (2πi/N + braid_number × φ)
        # This means each strand's coordinate system rotates as you go around φ

        self.strands = []
        dt_min = np.inf

        for i in range(n_strands):
            # Strand center angle at each φ
            if n_strands > 1:
                theta_offset = 2 * np.pi * i / n_strands + braid_number * self.phi_grid
            else:
                theta_offset = 0.0

            # Effective major radius for this strand
            # The strand center is at distance r_offset from the tube center
            R_eff = R + self.r_offset * np.cos(theta_offset)
            z_offset = self.r_offset * np.sin(theta_offset)

            # Scale factors for this strand's surface
            # h_phi depends on where the strand center is
            h_phi = R_eff + self.r_strand * np.cos(self.theta_grid)
            h_theta = self.r_strand
            sqrt_g = h_theta * h_phi

            # CFL
            min_h_phi = np.min(h_phi)
            if min_h_phi <= 0:
                min_h_phi = 0.01
            ds_min = min(self.r_strand * self.dtheta, min_h_phi * self.dphi)
            dt_shell = 0.4 * ds_min / np.sqrt(2)
            dt_min = min(dt_min, dt_shell)

            strand = {
                'index': i,
                'theta_offset': theta_offset,
                'R_eff': R_eff,
                'h_theta': h_theta,
                'h_phi': h_phi,
                'sqrt_g': sqrt_g,
                'E_theta': np.zeros((n_theta, n_phi)),
                'E_phi': np.zeros((n_theta, n_phi)),
                'B': np.zeros((n_theta, n_phi)),
                'P': np.zeros((n_theta, n_phi)),
            }
            self.strands.append(strand)

        self.dt = dt_min
        self.time = 0.0
        self.n_step = 0
        self.history = []

        braid_str = f"braid={braid_number}" if n_strands > 1 else "no braid"
        print(f"Braided torus: R={R}, r_tube={r_tube}, "
              f"{n_strands} strand(s), {braid_str}")
        if n_strands > 1:
            print(f"  r_offset={self.r_offset:.3f}, "
                  f"r_strand={self.r_strand:.3f}, "
                  f"coupling={coupling_strength}")
        print(f"  Grid: {n_theta}×{n_phi}, dt={self.dt:.6f}")

    def _roll_theta(self, f, shift):
        return np.roll(f, shift, axis=0)

    def _roll_phi(self, f, shift):
        return np.roll(f, shift, axis=1)

    def update(self):
        """One FDTD step for all strands."""
        dt = self.dt
        dtheta = self.dtheta
        dphi = self.dphi

        # Phase 1: Independent strand evolution
        for s in self.strands:
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

        # Phase 2: Inter-strand coupling (energy-conserving)
        # Strands that are close together couple through E fields
        if self.n_strands > 1 and self.coupling_strength > 0:
            Et_all = [s['E_theta'].copy() for s in self.strands]
            Ep_all = [s['E_phi'].copy() for s in self.strands]

            kappa = self.coupling_strength * dt
            for i in range(self.n_strands):
                for j in range(i + 1, self.n_strands):
                    # Coupling modulated by angular proximity
                    # Strands are closest when their theta offsets align
                    d_theta = self.strands[i]['theta_offset'] - \
                              self.strands[j]['theta_offset']
                    proximity = np.exp(-2.0 * (1 - np.cos(d_theta)))
                    k_eff = kappa * proximity

                    dEt = k_eff * (Et_all[j] - Et_all[i])
                    dEp = k_eff * (Ep_all[j] - Ep_all[i])
                    self.strands[i]['E_theta'] += dEt
                    self.strands[i]['E_phi'] += dEp
                    self.strands[j]['E_theta'] -= dEt
                    self.strands[j]['E_phi'] -= dEp

        self.time += dt
        self.n_step += 1

    def init_traveling_wave(self, n_toroidal=2, n_poloidal=1, amplitude=1.0,
                            strand_idx=None, phase_offset=0.0):
        """Initialize traveling wave on specified strands."""
        if strand_idx is None:
            strand_idx = list(range(self.n_strands))
        elif isinstance(strand_idx, int):
            strand_idx = [strand_idx]

        theta = self.theta_grid
        phi = self.phi_grid

        for i in strand_idx:
            # Each strand can have a phase offset (like color charge)
            phase = n_toroidal * phi + n_poloidal * theta + phase_offset
            if self.n_strands > 1:
                # Add strand-specific phase: 2π/N per strand
                phase += 2 * np.pi * i / self.n_strands

            self.strands[i]['E_theta'] = amplitude * np.cos(phase)
            self.strands[i]['B'] = amplitude * np.sin(phase)
            if n_poloidal > 0:
                self.strands[i]['E_phi'] = amplitude * 0.3 * np.sin(phase)

    def compute_energy(self, strand_idx=None):
        """Compute energy for specified strands."""
        if strand_idx is None:
            strand_idx = list(range(self.n_strands))
        elif isinstance(strand_idx, int):
            strand_idx = [strand_idx]

        total_em = 0.0
        total_pivot = 0.0
        per_strand = []

        for i in strand_idx:
            s = self.strands[i]
            dA = s['sqrt_g'] * self.dtheta * self.dphi
            e2 = s['E_theta']**2 + s['E_phi']**2
            b2 = s['B']**2
            p2 = s['P']**2
            em = 0.5 * np.sum((e2 + b2) * dA)
            pivot = 0.5 * np.sum(p2 * dA)
            total_em += em
            total_pivot += pivot
            per_strand.append({'em': em, 'pivot': pivot, 'total': em + pivot})

        return {
            'em': total_em, 'pivot': total_pivot,
            'total': total_em + total_pivot,
            'per_strand': per_strand,
        }

    def compute_braid_phase(self):
        """Compute the relative phase between strands as a function of φ.

        For braided strands, this should show a systematic rotation.
        For unbraided, phases should be independent.
        """
        if self.n_strands < 2:
            return None

        # Cross-correlation between strand 0 and strand 1 B fields
        # at each φ position
        B0 = self.strands[0]['B']
        B1 = self.strands[1]['B']

        # Phase difference via cross-correlation in θ at each φ
        phases = np.zeros(self.n_phi)
        for j in range(self.n_phi):
            b0 = B0[:, j]
            b1 = B1[:, j]
            # Cross-correlation to find phase shift
            corr = np.correlate(b0, np.tile(b1, 3), mode='valid')
            if len(corr) > 0:
                peak = np.argmax(corr)
                phases[j] = (peak - self.n_theta) * self.dtheta

        return phases

    def record_diagnostics(self):
        e = self.compute_energy()
        self.history.append({
            'time': self.time,
            'step': self.n_step,
            'em': e['em'],
            'pivot': e['pivot'],
            'total': e['total'],
            'per_strand': e['per_strand'],
        })


def run_single(R=3.0, r=1.0, steps=3000, save=False):
    """Single strand baseline (electron)."""
    print("=" * 60)
    print("Single Strand (Electron)")
    print("=" * 60)

    sim = BraidedTorusSim(R=R, r_tube=r, n_strands=1)
    sim.init_traveling_wave(n_toroidal=2, n_poloidal=1)
    sim.record_diagnostics()

    t_start = time.time()
    for step_i in range(steps):
        sim.update()
        if step_i % max(1, steps // 100) == 0:
            sim.record_diagnostics()
        if step_i % max(1, steps // 5) == 0:
            e = sim.compute_energy()
            print(f"  Step {step_i:5d}: E={e['total']:.4f} "
                  f"EM={e['em']:.4f} P²={e['pivot']:.4f}")

    elapsed = time.time() - t_start
    print(f"Done: {elapsed:.1f}s")

    h = sim.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Single Strand (Electron): R={R}, r={r}", fontsize=13)

    times = [d['time'] for d in h]
    axes[0].plot(times, [d['em'] for d in h], 'b-', label='EM')
    axes[0].plot(times, [d['pivot'] for d in h], 'r-', label='P²')
    axes[0].plot(times, [d['total'] for d in h], 'k--', label='Total')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Energy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    vmax = np.percentile(np.abs(sim.strands[0]['B']), 99)
    axes[1].imshow(sim.strands[0]['B'].T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto',
                   extent=[0, 360, 0, 360])
    axes[1].set_xlabel('θ (poloidal) [deg]')
    axes[1].set_ylabel('φ (toroidal) [deg]')
    axes[1].set_title('B field (final)')

    plt.tight_layout()
    if save:
        path = str(OUTPUT_DIR / 'braided_single.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

    return sim


def run_braided(R=3.0, r=1.0, n_strands=3, braid=1, coupling=0.5,
                steps=3000, save=False):
    """Run N braided strands and observe energy dynamics."""
    print("=" * 60)
    print(f"Braided Torus: {n_strands} strands, braid={braid}")
    print("=" * 60)

    sim = BraidedTorusSim(R=R, r_tube=r, n_strands=n_strands,
                          braid_number=braid,
                          coupling_strength=coupling)
    sim.init_traveling_wave(n_toroidal=2, n_poloidal=1)
    sim.record_diagnostics()

    t_start = time.time()
    for step_i in range(steps):
        sim.update()
        if step_i % max(1, steps // 100) == 0:
            sim.record_diagnostics()
        if step_i % max(1, steps // 5) == 0:
            e = sim.compute_energy()
            strand_str = " | ".join(
                f"S{i}:{s['total']:.3f}" for i, s in enumerate(e['per_strand']))
            print(f"  Step {step_i:5d}: total={e['total']:.4f} [{strand_str}]")

    elapsed = time.time() - t_start
    print(f"Done: {elapsed:.1f}s")

    # Plot
    h = sim.history
    n_cols = min(n_strands + 1, 4)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    fig.suptitle(
        f"Braided Torus: {n_strands} strands, braid={braid}, "
        f"R={R}, κ={coupling}\n"
        f"Traveling wave (2,1), {steps} steps",
        fontsize=12, fontweight='bold')

    times = [d['time'] for d in h]
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_strands, 3)))

    # Energy per strand
    ax = axes[0, 0]
    for i in range(n_strands):
        strand_e = [d['per_strand'][i]['total'] for d in h]
        ax.plot(times, strand_e, color=colors[i], linewidth=2,
                label=f'Strand {i}')
    ax.plot(times, [d['total'] for d in h], 'k--', linewidth=1.5, label='Total')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy per strand')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # EM vs P breakdown
    ax = axes[1, 0]
    ax.plot(times, [d['em'] for d in h], 'b-', linewidth=2, label='EM')
    ax.plot(times, [d['pivot'] for d in h], 'r-', linewidth=2, label='P²')
    ax.plot(times, [d['total'] for d in h], 'k--', linewidth=1.5, label='Total')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('EM vs Pivot')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # B field for each strand
    for col in range(1, n_cols):
        si = col - 1
        if si >= n_strands:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue

        ax = axes[0, col]
        B = sim.strands[si]['B']
        vmax = np.percentile(np.abs(B), 99)
        if vmax < 1e-10:
            vmax = 1.0
        ax.imshow(B.T, origin='lower', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, aspect='auto',
                  extent=[0, 360, 0, 360])
        ax.set_xlabel('θ [deg]')
        ax.set_ylabel('φ [deg]')
        ax.set_title(f'Strand {si} B field')

        ax = axes[1, col]
        P = sim.strands[si]['P']
        vmax_p = np.percentile(np.abs(P), 99)
        if vmax_p < 1e-10:
            vmax_p = 1.0
        ax.imshow(P.T, origin='lower', cmap='RdBu_r',
                  vmin=-vmax_p, vmax=vmax_p, aspect='auto',
                  extent=[0, 360, 0, 360])
        ax.set_xlabel('θ [deg]')
        ax.set_ylabel('φ [deg]')
        ax.set_title(f'Strand {si} P field')

    plt.tight_layout()
    if save:
        path = str(OUTPUT_DIR / f'braided_{n_strands}s_b{braid}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

    return sim


def run_spectrum(R=3.0, r=1.0, steps=3000, coupling=0.5, save=False):
    """Mass spectrum from braided strands.

    Compare:
        1 strand, no braid: electron
        2 strands, braid=0: unlinked pair (unstable meson?)
        2 strands, braid=1: linked pair (pion?)
        3 strands, braid=0: unlinked triplet (unstable baryon?)
        3 strands, braid=1: braided triplet (proton?)
        3 strands, braid=2: double-braided triplet

    Key prediction: braided configurations should have higher
    P² (rest mass) energy than unbraided, because the braid
    prevents the strands from decoupling.
    """
    print("=" * 60)
    print("Braided Strand Mass Spectrum")
    print("=" * 60)

    configs = [
        (1, 0, "electron"),
        (2, 0, "2s unlinked"),
        (2, 1, "2s linked (meson?)"),
        (3, 0, "3s unlinked"),
        (3, 1, "3s braided (proton?)"),
        (3, 2, "3s double-braid"),
    ]

    results = []
    for n_strands, braid, name in configs:
        print(f"\n--- {name}: {n_strands} strands, braid={braid} ---")

        sim = BraidedTorusSim(R=R, r_tube=r, n_strands=n_strands,
                              braid_number=braid,
                              coupling_strength=coupling)
        sim.init_traveling_wave(n_toroidal=2, n_poloidal=1)
        sim.record_diagnostics()

        t_start = time.time()
        for step_i in range(steps):
            sim.update()
            if step_i % max(1, steps // 50) == 0:
                sim.record_diagnostics()

        elapsed = time.time() - t_start
        h = sim.history

        # Time-averaged energy (last quarter)
        quarter = len(h) // 4
        avg_total = np.mean([d['total'] for d in h[-quarter:]])
        avg_em = np.mean([d['em'] for d in h[-quarter:]])
        avg_pivot = np.mean([d['pivot'] for d in h[-quarter:]])

        # P² fraction = rest mass fraction
        p2_frac = avg_pivot / avg_total if avg_total > 0 else 0

        # Energy conservation
        e0 = h[0]['total']
        ef = h[-1]['total']
        cons = abs(ef - e0) / (e0 + 1e-15) * 100

        results.append({
            'n_strands': n_strands, 'braid': braid, 'name': name,
            'avg_total': avg_total, 'avg_em': avg_em, 'avg_pivot': avg_pivot,
            'p2_frac': p2_frac, 'conservation': cons,
            'history': h,
        })

        print(f"  E={avg_total:.4f} (EM={avg_em:.4f}, P²={avg_pivot:.4f}) "
              f"P²frac={p2_frac:.3f} cons={cons:.1f}% [{elapsed:.1f}s]")

    # Mass ratios
    e_electron = results[0]['avg_total']
    print("\n" + "=" * 70)
    print("BRAIDED STRAND MASS SPECTRUM")
    print("=" * 70)
    print(f"{'Config':>22s} {'Strands':>7s} {'Braid':>6s} {'Energy':>10s} "
          f"{'Ratio':>8s} {'P²':>10s} {'P²frac':>8s}")
    print("-" * 70)

    for r_item in results:
        ratio = r_item['avg_total'] / e_electron if e_electron > 0 else 0
        print(f"{r_item['name']:>22s} {r_item['n_strands']:>7d} "
              f"{r_item['braid']:>6d} {r_item['avg_total']:>10.4f} "
              f"{ratio:>8.3f}x {r_item['avg_pivot']:>10.4f} "
              f"{r_item['p2_frac']:>8.3f}")

    print("\nKnown mass ratios:")
    print("  pion±/electron  = 273.1")
    print("  proton/electron = 1836.2")
    print("  neutron/electron = 1838.7")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Braided Strand Mass Spectrum — Williamson Model\n"
        f"R={R}, r={r}, κ={coupling}, {steps} steps",
        fontsize=13, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

    # Top-left: Energy vs time
    ax = axes[0, 0]
    for i, r_item in enumerate(results):
        h = r_item['history']
        ax.plot([d['time'] for d in h], [d['total'] for d in h],
                color=colors[i], linewidth=2, label=r_item['name'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Evolution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Top-middle: P² vs time
    ax = axes[0, 1]
    for i, r_item in enumerate(results):
        h = r_item['history']
        ax.plot([d['time'] for d in h], [d['pivot'] for d in h],
                color=colors[i], linewidth=2, label=r_item['name'])
    ax.set_xlabel('Time')
    ax.set_ylabel('P² (rest mass)')
    ax.set_title('Pivot Energy (Mass)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Top-right: P² fraction vs time
    ax = axes[0, 2]
    for i, r_item in enumerate(results):
        h = r_item['history']
        fracs = [d['pivot'] / (d['total'] + 1e-15) for d in h]
        ax.plot([d['time'] for d in h], fracs,
                color=colors[i], linewidth=2, label=r_item['name'])
    ax.set_xlabel('Time')
    ax.set_ylabel('P² / Total')
    ax.set_title('Rest Mass Fraction')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Mass ratio bar chart
    ax = axes[1, 0]
    labels = [r_item['name'] for r_item in results]
    ratios = [r_item['avg_total'] / e_electron for r_item in results]
    x = np.arange(len(results))
    bars = ax.bar(x, ratios, color=colors[:len(results)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Mass ratio (m / m_electron)')
    ax.set_title('Mass Ratios')
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-middle: braided vs unbraided comparison
    ax = axes[1, 1]
    # Group by strand count
    for ns in [2, 3]:
        sub = [r_item for r_item in results if r_item['n_strands'] == ns]
        braids = [r_item['braid'] for r_item in sub]
        energies = [r_item['avg_total'] for r_item in sub]
        pivots = [r_item['avg_pivot'] for r_item in sub]
        ax.plot(braids, energies, 'o-', markersize=10, linewidth=2,
                label=f'{ns} strands (total)')
        ax.plot(braids, pivots, 's--', markersize=8, linewidth=1.5,
                label=f'{ns} strands (P²)')
    ax.set_xlabel('Braid Number')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs Braiding')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: energy composition
    ax = axes[1, 2]
    em_vals = [r_item['avg_em'] for r_item in results]
    pv_vals = [r_item['avg_pivot'] for r_item in results]
    ax.bar(x - 0.15, em_vals, 0.3, color='blue', alpha=0.7, label='EM')
    ax.bar(x + 0.15, pv_vals, 0.3, color='red', alpha=0.7, label='P²')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Energy')
    ax.set_title('Energy Composition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = str(OUTPUT_DIR / 'braided_spectrum.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Braided torus — multi-strand Williamson model')
    parser.add_argument('--mode', choices=['single', 'braided', 'spectrum'],
                        default='spectrum')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--strands', type=int, default=3)
    parser.add_argument('--braid', type=int, default=1)
    parser.add_argument('--coupling', type=float, default=0.5)
    parser.add_argument('--R', type=float, default=3.0)
    parser.add_argument('--r', type=float, default=1.0)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.mode == 'single':
        run_single(R=args.R, r=args.r, steps=args.steps, save=args.save)
    elif args.mode == 'braided':
        run_braided(R=args.R, r=args.r, n_strands=args.strands,
                    braid=args.braid, coupling=args.coupling,
                    steps=args.steps, save=args.save)
    elif args.mode == 'spectrum':
        run_spectrum(R=args.R, r=args.r, steps=args.steps,
                     coupling=args.coupling, save=args.save)


if __name__ == '__main__':
    main()
