#!/usr/bin/env python3
"""
Williamson Pivot Field Simulation
==================================

Implements the extended Maxwell equations from John Williamson's
"A new theory of light and matter" (Frontiers of Fundamental Physics 14, 2014).

The pivot field P is a scalar field that extends Maxwell's equations:

    ∂B/∂t = -∇×E                    (Faraday, unchanged)
    ∂E/∂t = ∇×B - ∇P               (Ampere + pivot gradient)
    ∂P/∂t = -∇·E                    (pivot evolution)
    ∇·B = 0                         (no monopoles, unchanged)

Energy-momentum density:
    M = ½(E² + B² + P²) + (E×B + P·E)

The P² term is rest mass-energy density.
The P·E term redirects momentum toward E, enabling closed flows (confinement).

2D TE mode: Ex, Ey, Bz, P on Yee grid.

Usage:
    python3 simulations/williamson_pivot.py [--no-pivot] [--steps N] [--save]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class PivotFieldSimulation:
    """2D FDTD simulation with Williamson pivot field."""

    def __init__(self, nx=200, ny=200, dx=1.0, courant=0.5, pivot_enabled=True):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx  # square cells
        self.dt = courant * dx  # CFL condition: c*dt/dx <= 1/sqrt(2) for 2D
        self.courant = courant
        self.pivot_enabled = pivot_enabled

        # TE mode fields on Yee grid:
        #   Ex at (i+½, j)    -> stored as Ex[i, j] represents (i+0.5, j)
        #   Ey at (i, j+½)    -> stored as Ey[i, j] represents (i, j+0.5)
        #   Bz at (i+½, j+½)  -> stored as Bz[i, j] represents (i+0.5, j+0.5)
        #   P  at (i, j)      -> stored as P[i, j]  represents (i, j)
        self.Ex = np.zeros((nx, ny))
        self.Ey = np.zeros((nx, ny))
        self.Bz = np.zeros((nx, ny))
        self.P = np.zeros((nx, ny))

        # Diagnostics
        self.time = 0.0
        self.step = 0
        self.energy_history = []

    def _apply_abc(self):
        """Simple first-order Mur absorbing boundary conditions."""
        c = self.courant
        factor = (c - 1) / (c + 1)

        # Store old boundary values (set before update)
        # For simplicity, use zero-gradient (open) boundaries
        self.Ex[0, :] = self.Ex[1, :]
        self.Ex[-1, :] = self.Ex[-2, :]
        self.Ex[:, 0] = self.Ex[:, 1]
        self.Ex[:, -1] = self.Ex[:, -2]

        self.Ey[0, :] = self.Ey[1, :]
        self.Ey[-1, :] = self.Ey[-2, :]
        self.Ey[:, 0] = self.Ey[:, 1]
        self.Ey[:, -1] = self.Ey[:, -2]

        self.P[0, :] = self.P[1, :]
        self.P[-1, :] = self.P[-2, :]
        self.P[:, 0] = self.P[:, 1]
        self.P[:, -1] = self.P[:, -2]

    def update(self):
        """Single FDTD time step with pivot field."""
        dt = self.dt
        dx = self.dx
        dy = self.dy
        Ex, Ey, Bz, P = self.Ex, self.Ey, self.Bz, self.P

        # 1. Update Bz: ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)
        #    Bz at (i+½, j+½), Ey at (i, j+½), Ex at (i+½, j)
        Bz[:-1, :-1] -= dt / dx * (Ey[1:, :-1] - Ey[:-1, :-1]) \
                       - dt / dy * (Ex[:-1, 1:] - Ex[:-1, :-1])

        # 2. Update Ex: ∂Ex/∂t = ∂Bz/∂y - ∂P/∂x
        #    Ex at (i+½, j), Bz at (i+½, j+½), P at (i, j)
        #    ∂Bz/∂y at (i+½, j): (Bz[i, j] - Bz[i, j-1]) / dy
        #    ∂P/∂x at (i+½, j): (P[i+1, j] - P[i, j]) / dx
        Ex[:-1, 1:-1] += dt / dy * (Bz[:-1, 1:-1] - Bz[:-1, :-2])
        if self.pivot_enabled:
            Ex[:-1, :] -= dt / dx * (P[1:, :] - P[:-1, :])

        # 3. Update Ey: ∂Ey/∂t = -∂Bz/∂x - ∂P/∂y
        #    Ey at (i, j+½), Bz at (i+½, j+½), P at (i, j)
        #    ∂Bz/∂x at (i, j+½): (Bz[i, j] - Bz[i-1, j]) / dx
        #    ∂P/∂y at (i, j+½): (P[i, j+1] - P[i, j]) / dy
        Ey[1:-1, :-1] -= dt / dx * (Bz[1:-1, :-1] - Bz[:-2, :-1])
        if self.pivot_enabled:
            Ey[:, :-1] -= dt / dy * (P[:, 1:] - P[:, :-1])

        # 4. Update P: ∂P/∂t = -(∂Ex/∂x + ∂Ey/∂y)
        #    P at (i, j), Ex at (i+½, j), Ey at (i, j+½)
        #    ∂Ex/∂x at (i, j): (Ex[i, j] - Ex[i-1, j]) / dx
        #    ∂Ey/∂y at (i, j): (Ey[i, j] - Ey[i, j-1]) / dy
        if self.pivot_enabled:
            P[1:, 1:] -= dt * (
                (Ex[1:, 1:] - Ex[:-1, 1:]) / dx +
                (Ey[1:, 1:] - Ey[1:, :-1]) / dy
            )

        # Absorbing boundaries
        self._apply_abc()

        self.time += dt
        self.step += 1

    def compute_energy(self):
        """Compute field energy components."""
        em_energy = 0.5 * np.sum(self.Ex**2 + self.Ey**2 + self.Bz**2) * self.dx * self.dy
        pivot_energy = 0.5 * np.sum(self.P**2) * self.dx * self.dy
        total = em_energy + pivot_energy
        return {"em": em_energy, "pivot": pivot_energy, "total": total}

    def compute_energy_density(self):
        """Compute total energy density ½(E² + B² + P²)."""
        return 0.5 * (self.Ex**2 + self.Ey**2 + self.Bz**2 + self.P**2)

    def compute_poynting_magnitude(self):
        """Compute |E×B + P·E| (magnitude of momentum density)."""
        # In 2D TE: E×B has only z-component contributions to in-plane Poynting
        # Standard Poynting: S = E×B (in-plane for TE mode)
        # Sx ~ Ey*Bz, Sy ~ -Ex*Bz
        # Pivot momentum: P*Ex, P*Ey
        Sx = self.Ey * self.Bz + self.P * self.Ex
        Sy = -self.Ex * self.Bz + self.P * self.Ey
        return np.sqrt(Sx**2 + Sy**2)

    def compute_confined_energy_fraction(self, radius=None):
        """Fraction of total energy within given radius of center."""
        if radius is None:
            radius = self.nx // 4
        cx, cy = self.nx // 2, self.ny // 2
        y, x = np.ogrid[:self.nx, :self.ny]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        total = self.compute_energy()["total"]
        if total < 1e-15:
            return 0.0
        confined = 0.5 * np.sum(
            (self.Ex**2 + self.Ey**2 + self.Bz**2 + self.P**2) * mask
        ) * self.dx * self.dy
        return confined / total


def init_gaussian_pulse(sim, sigma=10.0, k0=0.5):
    """Initialize with a Gaussian EM pulse at center.

    Creates a localized wave packet with non-zero ∇·E to seed the pivot.
    Uses a radially-polarized E field (divergent) to maximize pivot coupling.
    """
    cx, cy = sim.nx // 2, sim.ny // 2
    x = np.arange(sim.nx) - cx
    y = np.arange(sim.ny) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2) + 1e-10

    # Gaussian envelope
    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Radially polarized E field (has non-zero divergence)
    # E = E0 * r_hat * f(r) * cos(k0*r) for radial component
    # Plus a circulating component for angular momentum
    radial = envelope * np.cos(k0 * R)
    azimuthal = envelope * np.sin(k0 * R)

    sim.Ex = (X / R * radial - Y / R * azimuthal) * 0.5
    sim.Ey = (Y / R * radial + X / R * azimuthal) * 0.5

    # Consistent Bz from Faraday's law (approximate)
    # For a radially polarized pulse, Bz ~ 0 initially
    # For the azimuthal component, Bz is non-zero
    sim.Bz = azimuthal * 0.3


def init_circular_pulse(sim, radius=20.0, width=5.0, k0=1.0):
    """Initialize with a circularly propagating EM pulse.

    This more directly mimics Williamson's confined photon —
    a wave packet propagating around a circle.
    """
    cx, cy = sim.nx // 2, sim.ny // 2
    x = np.arange(sim.nx) - cx
    y = np.arange(sim.ny) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2) + 1e-10
    theta = np.arctan2(Y, X)

    # Ring-shaped envelope centered at radius
    ring = np.exp(-((R - radius)**2) / (2 * width**2))

    # Circulating wave: fields propagate azimuthally
    # E is radial, B is axial (z), wave propagates in theta direction
    phase = k0 * radius * theta  # phase increases with angle

    # Radial E field (points inward/outward depending on phase)
    Er = ring * np.cos(phase)
    sim.Ex = Er * X / R
    sim.Ey = Er * Y / R

    # Bz from the circulating wave
    sim.Bz = ring * np.sin(phase)


def init_dipole_vortex(sim, sigma=15.0, strength=1.0):
    """Initialize with a vortex-like E field configuration.

    Creates a configuration with strong ∇·E to maximally excite the pivot.
    """
    cx, cy = sim.nx // 2, sim.ny // 2
    x = np.arange(sim.nx) - cx
    y = np.arange(sim.ny) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2) + 1e-10

    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Divergent + rotational E field
    # Divergent part (sources pivot)
    sim.Ex = strength * X / sigma * envelope
    sim.Ey = strength * Y / sigma * envelope

    # Bz consistent with a pulse
    sim.Bz = strength * envelope * 0.5


def run_comparison(steps=500, save=False):
    """Run pivot vs no-pivot comparison."""
    print("Williamson Pivot Field Simulation")
    print("=" * 50)
    print(f"Grid: 200x200, Steps: {steps}")
    print()

    # Create two simulations
    sim_pivot = PivotFieldSimulation(nx=200, ny=200, pivot_enabled=True)
    sim_nopivot = PivotFieldSimulation(nx=200, ny=200, pivot_enabled=False)

    # Same initial conditions
    init_gaussian_pulse(sim_pivot)
    init_gaussian_pulse(sim_nopivot)

    # Track energy confinement
    pivot_confinement = []
    nopivot_confinement = []
    pivot_energies = []
    nopivot_energies = []
    pivot_mass = []  # P² energy (rest mass density)

    measure_interval = 5
    for step in range(steps):
        sim_pivot.update()
        sim_nopivot.update()

        if step % measure_interval == 0:
            pivot_confinement.append(sim_pivot.compute_confined_energy_fraction(radius=50))
            nopivot_confinement.append(sim_nopivot.compute_confined_energy_fraction(radius=50))
            pe = sim_pivot.compute_energy()
            ne = sim_nopivot.compute_energy()
            pivot_energies.append(pe["total"])
            nopivot_energies.append(ne["total"])
            pivot_mass.append(pe["pivot"])

        if step % 100 == 0:
            pf = sim_pivot.compute_confined_energy_fraction(radius=50)
            nf = sim_nopivot.compute_confined_energy_fraction(radius=50)
            pe = sim_pivot.compute_energy()
            print(f"  Step {step:4d}: confined(pivot)={pf:.3f}  "
                  f"confined(maxwell)={nf:.3f}  "
                  f"P²_energy={pe['pivot']:.4f}")

    times = np.arange(0, len(pivot_confinement)) * measure_interval * sim_pivot.dt

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Williamson Pivot Field: Extended Maxwell Equations\n"
                 "d(F + P) = 0  →  ∂E/∂t = ∇×B - ∇P,  ∂P/∂t = -∇·E",
                 fontsize=13, fontweight='bold')

    # Energy density snapshots
    ed_pivot = sim_pivot.compute_energy_density()
    ed_nopivot = sim_nopivot.compute_energy_density()
    vmax = max(ed_pivot.max(), ed_nopivot.max(), 1e-10)

    ax = axes[0, 0]
    im = ax.imshow(ed_pivot.T, origin='lower', cmap='inferno',
                   vmin=0, vmax=vmax * 0.5)
    ax.set_title(f'Energy density (with pivot)\nt = {sim_pivot.time:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='½(E²+B²+P²)')

    ax = axes[0, 1]
    im = ax.imshow(ed_nopivot.T, origin='lower', cmap='inferno',
                   vmin=0, vmax=vmax * 0.5)
    ax.set_title(f'Energy density (standard Maxwell)\nt = {sim_nopivot.time:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='½(E²+B²)')

    # Pivot field
    ax = axes[0, 2]
    pmax = max(abs(sim_pivot.P).max(), 1e-10)
    im = ax.imshow(sim_pivot.P.T, origin='lower', cmap='RdBu_r',
                   vmin=-pmax, vmax=pmax)
    ax.set_title('Pivot field P (rest-mass density)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='P')

    # Energy confinement
    ax = axes[1, 0]
    ax.plot(times, pivot_confinement, 'b-', linewidth=2, label='With pivot (P ≠ 0)')
    ax.plot(times, nopivot_confinement, 'r--', linewidth=2, label='Standard Maxwell (P = 0)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy fraction within r=50')
    ax.set_title('Energy Confinement')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Momentum density
    ax = axes[1, 1]
    mom_pivot = sim_pivot.compute_poynting_magnitude()
    im = ax.imshow(mom_pivot.T, origin='lower', cmap='viridis',
                   vmin=0, vmax=mom_pivot.max() * 0.5 + 1e-10)
    ax.set_title('|E×B + P·E| momentum density\n(pivot redirects flow)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='|S|')

    # Energy components
    ax = axes[1, 2]
    ax.plot(times, pivot_energies, 'b-', linewidth=2, label='Total (pivot)')
    ax.plot(times, nopivot_energies, 'r--', linewidth=2, label='Total (Maxwell)')
    ax.plot(times, pivot_mass, 'g-', linewidth=1.5, label='P² (rest mass)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(str(OUTPUT_DIR / 'pivot_comparison.png'),
                    dpi=150, bbox_inches='tight')
        print(f"\nSaved to simulations/pivot_comparison.png")
    else:
        plt.show()


def run_animation(init_type='gaussian', steps=1000, save=False):
    """Run animated simulation."""
    print(f"Animating pivot field simulation ({init_type} init)...")

    sim = PivotFieldSimulation(nx=200, ny=200, pivot_enabled=True)

    if init_type == 'circular':
        init_circular_pulse(sim)
    elif init_type == 'vortex':
        init_dipole_vortex(sim)
    else:
        init_gaussian_pulse(sim)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Williamson Pivot Field: d(F + P) = 0", fontsize=13, fontweight='bold')

    # Initial plots
    ed = sim.compute_energy_density()
    im_energy = axes[0].imshow(ed.T, origin='lower', cmap='inferno',
                                vmin=0, vmax=0.1)
    axes[0].set_title('Energy density')
    plt.colorbar(im_energy, ax=axes[0])

    pmax = 0.05
    im_pivot = axes[1].imshow(sim.P.T, origin='lower', cmap='RdBu_r',
                               vmin=-pmax, vmax=pmax)
    axes[1].set_title('Pivot field P')
    plt.colorbar(im_pivot, ax=axes[1])

    mom = sim.compute_poynting_magnitude()
    im_mom = axes[2].imshow(mom.T, origin='lower', cmap='viridis',
                             vmin=0, vmax=0.05)
    axes[2].set_title('Momentum density')
    plt.colorbar(im_mom, ax=axes[2])

    steps_per_frame = 5

    def animate(frame):
        for _ in range(steps_per_frame):
            sim.update()

        ed = sim.compute_energy_density()
        im_energy.set_data(ed.T)
        vmax_e = max(ed.max() * 0.8, 1e-10)
        im_energy.set_clim(0, vmax_e)

        im_pivot.set_data(sim.P.T)
        pmax = max(abs(sim.P).max() * 0.8, 1e-10)
        im_pivot.set_clim(-pmax, pmax)

        mom = sim.compute_poynting_magnitude()
        im_mom.set_data(mom.T)
        im_mom.set_clim(0, max(mom.max() * 0.8, 1e-10))

        fig.suptitle(f"Williamson Pivot Field: d(F + P) = 0    "
                     f"t = {sim.time:.1f}  step = {sim.step}",
                     fontsize=13, fontweight='bold')
        return im_energy, im_pivot, im_mom

    n_frames = steps // steps_per_frame
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                    interval=50, blit=False)

    if save:
        outpath = str(OUTPUT_DIR / f'pivot_{init_type}.mp4')
        anim.save(outpath, writer='ffmpeg', fps=30, dpi=100)
        print(f"Saved to {outpath}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Williamson Pivot Field Simulation')
    parser.add_argument('--mode', choices=['compare', 'animate'], default='compare',
                        help='Run mode: compare pivot vs Maxwell, or animate')
    parser.add_argument('--init', choices=['gaussian', 'circular', 'vortex'],
                        default='gaussian', help='Initial condition type')
    parser.add_argument('--steps', type=int, default=500, help='Number of time steps')
    parser.add_argument('--save', action='store_true', help='Save output to file')

    args = parser.parse_args()

    if args.mode == 'compare':
        run_comparison(steps=args.steps, save=args.save)
    else:
        run_animation(init_type=args.init, steps=args.steps, save=args.save)
