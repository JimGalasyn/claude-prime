#!/usr/bin/env python3
"""
3D Williamson Pivot Field Simulation
======================================

Full 3D FDTD implementation of the extended Maxwell equations from
John Williamson's "A new theory of light and matter" (FFP14, 2014).

    ∂B/∂t = -∇×E                    (Faraday, unchanged)
    ∂E/∂t = ∇×B - ∇P               (Ampere + pivot gradient)
    ∂P/∂t = -∇·E                    (pivot evolution)

7 field components on Yee grid: Ex, Ey, Ez, Bx, By, Bz, P

Energy-momentum density:
    M = ½(E² + B² + P²) + (E×B + P·E)

Usage:
    python3 williamson_pivot_3d.py --mode compare --steps 300 --save
    python3 williamson_pivot_3d.py --mode animate --init toroidal --steps 500
    python3 williamson_pivot_3d.py --mode snapshot --init toroidal --steps 200 --save
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import time


class PivotFieldSimulation3D:
    """Full 3D FDTD simulation with Williamson pivot field.

    Yee grid staggering:
        Ex at (i+½, j,   k  )
        Ey at (i,   j+½, k  )
        Ez at (i,   j,   k+½)
        Bx at (i,   j+½, k+½)
        By at (i+½, j,   k+½)
        Bz at (i+½, j+½, k  )
        P  at (i,   j,   k  )
    """

    def __init__(self, n=64, dx=1.0, courant=0.3, pivot_enabled=True,
                 boundary='absorbing'):
        self.n = n
        self.nx = self.ny = self.nz = n
        self.dx = self.dy = self.dz = dx
        # CFL stability: dt <= dx / (c * sqrt(3)) for 3D
        self.dt = courant * dx
        self.courant = courant
        self.pivot_enabled = pivot_enabled
        self.boundary = boundary  # 'absorbing' or 'periodic'

        # Allocate fields
        shape = (n, n, n)
        self.Ex = np.zeros(shape)
        self.Ey = np.zeros(shape)
        self.Ez = np.zeros(shape)
        self.Bx = np.zeros(shape)
        self.By = np.zeros(shape)
        self.Bz = np.zeros(shape)
        self.P = np.zeros(shape)

        # Absorbing boundary damping profile
        if boundary == 'absorbing':
            self._build_damping_profile(width=8)
        else:
            self.damping = np.ones(shape)

        self.time = 0.0
        self.step = 0
        self.energy_history = []

        mem_mb = 7 * n**3 * 8 / 1e6
        print(f"3D Pivot Field: {n}³ grid, {mem_mb:.1f} MB, "
              f"dt={self.dt:.4f}, pivot={'ON' if pivot_enabled else 'OFF'}, "
              f"BC={boundary}")

    def _build_damping_profile(self, width=8):
        """Build a 3D damping profile for absorbing boundaries."""
        n = self.n
        profile_1d = np.ones(n)
        for i in range(width):
            damp = (i / width) ** 2  # quadratic ramp
            profile_1d[i] = damp
            profile_1d[n - 1 - i] = damp
        # 3D profile is product of three 1D profiles
        px = profile_1d[:, np.newaxis, np.newaxis]
        py = profile_1d[np.newaxis, :, np.newaxis]
        pz = profile_1d[np.newaxis, np.newaxis, :]
        self.damping = px * py * pz

    def update(self):
        """Single 3D FDTD time step with pivot field."""
        dt = self.dt
        dx, dy, dz = self.dx, self.dy, self.dz
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        Bx, By, Bz = self.Bx, self.By, self.Bz
        P = self.P

        # === 1. Update B: ∂B/∂t = -∇×E ===
        # Bx at (i, j+½, k+½)
        # ∂Ez/∂y at (i, j+½, k+½): Ez(i,j+1,k+½) - Ez(i,j,k+½)
        # ∂Ey/∂z at (i, j+½, k+½): Ey(i,j+½,k+1) - Ey(i,j+½,k)
        Bx[:, :-1, :-1] -= dt * (
            (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dy -
            (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dz
        )

        # By at (i+½, j, k+½)
        # ∂Ex/∂z - ∂Ez/∂x
        By[:-1, :, :-1] -= dt * (
            (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / dz -
            (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx
        )

        # Bz at (i+½, j+½, k)
        # ∂Ey/∂x - ∂Ex/∂y
        Bz[:-1, :-1, :] -= dt * (
            (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dx -
            (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / dy
        )

        # === 2. Update E: ∂E/∂t = ∇×B - ∇P ===
        # Ex at (i+½, j, k)
        # ∂Bz/∂y - ∂By/∂z - ∂P/∂x
        Ex[:-1, 1:, 1:] += dt * (
            (Bz[:-1, 1:, 1:] - Bz[:-1, :-1, 1:]) / dy -
            (By[:-1, 1:, 1:] - By[:-1, 1:, :-1]) / dz
        )
        if self.pivot_enabled:
            Ex[:-1, :, :] -= dt / dx * (P[1:, :, :] - P[:-1, :, :])

        # Ey at (i, j+½, k)
        # ∂Bx/∂z - ∂Bz/∂x - ∂P/∂y
        Ey[1:, :-1, 1:] += dt * (
            (Bx[1:, :-1, 1:] - Bx[1:, :-1, :-1]) / dz -
            (Bz[1:, :-1, 1:] - Bz[:-1, :-1, 1:]) / dx
        )
        if self.pivot_enabled:
            Ey[:, :-1, :] -= dt / dy * (P[:, 1:, :] - P[:, :-1, :])

        # Ez at (i, j, k+½)
        # ∂By/∂x - ∂Bx/∂y - ∂P/∂z
        Ez[1:, 1:, :-1] += dt * (
            (By[1:, 1:, :-1] - By[:-1, 1:, :-1]) / dx -
            (Bx[1:, 1:, :-1] - Bx[1:, :-1, :-1]) / dy
        )
        if self.pivot_enabled:
            Ez[:, :, :-1] -= dt / dz * (P[:, :, 1:] - P[:, :, :-1])

        # === 3. Update P: ∂P/∂t = -∇·E ===
        if self.pivot_enabled:
            P[1:, 1:, 1:] -= dt * (
                (Ex[1:, 1:, 1:] - Ex[:-1, 1:, 1:]) / dx +
                (Ey[1:, 1:, 1:] - Ey[1:, :-1, 1:]) / dy +
                (Ez[1:, 1:, 1:] - Ez[1:, 1:, :-1]) / dz
            )

        # === 4. Absorbing boundaries ===
        self.Ex *= self.damping
        self.Ey *= self.damping
        self.Ez *= self.damping
        self.Bx *= self.damping
        self.By *= self.damping
        self.Bz *= self.damping
        self.P *= self.damping

        self.time += dt
        self.step += 1

    def compute_energy(self):
        """Compute field energy components."""
        dv = self.dx * self.dy * self.dz
        e2 = np.sum(self.Ex**2 + self.Ey**2 + self.Ez**2)
        b2 = np.sum(self.Bx**2 + self.By**2 + self.Bz**2)
        p2 = np.sum(self.P**2)
        em = 0.5 * (e2 + b2) * dv
        pivot = 0.5 * p2 * dv
        return {"em": em, "pivot": pivot, "total": em + pivot}

    def compute_energy_density_slice(self, axis='z', index=None):
        """Compute energy density on a 2D slice."""
        if index is None:
            index = self.n // 2
        sl = [slice(None)] * 3
        sl[{'x': 0, 'y': 1, 'z': 2}[axis]] = index

        e2 = self.Ex[tuple(sl)]**2 + self.Ey[tuple(sl)]**2 + self.Ez[tuple(sl)]**2
        b2 = self.Bx[tuple(sl)]**2 + self.By[tuple(sl)]**2 + self.Bz[tuple(sl)]**2
        p2 = self.P[tuple(sl)]**2
        return 0.5 * (e2 + b2 + p2)

    def compute_pivot_slice(self, axis='z', index=None):
        """Get pivot field on a 2D slice."""
        if index is None:
            index = self.n // 2
        sl = [slice(None)] * 3
        sl[{'x': 0, 'y': 1, 'z': 2}[axis]] = index
        return self.P[tuple(sl)]

    def compute_confined_fraction(self, center=None, radius=None):
        """Fraction of total energy within a sphere of given radius."""
        n = self.n
        if center is None:
            center = (n // 2, n // 2, n // 2)
        if radius is None:
            radius = n // 4

        z, y, x = np.ogrid[:n, :n, :n]
        r2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        mask = r2 <= radius**2

        total_e2 = self.Ex**2 + self.Ey**2 + self.Ez**2 + \
                   self.Bx**2 + self.By**2 + self.Bz**2 + self.P**2
        total = np.sum(total_e2)
        if total < 1e-20:
            return 0.0
        return np.sum(total_e2 * mask) / total


# ============================================================
# Initial Conditions
# ============================================================

def init_toroidal(sim, major_radius=None, minor_radius=None,
                  n_wavelengths=2, amplitude=1.0):
    """Initialize with a photon circulating on a torus.

    Williamson's model: a single-wavelength photon confined to a
    toroidal path. The photon makes a double loop (4π phase per circuit)
    to produce the half-integer spin topology.

    Args:
        major_radius: Distance from torus center to tube center
        minor_radius: Radius of the tube
        n_wavelengths: Number of wavelengths around the torus (2 = double loop)
        amplitude: Field strength
    """
    n = sim.n
    if major_radius is None:
        major_radius = n * 0.2
    if minor_radius is None:
        minor_radius = major_radius * 0.3

    cx, cy, cz = n // 2, n // 2, n // 2

    # Build coordinate arrays
    x = np.arange(n) - cx
    y = np.arange(n) - cy
    z = np.arange(n) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Cylindrical coordinates from torus axis (z-axis)
    rho = np.sqrt(X**2 + Y**2) + 1e-10  # distance from z-axis
    phi = np.arctan2(Y, X)              # toroidal angle

    # Distance from torus ring (in the rho-z plane)
    d_rho = rho - major_radius
    d_torus = np.sqrt(d_rho**2 + Z**2)  # distance from torus skeleton
    theta = np.arctan2(Z, d_rho)        # poloidal angle

    # Toroidal envelope: Gaussian tube around the ring
    envelope = amplitude * np.exp(-d_torus**2 / (2 * minor_radius**2))

    # Phase: n_wavelengths complete oscillations around the torus
    # For Williamson double-loop: n_wavelengths=2 gives 4π per circuit
    phase = n_wavelengths * phi

    # E field: radial (pointing toward/away from torus axis)
    # In Williamson's model, E is always radial for the confined state
    # This creates ∇·E ≠ 0, which seeds the pivot field
    Er = envelope * np.cos(phase)

    # Convert radial E to Cartesian
    sim.Ex = Er * X / rho
    sim.Ey = Er * Y / rho
    sim.Ez = np.zeros_like(Er)

    # B field: poloidal (circulating around the tube cross-section)
    # B_theta in the poloidal direction
    Bp = envelope * np.sin(phase)

    # Convert poloidal B to Cartesian
    # Poloidal direction in (rho, z) plane: (-sin(theta), cos(theta))
    # In Cartesian: (-sin(theta)*cos(phi), -sin(theta)*sin(phi), cos(theta))
    cos_phi = X / rho
    sin_phi = Y / rho
    cos_theta = d_rho / (d_torus + 1e-10)
    sin_theta = Z / (d_torus + 1e-10)

    sim.Bx = Bp * (-sin_theta * cos_phi)
    sim.By = Bp * (-sin_theta * sin_phi)
    sim.Bz = Bp * cos_theta

    # Add a toroidal B component for angular momentum
    Bt = envelope * np.sin(phase) * 0.3
    sim.Bx += Bt * (-sin_phi)
    sim.By += Bt * cos_phi

    print(f"Toroidal init: R={major_radius:.1f}, a={minor_radius:.1f}, "
          f"n_λ={n_wavelengths}, E_max={np.max(np.abs(Er)):.3f}")


def init_gaussian(sim, sigma=None, amplitude=1.0):
    """Initialize with a 3D Gaussian EM pulse (radially polarized)."""
    n = sim.n
    if sigma is None:
        sigma = n * 0.08

    cx, cy, cz = n // 2, n // 2, n // 2
    x = np.arange(n) - cx
    y = np.arange(n) - cy
    z = np.arange(n) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-10

    envelope = amplitude * np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))

    # Radially polarized E (divergent → seeds pivot)
    sim.Ex = envelope * X / sigma
    sim.Ey = envelope * Y / sigma
    sim.Ez = envelope * Z / sigma

    print(f"Gaussian init: σ={sigma:.1f}")


def init_ring_current(sim, radius=None, width=None, amplitude=1.0):
    """Initialize with a current ring in the xy-plane.

    Creates a magnetic dipole configuration — simpler than full toroidal
    but still has the right symmetry for testing confinement.
    """
    n = sim.n
    if radius is None:
        radius = n * 0.2
    if width is None:
        width = radius * 0.2

    cx, cy, cz = n // 2, n // 2, n // 2
    x = np.arange(n) - cx
    y = np.arange(n) - cy
    z = np.arange(n) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    rho = np.sqrt(X**2 + Y**2) + 1e-10
    phi = np.arctan2(Y, X)

    # Ring-shaped current in xy-plane
    ring = amplitude * np.exp(-((rho - radius)**2 + Z**2) / (2 * width**2))

    # Azimuthal E field (like a current ring)
    sim.Ex = ring * (-Y / rho)
    sim.Ey = ring * (X / rho)
    sim.Ez = np.zeros_like(ring)

    # Bz from the ring (dipole-like)
    sim.Bz = ring * 0.5

    print(f"Ring current init: R={radius:.1f}, w={width:.1f}")


def _poisson_solve_3d(rhs, dx=1.0):
    """Solve ∇²P = rhs using FFT (periodic BCs).

    Returns P such that the Laplacian of P equals rhs.
    """
    n = rhs.shape[0]
    rhs_hat = np.fft.fftn(rhs)

    # Laplacian eigenvalues in Fourier space
    kx = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # avoid division by zero (DC component)

    P_hat = rhs_hat / (-k2)
    P_hat[0, 0, 0] = 0.0  # zero-mean solution

    return np.real(np.fft.ifftn(P_hat))


def _compute_div_E(sim):
    """Compute ∇·E on the grid."""
    dx, dy, dz = sim.dx, sim.dy, sim.dz
    divE = np.zeros_like(sim.Ex)
    divE[1:, :, :] += (sim.Ex[1:, :, :] - sim.Ex[:-1, :, :]) / dx
    divE[:, 1:, :] += (sim.Ey[:, 1:, :] - sim.Ey[:, :-1, :]) / dy
    divE[:, :, 1:] += (sim.Ez[:, :, 1:] - sim.Ez[:, :, :-1]) / dz
    return divE


def init_toroidal_propagating(sim, major_radius=None, minor_radius=None,
                               n_wavelengths=2, amplitude=0.5,
                               self_consistent_P=True):
    """Initialize a PROPAGATING photon circulating on a torus.

    Key difference from init_toroidal: E and B are in-phase (traveling wave),
    so E×B points in the toroidal direction — energy actually circulates.

    For a wave propagating in the φ̂ direction:
        E = E₀ f(d) cos(nφ) ρ̂        (radial → creates ∇·E ≠ 0)
        B = -E₀ f(d) cos(nφ) ẑ        (from B = (1/c)φ̂×E)
        E×B = E₀² f² cos²(nφ) φ̂       (always positive → circulating)

    Optionally solves for self-consistent P₀ so there's no initial transient.
    """
    n = sim.n
    if major_radius is None:
        major_radius = n * 0.25
    if minor_radius is None:
        minor_radius = major_radius * 0.25

    cx, cy, cz = n // 2, n // 2, n // 2
    x = np.arange(n) - cx
    y = np.arange(n) - cy
    z = np.arange(n) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    rho = np.sqrt(X**2 + Y**2) + 1e-10
    phi = np.arctan2(Y, X)

    d_rho = rho - major_radius
    d_torus = np.sqrt(d_rho**2 + Z**2)

    # Gaussian envelope around torus skeleton
    envelope = amplitude * np.exp(-d_torus**2 / (2 * minor_radius**2))

    # Traveling wave phase: cos(nφ) at t=0
    phase = n_wavelengths * phi
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    cos_phi = X / rho
    sin_phi = Y / rho

    # E field: radial polarization (creates ∇·E for pivot sourcing)
    # E_ρ = A cos(nφ) envelope
    sim.Ex = envelope * cos_phase * cos_phi
    sim.Ey = envelope * cos_phase * sin_phi
    sim.Ez = np.zeros_like(envelope)

    # B field: B = (1/c)φ̂ × E for traveling wave in +φ direction
    # φ̂ × ρ̂ = ẑ, so B_z = -E_ρ/c (we use c=1)
    sim.Bx = np.zeros_like(envelope)
    sim.By = np.zeros_like(envelope)
    sim.Bz = -envelope * cos_phase

    # Self-consistent P: for a mode ∝ exp(i(nφ - ωt)), P is 90° out of phase
    # ∂P/∂t = -∇·E → -iωP = -∇·E → P = ∇·E/(iω) → P ∝ sin(nφ) × |∇·E|/ω
    if self_consistent_P and sim.pivot_enabled:
        divE = _compute_div_E(sim)
        # Angular frequency for n wavelengths around radius R: ω = n/R
        omega = n_wavelengths / major_radius
        # P₀ from Poisson solve: more accurate than analytic approximation
        sim.P = _poisson_solve_3d(divE, dx=sim.dx)
        # Scale P so that ∂P/∂t ≈ -∇·E is approximately satisfied
        # for a quasi-steady rotating mode
        sim.P *= 1.0 / omega
        p_max = np.max(np.abs(sim.P))
        print(f"  Self-consistent P₀: max|P|={p_max:.4f}, ω={omega:.3f}")

    poynting_phi = np.sum(envelope**2 * cos_phase**2)
    print(f"Toroidal propagating: R={major_radius:.1f}, a={minor_radius:.1f}, "
          f"n_λ={n_wavelengths}, Poynting~{poynting_phi:.0f}")


def init_toroidal_eigenmode(sim, major_radius=None, minor_radius=None,
                             n_wavelengths=1, amplitude=0.3):
    """Initialize with a single-wavelength toroidal eigenmode.

    Uses n_wavelengths=1 (single loop) and careful field configuration
    to minimize initial radiation. The key insight: use both radial AND
    poloidal E components with specific phase relationships to create
    a mode that's closer to the self-consistent solution.

    E = envelope × [cos(nφ) ρ̂ + sin(nφ) θ̂ₚ]   (rotating polarization)
    B = envelope × [cos(nφ) ẑ + sin(nφ) × (poloidal)]

    The rotating polarization means ∇·E is smaller (partial cancellation)
    while E×B still circulates.
    """
    n = sim.n
    if major_radius is None:
        major_radius = n * 0.25
    if minor_radius is None:
        minor_radius = major_radius * 0.3

    cx, cy, cz = n // 2, n // 2, n // 2
    x = np.arange(n) - cx
    y = np.arange(n) - cy
    z = np.arange(n) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    rho = np.sqrt(X**2 + Y**2) + 1e-10
    phi = np.arctan2(Y, X)
    d_rho = rho - major_radius
    d_torus = np.sqrt(d_rho**2 + Z**2) + 1e-10
    theta = np.arctan2(Z, d_rho)

    envelope = amplitude * np.exp(-d_torus**2 / (2 * minor_radius**2))
    phase = n_wavelengths * phi

    cos_phi = X / rho
    sin_phi = Y / rho
    cos_theta = d_rho / d_torus
    sin_theta = Z / d_torus

    # Component 1: E radial, B axial — cos(nφ) part
    E_rho_1 = envelope * np.cos(phase)
    sim.Ex = E_rho_1 * cos_phi
    sim.Ey = E_rho_1 * sin_phi
    sim.Ez = np.zeros_like(envelope)
    sim.Bz = -envelope * np.cos(phase)

    # Component 2: E poloidal, B toroidal — sin(nφ) part (90° rotated)
    # Poloidal E: in the (d_rho, Z) plane, perpendicular to toroidal direction
    E_pol = envelope * np.sin(phase)
    # Poloidal direction: cos(θ)ρ̂ + sin(θ)ẑ in the torus cross-section plane
    # But we want the component perpendicular to ρ̂ in the poloidal plane = ẑ·cos(θ) - ρ̂·sin(θ)
    # Actually the poloidal direction tangent to the torus tube is:
    #   θ̂ = -sin(θ)ρ̂ + cos(θ)ẑ
    sim.Ex += E_pol * (-sin_theta * cos_phi)
    sim.Ey += E_pol * (-sin_theta * sin_phi)
    sim.Ez += E_pol * cos_theta

    # Corresponding B for the poloidal E: B should be toroidal (φ̂)
    B_tor = envelope * np.sin(phase)
    sim.Bx = B_tor * (-sin_phi)
    sim.By = B_tor * cos_phi

    # Self-consistent P
    if sim.pivot_enabled:
        divE = _compute_div_E(sim)
        omega = n_wavelengths / major_radius
        sim.P = _poisson_solve_3d(divE, dx=sim.dx) / omega
        print(f"  Eigenmode P₀: max|P|={np.max(np.abs(sim.P)):.4f}")

    print(f"Toroidal eigenmode: R={major_radius:.1f}, a={minor_radius:.1f}, "
          f"n_λ={n_wavelengths}, rotating polarization")


def init_toroidal_relaxed(sim, major_radius=None, minor_radius=None,
                           n_wavelengths=2, amplitude=0.3, relax_steps=50):
    """Initialize toroidal fields, then 'relax' by running with strong damping.

    Strategy: start with propagating init, run briefly with artificial damping
    to let P build up and fields settle, then continue normally. This lets the
    system find its own quasi-equilibrium.
    """
    # Start with propagating init
    init_toroidal_propagating(sim, major_radius=major_radius,
                               minor_radius=minor_radius,
                               n_wavelengths=n_wavelengths,
                               amplitude=amplitude,
                               self_consistent_P=True)

    print(f"  Relaxing for {relax_steps} steps with enhanced damping...")
    # Save original damping
    orig_damping = sim.damping.copy()
    # Apply stronger damping at edges during relaxation
    sim._build_damping_profile(width=12)

    e0 = sim.compute_energy()
    for _ in range(relax_steps):
        sim.update()
    e1 = sim.compute_energy()

    # Restore original damping and reset time
    sim.damping = orig_damping
    sim.time = 0.0
    sim.step = 0
    sim.energy_history = []
    print(f"  Relaxed: E {e0['total']:.2f} → {e1['total']:.2f}, "
          f"P²={e1['pivot']:.4f}")


# ============================================================
# Visualization
# ============================================================

def plot_slices(sim, title_suffix="", save_path=None):
    """Plot energy density and pivot field on three orthogonal slices."""
    n = sim.n
    mid = n // 2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Williamson Pivot Field 3D — t={sim.time:.1f}, step={sim.step}\n"
        f"d(F+P)=0: ∂E/∂t = ∇×B - ∇P,  ∂P/∂t = -∇·E  "
        f"{'[PIVOT ON]' if sim.pivot_enabled else '[PIVOT OFF]'} {title_suffix}",
        fontsize=12, fontweight='bold'
    )

    slices = [('z', mid, 'XY plane (z=mid)'),
              ('y', mid, 'XZ plane (y=mid)'),
              ('x', mid, 'YZ plane (x=mid)')]

    for col, (axis, idx, label) in enumerate(slices):
        # Energy density
        ed = sim.compute_energy_density_slice(axis=axis, index=idx)
        ax = axes[0, col]
        vmax = max(ed.max(), 1e-10)
        im = ax.imshow(ed.T, origin='lower', cmap='inferno',
                       vmin=0, vmax=vmax * 0.7)
        ax.set_title(f'Energy density\n{label}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Pivot field
        pv = sim.compute_pivot_slice(axis=axis, index=idx)
        ax = axes[1, col]
        pmax = max(abs(pv).max(), 1e-10)
        im = ax.imshow(pv.T, origin='lower', cmap='RdBu_r',
                       vmin=-pmax, vmax=pmax)
        ax.set_title(f'Pivot field P\n{label}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_comparison(sim_pivot, sim_maxwell, save_path=None):
    """Compare pivot vs standard Maxwell with energy tracking."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "Williamson Pivot Field 3D: Pivot vs Standard Maxwell\n"
        "d(F+P)=0 vs dF=0",
        fontsize=13, fontweight='bold'
    )

    mid = sim_pivot.n // 2

    # Row 0: Energy density slices (z=mid)
    for col, (sim, label) in enumerate(
            [(sim_pivot, "With Pivot (P≠0)"), (sim_maxwell, "Standard Maxwell (P=0)")]):
        ed = sim.compute_energy_density_slice(axis='z', index=mid)
        ax = axes[0, col]
        vmax = max(ed.max(), 1e-10)
        im = ax.imshow(ed.T, origin='lower', cmap='inferno', vmin=0, vmax=vmax * 0.7)
        ax.set_title(f'Energy XY slice\n{label}')
        plt.colorbar(im, ax=ax)

    # XZ slices
    for col_offset, (sim, label) in enumerate(
            [(sim_pivot, "Pivot XZ"), (sim_maxwell, "Maxwell XZ")]):
        ed = sim.compute_energy_density_slice(axis='y', index=mid)
        ax = axes[0, 2 + col_offset]
        vmax = max(ed.max(), 1e-10)
        im = ax.imshow(ed.T, origin='lower', cmap='inferno', vmin=0, vmax=vmax * 0.7)
        ax.set_title(f'Energy XZ slice\n{label}')
        plt.colorbar(im, ax=ax)

    # Row 1: Pivot field, energy curves
    pv = sim_pivot.compute_pivot_slice(axis='z', index=mid)
    ax = axes[1, 0]
    pmax = max(abs(pv).max(), 1e-10)
    im = ax.imshow(pv.T, origin='lower', cmap='RdBu_r', vmin=-pmax, vmax=pmax)
    ax.set_title('Pivot field P (XY)')
    plt.colorbar(im, ax=ax)

    pv_xz = sim_pivot.compute_pivot_slice(axis='y', index=mid)
    ax = axes[1, 1]
    pmax = max(abs(pv_xz).max(), 1e-10)
    im = ax.imshow(pv_xz.T, origin='lower', cmap='RdBu_r', vmin=-pmax, vmax=pmax)
    ax.set_title('Pivot field P (XZ)')
    plt.colorbar(im, ax=ax)

    # Energy confinement history
    if sim_pivot.energy_history:
        ax = axes[1, 2]
        times = [e['time'] for e in sim_pivot.energy_history]
        ax.plot(times, [e['confined_pivot'] for e in sim_pivot.energy_history],
                'b-', linewidth=2, label='Pivot')
        ax.plot(times, [e['confined_maxwell'] for e in sim_pivot.energy_history],
                'r--', linewidth=2, label='Maxwell')
        ax.set_xlabel('Time')
        ax.set_ylabel('Confined fraction')
        ax.set_title('Energy Confinement (r<N/4)')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 3]
        ax.plot(times, [e['total_pivot'] for e in sim_pivot.energy_history],
                'b-', linewidth=2, label='Total (pivot)')
        ax.plot(times, [e['total_maxwell'] for e in sim_pivot.energy_history],
                'r--', linewidth=2, label='Total (Maxwell)')
        ax.plot(times, [e['mass_pivot'] for e in sim_pivot.energy_history],
                'g-', linewidth=1.5, label='P² (rest mass)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# Run Modes
# ============================================================

def run_comparison(n=64, steps=300, init_type='toroidal', save=False,
                   boundary='absorbing'):
    """Run pivot vs standard Maxwell comparison."""
    print("=" * 60)
    print("Williamson Pivot Field — 3D Comparison")
    print("=" * 60)

    sim_p = PivotFieldSimulation3D(n=n, pivot_enabled=True, boundary=boundary)
    sim_m = PivotFieldSimulation3D(n=n, pivot_enabled=False, boundary=boundary)

    init_fn = {'toroidal': init_toroidal, 'gaussian': init_gaussian,
               'ring': init_ring_current, 'propagating': init_toroidal_propagating,
               'eigenmode': init_toroidal_eigenmode,
               'relaxed': init_toroidal_relaxed}[init_type]
    init_fn(sim_p)
    init_fn(sim_m)

    measure_interval = max(1, steps // 100)
    t_start = time.time()

    for step in range(steps):
        sim_p.update()
        sim_m.update()

        if step % measure_interval == 0:
            ep = sim_p.compute_energy()
            em = sim_m.compute_energy()
            cp = sim_p.compute_confined_fraction()
            cm = sim_m.compute_confined_fraction()
            sim_p.energy_history.append({
                'time': sim_p.time,
                'total_pivot': ep['total'],
                'total_maxwell': em['total'],
                'mass_pivot': ep['pivot'],
                'confined_pivot': cp,
                'confined_maxwell': cm,
            })

        if step % max(1, steps // 5) == 0:
            ep = sim_p.compute_energy()
            cp = sim_p.compute_confined_fraction()
            cm = sim_m.compute_confined_fraction()
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  Step {step:4d}/{steps}: "
                  f"confined(P)={cp:.3f} confined(M)={cm:.3f} "
                  f"P²={ep['pivot']:.4f} "
                  f"[{rate:.0f} steps/s]")

    elapsed = time.time() - t_start
    print(f"\nCompleted {steps} steps in {elapsed:.1f}s "
          f"({steps/elapsed:.0f} steps/s)")

    save_path = None
    if save:
        save_path = f'/home/jim/repos/claude-prime/simulations/pivot_3d_{init_type}.png'
    plot_comparison(sim_p, sim_m, save_path=save_path)


def run_snapshot(n=64, steps=200, init_type='toroidal', save=False,
                 boundary='absorbing'):
    """Run simulation and save cross-section snapshots."""
    print("=" * 60)
    print(f"Williamson Pivot Field — 3D Snapshots ({init_type})")
    print("=" * 60)

    sim = PivotFieldSimulation3D(n=n, pivot_enabled=True, boundary=boundary)
    init_fn = {'toroidal': init_toroidal, 'gaussian': init_gaussian,
               'ring': init_ring_current, 'propagating': init_toroidal_propagating,
               'eigenmode': init_toroidal_eigenmode,
               'relaxed': init_toroidal_relaxed}[init_type]
    init_fn(sim)

    # Snapshot at t=0
    if save:
        plot_slices(sim, title_suffix="(initial)",
                    save_path=f'/home/jim/repos/claude-prime/simulations/pivot_3d_{init_type}_t0.png')

    t_start = time.time()
    for step in range(steps):
        sim.update()
        if step % max(1, steps // 5) == 0:
            e = sim.compute_energy()
            cf = sim.compute_confined_fraction()
            print(f"  Step {step:4d}/{steps}: "
                  f"E_total={e['total']:.4f} P²={e['pivot']:.4f} "
                  f"confined={cf:.3f}")

    elapsed = time.time() - t_start
    print(f"\nCompleted {steps} steps in {elapsed:.1f}s")

    save_path = None
    if save:
        save_path = f'/home/jim/repos/claude-prime/simulations/pivot_3d_{init_type}_t{steps}.png'
    plot_slices(sim, save_path=save_path)


def run_animate(n=48, steps=500, init_type='toroidal', save=False):
    """Run animated cross-section visualization."""
    print("=" * 60)
    print(f"Williamson Pivot Field — 3D Animation ({init_type})")
    print("=" * 60)

    sim = PivotFieldSimulation3D(n=n, pivot_enabled=True)
    init_fn = {'toroidal': init_toroidal, 'gaussian': init_gaussian,
               'ring': init_ring_current, 'propagating': init_toroidal_propagating,
               'eigenmode': init_toroidal_eigenmode,
               'relaxed': init_toroidal_relaxed}[init_type]
    init_fn(sim)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Williamson Pivot Field 3D", fontsize=13, fontweight='bold')

    mid = n // 2

    # Initial plots
    ed_xy = sim.compute_energy_density_slice('z', mid)
    ed_xz = sim.compute_energy_density_slice('y', mid)
    pv = sim.compute_pivot_slice('z', mid)

    im0 = axes[0].imshow(ed_xy.T, origin='lower', cmap='inferno', vmin=0, vmax=0.1)
    axes[0].set_title('Energy (XY)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ed_xz.T, origin='lower', cmap='inferno', vmin=0, vmax=0.1)
    axes[1].set_title('Energy (XZ)')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pv.T, origin='lower', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    axes[2].set_title('Pivot P (XY)')
    plt.colorbar(im2, ax=axes[2])

    steps_per_frame = 3

    def animate(frame):
        for _ in range(steps_per_frame):
            sim.update()

        ed_xy = sim.compute_energy_density_slice('z', mid)
        ed_xz = sim.compute_energy_density_slice('y', mid)
        pv = sim.compute_pivot_slice('z', mid)

        im0.set_data(ed_xy.T)
        im0.set_clim(0, max(ed_xy.max() * 0.8, 1e-10))

        im1.set_data(ed_xz.T)
        im1.set_clim(0, max(ed_xz.max() * 0.8, 1e-10))

        im2.set_data(pv.T)
        pmax = max(abs(pv).max() * 0.8, 1e-10)
        im2.set_clim(-pmax, pmax)

        fig.suptitle(f"Williamson Pivot Field 3D — t={sim.time:.1f}  step={sim.step}",
                     fontsize=13, fontweight='bold')
        return im0, im1, im2

    n_frames = steps // steps_per_frame
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                    interval=80, blit=False)
    if save:
        outpath = f'/home/jim/repos/claude-prime/simulations/pivot_3d_{init_type}.mp4'
        anim.save(outpath, writer='ffmpeg', fps=25, dpi=100)
        print(f"Saved: {outpath}")
    else:
        plt.show()


def run_confinement_sweep(n=64, steps=500, save=False):
    """Sweep over initial conditions and compare confinement behavior.

    This is the key experiment: which init gives the best self-confinement
    with Williamson's pivot field?
    """
    print("=" * 60)
    print("Williamson Pivot Field — Confinement Sweep")
    print("=" * 60)

    configs = [
        ('toroidal', 'absorbing', init_toroidal, "Original toroidal"),
        ('propagating', 'absorbing', init_toroidal_propagating, "Propagating (abs BC)"),
        ('propagating', 'periodic', init_toroidal_propagating, "Propagating (periodic)"),
        ('eigenmode', 'absorbing', init_toroidal_eigenmode, "Eigenmode (abs BC)"),
        ('eigenmode', 'periodic', init_toroidal_eigenmode, "Eigenmode (periodic)"),
        ('relaxed', 'absorbing', init_toroidal_relaxed, "Relaxed (abs BC)"),
    ]

    results = {}
    measure_interval = max(1, steps // 100)

    for name, bc, init_fn, label in configs:
        print(f"\n--- {label} ---")
        sim = PivotFieldSimulation3D(n=n, pivot_enabled=True, boundary=bc)
        init_fn(sim)

        history = []
        t_start = time.time()

        for step in range(steps):
            sim.update()
            if step % measure_interval == 0:
                e = sim.compute_energy()
                cf = sim.compute_confined_fraction()
                history.append({
                    'step': step,
                    'time': sim.time,
                    'total': e['total'],
                    'pivot': e['pivot'],
                    'em': e['em'],
                    'confined': cf,
                })

        elapsed = time.time() - t_start
        final_e = sim.compute_energy()
        final_cf = sim.compute_confined_fraction()
        print(f"  Final: confined={final_cf:.3f}, E_total={final_e['total']:.2f}, "
              f"P²={final_e['pivot']:.4f} [{elapsed:.1f}s]")

        results[label] = {
            'history': history,
            'sim': sim,
        }

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Williamson Pivot Confinement Sweep — {n}³ grid, {steps} steps\n"
        "Which initial conditions produce self-confinement?",
        fontsize=13, fontweight='bold'
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Top row: confinement fraction, total energy, P² energy
    ax = axes[0, 0]
    for i, (_, _, _, label) in enumerate(configs):
        h = results[label]['history']
        ax.plot([d['time'] for d in h], [d['confined'] for d in h],
                color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Confined fraction (r<N/4)')
    ax.set_title('Energy Confinement')
    ax.legend(fontsize=7, loc='best')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, (_, _, _, label) in enumerate(configs):
        h = results[label]['history']
        ax.plot([d['time'] for d in h], [d['total'] for d in h],
                color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for i, (_, _, _, label) in enumerate(configs):
        h = results[label]['history']
        ax.plot([d['time'] for d in h], [d['pivot'] for d in h],
                color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('P² Energy (rest mass)')
    ax.set_title('Pivot (Mass) Energy')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Bottom row: final XY slices for top 3 most-confined
    ranked = sorted(results.items(),
                    key=lambda x: x[1]['history'][-1]['confined'], reverse=True)
    for col, (label, data) in enumerate(ranked[:3]):
        sim = data['sim']
        mid = sim.n // 2
        ed = sim.compute_energy_density_slice('z', mid)
        ax = axes[1, col]
        vmax = max(ed.max(), 1e-10)
        im = ax.imshow(ed.T, origin='lower', cmap='inferno', vmin=0, vmax=vmax * 0.7)
        cf = data['history'][-1]['confined']
        ax.set_title(f'{label}\nconfined={cf:.3f}', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    if save:
        save_path = '/home/jim/repos/claude-prime/simulations/pivot_3d_sweep.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    else:
        plt.show()
    plt.close()

    # Print ranking
    print("\n" + "=" * 60)
    print("CONFINEMENT RANKING (final confined fraction):")
    print("=" * 60)
    for rank, (label, data) in enumerate(ranked, 1):
        h = data['history']
        print(f"  {rank}. {label}: confined={h[-1]['confined']:.3f}, "
              f"E_total={h[-1]['total']:.2f}, P²={h[-1]['pivot']:.4f}")


def run_rebound_study(n=64, steps=1000, save=False):
    """Study the confinement rebound phenomenon in detail.

    The original toroidal init shows energy dispersing then RE-CONFINING.
    This is the signature of Williamson's pivot mechanism at work.
    Let's explore what parameters control this.
    """
    print("=" * 60)
    print("Williamson Pivot Field — Rebound Confinement Study")
    print("=" * 60)

    configs = [
        # (label, R_frac, a_frac, n_wavelengths, amplitude, P0)
        # R_frac = fraction of n for major radius
        # a_frac = fraction of R for minor radius
        ("R=0.15, a/R=0.3, n=2", 0.15, 0.3, 2, 1.0, False),
        ("R=0.20, a/R=0.3, n=2", 0.20, 0.3, 2, 1.0, False),
        ("R=0.25, a/R=0.3, n=2", 0.25, 0.3, 2, 1.0, False),
        ("R=0.20, a/R=0.2, n=2", 0.20, 0.2, 2, 1.0, False),
        ("R=0.20, a/R=0.4, n=2", 0.20, 0.4, 2, 1.0, False),
        ("R=0.20, a/R=0.3, n=1", 0.20, 0.3, 1, 1.0, False),
        ("R=0.20, a/R=0.3, n=3", 0.20, 0.3, 3, 1.0, False),
        ("R=0.20, amp=0.3, n=2", 0.20, 0.3, 2, 0.3, False),
        ("Prop no-P₀, R=0.2", 0.20, 0.3, 2, 0.5, 'prop_noP'),
    ]

    results = {}
    measure_interval = max(1, steps // 200)

    for label, R_frac, a_frac, n_wl, amp, special in configs:
        print(f"\n--- {label} ---")
        sim = PivotFieldSimulation3D(n=n, pivot_enabled=True, boundary='absorbing')

        R = n * R_frac
        a = R * a_frac

        if special == 'prop_noP':
            # Propagating init but with P₀ = 0 (let it build naturally)
            init_toroidal_propagating(sim, major_radius=R, minor_radius=a,
                                       n_wavelengths=n_wl, amplitude=amp,
                                       self_consistent_P=False)
        else:
            init_toroidal(sim, major_radius=R, minor_radius=a,
                          n_wavelengths=n_wl, amplitude=amp)

        history = []
        t_start = time.time()

        for step in range(steps):
            sim.update()
            if step % measure_interval == 0:
                e = sim.compute_energy()
                cf = sim.compute_confined_fraction()
                history.append({
                    'step': step,
                    'time': sim.time,
                    'total': e['total'],
                    'pivot': e['pivot'],
                    'em': e['em'],
                    'confined': cf,
                })

        elapsed = time.time() - t_start
        # Find peak rebound (max confinement after initial drop)
        confs = [h['confined'] for h in history]
        min_idx = np.argmin(confs[:len(confs)//2]) if len(confs) > 2 else 0
        rebound_max = max(confs[min_idx:]) if min_idx < len(confs) else confs[-1]
        rebound_idx = min_idx + np.argmax(confs[min_idx:])

        print(f"  Final: confined={confs[-1]:.3f}, rebound_peak={rebound_max:.3f} "
              f"(step ~{history[rebound_idx]['step']}), [{elapsed:.1f}s]")

        results[label] = {
            'history': history,
            'rebound_max': rebound_max,
            'rebound_step': history[rebound_idx]['step'],
            'final_confined': confs[-1],
        }

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Williamson Pivot — Rebound Confinement Study\n"
        f"{n}³ grid, {steps} steps — Which parameters maximize self-confinement?",
        fontsize=13, fontweight='bold'
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Top-left: Confinement vs time
    ax = axes[0, 0]
    for i, (label, *_) in enumerate(configs):
        h = results[label]['history']
        ax.plot([d['time'] for d in h], [d['confined'] for d in h],
                color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Confined fraction (r<N/4)')
    ax.set_title('Energy Confinement — Rebound Dynamics')
    ax.legend(fontsize=6, loc='best')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Top-right: P² vs time
    ax = axes[0, 1]
    for i, (label, *_) in enumerate(configs):
        h = results[label]['history']
        ax.plot([d['time'] for d in h], [d['pivot'] for d in h],
                color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('P² (rest mass energy)')
    ax.set_title('Pivot Field Energy')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    # Bottom-left: Total energy vs time
    ax = axes[1, 0]
    for i, (label, *_) in enumerate(configs):
        h = results[label]['history']
        ax.plot([d['time'] for d in h], [d['total'] for d in h],
                color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Bar chart of rebound peak confinement
    ax = axes[1, 1]
    labels_short = [l.split(',')[0] for l in results.keys()]
    rebound_vals = [results[l]['rebound_max'] for l in results]
    final_vals = [results[l]['final_confined'] for l in results]
    x = np.arange(len(labels_short))
    width = 0.35
    ax.bar(x - width/2, rebound_vals, width, label='Peak rebound', color='steelblue')
    ax.bar(x + width/2, final_vals, width, label='Final', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Confined fraction')
    ax.set_title('Confinement: Peak Rebound vs Final')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        save_path = '/home/jim/repos/claude-prime/simulations/pivot_3d_rebound.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    else:
        plt.show()
    plt.close()

    # Print ranking
    print("\n" + "=" * 60)
    print("REBOUND RANKING:")
    print("=" * 60)
    ranked = sorted(results.items(), key=lambda x: x[1]['rebound_max'], reverse=True)
    for rank, (label, data) in enumerate(ranked, 1):
        print(f"  {rank}. {label}")
        print(f"     rebound_peak={data['rebound_max']:.3f} "
              f"(step {data['rebound_step']}), "
              f"final={data['final_confined']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='3D Williamson Pivot Field Simulation')
    parser.add_argument('--mode', choices=['compare', 'snapshot', 'animate', 'sweep', 'rebound'],
                        default='snapshot', help='Run mode')
    parser.add_argument('--init',
                        choices=['toroidal', 'gaussian', 'ring',
                                 'propagating', 'eigenmode', 'relaxed'],
                        default='toroidal', help='Initial condition')
    parser.add_argument('--grid', type=int, default=64,
                        help='Grid size N (NxNxN)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of time steps')
    parser.add_argument('--save', action='store_true',
                        help='Save output to files')
    parser.add_argument('--boundary', choices=['absorbing', 'periodic'],
                        default='absorbing', help='Boundary condition type')

    args = parser.parse_args()

    if args.mode == 'rebound':
        run_rebound_study(n=args.grid, steps=args.steps, save=args.save)
    elif args.mode == 'sweep':
        run_confinement_sweep(n=args.grid, steps=args.steps, save=args.save)
    elif args.mode == 'compare':
        run_comparison(n=args.grid, steps=args.steps,
                       init_type=args.init, save=args.save,
                       boundary=args.boundary)
    elif args.mode == 'animate':
        run_animate(n=min(args.grid, 48), steps=args.steps,
                    init_type=args.init, save=args.save)
    else:
        run_snapshot(n=args.grid, steps=args.steps,
                     init_type=args.init, save=args.save,
                     boundary=args.boundary)
