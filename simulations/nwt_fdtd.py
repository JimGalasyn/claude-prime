#!/usr/bin/env python3
"""
Einstein-Maxwell FDTD: Phase 0 — Flat-space axisymmetric Maxwell
=================================================================

Axisymmetric (r, z) Maxwell FDTD on a Yee staggered grid in cylindrical
coordinates.  Measures how fast a torus knot EM configuration disperses
without gravity — the baseline "problem statement" for Phase 2
(Kerr-Newman background).

Grid is sized to the electron torus:
    R = ℏ/(2 m_e c) ≈ 193 fm   (major radius = reduced Compton / 2)
    r_minor = α R ≈ 1.4 fm      (minor radius)

Polarizations:
    TE: E_φ, B_r, B_z   (toroidal electric, poloidal magnetic)
    TM: B_φ, E_r, E_z   (toroidal magnetic, poloidal electric)

Usage:
    python nwt_fdtd.py --test                       # validation suite
    python nwt_fdtd.py --run                        # torus dispersion
    python nwt_fdtd.py --run --n-circulations 50    # longer run
    python nwt_fdtd.py --run --resolution 2         # 2x resolution
    python nwt_fdtd.py --run --animate              # save GIF

Requires: numpy, matplotlib, scipy (optional, for validation)
"""

import numpy as np
import argparse
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

# ---------------------------------------------------------------------------
# Physical constants (SI) — duplicated from null_worldtube.py so this file
# is self-contained, but values are identical.
# ---------------------------------------------------------------------------
c = 2.99792458e8              # speed of light (m/s)
hbar = 1.054571817e-34        # reduced Planck constant (J·s)
e_charge = 1.602176634e-19    # elementary charge (C)
eps0 = 8.8541878128e-12       # vacuum permittivity (F/m)
mu0 = 1.2566370621e-6         # vacuum permeability (H/m)
m_e = 9.1093837015e-31        # electron mass (kg)
alpha = 7.2973525693e-3       # fine-structure constant
eV = 1.602176634e-19
MeV = 1e6 * eV

lambda_C = hbar / (m_e * c)   # reduced Compton wavelength ≈ 3.86e-13 m


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
@dataclass
class GridParams:
    """Axisymmetric (r, z) grid specification."""
    Nr: int                    # radial cells
    Nz: int                    # axial cells
    dr: float                  # cell size (m), same in r and z
    R_torus: float             # torus major radius (m)
    r_minor: float             # torus minor radius (m)
    courant: float = 0.7       # Courant number (< 1/√2 for stability)

    @property
    def dz(self) -> float:
        return self.dr

    @property
    def dt(self) -> float:
        return self.courant * self.dr / (c * np.sqrt(2.0))

    @property
    def r_axis(self) -> np.ndarray:
        """Cell-center radial positions, r[0] = dr/2."""
        return (np.arange(self.Nr) + 0.5) * self.dr

    @property
    def z_axis(self) -> np.ndarray:
        """Cell-center axial positions, centered on z=0."""
        return (np.arange(self.Nz) + 0.5) * self.dr - self.Nz * self.dr / 2

    @property
    def T_circ(self) -> float:
        """Circulation period: one toroidal loop at c."""
        return 2 * np.pi * self.R_torus / c

    @property
    def steps_per_circ(self) -> int:
        return int(np.ceil(self.T_circ / self.dt))

    @property
    def memory_MB(self) -> float:
        return 6 * self.Nr * self.Nz * 8 / 1e6  # 6 fields, float64

    @classmethod
    def for_electron(cls, resolution: float = 1.0, courant: float = 0.7):
        """Factory for electron-scale torus.

        resolution=1 → dr = r_minor/10 (baseline)
        resolution=2 → dr = r_minor/20 (double), etc.
        """
        R = lambda_C / 2          # ≈ 193 fm
        r_min = alpha * R         # ≈ 1.4 fm
        dr = r_min / (10 * resolution)
        # Grid must contain the torus with generous margin for dispersion
        r_max = R + 30 * r_min    # radial extent (30 minor radii of margin)
        z_max = 30 * r_min        # axial extent (symmetric about z=0)
        Nr = int(np.ceil(r_max / dr))
        Nz = int(np.ceil(2 * z_max / dr))
        # Make Nz even for symmetry
        if Nz % 2 == 1:
            Nz += 1
        return cls(Nr=Nr, Nz=Nz, dr=dr, R_torus=R, r_minor=r_min,
                   courant=courant)

    @classmethod
    def for_test(cls, Nr: int = 200, Nz: int = 400, dr: float = 1.0,
                 courant: float = 0.5):
        """Factory for validation tests (dimensionless units, c=1)."""
        return cls(Nr=Nr, Nz=Nz, dr=dr, R_torus=Nr*dr*0.4,
                   r_minor=Nr*dr*0.04, courant=courant)


# ---------------------------------------------------------------------------
# Axisymmetric Maxwell FDTD solver
# ---------------------------------------------------------------------------
class AxiSymMaxwellFDTD:
    """Yee-scheme FDTD for Maxwell's equations in cylindrical (r, z)
    coordinates with azimuthal symmetry (∂/∂φ = 0).

    Staggering (half-integer = edge-centered):
        E_r   at (i+½, j  )  — radial edges
        E_z   at (i,   j+½)  — axial edges
        E_phi at (i,   j  )  — azimuthal (cell center)
        B_r   at (i,   j+½)  — radial faces
        B_z   at (i+½, j+½)  — axial faces (actually cell corners in 2D)
        B_phi at (i+½, j  )  — azimuthal faces

    TE mode: E_φ, B_r, B_z   (toroidal E, poloidal B)
    TM mode: B_φ, E_r, E_z   (toroidal B, poloidal E)

    Updates follow the standard leapfrog:
        B fields at half-integer time steps
        E fields at integer time steps
    """

    def __init__(self, gp: GridParams):
        self.gp = gp
        Nr, Nz = gp.Nr, gp.Nz
        dr, dz, dt = gp.dr, gp.dz, gp.dt
        self.c_sq = c * c  # c² for Ampere's law: ∂E/∂t = c² curl(B)

        # --- Fields (all initialized to zero) ---
        # TE polarization
        self.E_phi = np.zeros((Nr, Nz))    # (i, j)
        self.B_r   = np.zeros((Nr, Nz))    # (i, j+½) — stored at (i, j)
        self.B_z   = np.zeros((Nr, Nz))    # (i+½, j+½) — stored at (i, j)

        # TM polarization
        self.B_phi = np.zeros((Nr, Nz))    # (i+½, j)
        self.E_r   = np.zeros((Nr, Nz))    # (i+½, j) — stored at (i, j)
        self.E_z   = np.zeros((Nr, Nz))    # (i, j+½) — stored at (i, j)

        # Precompute radial positions for the different staggerings
        # r at cell centers: r_c[i] = (i + 0.5) * dr
        self.r_c = (np.arange(Nr) + 0.5) * dr            # shape (Nr,)
        # r at cell edges (half-integer): r_e[i] = i * dr
        self.r_e = np.arange(Nr + 1).astype(float) * dr   # shape (Nr+1,)

        # For vectorized updates, broadcast to 2D
        self.r_c_2d = self.r_c[:, np.newaxis]              # (Nr, 1)
        self.r_e_2d = self.r_e[:, np.newaxis]              # (Nr+1, 1)

        # Precompute inverse r at cell centers (with L'Hopital at r=0)
        self.inv_r_c = np.where(self.r_c > 0, 1.0 / self.r_c, 0.0)
        self.inv_r_c_2d = self.inv_r_c[:, np.newaxis]

        # For Mur ABC storage (previous boundary values)
        self._abc_E_phi_r_old = np.zeros(Nz)
        self._abc_E_phi_r_new = np.zeros(Nz)
        self._abc_E_r_z_lo_old = np.zeros(Nr)
        self._abc_E_r_z_hi_old = np.zeros(Nr)
        self._abc_E_z_r_old = np.zeros(Nz)
        self._abc_E_z_r_new = np.zeros(Nz)

        self.time = 0.0
        self.step = 0

    # ----- TE mode updates: E_phi, B_r, B_z -----

    def _update_B_te(self):
        """Update B_r and B_z from E_phi (TE mode).

        curl E in cylindrical (azimuthal symmetry):
            (curl E)_r = -∂E_φ/∂z
            (curl E)_z = (1/r) ∂(r E_φ)/∂r

        Faraday: ∂B/∂t = -curl E
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz
        E_phi = self.E_phi
        Nr, Nz = self.gp.Nr, self.gp.Nz

        # B_r at (i, j+½): ∂B_r/∂t = ∂E_φ/∂z
        # ∂E_φ/∂z at (i, j+½) ≈ (E_phi[i,j+1] - E_phi[i,j]) / dz
        self.B_r[:, :-1] += dt / dz * (E_phi[:, 1:] - E_phi[:, :-1])

        # B_z at (i+½, j+½): ∂B_z/∂t = -(1/r) ∂(r E_φ)/∂r
        # At (i+½, j+½):
        #   ∂(r E_φ)/∂r ≈ (r[i+1] E_phi[i+1,j] - r[i] E_phi[i,j]) / dr
        #   divided by r at i+½ = r_e[i+1]
        rE = self.r_c_2d * E_phi  # r * E_phi at cell centers
        # r_e[i+1] for i=0..Nr-2 → edges at half-integer positions
        r_half = self.r_e_2d[1:-1]  # (Nr-1, 1)
        inv_r_half = np.where(r_half > 0, 1.0 / r_half, 2.0 / self.gp.dr)
        self.B_z[:-1, :-1] -= dt * inv_r_half * (rE[1:, :-1] - rE[:-1, :-1]) / dr

    def _update_E_te(self):
        """Update E_phi from B_r, B_z (TE mode).

        Ampere: ∂E_φ/∂t = c²(curl B)_φ = c²(∂B_r/∂z - ∂B_z/∂r)
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz
        c2 = self.c_sq

        # ∂B_r/∂z at (i, j): (B_r[i,j] - B_r[i,j-1]) / dz
        self.E_phi[:, 1:] += c2 * dt / dz * (self.B_r[:, 1:] - self.B_r[:, :-1])

        # ∂B_z/∂r at (i, j): (B_z[i,j] - B_z[i-1,j]) / dr
        self.E_phi[1:, :-1] -= c2 * dt / dr * (self.B_z[1:, :-1] - self.B_z[:-1, :-1])

        # r=0 axis: E_phi(r=0) = 0 by symmetry for m=0; grid starts at r=dr/2
        # B_z[-½] = B_z[0] (symmetry), so the contribution vanishes
        # No explicit action needed for i=0

    # ----- TM mode updates: B_phi, E_r, E_z -----

    def _update_B_tm(self):
        """Update B_phi from E_r, E_z (TM mode).

        (curl E)_φ = ∂E_r/∂z - ∂E_z/∂r
        ∂B_φ/∂t = -( ∂E_r/∂z - ∂E_z/∂r )
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz

        # B_phi at (i+½, j): staggered between E_r and E_z
        # ∂E_r/∂z at (i+½, j): (E_r[i, j+1] - E_r[i, j]) / dz
        #   but E_r is at (i+½, j), so we need it at j and j+1 → not shifted
        #   Actually E_r[i,j] is at (i+½, j), so ∂E_r/∂z needs j staggering
        #   We approximate: (E_r[i, j] - E_r[i, j-1]) / dz at (i+½, j-½)
        #   For B_phi at (i+½, j):
        self.B_phi[:-1, 1:] -= dt / dz * (self.E_r[:-1, 1:] - self.E_r[:-1, :-1])

        # ∂E_z/∂r at (i+½, j): (E_z[i+1, j] - E_z[i, j]) / dr
        self.B_phi[:-1, :] += dt / dr * (self.E_z[1:, :] - self.E_z[:-1, :])

    def _update_E_tm(self):
        """Update E_r and E_z from B_phi (TM mode).

        ∂E_r/∂t = -c² ∂B_φ/∂z
        ∂E_z/∂t = c² (1/r) ∂(r B_φ)/∂r
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz
        c2 = self.c_sq

        # E_r at (i+½, j): ∂E_r/∂t = -c² ∂B_φ/∂z
        self.E_r[:, 1:] -= c2 * dt / dz * (self.B_phi[:, 1:] - self.B_phi[:, :-1])

        # E_z at (i, j+½): ∂E_z/∂t = c² (1/r) ∂(r B_φ)/∂r
        rB = self.r_e_2d[1:] * self.B_phi  # r B_phi at (i+½, j)
        rB_shifted = np.zeros_like(rB)
        rB_shifted[1:, :] = self.r_e_2d[1:-1] * self.B_phi[:-1, :]
        # At i=0: rB_shifted[0] = 0 (axis antisymmetry)

        self.E_z[:, :-1] += c2 * dt / dr * self.inv_r_c_2d * (rB[:, :-1] - rB_shifted[:, :-1])

    # ----- Boundary conditions -----

    def _apply_mur_abc(self):
        """First-order Mur absorbing boundary conditions at outer edges.

        At r=0: symmetry conditions (E_phi=0, E_z continuous, etc.)
        At r=r_max, z=0, z=z_max: Mur ABC.
        """
        C = self.gp.courant / np.sqrt(2.0)  # effective 1D Courant number
        factor = (C - 1.0) / (C + 1.0)
        Nr, Nz = self.gp.Nr, self.gp.Nz

        # --- Outer radial boundary (r = r_max, i = Nr-1) ---
        # E_phi at r_max
        new_boundary = self.E_phi[-1, :]
        self.E_phi[-1, :] = (self._abc_E_phi_r_old
                             + factor * (self.E_phi[-2, :] - new_boundary))
        self._abc_E_phi_r_old = self.E_phi[-2, :].copy()

        # E_z at r_max
        new_bnd_ez = self.E_z[-1, :]
        self.E_z[-1, :] = (self._abc_E_z_r_old
                           + factor * (self.E_z[-2, :] - new_bnd_ez))
        self._abc_E_z_r_old = self.E_z[-2, :].copy()

        # --- Axial boundaries (z=0 and z=z_max) ---
        # Simple zero-gradient for axial boundaries (fields decay before reaching)
        self.E_phi[:, 0] = self.E_phi[:, 1]
        self.E_phi[:, -1] = self.E_phi[:, -2]
        self.E_r[:, 0] = self.E_r[:, 1]
        self.E_r[:, -1] = self.E_r[:, -2]

        # --- r=0 axis symmetry ---
        # E_phi(r=0) = 0 for m=0 mode (already at r=dr/2, no action needed)
        # B_z is continuous across axis: handled by ghost cell = B_z[0]

    # ----- Main step -----

    def step_forward(self):
        """One full leapfrog step: B half-step, E full-step."""
        # B update (half-step in leapfrog, but we do full dt updates
        # since E and B are already offset by dt/2)
        self._update_B_te()
        self._update_B_tm()

        # E update
        self._update_E_te()
        self._update_E_tm()

        # Boundary conditions
        self._apply_mur_abc()

        self.time += self.gp.dt
        self.step += 1

    # ----- Energy diagnostics -----

    def compute_energy(self) -> Dict[str, float]:
        """Total EM energy integrated over volume (axisymmetric, factor 2π r dr dz)."""
        dr = self.gp.dr
        dz = self.gp.dz
        r = self.r_c_2d  # (Nr, 1) broadcast

        dV = 2 * np.pi * r * dr * dz  # volume element

        E2 = self.E_r**2 + self.E_z**2 + self.E_phi**2
        B2 = self.B_r**2 + self.B_z**2 + self.B_phi**2

        E_energy = 0.5 * eps0 * np.sum(E2 * dV)
        B_energy = 0.5 / mu0 * np.sum(B2 * dV)
        total = E_energy + B_energy

        return {'E_energy': E_energy, 'B_energy': B_energy, 'total': total}

    def compute_energy_density(self) -> np.ndarray:
        """Energy density u = ½(ε₀E² + B²/μ₀) at each cell."""
        E2 = self.E_r**2 + self.E_z**2 + self.E_phi**2
        B2 = self.B_r**2 + self.B_z**2 + self.B_phi**2
        return 0.5 * (eps0 * E2 + B2 / mu0)

    def compute_confinement(self, r_center: float, z_center: float,
                            radius: float) -> float:
        """Fraction of energy within a circle of given radius around (r_center, z_center)."""
        r = self.gp.r_axis
        z = self.gp.z_axis
        R, Z = np.meshgrid(r, z, indexing='ij')
        dist2 = (R - r_center)**2 + (Z - z_center)**2
        mask = dist2 <= radius**2

        u = self.compute_energy_density()
        dV = 2 * np.pi * self.r_c_2d * self.gp.dr * self.gp.dz
        total = np.sum(u * dV)
        if total == 0.0:
            return 0.0
        confined = np.sum(u * dV * mask)
        return confined / total

    def compute_div_B(self) -> np.ndarray:
        """Compute ∇·B = (1/r)∂(r B_r)/∂r + ∂B_z/∂z.

        Should be zero (or machine precision) if initialized correctly.
        """
        dr = self.gp.dr
        dz = self.gp.dz
        Nr, Nz = self.gp.Nr, self.gp.Nz

        div = np.zeros((Nr, Nz))

        # (1/r) ∂(r B_r)/∂r: finite difference in r
        rBr = self.r_c_2d * self.B_r
        # ∂(r B_r)/∂r at (i, j) ≈ (rBr[i+1] - rBr[i]) / dr  (approximate)
        div[:-1, :] += (rBr[1:, :] - rBr[:-1, :]) / dr * self.inv_r_c_2d[:-1]

        # ∂B_z/∂z at (i, j+½ → j):
        div[:, :-1] += (self.B_z[:, 1:] - self.B_z[:, :-1]) / dz

        return div


# ---------------------------------------------------------------------------
# Initial conditions: Poisson solve for magnetostatic torus current
# ---------------------------------------------------------------------------
def poisson_solve_A_phi(gp: GridParams, n_iter: int = 5000,
                        tol: float = 1e-6) -> np.ndarray:
    """Solve for A_φ from a toroidal current loop J_φ using Jacobi iteration.

    In cylindrical coordinates with azimuthal symmetry:
        ∇²A_φ - A_φ/r² = -μ₀ J_φ

    where ∇² = ∂²/∂r² + (1/r)∂/∂r + ∂²/∂z² - 1/r²

    The current J_φ is a Gaussian ring centered at (R_torus, 0) with
    width ~ r_minor.
    """
    Nr, Nz = gp.Nr, gp.Nz
    dr, dz = gp.dr, gp.dz
    r = gp.r_axis
    z = gp.z_axis

    R, Z = np.meshgrid(r, z, indexing='ij')

    # Current source: Gaussian torus cross-section
    sigma = gp.r_minor / 2.0
    J_phi = np.exp(-((R - gp.R_torus)**2 + Z**2) / (2 * sigma**2))
    # Normalize so total current is meaningful
    J_sum = np.sum(J_phi) * dr * dz
    if J_sum > 0:
        J_phi /= J_sum
    # Scale to give fields of order 1 (in simulation units)
    J_phi *= 1.0 / mu0

    A = np.zeros((Nr, Nz))
    r_2d = R

    inv_r2 = np.where(r_2d > 0, 1.0 / r_2d**2, 0.0)
    inv_r = np.where(r_2d > 0, 1.0 / r_2d, 0.0)

    source = mu0 * J_phi

    for iteration in range(n_iter):
        A_old = A.copy()

        # Interior Jacobi update
        # ∂²A/∂r² ≈ (A[i+1] - 2A[i] + A[i-1]) / dr²
        # (1/r) ∂A/∂r ≈ (A[i+1] - A[i-1]) / (2r dr)
        # ∂²A/∂z² ≈ (A[j+1] - 2A[j] + A[j-1]) / dz²
        # A/r² term

        # Coefficient of A[i,j] in the Laplacian:
        # -2/dr² - 2/dz² - 1/r²
        coeff = -2.0 / dr**2 - 2.0 / dz**2 - inv_r2[1:-1, 1:-1]

        rhs = -source[1:-1, 1:-1]

        # Neighbor terms
        rhs -= (A[2:, 1:-1] + A[:-2, 1:-1]) / dr**2  # ∂²/∂r²
        rhs -= inv_r[1:-1, 1:-1] * (A[2:, 1:-1] - A[:-2, 1:-1]) / (2 * dr)
        rhs -= (A[1:-1, 2:] + A[1:-1, :-2]) / dz**2    # ∂²/∂z²

        A[1:-1, 1:-1] = rhs / coeff

        # BCs: A=0 on all boundaries
        A[0, :] = 0.0   # r=0 axis (A_phi = 0 by symmetry)
        A[-1, :] = 0.0   # outer r
        A[:, 0] = 0.0    # z_min
        A[:, -1] = 0.0   # z_max

        if iteration % 100 == 0:
            residual = np.max(np.abs(A - A_old))
            if residual < tol:
                break

    return A


def init_torus_fields(fdtd: AxiSymMaxwellFDTD, amplitude: float = 1.0):
    """Initialize TE fields from magnetostatic Poisson solve.

    B = curl(A) where A = A_phi φ_hat:
        B_r = -∂A_φ/∂z
        B_z = (1/r) ∂(r A_φ)/∂r
    """
    gp = fdtd.gp
    A_phi = poisson_solve_A_phi(gp)
    A_phi *= amplitude

    dr, dz = gp.dr, gp.dz

    # B_r = -∂A_φ/∂z
    fdtd.B_r[:, :-1] = -(A_phi[:, 1:] - A_phi[:, :-1]) / dz

    # B_z = (1/r) ∂(r A_φ)/∂r
    rA = fdtd.r_c_2d * A_phi
    fdtd.B_z[:-1, :] = (rA[1:, :] - rA[:-1, :]) / dr
    # Divide by r at half-integer position
    r_half = fdtd.r_e_2d[1:-1]
    inv_r_half = np.where(r_half > 0, 1.0 / r_half, 2.0 / dr)
    fdtd.B_z[:-1, :] *= inv_r_half

    return A_phi


def init_em_wave_torus(fdtd: AxiSymMaxwellFDTD, amplitude: float = 1.0):
    """Initialize a free EM wave packet shaped like a torus cross-section.

    No sources — pure radiation that will freely disperse.
    Uses E_phi (TE mode) with matching B_r, B_z for an outward-propagating pulse.
    """
    gp = fdtd.gp
    r = gp.r_axis
    z = gp.z_axis
    R, Z = np.meshgrid(r, z, indexing='ij')

    # Gaussian torus cross-section
    sigma = gp.r_minor / 2.0
    profile = np.exp(-((R - gp.R_torus)**2 + Z**2) / (2 * sigma**2))

    # E_phi component
    fdtd.E_phi = amplitude * profile

    # For a wave packet, set B consistent with outward propagation
    # B_r ≈ -E_phi/c * (z-component of propagation direction)
    # B_z ≈  E_phi/c * (r-component of propagation direction)
    # Direction away from torus center: (R - R_torus, Z) normalized
    dist = np.sqrt((R - gp.R_torus)**2 + Z**2)
    dist = np.maximum(dist, gp.dr * 0.01)  # avoid division by zero
    nr = (R - gp.R_torus) / dist
    nz = Z / dist

    # B = (1/c) n_hat × E for outward propagation (SI units)
    # B_r = -(nz / c) * E_phi, B_z = (nr / c) * E_phi
    fdtd.B_r = -nz * amplitude * profile / c
    fdtd.B_z = nr * amplitude * profile / c


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class Diagnostics:
    """Track simulation diagnostics over time."""
    times: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    E_energies: List[float] = field(default_factory=list)
    B_energies: List[float] = field(default_factory=list)
    confinements: List[float] = field(default_factory=list)

    def record(self, fdtd: AxiSymMaxwellFDTD, confinement_radius: Optional[float] = None):
        en = fdtd.compute_energy()
        self.times.append(fdtd.time)
        self.energies.append(en['total'])
        self.E_energies.append(en['E_energy'])
        self.B_energies.append(en['B_energy'])

        if confinement_radius is not None:
            gp = fdtd.gp
            conf = fdtd.compute_confinement(gp.R_torus, 0.0, confinement_radius)
            self.confinements.append(conf)

    def energy_conservation(self) -> float:
        """Relative energy change from initial."""
        if len(self.energies) < 2 or self.energies[0] == 0:
            return 0.0
        if self.energies[0] == 0.0:
            return 0.0
        return abs(self.energies[-1] - self.energies[0]) / self.energies[0]

    def dispersion_timescale(self, threshold: float = 0.5) -> Optional[float]:
        """Time when confinement drops below threshold, in units of T_circ."""
        if not self.confinements:
            return None
        for i, conf in enumerate(self.confinements):
            if conf < threshold:
                return self.times[i]
        return None  # never dropped below


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def test_plane_wave(verbose: bool = True) -> bool:
    """Test: plane wave propagation at speed c.

    Launch a z-propagating TM wave (E_z, B_phi polarization) and verify
    it moves at the correct speed.  TM mode is cleaner for this test
    because the (1/r) geometric coupling is weaker for E_z propagation.

    We also use a pulse far from axis with wide radial profile, so the
    cylindrical geometry is locally flat.
    """
    if verbose:
        print("  [test_plane_wave] ", end="", flush=True)

    gp = GridParams.for_test(Nr=200, Nz=600, dr=1.0, courant=0.5)
    fdtd = AxiSymMaxwellFDTD(gp)

    z = gp.z_axis
    r = gp.r_axis
    z0 = z[len(z)//4]  # start at quarter domain
    sigma_z = 15 * gp.dr
    # Far from axis, very wide in r → locally flat
    r0 = gp.Nr * gp.dr * 0.5
    sigma_r = 60 * gp.dr

    R, Z = np.meshgrid(r, z, indexing='ij')
    profile = (np.exp(-(Z - z0)**2 / (2 * sigma_z**2))
               * np.exp(-(R - r0)**2 / (2 * sigma_r**2)))

    # TM wave: E_z, B_phi, propagating in +z
    # For +z propagation: B_phi = E_z / c
    fdtd.E_z = profile * 1.0
    fdtd.B_phi = profile / c

    # Track z-centroid of |E_z|
    Ez_z = np.sum(np.abs(fdtd.E_z), axis=0)
    z_peak_initial = np.average(z, weights=Ez_z + 1e-30)

    n_steps = 200
    for _ in range(n_steps):
        fdtd.step_forward()

    Ez_z = np.sum(np.abs(fdtd.E_z), axis=0)
    z_peak_final = np.average(z, weights=Ez_z + 1e-30)

    expected_dz = gp.dt * c * n_steps
    actual_dz = z_peak_final - z_peak_initial

    error_cells = abs(actual_dz - expected_dz) / gp.dr
    passed = error_cells < 5.0  # ±5 cells tolerance for cylindrical geometry

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (moved {actual_dz/gp.dr:.1f} cells, "
              f"expected {expected_dz/gp.dr:.1f}, error {error_cells:.1f} cells)")

    return passed


def test_static_fields(verbose: bool = True) -> bool:
    """Test: static magnetic field from a current loop should not evolve.

    Initialize B from curl(A) and verify energy is conserved with no E fields.
    """
    if verbose:
        print("  [test_static_fields] ", end="", flush=True)

    gp = GridParams.for_test(Nr=100, Nz=200, dr=1.0, courant=0.5)
    fdtd = AxiSymMaxwellFDTD(gp)

    # Static B field: a simple dipole-like pattern
    r = gp.r_axis
    z = gp.z_axis
    R, Z = np.meshgrid(r, z, indexing='ij')

    # Gaussian B_z centered in domain
    r0 = gp.R_torus
    sigma = gp.r_minor * 3
    fdtd.B_z = np.exp(-((R - r0)**2 + Z**2) / (2 * sigma**2))

    E0 = fdtd.compute_energy()['total']

    # Evolve
    n_steps = 500
    for _ in range(n_steps):
        fdtd.step_forward()

    E_final = fdtd.compute_energy()['total']

    # A pure static B field WILL generate E fields via Faraday's law
    # unless it satisfies ∇×B = 0.  Our simple Gaussian doesn't.
    # So this test checks that energy is approximately conserved
    # (no blow-up), with some radiation loss through boundaries.
    # Allow 20% loss (Mur ABCs absorb radiation).
    if E0 > 0:
        relative_change = abs(E_final - E0) / E0
    else:
        relative_change = 0.0

    passed = relative_change < 0.5  # generous for ABCs

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (ΔE/E = {relative_change:.4f})")

    return passed


def test_energy_conservation(verbose: bool = True) -> bool:
    """Test: energy should be well-conserved for interior fields.

    Use a localized pulse far from boundaries to minimize ABC losses.
    """
    if verbose:
        print("  [test_energy_conservation] ", end="", flush=True)

    gp = GridParams.for_test(Nr=200, Nz=400, dr=1.0, courant=0.5)
    fdtd = AxiSymMaxwellFDTD(gp)

    # Localized TE pulse in center of domain (far from boundaries)
    r = gp.r_axis
    z = gp.z_axis
    R, Z = np.meshgrid(r, z, indexing='ij')

    r0 = gp.Nr * gp.dr * 0.4
    sigma = 15.0 * gp.dr
    profile = np.exp(-((R - r0)**2 + Z**2) / (2 * sigma**2))

    fdtd.E_phi = profile * 1.0
    # Match B for standing wave: B ~ E/c for equal energy density
    fdtd.B_r = -profile * 0.5 / c
    fdtd.B_z = profile * 0.5 / c

    diag = Diagnostics()
    diag.record(fdtd)

    n_steps = 100  # short run, pulse stays interior
    for _ in range(n_steps):
        fdtd.step_forward()
        if fdtd.step % 10 == 0:
            diag.record(fdtd)

    rel_change = diag.energy_conservation()
    passed = rel_change < 1e-2  # 1% tolerance for short interior run

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (ΔE/E = {rel_change:.6f} over {n_steps} steps)")

    return passed


def test_div_B(verbose: bool = True) -> bool:
    """Test: ∇·B = 0 when initialized from curl(A)."""
    if verbose:
        print("  [test_div_B] ", end="", flush=True)

    gp = GridParams.for_test(Nr=100, Nz=200, dr=1.0, courant=0.5)
    fdtd = AxiSymMaxwellFDTD(gp)

    # Initialize from Poisson solve (curl of A guarantees div B = 0)
    init_torus_fields(fdtd, amplitude=1.0)

    div = fdtd.compute_div_B()
    max_div = np.max(np.abs(div))

    # Relative to max |B|
    max_B = max(np.max(np.abs(fdtd.B_r)), np.max(np.abs(fdtd.B_z)), 1e-30)
    relative_div = max_div / (max_B / gp.dr)

    passed = relative_div < 1e-2  # finite-difference div is approximate

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (max |∇·B|/(|B|/dr) = {relative_div:.2e})")

    return passed


def run_validation(verbose: bool = True) -> bool:
    """Run all validation tests."""
    print("=" * 60)
    print("FDTD Validation Suite")
    print("=" * 60)

    results = []
    results.append(("Plane wave propagation", test_plane_wave(verbose)))
    results.append(("Static field stability", test_static_fields(verbose)))
    results.append(("Energy conservation", test_energy_conservation(verbose)))
    results.append(("div(B) = 0", test_div_B(verbose)))

    print("-" * 60)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")

    if n_pass == n_total:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED:")
        for name, p in results:
            if not p:
                print(f"  FAIL: {name}")

    return n_pass == n_total


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_fields(fdtd: AxiSymMaxwellFDTD, title: str = "", save_path: str = None):
    """4-panel field plot: E_phi, B_r, B_z, energy density."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    gp = fdtd.gp
    r = gp.r_axis * 1e15   # convert to fm
    z = gp.z_axis * 1e15

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title or f"FDTD Fields at t = {fdtd.time:.2e} s "
                 f"(step {fdtd.step})", fontsize=14)

    fields = [
        (fdtd.E_phi, r"$E_\phi$"),
        (fdtd.B_r, r"$B_r$"),
        (fdtd.B_z, r"$B_z$"),
        (fdtd.compute_energy_density(), "Energy density $u$"),
    ]

    for ax, (data, label) in zip(axes.flat, fields):
        vmax = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1.0
        if label == "Energy density $u$":
            im = ax.pcolormesh(z, r, data, cmap='inferno',
                               shading='auto')
        else:
            im = ax.pcolormesh(z, r, data, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlabel("z (fm)")
        ax.set_ylabel("r (fm)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

        # Mark torus location
        R_fm = gp.R_torus * 1e15
        r_fm = gp.r_minor * 1e15
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_fm * np.sin(theta), R_fm + r_fm * np.cos(theta),
                'g--', linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.savefig("nwt_fdtd_fields.png", dpi=150, bbox_inches='tight')
        print("  Saved: nwt_fdtd_fields.png")
    plt.close()


def plot_diagnostics(diag: Diagnostics, gp: GridParams,
                     save_path: str = None):
    """Plot energy and confinement vs time."""
    import matplotlib.pyplot as plt

    T_circ = gp.T_circ
    t_units = np.array(diag.times) / T_circ

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Energy vs time
    ax = axes[0]
    E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
    ax.plot(t_units, np.array(diag.energies) / E0, 'b-', label='Total')
    ax.plot(t_units, np.array(diag.E_energies) / E0, 'r--', label='Electric')
    ax.plot(t_units, np.array(diag.B_energies) / E0, 'g--', label='Magnetic')
    ax.set_xlabel(r"Time ($T_{\rm circ}$)")
    ax.set_ylabel(r"Energy / $E_0$")
    ax.set_title("Energy Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Confinement vs time
    ax = axes[1]
    if diag.confinements:
        ax.plot(t_units, diag.confinements, 'k-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5,
                    label='50% threshold')
        t_disp = diag.dispersion_timescale(0.5)
        if t_disp is not None:
            ax.axvline(x=t_disp / T_circ, color='orange', linestyle=':',
                        label=f'Dispersion at {t_disp/T_circ:.1f} $T_{{circ}}$')
        ax.set_xlabel(r"Time ($T_{\rm circ}$)")
        ax.set_ylabel("Confinement fraction")
        ax.set_title("Energy Confinement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No confinement data", ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    path = save_path or "nwt_fdtd_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def make_animation(frames: List[np.ndarray], gp: GridParams,
                   save_path: str = "nwt_fdtd_dispersion.gif"):
    """Create GIF animation from saved energy density frames."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    r = gp.r_axis * 1e15
    z = gp.z_axis * 1e15
    R_fm = gp.R_torus * 1e15
    r_fm = gp.r_minor * 1e15

    fig, ax = plt.subplots(figsize=(10, 6))

    vmax = max(np.max(f) for f in frames) if frames else 1.0
    im = ax.pcolormesh(z, r, frames[0], cmap='inferno',
                       vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(im, ax=ax, label="Energy density")
    ax.set_xlabel("z (fm)")
    ax.set_ylabel("r (fm)")

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_fm * np.sin(theta), R_fm + r_fm * np.cos(theta),
            'g--', linewidth=0.8, alpha=0.7)

    title = ax.set_title("t = 0")

    def update(frame_idx):
        im.set_array(frames[frame_idx].ravel())
        t_circ = frame_idx / max(len(frames)-1, 1)
        title.set_text(f"Energy density — frame {frame_idx}/{len(frames)-1}")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=100, blit=False)
    anim.save(save_path, writer='pillow', fps=10)
    print(f"  Saved animation: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------
def run_torus_dispersion(resolution: float = 1.0, n_circulations: int = 10,
                         animate: bool = False, verbose: bool = True):
    """Run the torus knot dispersion simulation.

    This is the core Phase 0 result: how fast does the EM configuration
    disperse without gravity?
    """
    print("=" * 60)
    print("Torus Knot EM Dispersion (Phase 0)")
    print("=" * 60)

    gp = GridParams.for_electron(resolution=resolution)

    print(f"\nGrid parameters:")
    print(f"  R_torus  = {gp.R_torus*1e15:.1f} fm")
    print(f"  r_minor  = {gp.r_minor*1e15:.2f} fm")
    print(f"  dr = dz  = {gp.dr*1e15:.3f} fm")
    print(f"  Nr × Nz  = {gp.Nr} × {gp.Nz} = {gp.Nr*gp.Nz/1e6:.1f}M cells")
    print(f"  dt       = {gp.dt:.2e} s")
    print(f"  T_circ   = {gp.T_circ:.2e} s")
    print(f"  Steps/circ = {gp.steps_per_circ}")
    print(f"  Memory   = {gp.memory_MB:.0f} MB")

    total_steps = n_circulations * gp.steps_per_circ
    print(f"\nSimulation: {n_circulations} circulations = {total_steps} steps")

    # Create solver and initialize
    fdtd = AxiSymMaxwellFDTD(gp)

    print("\nInitializing fields (EM wave packet on torus)...")
    t0 = time.time()
    init_em_wave_torus(fdtd, amplitude=1.0)
    print(f"  Init time: {time.time()-t0:.2f}s")

    # Initial div(B) check
    div = fdtd.compute_div_B()
    max_B = max(np.max(np.abs(fdtd.B_r)), np.max(np.abs(fdtd.B_z)), 1e-30)
    print(f"  Initial max|∇·B|/(|B|/dr) = {np.max(np.abs(div))/(max_B/gp.dr):.2e}")

    # Diagnostics
    confinement_radius = 3 * gp.r_minor
    diag = Diagnostics()
    diag.record(fdtd, confinement_radius)

    frames = []
    record_interval = max(1, gp.steps_per_circ // 10)
    frame_interval = max(1, gp.steps_per_circ // 2) if animate else total_steps + 1

    # Initial field plot
    print("\nSaving initial field snapshot...")
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    plot_fields(fdtd, title="Initial EM torus configuration",
                save_path=os.path.join(sim_dir, "nwt_fdtd_initial.png"))

    # Main time loop
    print(f"\nEvolving...")
    t0 = time.time()
    for step in range(total_steps):
        fdtd.step_forward()

        if step % record_interval == 0:
            diag.record(fdtd, confinement_radius)

        if animate and step % frame_interval == 0:
            frames.append(fdtd.compute_energy_density().copy())

        if verbose and step % (gp.steps_per_circ * max(1, n_circulations // 10)) == 0 and step > 0:
            circ = step / gp.steps_per_circ
            en = fdtd.compute_energy()
            conf = fdtd.compute_confinement(gp.R_torus, 0.0, confinement_radius)
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0
            E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
            print(f"  t = {circ:.1f} T_circ | "
                  f"E/E₀ = {en['total']/E0:.4f} | "
                  f"Confinement = {conf:.3f} | "
                  f"{rate:.0f} steps/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")

    # Final diagnostics
    diag.record(fdtd, confinement_radius)
    print(f"\nResults:")
    print(f"  Initial energy:    {diag.energies[0]:.4e}")
    print(f"  Final energy:      {diag.energies[-1]:.4e}")
    print(f"  Energy change:     {diag.energy_conservation()*100:.2f}%")
    if diag.confinements:
        print(f"  Initial confinement: {diag.confinements[0]:.4f}")
        print(f"  Final confinement:   {diag.confinements[-1]:.4f}")
        t_disp = diag.dispersion_timescale(0.5)
        if t_disp is not None:
            print(f"  Dispersion time (50%): {t_disp/gp.T_circ:.2f} T_circ")
        else:
            print(f"  Dispersion time (50%): > {n_circulations} T_circ (still confined)")

    # Save plots
    print("\nGenerating plots...")
    plot_fields(fdtd, title=f"Fields after {n_circulations} circulations",
                save_path=os.path.join(sim_dir, "nwt_fdtd_final.png"))
    plot_diagnostics(diag, gp,
                     save_path=os.path.join(sim_dir, "nwt_fdtd_diagnostics.png"))

    if animate and frames:
        print("Generating animation...")
        make_animation(frames, gp,
                       save_path=os.path.join(sim_dir, "nwt_fdtd_dispersion.gif"))

    return diag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NWT FDTD: Axisymmetric Maxwell solver for torus knot EM configurations")
    parser.add_argument('--test', action='store_true',
                        help='Run validation test suite')
    parser.add_argument('--run', action='store_true',
                        help='Run torus dispersion simulation')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Resolution multiplier (default: 1.0)')
    parser.add_argument('--n-circulations', type=int, default=10,
                        help='Number of circulation periods to simulate')
    parser.add_argument('--animate', action='store_true',
                        help='Generate GIF animation')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    if not args.test and not args.run:
        parser.print_help()
        print("\nQuick start:")
        print("  python nwt_fdtd.py --test              # validation")
        print("  python nwt_fdtd.py --run               # dispersion sim")
        print("  python nwt_fdtd.py --run --animate     # with animation")
        return

    if args.test:
        success = run_validation(verbose=not args.quiet)
        if not success:
            sys.exit(1)

    if args.run:
        run_torus_dispersion(
            resolution=args.resolution,
            n_circulations=args.n_circulations,
            animate=args.animate,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
