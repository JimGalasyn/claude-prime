#!/usr/bin/env python3
"""
Einstein-Maxwell FDTD: Phase 2 — Maxwell on Kerr-Newman Background
====================================================================

Extends Phase 0 (nwt_fdtd.py) to propagate EM fields on a Kerr-Newman
spacetime background. The KN solution for electron parameters
(m_e, e, hbar/2) has a ring singularity at a = hbar/(2 m_e c) = 193 fm
-- exactly the NWT torus radius.

At G_N the metric correction is ~10^-44 (flat to 44 decimal places).
The key scientific output is a G_eff scan: at what effective gravitational
coupling does confinement appear?

Architecture: subclasses AxiSymMaxwellFDTD from Phase 0.  Overrides
E-field updates (Ampere: alpha^2 * c^2) and B-field updates (Faraday:
alpha * E in curl).  With --frame-dragging, shift vector beta^phi
couples TE and TM modes via Lie derivative terms.

Phase 2a: lapse-only (alpha in both Ampere and Faraday)
Phase 2b: lapse + frame-dragging (beta^phi shift coupling)

Usage:
    python nwt_fdtd_kn.py --test                          # validation
    python nwt_fdtd_kn.py --single --g-ratio 1e44         # single run
    python nwt_fdtd_kn.py --single --g-ratio 1.5e43 --frame-dragging
    python nwt_fdtd_kn.py --scan                          # G_eff sweep
    python nwt_fdtd_kn.py --scan --frame-dragging         # Phase 2b scan
    python nwt_fdtd_kn.py --dynamic                       # dynamic G_eff
    python nwt_fdtd_kn.py --metric-viz --g-ratio 3e44     # lapse plot

Requires: numpy, matplotlib
"""

import numpy as np
import argparse
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List

# Import Phase 0 infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_fdtd import (
    AxiSymMaxwellFDTD, GridParams, Diagnostics,
    init_em_wave_torus, plot_fields, plot_diagnostics,
    c, hbar, e_charge, eps0, mu0, m_e, alpha as alpha_fs, lambda_C,
)

# ---------------------------------------------------------------------------
# Additional physical constants
# ---------------------------------------------------------------------------
G_N = 6.67430e-11  # Newton's gravitational constant (m^3 kg^-1 s^-2)


# ---------------------------------------------------------------------------
# Step 1: Coordinate Transform + Metric
# ---------------------------------------------------------------------------
def cyl_to_bl(rho, z, a):
    """Convert cylindrical (rho, z) to Boyer-Lindquist (r, theta).

    Parameters
    ----------
    rho : array_like  -- cylindrical radial coordinate
    z   : array_like  -- cylindrical axial coordinate
    a   : float       -- Kerr spin parameter (m)

    Returns
    -------
    r_bl    : ndarray -- Boyer-Lindquist radial coordinate
    cos_th  : ndarray -- cos(theta)
    """
    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # r^2 = [(rho^2 + z^2 - a^2) + sqrt((rho^2+z^2-a^2)^2 + 4 a^2 z^2)] / 2
    w = rho**2 + z**2 - a**2
    r_sq = 0.5 * (w + np.sqrt(w**2 + 4.0 * a**2 * z**2))
    r_bl = np.sqrt(np.maximum(r_sq, 0.0))

    # cos(theta) from rho = sqrt(r^2+a^2) * sin(theta):
    #   cos^2(theta) = 1 - rho^2 / (r^2 + a^2)
    # This avoids the 0/0 issue with z/r at the ring singularity (r=0).
    r2_a2 = r_bl**2 + a**2
    cos_th_sq = np.clip(1.0 - rho**2 / r2_a2, 0.0, 1.0)
    cos_th = np.sqrt(cos_th_sq)
    # Sign: z = r * cos(theta), so cos(theta) has the same sign as z
    cos_th = np.where(z < 0, -cos_th, cos_th)

    return r_bl, cos_th


def bl_to_cyl(r_bl, cos_th, a):
    """Convert Boyer-Lindquist (r, theta) back to cylindrical (rho, z).

    rho = sqrt(r^2 + a^2) * sin(theta)
    z   = r * cos(theta)
    """
    sin_th = np.sqrt(np.maximum(1.0 - cos_th**2, 0.0))
    rho = np.sqrt(r_bl**2 + a**2) * sin_th
    z = r_bl * cos_th
    return rho, z


def compute_kn_metric(r_bl, cos_th, M_geom, a, Q_geom, sigma_floor):
    """Compute Kerr-Newman metric components.

    Parameters
    ----------
    r_bl        : ndarray -- BL radial coordinate
    cos_th      : ndarray -- cos(theta)
    M_geom      : float   -- geometric mass = G_eff * m_e / c^2
    a           : float   -- spin parameter (m), FIXED independent of G_eff
    Q_geom      : float   -- geometric charge = sqrt(G_eff * e^2 / (4 pi eps0 c^4))
    sigma_floor : float   -- minimum Sigma value for regularization

    Returns
    -------
    alpha   : ndarray -- lapse function
    beta_phi: ndarray -- azimuthal shift (frame-dragging)
    """
    sin2_th = np.maximum(1.0 - cos_th**2, 0.0)
    r = r_bl

    # Sigma = r^2 + a^2 cos^2(theta) -- regularized
    Sigma = r**2 + a**2 * cos_th**2
    Sigma = np.maximum(Sigma, sigma_floor**2)

    # Delta = r^2 - 2Mr + a^2 + Q^2
    Delta = r**2 - 2.0 * M_geom * r + a**2 + Q_geom**2

    # f = 2Mr - Q^2 ("mass-charge" function; controls deviation from flat)
    f = 2.0 * M_geom * r - Q_geom**2

    # A = (r^2+a^2)*Sigma + f*a^2*sin^2(theta)
    # This is algebraically equivalent to (r^2+a^2)^2 - Delta*a^2*sin^2(theta),
    # but uses the REGULARIZED Sigma, which ensures that at the ring singularity
    # (Sigma_raw -> 0), A remains proportional to Sigma and alpha^2 -> 1
    # in the flat limit (f -> 0).  The direct formula A = r2_a2^2 - Delta*a^2*sin^2
    # suffers from catastrophic cancellation: both terms -> a^4 near r=0,theta=pi/2.
    r2_a2 = r**2 + a**2
    A = r2_a2 * Sigma + f * a**2 * sin2_th
    # A has units m^4.  Floor must also be m^4: use Sigma_floor * r2_a2.
    # At the ring singularity (r=0): A_floor = eps^2*a^4, giving alpha^2 -> 1
    # in the flat limit.  For strong G_eff, physical deviations occur at r > 0.
    A = np.maximum(A, sigma_floor**2 * r2_a2)

    # Lapse: alpha^2 = Delta * Sigma / A
    alpha_sq = Delta * Sigma / A
    alpha_sq = np.clip(alpha_sq, 0.0, 4.0)
    lapse = np.sqrt(alpha_sq)

    # Shift: beta^phi = -f * a / A
    beta_phi = -f * a / A

    return lapse, beta_phi


def electron_kn_params(G_eff):
    """Compute Kerr-Newman geometric parameters for electron at given G_eff.

    Returns
    -------
    M_geom  : float -- geometric mass
    a       : float -- spin parameter (FIXED, independent of G_eff)
    Q_geom  : float -- geometric charge
    a_over_M: float -- a/M ratio (extremality parameter)
    """
    M_geom = G_eff * m_e / c**2
    a = hbar / (2.0 * m_e * c)  # always the same
    Q_geom = np.sqrt(G_eff * e_charge**2 / (4.0 * np.pi * eps0 * c**4))

    # Extremality: a/M (=1 is extremal, <1 has horizon, >1 is naked)
    a_over_M = a / M_geom if M_geom > 0 else np.inf

    return M_geom, a, Q_geom, a_over_M


def has_horizon(M_geom, a, Q_geom):
    """Check if KN spacetime has an event horizon.

    Horizon exists when Delta=0 has real roots: M^2 >= a^2 + Q^2
    """
    return M_geom**2 >= a**2 + Q_geom**2


# ---------------------------------------------------------------------------
# Step 2: KerrNewmanFDTD Class (Lapse-Only -- Phase 2a)
# ---------------------------------------------------------------------------
class KerrNewmanFDTD(AxiSymMaxwellFDTD):
    """FDTD solver on Kerr-Newman background.

    The lapse alpha modifies the local wave speed: c_eff(x) = alpha(x) * c.
    In the FDTD, both Ampere's and Faraday's laws get lapse corrections.

    When frame_dragging=True, the shift vector beta^phi couples TE and TM
    modes via Lie derivative terms, creating an angular momentum barrier.
    """

    def __init__(self, gp: GridParams, G_eff: float = G_N,
                 sigma_floor_frac: float = 0.01,
                 frame_dragging: bool = False):
        super().__init__(gp)

        self.G_eff = G_eff
        self.sigma_floor_frac = sigma_floor_frac
        self.frame_dragging = frame_dragging

        # Compute KN parameters
        M_geom, a, Q_geom, a_over_M = electron_kn_params(G_eff)
        self.M_geom = M_geom
        self.a_spin = a
        self.Q_geom = Q_geom
        self.a_over_M = a_over_M
        self.kn_has_horizon = has_horizon(M_geom, a, Q_geom)

        sigma_floor = sigma_floor_frac * a

        # --- Precompute lapse and shift at cell centers ---
        r_ax = gp.r_axis   # (Nr,)
        z_ax = gp.z_axis   # (Nz,)
        RHO, ZZ = np.meshgrid(r_ax, z_ax, indexing='ij')  # (Nr, Nz)

        r_bl, cos_th = cyl_to_bl(RHO, ZZ, a)
        alpha_center, beta_phi_center = compute_kn_metric(
            r_bl, cos_th, M_geom, a, Q_geom, sigma_floor)
        self.alpha_center = alpha_center  # shape (Nr, Nz)

        # --- alpha^2 interpolated to staggered E-field positions ---
        # Clip alpha for CFL stability: need alpha * courant < 1 in 2D
        max_safe_alpha = 0.95 / gp.courant  # ~1.36 for courant=0.7
        alpha_cfl = np.clip(alpha_center, 0.0, max_safe_alpha)

        # E_phi at cell centers (i, j) -- no interpolation needed
        self.alpha_sq_Ephi = alpha_cfl**2

        # E_r at (i+1/2, j) -- average adjacent cells in r
        # alpha_sq_Er[i,j] = 0.5*(alpha[i,j]^2 + alpha[i+1,j]^2)
        alpha_sq = alpha_cfl**2
        self.alpha_sq_Er = np.zeros_like(alpha_sq)
        self.alpha_sq_Er[:-1, :] = 0.5 * (alpha_sq[:-1, :] + alpha_sq[1:, :])
        self.alpha_sq_Er[-1, :] = alpha_sq[-1, :]  # boundary: copy last

        # E_z at (i, j+1/2) -- average adjacent cells in z
        self.alpha_sq_Ez = np.zeros_like(alpha_sq)
        self.alpha_sq_Ez[:, :-1] = 0.5 * (alpha_sq[:, :-1] + alpha_sq[:, 1:])
        self.alpha_sq_Ez[:, -1] = alpha_sq[:, -1]  # boundary: copy last

        # --- Alpha at staggered positions (unclipped, for Faraday) ---
        # alpha at E_r position (i+1/2, j)
        self.alpha_at_Er = np.zeros_like(alpha_center)
        self.alpha_at_Er[:-1, :] = 0.5 * (alpha_center[:-1, :] + alpha_center[1:, :])
        self.alpha_at_Er[-1, :] = alpha_center[-1, :]

        # alpha at E_z position (i, j+1/2)
        self.alpha_at_Ez = np.zeros_like(alpha_center)
        self.alpha_at_Ez[:, :-1] = 0.5 * (alpha_center[:, :-1] + alpha_center[:, 1:])
        self.alpha_at_Ez[:, -1] = alpha_center[:, -1]

        # --- Frame-dragging arrays ---
        self._precompute_frame_dragging(beta_phi_center)

        # Track max deviation from flat
        self.max_alpha_deviation = float(np.max(np.abs(alpha_center - 1.0)))

        # CFL check: alpha <= 1 for sub-extremal -> existing Courant is fine
        max_alpha = float(np.max(alpha_center))
        if max_alpha > 1.0:
            needed_courant = gp.courant / max_alpha
            if needed_courant < 0.1:
                print(f"WARNING: max(alpha)={max_alpha:.4f} > 1, "
                      f"CFL may be violated. Consider reducing courant to "
                      f"<= {needed_courant:.4f}")

    def _precompute_frame_dragging(self, beta_phi_center):
        """Precompute beta^phi gradients for frame-dragging terms.

        When max|beta^phi| is below a threshold (1e-20), all shift arrays
        are zeroed.  At G_N, beta_phi ~ 10^-31 is physically zero but
        numerically nonzero; the c^2 amplification in Ampere's law would
        convert these rounding-level gradients into exponentially growing
        TM fields.  The threshold eliminates this artifact.
        """
        Nr, Nz = self.gp.Nr, self.gp.Nz
        dr = self.gp.dr
        dz = self.gp.dz

        # Threshold: if beta_phi is negligibly small, zero everything
        beta_phi_max = float(np.max(np.abs(beta_phi_center)))
        if beta_phi_max < 1e-20:
            self.beta_phi_center = np.zeros((Nr, Nz))
            self.dbeta_dr = np.zeros((Nr - 1, Nz))
            self.dbeta_dz = np.zeros((Nr, Nz - 1))
            self.dbeta_dz_at_Bphi = np.zeros((Nr - 1, Nz))
            return

        self.beta_phi_center = beta_phi_center  # (Nr, Nz)

        # d_r(beta^phi) at (i+1/2, j) -- natural for E_r and B_phi positions
        self.dbeta_dr = np.zeros((Nr - 1, Nz))
        self.dbeta_dr[:, :] = (beta_phi_center[1:, :] - beta_phi_center[:-1, :]) / dr

        # d_z(beta^phi) at (i, j+1/2) -- natural for E_z position
        self.dbeta_dz = np.zeros((Nr, Nz - 1))
        self.dbeta_dz[:, :] = (beta_phi_center[:, 1:] - beta_phi_center[:, :-1]) / dz

        # d_z(beta^phi) at B_phi position (i+1/2, j):
        # centered diff at cell centers: (beta[i,j+1]-beta[i,j-1])/(2*dz)
        # then average in r to (i+1/2, j)
        dbdz_center = np.zeros((Nr, Nz))
        dbdz_center[:, 1:-1] = (beta_phi_center[:, 2:] - beta_phi_center[:, :-2]) / (2.0 * dz)
        # Boundary: one-sided at j=0 and j=Nz-1
        dbdz_center[:, 0] = (beta_phi_center[:, 1] - beta_phi_center[:, 0]) / dz
        dbdz_center[:, -1] = (beta_phi_center[:, -1] - beta_phi_center[:, -2]) / dz
        # Average to (i+1/2, j)
        self.dbeta_dz_at_Bphi = 0.5 * (dbdz_center[:-1, :] + dbdz_center[1:, :])

        # Stability clipping: the explicit shift coupling dB_phi/dt ~ B * dbeta/dx
        # requires |dt * dbeta/dx| << 1 to prevent exponential blowup.
        # The TE→TM coupling feeds through the Ampere c^2 amplification,
        # so the effective stability limit is more restrictive than CFL alone.
        # Analogous to the alpha CFL clipping for wave speed.
        dt = self.gp.dt
        max_safe_grad = 0.05 / dt
        self.dbeta_dr = np.clip(self.dbeta_dr, -max_safe_grad, max_safe_grad)
        self.dbeta_dz = np.clip(self.dbeta_dz, -max_safe_grad, max_safe_grad)
        self.dbeta_dz_at_Bphi = np.clip(self.dbeta_dz_at_Bphi,
                                         -max_safe_grad, max_safe_grad)

    def _update_B_te(self):
        """Override: B_r, B_z update with Faraday lapse correction.

        dB_r/dt = d_z(alpha * E_phi)
        dB_z/dt = -(1/r) d_r(r * alpha * E_phi)
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz

        # alpha * E_phi at cell centers
        aE = self.alpha_center * self.E_phi

        # B_r at (i, j+1/2): dB_r/dt = d_z(alpha * E_phi)
        self.B_r[:, :-1] += dt / dz * (aE[:, 1:] - aE[:, :-1])

        # B_z at (i+1/2, j+1/2): dB_z/dt = -(1/r) d_r(r * alpha * E_phi)
        raE = self.r_c_2d * aE
        r_half = self.r_e_2d[1:-1]  # (Nr-1, 1)
        inv_r_half = np.where(r_half > 0, 1.0 / r_half, 2.0 / dr)
        self.B_z[:-1, :-1] -= dt * inv_r_half * (raE[1:, :-1] - raE[:-1, :-1]) / dr

    def _update_E_te(self):
        """Override: E_phi update with lapse modification.

        Ampere: dE_phi/dt = alpha^2 * c^2 * (curl B)_phi
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz

        # alpha^2 * c^2 at E_phi positions (cell centers)
        ac2 = self.alpha_sq_Ephi * self.c_sq

        # dB_r/dz at (i, j): (B_r[i,j] - B_r[i,j-1]) / dz
        self.E_phi[:, 1:] += ac2[:, 1:] * dt / dz * (
            self.B_r[:, 1:] - self.B_r[:, :-1])

        # dB_z/dr at (i, j): (B_z[i,j] - B_z[i-1,j]) / dr
        self.E_phi[1:, :-1] -= ac2[1:, :-1] * dt / dr * (
            self.B_z[1:, :-1] - self.B_z[:-1, :-1])

    def _update_B_tm(self):
        """Override: B_phi update with Faraday lapse + shift coupling.

        B_phi lives at (i+½, j+½) — the Yee-correct stagger that ensures
        centered differences for both dE_r/dz and dE_z/dr, preserving the
        discrete energy conservation property.

        Lapse: dB_phi/dt = -alpha * [d_z(E_r) - d_r(E_z)]
        Shift: dB_phi/dt += -(B_r * d_r(beta^phi) + B_z * d_z(beta^phi))
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz

        # Lapse-corrected curl: use alpha at E_r and E_z positions
        aEr = self.alpha_at_Er * self.E_r
        aEz = self.alpha_at_Ez * self.E_z

        # B_phi at (i+½, j+½):
        # d(alpha*E_r)/dz centered at j+½: (aEr[j+1] - aEr[j]) / dz
        self.B_phi[:-1, :-1] -= dt / dz * (aEr[:-1, 1:] - aEr[:-1, :-1])

        # d(alpha*E_z)/dr centered at i+½: (aEz[i+1] - aEz[i]) / dr
        self.B_phi[:-1, :-1] += dt / dr * (aEz[1:, :-1] - aEz[:-1, :-1])

        # Shift coupling: dB_phi/dt += -(B_r * d_r(beta^phi) + B_z * d_z(beta^phi))
        if self.frame_dragging:
            # B_r at (i, j+½) → average in r to (i+½, j+½)
            # B_r valid at j=0..Nz-2, so use :-1 in z
            B_r_interp = 0.5 * (self.B_r[:-1, :-1] + self.B_r[1:, :-1])  # (Nr-1, Nz-1)

            # B_z already at (i+½, j+½) — no interpolation needed
            # B_z valid range: [:-1, :-1] = (Nr-1, Nz-1)

            # Interpolate gradients to (i+½, j+½)
            # dbeta_dr at (i+½, j) → average in z to (i+½, j+½)
            dbdr = 0.5 * (self.dbeta_dr[:, :-1] + self.dbeta_dr[:, 1:])  # (Nr-1, Nz-1)
            # dbeta_dz at (i, j+½) → average in r to (i+½, j+½)
            dbdz = 0.5 * (self.dbeta_dz[:-1, :] + self.dbeta_dz[1:, :])  # (Nr-1, Nz-1)

            self.B_phi[:-1, :-1] -= dt * (
                B_r_interp * dbdr + self.B_z[:-1, :-1] * dbdz)

    def _update_E_tm(self):
        """Override: E_r, E_z update with lapse + shift.

        Ampere: dE_r/dt = -alpha^2 * c^2 * dB_phi/dz
                dE_z/dt =  alpha^2 * c^2 * (1/r) d(r B_phi)/dr
        Shift:  dE_r/dt += r * E_phi * d_r(beta^phi)
                dE_z/dt += r * E_phi * d_z(beta^phi)
        """
        dt = self.gp.dt
        dr = self.gp.dr
        dz = self.gp.dz

        # E_r at (i+1/2, j)
        ac2_r = self.alpha_sq_Er * self.c_sq
        self.E_r[:, 1:] -= ac2_r[:, 1:] * dt / dz * (
            self.B_phi[:, 1:] - self.B_phi[:, :-1])

        # E_z at (i, j+1/2)
        ac2_z = self.alpha_sq_Ez * self.c_sq
        rB = self.r_e_2d[1:] * self.B_phi  # r B_phi at (i+1/2, j)
        rB_shifted = np.zeros_like(rB)
        rB_shifted[1:, :] = self.r_e_2d[1:-1] * self.B_phi[:-1, :]

        self.E_z[:, :-1] += ac2_z[:, :-1] * dt / dr * self.inv_r_c_2d * (
            rB[:, :-1] - rB_shifted[:, :-1])

        # Shift coupling terms
        if self.frame_dragging:
            # E_r += dt * r * E_phi_interp * d_r(beta^phi)
            # E_phi at (i,j) -> interpolate to E_r at (i+1/2, j)
            E_phi_at_Er = 0.5 * (self.E_phi[:-1, :] + self.E_phi[1:, :])
            r_Er = self.r_e_2d[1:-1]  # r at (i+1/2), shape (Nr-1, 1)
            self.E_r[:-1, :] += dt * r_Er * E_phi_at_Er * self.dbeta_dr

            # E_z += dt * r * E_phi_interp * d_z(beta^phi)
            # E_phi at (i,j) -> interpolate to E_z at (i, j+1/2)
            E_phi_at_Ez = 0.5 * (self.E_phi[:, :-1] + self.E_phi[:, 1:])
            r_Ez = self.r_c_2d  # r at (i), shape (Nr, 1)
            self.E_z[:, :-1] += dt * r_Ez * E_phi_at_Ez * self.dbeta_dz


# ---------------------------------------------------------------------------
# Step 4: Dynamic G_eff (Phase 2c)
# ---------------------------------------------------------------------------
class DynamicGeffFDTD(KerrNewmanFDTD):
    """G_eff(u) = G_N * (1 + (u/u_crit)^n) -- "violent creation" model.

    Strong coupling during transient energy spike, relaxes to G_N after
    topology forms.  Metric recomputed every recompute_interval steps.
    """

    def __init__(self, gp: GridParams, u_crit: float = 1.0,
                 exponent: float = 2.0, recompute_interval: int = 10,
                 sigma_floor_frac: float = 0.01):
        # Initialize with G_N first
        super().__init__(gp, G_eff=G_N, sigma_floor_frac=sigma_floor_frac)
        self.u_crit = u_crit
        self.exponent = exponent
        self.recompute_interval = recompute_interval
        self.sigma_floor_frac = sigma_floor_frac
        self.geff_history = []

    def _recompute_metric(self):
        """Recompute metric from local EM energy density."""
        u = self.compute_energy_density()
        # Peak energy density determines effective G
        u_max = float(np.max(u))
        G_eff_local = G_N * (1.0 + (u_max / self.u_crit) ** self.exponent)

        # Recompute KN metric
        M_geom, a, Q_geom, a_over_M = electron_kn_params(G_eff_local)
        self.G_eff = G_eff_local
        self.M_geom = M_geom
        self.Q_geom = Q_geom
        self.a_over_M = a_over_M
        self.kn_has_horizon = has_horizon(M_geom, a, Q_geom)

        sigma_floor = self.sigma_floor_frac * a

        r_ax = self.gp.r_axis
        z_ax = self.gp.z_axis
        RHO, ZZ = np.meshgrid(r_ax, z_ax, indexing='ij')
        r_bl, cos_th = cyl_to_bl(RHO, ZZ, a)
        alpha_center, beta_phi_center = compute_kn_metric(
            r_bl, cos_th, M_geom, a, Q_geom, sigma_floor)
        self.alpha_center = alpha_center
        max_safe_alpha = 0.95 / self.gp.courant
        alpha_cfl = np.clip(alpha_center, 0.0, max_safe_alpha)
        alpha_sq = alpha_cfl**2
        self.alpha_sq_Ephi = alpha_sq
        self.alpha_sq_Er[:-1, :] = 0.5 * (alpha_sq[:-1, :] + alpha_sq[1:, :])
        self.alpha_sq_Er[-1, :] = alpha_sq[-1, :]
        self.alpha_sq_Ez[:, :-1] = 0.5 * (alpha_sq[:, :-1] + alpha_sq[:, 1:])
        self.alpha_sq_Ez[:, -1] = alpha_sq[:, -1]

        # Recompute alpha at staggered positions (for Faraday)
        self.alpha_at_Er[:-1, :] = 0.5 * (alpha_center[:-1, :] + alpha_center[1:, :])
        self.alpha_at_Er[-1, :] = alpha_center[-1, :]
        self.alpha_at_Ez[:, :-1] = 0.5 * (alpha_center[:, :-1] + alpha_center[:, 1:])
        self.alpha_at_Ez[:, -1] = alpha_center[:, -1]

        # Recompute frame-dragging gradients
        self._precompute_frame_dragging(beta_phi_center)

        self.max_alpha_deviation = float(np.max(np.abs(alpha_center - 1.0)))

        self.geff_history.append((self.time, G_eff_local))

    def step_forward(self):
        """Override to recompute metric periodically."""
        if self.step % self.recompute_interval == 0:
            self._recompute_metric()
        super().step_forward()


# ---------------------------------------------------------------------------
# Step 3: G_eff Scan
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    """Result from a single G_eff point in the scan."""
    G_eff: float
    G_ratio: float       # G_eff / G_N
    a_over_M: float
    has_horizon: bool
    final_confinement: float
    dispersion_time: Optional[float]  # in units of T_circ, or None
    max_alpha_deviation: float
    final_energy_ratio: float  # E_final / E_initial


def run_single(G_ratio: float = 1.0, resolution: float = 0.5,
               n_circulations: int = 5, verbose: bool = True,
               frame_dragging: bool = False):
    """Run a single KN simulation at specified G_eff/G_N ratio.

    Returns (fdtd, diagnostics).
    """
    G_eff = G_ratio * G_N
    gp = GridParams.for_electron(resolution=resolution)

    if verbose:
        M, a, Q, aM = electron_kn_params(G_eff)
        print(f"\nKerr-Newman FDTD: G_eff/G_N = {G_ratio:.2e}")
        print(f"  M_geom  = {M:.4e} m")
        print(f"  a       = {a:.4e} m  (= R_torus)")
        print(f"  Q_geom  = {Q:.4e} m")
        print(f"  a/M     = {aM:.4e}")
        print(f"  Horizon = {has_horizon(M, a, Q)}")
        print(f"  Grid    = {gp.Nr} x {gp.Nz}")
        print(f"  Frame-dragging = {frame_dragging}")

    fdtd = KerrNewmanFDTD(gp, G_eff=G_eff, frame_dragging=frame_dragging)

    if verbose:
        print(f"  max|alpha-1| = {fdtd.max_alpha_deviation:.6e}")

    init_em_wave_torus(fdtd, amplitude=1.0)

    confinement_radius = 3 * gp.r_minor
    diag = Diagnostics()
    diag.record(fdtd, confinement_radius)

    total_steps = n_circulations * gp.steps_per_circ
    record_interval = max(1, gp.steps_per_circ // 10)

    t0 = time.time()
    for step_i in range(total_steps):
        fdtd.step_forward()

        if step_i % record_interval == 0:
            diag.record(fdtd, confinement_radius)

        if (verbose and step_i > 0 and
                step_i % (gp.steps_per_circ * max(1, n_circulations // 5)) == 0):
            circ = step_i / gp.steps_per_circ
            en = fdtd.compute_energy()
            conf = fdtd.compute_confinement(gp.R_torus, 0.0, confinement_radius)
            E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
            elapsed = time.time() - t0
            rate = step_i / elapsed if elapsed > 0 else 0
            print(f"  t={circ:.1f} T_circ | E/E0={en['total']/E0:.4f} | "
                  f"conf={conf:.3f} | {rate:.0f} steps/s")

    diag.record(fdtd, confinement_radius)
    elapsed = time.time() - t0

    if verbose:
        E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
        print(f"\n  Done in {elapsed:.1f}s ({total_steps/elapsed:.0f} steps/s)")
        print(f"  Final confinement: {diag.confinements[-1]:.4f}")
        print(f"  Energy change: {diag.energy_conservation()*100:.2f}%")
        t_disp = diag.dispersion_timescale(0.5)
        if t_disp is not None:
            print(f"  Dispersion time (50%): {t_disp/gp.T_circ:.2f} T_circ")
        else:
            print(f"  Dispersion time (50%): > {n_circulations} T_circ")

    return fdtd, diag


def run_geff_scan(n_points: int = 50, resolution: float = 0.5,
                  n_circulations: int = 5,
                  G_min_ratio: float = 1.0, G_max_ratio: float = 1e46,
                  verbose: bool = True, frame_dragging: bool = False):
    """Scan G_eff from G_N to 10^46 G_N.

    Returns list of ScanResult.
    """
    G_ratios = np.logspace(np.log10(G_min_ratio), np.log10(G_max_ratio),
                           n_points)

    gp = GridParams.for_electron(resolution=resolution)
    confinement_radius = 3 * gp.r_minor
    T_circ = gp.T_circ

    phase_label = "Phase 2b (lapse+shift)" if frame_dragging else "Phase 2a (lapse-only)"
    print("=" * 70)
    print(f"G_eff Scan: {phase_label}")
    print("=" * 70)
    print(f"  Points:        {n_points}")
    print(f"  G range:       {G_min_ratio:.0e} - {G_max_ratio:.0e} G_N")
    print(f"  Resolution:    {resolution}")
    print(f"  Circulations:  {n_circulations}")
    print(f"  Grid:          {gp.Nr} x {gp.Nz}")
    print(f"  Steps/point:   ~{n_circulations * gp.steps_per_circ}")
    print(f"  Frame-dragging: {frame_dragging}")

    # Extremal transition
    # a/M = 1 when G_eff = a * c^2 / m_e = hbar c / (2 m_e^2)
    a = hbar / (2.0 * m_e * c)
    G_extremal = a * c**2 / m_e
    G_extremal_ratio = G_extremal / G_N
    print(f"  Extremal G_eff/G_N = {G_extremal_ratio:.4e}")
    print()

    results = []
    t0_total = time.time()

    for idx, G_ratio in enumerate(G_ratios):
        G_eff = G_ratio * G_N
        M, a_spin, Q, aM = electron_kn_params(G_eff)
        hz = has_horizon(M, a_spin, Q)

        if verbose:
            print(f"  [{idx+1:3d}/{n_points}] G/G_N={G_ratio:.2e} "
                  f"a/M={aM:.2e} horizon={hz}", end=" ", flush=True)

        t0 = time.time()

        try:
            fdtd = KerrNewmanFDTD(gp, G_eff=G_eff,
                                    frame_dragging=frame_dragging)
            init_em_wave_torus(fdtd, amplitude=1.0)

            diag = Diagnostics()
            diag.record(fdtd, confinement_radius)

            total_steps = n_circulations * gp.steps_per_circ
            record_interval = max(1, gp.steps_per_circ // 10)

            stable = True
            for step_i in range(total_steps):
                fdtd.step_forward()
                if step_i % record_interval == 0:
                    diag.record(fdtd, confinement_radius)
                # Check for numerical blowup (NaN/Inf only — metric-induced
                # energy growth is expected and not a stability issue)
                if step_i % (total_steps // 5) == 0 and step_i > 0:
                    en = fdtd.compute_energy()
                    if np.isnan(en['total']) or np.isinf(en['total']):
                        stable = False
                        break

            diag.record(fdtd, confinement_radius)

            if stable:
                E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
                t_disp = diag.dispersion_timescale(0.5)
                result = ScanResult(
                    G_eff=G_eff,
                    G_ratio=G_ratio,
                    a_over_M=aM,
                    has_horizon=hz,
                    final_confinement=diag.confinements[-1],
                    dispersion_time=t_disp / T_circ if t_disp else None,
                    max_alpha_deviation=fdtd.max_alpha_deviation,
                    final_energy_ratio=diag.energies[-1] / E0,
                )
            else:
                result = ScanResult(
                    G_eff=G_eff, G_ratio=G_ratio, a_over_M=aM,
                    has_horizon=hz, final_confinement=0.0,
                    dispersion_time=0.0, max_alpha_deviation=fdtd.max_alpha_deviation,
                    final_energy_ratio=np.nan,
                )
        except Exception as exc:
            result = ScanResult(
                G_eff=G_eff, G_ratio=G_ratio, a_over_M=aM,
                has_horizon=hz, final_confinement=0.0,
                dispersion_time=0.0, max_alpha_deviation=np.nan,
                final_energy_ratio=np.nan,
            )
            if verbose:
                print(f"ERROR: {exc}")

        results.append(result)
        elapsed = time.time() - t0

        if verbose:
            print(f"conf={result.final_confinement:.3f} "
                  f"|a-1|={result.max_alpha_deviation:.2e} "
                  f"({elapsed:.1f}s)")

    total_time = time.time() - t0_total
    print(f"\nScan complete in {total_time/60:.1f} minutes")

    return results


# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------
def plot_metric(G_ratio: float, resolution: float = 0.5,
                save_path: Optional[str] = None):
    """Visualize the lapse function alpha(rho, z) for a given G_eff/G_N."""
    import matplotlib.pyplot as plt

    G_eff = G_ratio * G_N
    gp = GridParams.for_electron(resolution=resolution)
    M, a, Q, aM = electron_kn_params(G_eff)
    sigma_floor = 0.01 * a

    r = gp.r_axis
    z = gp.z_axis
    RHO, ZZ = np.meshgrid(r, z, indexing='ij')
    r_bl, cos_th = cyl_to_bl(RHO, ZZ, a)
    lapse, beta_phi = compute_kn_metric(r_bl, cos_th, M, a, Q, sigma_floor)

    r_fm = r * 1e15
    z_fm = z * 1e15
    R_fm = gp.R_torus * 1e15
    rm_fm = gp.r_minor * 1e15

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Kerr-Newman Metric: G_eff/G_N = {G_ratio:.2e}\n"
        f"a/M = {aM:.2e}, horizon = {has_horizon(M, a, Q)}, "
        f"max|alpha-1| = {np.max(np.abs(lapse-1.0)):.4e}",
        fontsize=13)

    # Panel 1: Lapse alpha
    ax = axes[0]
    im = ax.pcolormesh(z_fm, r_fm, lapse, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label=r'$\alpha$')
    ax.set_xlabel('z (fm)')
    ax.set_ylabel(r'$\rho$ (fm)')
    ax.set_title(r'Lapse $\alpha(\rho, z)$')
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(rm_fm * np.sin(theta), R_fm + rm_fm * np.cos(theta),
            'r--', linewidth=1, alpha=0.7, label='torus')
    ax.legend(fontsize=8)

    # Panel 2: alpha - 1 (deviation from flat)
    ax = axes[1]
    dev = lapse - 1.0
    vmax = max(np.max(np.abs(dev)), 1e-50)
    im = ax.pcolormesh(z_fm, r_fm, dev, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')
    plt.colorbar(im, ax=ax, label=r'$\alpha - 1$')
    ax.set_xlabel('z (fm)')
    ax.set_ylabel(r'$\rho$ (fm)')
    ax.set_title(r'$\alpha - 1$ (deviation from flat)')
    ax.plot(rm_fm * np.sin(theta), R_fm + rm_fm * np.cos(theta),
            'r--', linewidth=1, alpha=0.7)

    # Panel 3: Frame-dragging beta^phi
    ax = axes[2]
    vmax_b = max(np.max(np.abs(beta_phi)), 1e-50)
    im = ax.pcolormesh(z_fm, r_fm, beta_phi, cmap='coolwarm',
                       vmin=-vmax_b, vmax=vmax_b, shading='auto')
    plt.colorbar(im, ax=ax, label=r'$\beta^\phi$')
    ax.set_xlabel('z (fm)')
    ax.set_ylabel(r'$\rho$ (fm)')
    ax.set_title(r'Frame-dragging $\beta^\phi$')
    ax.plot(rm_fm * np.sin(theta), R_fm + rm_fm * np.cos(theta),
            'r--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    if save_path is None:
        sim_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(sim_dir,
                                 f"nwt_fdtd_kn_metric_G{G_ratio:.0e}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_scan_results(results: List[ScanResult],
                      save_path: Optional[str] = None,
                      comparison_results: Optional[List[ScanResult]] = None,
                      comparison_label: str = "Phase 2a (lapse-only)"):
    """4-panel scan summary plot with optional comparison overlay."""
    import matplotlib.pyplot as plt

    G_ratios = np.array([r.G_ratio for r in results])
    confinements = np.array([r.final_confinement for r in results])
    disp_times = np.array([r.dispersion_time if r.dispersion_time is not None
                           else np.nan for r in results])
    alpha_devs = np.array([r.max_alpha_deviation for r in results])
    a_over_Ms = np.array([r.a_over_M for r in results])

    # Comparison data (Phase 2a overlay)
    if comparison_results is not None:
        comp_G = np.array([r.G_ratio for r in comparison_results])
        comp_conf = np.array([r.final_confinement for r in comparison_results])
        comp_disp = np.array([r.dispersion_time if r.dispersion_time is not None
                              else np.nan for r in comparison_results])

    # Extremal transition line
    a = hbar / (2.0 * m_e * c)
    G_extremal_ratio = a * c**2 / m_e / G_N

    # Phase 0 baseline confinement (flat space ~ 0.1 T_circ dispersion)
    baseline_conf = confinements[0] if len(confinements) > 0 else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("G_eff Scan: EM Confinement on Kerr-Newman Background",
                 fontsize=14, fontweight='bold')

    # Panel 1: Confinement vs G_eff/G_N
    ax = axes[0, 0]
    ax.semilogx(G_ratios, confinements, 'b.-', linewidth=1.5, markersize=4,
                label='Current')
    if comparison_results is not None:
        ax.semilogx(comp_G, comp_conf, 'c--', linewidth=1, markersize=3,
                    alpha=0.7, label=comparison_label)
    ax.axhline(y=baseline_conf, color='gray', linestyle='--', alpha=0.5,
               label=f'Phase 0 baseline ({baseline_conf:.3f})')
    ax.axvline(x=G_extremal_ratio, color='red', linestyle=':', alpha=0.7,
               label=f'Extremal (a/M=1)')
    ax.set_xlabel(r'$G_{\rm eff} / G_N$')
    ax.set_ylabel('Final confinement')
    ax.set_title('Confinement vs Coupling Strength')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Dispersion time vs G_eff/G_N
    ax = axes[0, 1]
    valid = ~np.isnan(disp_times)
    if np.any(valid):
        ax.semilogx(G_ratios[valid], disp_times[valid], 'g.-',
                     linewidth=1.5, markersize=4, label='Current')
    if comparison_results is not None:
        valid_comp = ~np.isnan(comp_disp)
        if np.any(valid_comp):
            ax.semilogx(comp_G[valid_comp], comp_disp[valid_comp], 'c--',
                        linewidth=1, markersize=3, alpha=0.7,
                        label=comparison_label)
    ax.axvline(x=G_extremal_ratio, color='red', linestyle=':', alpha=0.7,
               label='Extremal')
    ax.set_xlabel(r'$G_{\rm eff} / G_N$')
    ax.set_ylabel(r'Dispersion time ($T_{\rm circ}$)')
    ax.set_title('Dispersion Timescale')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: max|alpha-1| vs G_eff/G_N
    ax = axes[1, 0]
    valid_a = ~np.isnan(alpha_devs) & (alpha_devs > 0)
    if np.any(valid_a):
        ax.loglog(G_ratios[valid_a], alpha_devs[valid_a], 'r.-',
                  linewidth=1.5, markersize=4)
    ax.axvline(x=G_extremal_ratio, color='red', linestyle=':', alpha=0.7,
               label='Extremal')
    ax.set_xlabel(r'$G_{\rm eff} / G_N$')
    ax.set_ylabel(r'max$|\alpha - 1|$')
    ax.set_title('Metric Strength')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: a/M vs G_eff/G_N
    ax = axes[1, 1]
    valid_aM = np.isfinite(a_over_Ms)
    if np.any(valid_aM):
        ax.loglog(G_ratios[valid_aM], a_over_Ms[valid_aM], 'm.-',
                  linewidth=1.5, markersize=4)
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.7,
               label='Extremal (a/M=1)')
    ax.axvline(x=G_extremal_ratio, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel(r'$G_{\rm eff} / G_N$')
    ax.set_ylabel('a/M')
    ax.set_title('Extremality Parameter')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Color horizon/no-horizon regions
    for ax in axes.flat:
        ax.axvspan(G_extremal_ratio, G_ratios[-1] * 2,
                   alpha=0.05, color='gray')

    plt.tight_layout()
    if save_path is None:
        sim_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(sim_dir, "nwt_fdtd_kn_scan.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------
def test_coordinate_roundtrip(verbose: bool = True) -> bool:
    """Test: BL -> cylindrical -> BL preserves (r, theta) to machine precision."""
    if verbose:
        print("  [test_coordinate_roundtrip] ", end="", flush=True)

    a = hbar / (2.0 * m_e * c)

    # Test points: various (rho, z) including near-axis, near-ring, far-field
    rho_vals = np.array([0.0, a * 0.1, a * 0.5, a, a * 2, a * 10])
    z_vals = np.array([0.0, a * 0.1, -a * 0.5, a, -a * 2, a * 5])
    RHO, ZZ = np.meshgrid(rho_vals, z_vals, indexing='ij')

    # Forward: cyl -> BL
    r_bl, cos_th = cyl_to_bl(RHO, ZZ, a)

    # Backward: BL -> cyl
    rho_back, z_back = bl_to_cyl(r_bl, cos_th, a)

    # Check roundtrip
    err_rho = np.max(np.abs(rho_back - np.abs(RHO)))  # rho is always >= 0
    err_z = np.max(np.abs(z_back - ZZ))

    # Normalize by a
    rel_err = max(err_rho, err_z) / a

    passed = rel_err < 1e-10

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (relative roundtrip error = {rel_err:.2e})")

    return passed


def test_flat_recovery(verbose: bool = True) -> bool:
    """Test: G_eff = G_N reproduces Phase 0 fields (|alpha-1| ~ 10^-44)."""
    if verbose:
        print("  [test_flat_recovery] ", end="", flush=True)

    gp = GridParams.for_electron(resolution=0.5)

    # Phase 0 reference
    from nwt_fdtd import AxiSymMaxwellFDTD as FlatFDTD
    flat = FlatFDTD(gp)
    init_em_wave_torus(flat, amplitude=1.0)

    # Phase 2 at G_N
    kn = KerrNewmanFDTD(gp, G_eff=G_N)
    init_em_wave_torus(kn, amplitude=1.0)

    # Check alpha is essentially 1
    alpha_dev = kn.max_alpha_deviation

    # Evolve both for 100 steps
    n_steps = 100
    for _ in range(n_steps):
        flat.step_forward()
        kn.step_forward()

    # Compare E_phi fields
    diff = np.max(np.abs(kn.E_phi - flat.E_phi))
    max_field = max(np.max(np.abs(flat.E_phi)), 1e-30)
    rel_diff = diff / max_field

    # At G_N, alpha-1 ~ 10^-44, so fields should be identical
    # to machine precision (float64 ~ 10^-16 << 10^-44 correction)
    passed = rel_diff < 1e-10 and alpha_dev < 1e-30

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (|alpha-1|={alpha_dev:.2e}, "
              f"field diff={rel_diff:.2e})")

    return passed


def test_metric_symmetry(verbose: bool = True) -> bool:
    """Test: alpha(rho, z) = alpha(rho, -z) (equatorial symmetry)."""
    if verbose:
        print("  [test_metric_symmetry] ", end="", flush=True)

    a = hbar / (2.0 * m_e * c)
    G_eff = 1e44 * G_N  # strong enough to see structure
    M, _, Q, _ = electron_kn_params(G_eff)
    sigma_floor = 0.01 * a

    rho_vals = np.linspace(0, 5 * a, 50)
    z_vals = np.linspace(0.01 * a, 3 * a, 30)  # avoid exact z=0

    max_asymmetry = 0.0
    for rho in rho_vals:
        for z in z_vals:
            r_p, ct_p = cyl_to_bl(rho, z, a)
            r_m, ct_m = cyl_to_bl(rho, -z, a)
            a_p, _ = compute_kn_metric(r_p, ct_p, M, a, Q, sigma_floor)
            a_m, _ = compute_kn_metric(r_m, ct_m, M, a, Q, sigma_floor)
            diff = abs(float(a_p) - float(a_m))
            max_asymmetry = max(max_asymmetry, diff)

    passed = max_asymmetry < 1e-12

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (max asymmetry = {max_asymmetry:.2e})")

    return passed


def test_energy_not_degraded(verbose: bool = True) -> bool:
    """Test: energy conservation not degraded compared to Phase 0.

    Runs both flat-space (Phase 0) and KN at modest G_eff for the same
    number of steps, and checks that KN energy loss is not significantly
    worse.  Both lose ~10% to ABC boundary absorption.
    """
    if verbose:
        print("  [test_energy_not_degraded] ", end="", flush=True)

    gp = GridParams.for_electron(resolution=0.5)
    n_steps = 200

    # Phase 0 baseline
    from nwt_fdtd import AxiSymMaxwellFDTD as FlatFDTD
    flat = FlatFDTD(gp)
    init_em_wave_torus(flat, amplitude=1.0)
    diag_flat = Diagnostics()
    diag_flat.record(flat)
    for _ in range(n_steps):
        flat.step_forward()
    diag_flat.record(flat)
    flat_change = diag_flat.energy_conservation()

    # KN at modest G_eff (|alpha-1| ~ 10^-4)
    G_eff = 1e40 * G_N
    kn = KerrNewmanFDTD(gp, G_eff=G_eff)
    init_em_wave_torus(kn, amplitude=1.0)
    diag_kn = Diagnostics()
    diag_kn.record(kn)
    for _ in range(n_steps):
        kn.step_forward()
    diag_kn.record(kn)
    kn_change = diag_kn.energy_conservation()

    # KN should not be significantly worse than Phase 0
    # Both have ~10% ABC loss; KN metric adds < 1% extra at this G_eff
    degradation = abs(kn_change - flat_change)
    passed = degradation < 0.02  # < 2% additional loss from metric

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (flat dE/E={flat_change:.4f}, "
              f"KN dE/E={kn_change:.4f}, "
              f"degradation={degradation:.4f}, "
              f"|alpha-1|={kn.max_alpha_deviation:.2e})")

    return passed


def test_flat_recovery_frame_dragging(verbose: bool = True) -> bool:
    """Test: G_eff=G_N with frame_dragging=True matches Phase 0.

    At G_N, beta^phi ~ 0 and alpha ~ 1, so frame-dragging terms vanish.
    """
    if verbose:
        print("  [test_flat_recovery_frame_dragging] ", end="", flush=True)

    gp = GridParams.for_electron(resolution=0.5)

    # Phase 0 reference
    from nwt_fdtd import AxiSymMaxwellFDTD as FlatFDTD
    flat = FlatFDTD(gp)
    init_em_wave_torus(flat, amplitude=1.0)

    # Phase 2b at G_N with frame-dragging enabled
    kn_fd = KerrNewmanFDTD(gp, G_eff=G_N, frame_dragging=True)
    init_em_wave_torus(kn_fd, amplitude=1.0)

    n_steps = 100
    for _ in range(n_steps):
        flat.step_forward()
        kn_fd.step_forward()

    # Compare E_phi fields
    diff = np.max(np.abs(kn_fd.E_phi - flat.E_phi))
    max_field = max(np.max(np.abs(flat.E_phi)), 1e-30)
    rel_diff = diff / max_field

    # Also check that TM fields didn't appear (beta^phi ~ 0 at G_N)
    tm_amplitude = max(np.max(np.abs(kn_fd.B_phi)),
                       np.max(np.abs(kn_fd.E_r)),
                       np.max(np.abs(kn_fd.E_z)))

    passed = rel_diff < 1e-10 and tm_amplitude < 1e-30

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (field diff={rel_diff:.2e}, "
              f"TM amplitude={tm_amplitude:.2e})")

    return passed


def test_shift_symmetry(verbose: bool = True) -> bool:
    """Test: beta^phi(rho,z) = beta^phi(rho,-z) and gradients antisymmetric."""
    if verbose:
        print("  [test_shift_symmetry] ", end="", flush=True)

    a = hbar / (2.0 * m_e * c)
    G_eff = 1e44 * G_N
    M, _, Q, _ = electron_kn_params(G_eff)
    sigma_floor = 0.01 * a

    rho_vals = np.linspace(0.1 * a, 5 * a, 30)
    z_vals = np.linspace(0.01 * a, 3 * a, 20)

    max_beta_asymmetry = 0.0
    for rho in rho_vals:
        for z in z_vals:
            r_p, ct_p = cyl_to_bl(rho, z, a)
            r_m, ct_m = cyl_to_bl(rho, -z, a)
            _, bp_p = compute_kn_metric(r_p, ct_p, M, a, Q, sigma_floor)
            _, bp_m = compute_kn_metric(r_m, ct_m, M, a, Q, sigma_floor)
            diff = abs(float(bp_p) - float(bp_m))
            max_beta_asymmetry = max(max_beta_asymmetry, diff)

    # Also verify dbeta_dz is antisymmetric in z via the precomputed arrays
    gp = GridParams.for_electron(resolution=0.5)
    fdtd = KerrNewmanFDTD(gp, G_eff=G_eff, frame_dragging=True)

    # beta_phi_center should be symmetric: beta(i, j) == beta(i, Nz-1-j)
    bp = fdtd.beta_phi_center
    bp_mirror = bp[:, ::-1]
    bp_max = max(float(np.max(np.abs(bp))), 1e-50)
    # Use relative asymmetry: near the ring singularity, sigma_floor
    # regularization creates O(dr/sigma_floor) relative differences
    beta_grid_asymmetry = float(np.max(np.abs(bp - bp_mirror))) / bp_max

    passed = max_beta_asymmetry < 1e-12 and beta_grid_asymmetry < 1e-6

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (beta asymmetry={max_beta_asymmetry:.2e}, "
              f"grid rel asymmetry={beta_grid_asymmetry:.2e})")

    return passed


def test_energy_stability_frame_dragging(verbose: bool = True) -> bool:
    """Test: frame-dragging doesn't cause numerical blowup.

    Uses G_eff=1e37 G_N where the shift coupling is naturally within the
    explicit scheme's stability range (dt * max|dbeta/dr| ~ 0.008).
    Frame-dragging physically transfers coordinate energy between TE/TM
    modes, so we only check for NaN/Inf and gross energy blowup, not
    strict conservation.
    """
    if verbose:
        print("  [test_energy_stability_frame_dragging] ", end="", flush=True)

    gp = GridParams.for_electron(resolution=0.5)
    n_steps = 200

    # Use G_eff where shift coupling is naturally within stable range
    G_eff = 1e37 * G_N
    kn_fd = KerrNewmanFDTD(gp, G_eff=G_eff, frame_dragging=True)
    init_em_wave_torus(kn_fd, amplitude=1.0)

    diag = Diagnostics()
    diag.record(kn_fd)

    stable = True
    for step_i in range(n_steps):
        kn_fd.step_forward()
        if step_i % 50 == 0:
            en = kn_fd.compute_energy()
            if np.isnan(en['total']) or np.isinf(en['total']):
                stable = False
                break
    diag.record(kn_fd)

    energy_change = diag.energy_conservation()
    # At 1e37, coupling is weak (dt*grad ~ 0.008); coordinate energy
    # change should be comparable to lapse-only (~10% ABC loss)
    passed = stable and energy_change < 0.20

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (stable={stable}, dE/E={energy_change:.4f})")

    return passed


def test_mode_coupling(verbose: bool = True) -> bool:
    """Test: pure TE init generates TM fields when frame_dragging=True at strong G_eff."""
    if verbose:
        print("  [test_mode_coupling] ", end="", flush=True)

    gp = GridParams.for_electron(resolution=0.5)

    # Strong G_eff where beta^phi gradients are significant
    G_eff = 1e43 * G_N
    kn_fd = KerrNewmanFDTD(gp, G_eff=G_eff, frame_dragging=True)

    # Pure TE initialization: only E_phi, B_r, B_z
    init_em_wave_torus(kn_fd, amplitude=1.0)

    # Verify TM fields start at zero
    assert np.max(np.abs(kn_fd.B_phi)) == 0.0
    assert np.max(np.abs(kn_fd.E_r)) == 0.0
    assert np.max(np.abs(kn_fd.E_z)) == 0.0

    # Evolve
    n_steps = 50
    for _ in range(n_steps):
        kn_fd.step_forward()

    # Check TM fields appeared
    tm_max = max(np.max(np.abs(kn_fd.B_phi)),
                 np.max(np.abs(kn_fd.E_r)),
                 np.max(np.abs(kn_fd.E_z)))

    # Also run without frame-dragging as control
    kn_no_fd = KerrNewmanFDTD(gp, G_eff=G_eff, frame_dragging=False)
    init_em_wave_torus(kn_no_fd, amplitude=1.0)
    for _ in range(n_steps):
        kn_no_fd.step_forward()
    tm_max_control = max(np.max(np.abs(kn_no_fd.B_phi)),
                         np.max(np.abs(kn_no_fd.E_r)),
                         np.max(np.abs(kn_no_fd.E_z)))

    # Frame-dragging should generate much more TM than control
    # Control may have tiny TM from Faraday lapse asymmetry, but frame-dragging
    # should produce orders of magnitude more
    passed = tm_max > 1e-20 and (tm_max_control < 1e-30 or tm_max > 100 * tm_max_control)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status} (TM with FD={tm_max:.2e}, "
              f"TM without={tm_max_control:.2e})")

    return passed


def run_validation(verbose: bool = True) -> bool:
    """Run all Phase 2 validation tests."""
    print("=" * 60)
    print("Phase 2 (Kerr-Newman) Validation Suite")
    print("=" * 60)

    results = []
    # Phase 2a tests
    results.append(("Coordinate roundtrip", test_coordinate_roundtrip(verbose)))
    results.append(("Flat recovery (G_N)", test_flat_recovery(verbose)))
    results.append(("Metric symmetry", test_metric_symmetry(verbose)))
    results.append(("Energy conservation", test_energy_not_degraded(verbose)))

    # Phase 2b tests
    print("  --- Phase 2b (frame-dragging) ---")
    results.append(("Flat recovery (FD)", test_flat_recovery_frame_dragging(verbose)))
    results.append(("Shift symmetry", test_shift_symmetry(verbose)))
    results.append(("Energy stability (FD)", test_energy_stability_frame_dragging(verbose)))
    results.append(("Mode coupling", test_mode_coupling(verbose)))

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
# Dynamic G_eff runner
# ---------------------------------------------------------------------------
def run_dynamic(resolution: float = 0.5, n_circulations: int = 5,
                u_crit: float = 1.0, exponent: float = 2.0,
                verbose: bool = True):
    """Run dynamic G_eff simulation ("violent creation" model)."""
    gp = GridParams.for_electron(resolution=resolution)

    print("=" * 60)
    print("Dynamic G_eff Simulation")
    print("=" * 60)
    print(f"  u_crit   = {u_crit}")
    print(f"  exponent = {exponent}")
    print(f"  Grid     = {gp.Nr} x {gp.Nz}")

    fdtd = DynamicGeffFDTD(gp, u_crit=u_crit, exponent=exponent)
    init_em_wave_torus(fdtd, amplitude=1.0)

    confinement_radius = 3 * gp.r_minor
    diag = Diagnostics()
    diag.record(fdtd, confinement_radius)

    total_steps = n_circulations * gp.steps_per_circ
    record_interval = max(1, gp.steps_per_circ // 10)

    t0 = time.time()
    for step_i in range(total_steps):
        fdtd.step_forward()

        if step_i % record_interval == 0:
            diag.record(fdtd, confinement_radius)

        if (verbose and step_i > 0 and
                step_i % (gp.steps_per_circ * max(1, n_circulations // 5)) == 0):
            circ = step_i / gp.steps_per_circ
            en = fdtd.compute_energy()
            conf = fdtd.compute_confinement(gp.R_torus, 0.0, confinement_radius)
            E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
            print(f"  t={circ:.1f} T_circ | E/E0={en['total']/E0:.4f} | "
                  f"conf={conf:.3f} | G/G_N={fdtd.G_eff/G_N:.2e}")

    diag.record(fdtd, confinement_radius)
    elapsed = time.time() - t0

    if verbose:
        E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Final confinement: {diag.confinements[-1]:.4f}")
        print(f"  Energy change: {diag.energy_conservation()*100:.2f}%")
        print(f"  Final G_eff/G_N: {fdtd.G_eff/G_N:.2e}")

    # Plot G_eff evolution
    if fdtd.geff_history:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not available, skipping plots)")
            return fdtd, diag
        sim_dir = os.path.dirname(os.path.abspath(__file__))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Dynamic G_eff Simulation", fontsize=13)

        times_g = [t / gp.T_circ for t, _ in fdtd.geff_history]
        g_vals = [g / G_N for _, g in fdtd.geff_history]

        ax = axes[0]
        ax.semilogy(times_g, g_vals, 'b-')
        ax.set_xlabel(r'Time ($T_{\rm circ}$)')
        ax.set_ylabel(r'$G_{\rm eff} / G_N$')
        ax.set_title('Effective Coupling')
        ax.grid(True, alpha=0.3)

        T_circ = gp.T_circ
        t_units = np.array(diag.times) / T_circ

        ax = axes[1]
        E0 = diag.energies[0] if diag.energies[0] > 0 else 1.0
        ax.plot(t_units, np.array(diag.energies) / E0, 'b-')
        ax.set_xlabel(r'Time ($T_{\rm circ}$)')
        ax.set_ylabel(r'Energy / $E_0$')
        ax.set_title('Energy Evolution')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(t_units, diag.confinements, 'k-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'Time ($T_{\rm circ}$)')
        ax.set_ylabel('Confinement')
        ax.set_title('Energy Confinement')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(sim_dir, "nwt_fdtd_kn_dynamic.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()

    return fdtd, diag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NWT FDTD Phase 2: Maxwell on Kerr-Newman background")

    parser.add_argument('--test', action='store_true',
                        help='Run validation test suite')
    parser.add_argument('--single', action='store_true',
                        help='Run single G_eff simulation')
    parser.add_argument('--scan', action='store_true',
                        help='Run G_eff scan (main scientific output)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Run dynamic G_eff simulation')
    parser.add_argument('--metric-viz', action='store_true',
                        help='Plot metric (lapse) for given G_eff')

    parser.add_argument('--g-ratio', type=float, default=1e44,
                        help='G_eff/G_N ratio (default: 1e44)')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='Resolution multiplier (default: 0.5)')
    parser.add_argument('--n-circulations', type=int, default=5,
                        help='Number of circulation periods (default: 5)')
    parser.add_argument('--n-points', type=int, default=50,
                        help='Number of scan points (default: 50)')
    parser.add_argument('--frame-dragging', action='store_true',
                        help='Enable frame-dragging (Phase 2b)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    if not any([args.test, args.single, args.scan, args.dynamic,
                args.metric_viz]):
        parser.print_help()
        print("\nQuick start:")
        print("  python nwt_fdtd_kn.py --test")
        print("  python nwt_fdtd_kn.py --single --g-ratio 1")
        print("  python nwt_fdtd_kn.py --single --g-ratio 3e44")
        print("  python nwt_fdtd_kn.py --single --g-ratio 1.5e43 --frame-dragging")
        print("  python nwt_fdtd_kn.py --scan --n-points 20 --frame-dragging")
        print("  python nwt_fdtd_kn.py --metric-viz --g-ratio 3e44")
        print("  python nwt_fdtd_kn.py --dynamic")
        return

    if args.test:
        success = run_validation(verbose=not args.quiet)
        if not success:
            sys.exit(1)

    if args.single:
        sim_dir = os.path.dirname(os.path.abspath(__file__))
        fdtd, diag = run_single(
            G_ratio=args.g_ratio,
            resolution=args.resolution,
            n_circulations=args.n_circulations,
            verbose=not args.quiet,
            frame_dragging=args.frame_dragging,
        )
        try:
            plot_fields(fdtd,
                        title=f"KN fields: G/G_N={args.g_ratio:.0e}, "
                              f"t={args.n_circulations} T_circ",
                        save_path=os.path.join(
                            sim_dir,
                            f"nwt_fdtd_kn_single_G{args.g_ratio:.0e}.png"))
            plot_diagnostics(diag, fdtd.gp,
                             save_path=os.path.join(
                                 sim_dir,
                                 f"nwt_fdtd_kn_diag_G{args.g_ratio:.0e}.png"))
        except ImportError:
            print("  (matplotlib not available, skipping plots)")

    if args.scan:
        comparison = None
        if args.frame_dragging:
            # Run Phase 2a (lapse-only) comparison first
            print("\n--- Phase 2a (lapse-only) comparison scan ---")
            comparison = run_geff_scan(
                n_points=args.n_points,
                resolution=args.resolution,
                n_circulations=args.n_circulations,
                verbose=not args.quiet,
                frame_dragging=False,
            )
            print("\n--- Phase 2b (lapse+shift) main scan ---")

        results = run_geff_scan(
            n_points=args.n_points,
            resolution=args.resolution,
            n_circulations=args.n_circulations,
            verbose=not args.quiet,
            frame_dragging=args.frame_dragging,
        )
        try:
            sim_dir = os.path.dirname(os.path.abspath(__file__))
            suffix = "_fd" if args.frame_dragging else ""
            plot_scan_results(
                results,
                save_path=os.path.join(sim_dir, f"nwt_fdtd_kn_scan{suffix}.png"),
                comparison_results=comparison,
            )
        except ImportError:
            print("  (matplotlib not available, skipping plots)")

    if args.dynamic:
        run_dynamic(
            resolution=args.resolution,
            n_circulations=args.n_circulations,
            verbose=not args.quiet,
        )

    if args.metric_viz:
        plot_metric(G_ratio=args.g_ratio, resolution=args.resolution)


if __name__ == "__main__":
    main()
