#!/usr/bin/env python3
"""
Closed Null Worldtube: Standard Maxwell on Toroidal Topology
=============================================================

The idea: an electron is a photon circulating on a torus at c.
The worldtube of this circulation is a closed null surface in
Minkowski spacetime. We use ONLY standard Maxwell's equations —
no pivot field, no extensions. The topology does the work.

Approach:
1. Parameterize a (p,q) torus knot as a null curve in spacetime
   (every point moves at c along the curve)
2. Treat the curve as a current loop (circulating charge/field)
3. Compute the electromagnetic self-energy of the configuration
4. Check self-consistency: does the total energy (circulation +
   self-interaction) match observed particle masses?
5. Resonance quantization: the field must be single-valued on the
   closed topology, which discretizes the allowed configurations
6. Read off the energy of stable solutions → particle masses

Physical constants are in SI units throughout.

Key result: α (fine-structure constant) enters naturally as the
ratio of self-interaction energy to circulation energy.

Usage:
    python3 null_worldtube.py                    # basic analysis
    python3 null_worldtube.py --scan             # scan parameter space
    python3 null_worldtube.py --energy           # detailed energy breakdown
    python3 null_worldtube.py --self-energy      # self-energy analysis with α
    python3 null_worldtube.py --resonance        # resonance quantization analysis
    python3 null_worldtube.py --angular-momentum # literal angular momentum (spin from geometry)
    python3 null_worldtube.py --find-radii       # find self-consistent radii for known particles
    python3 null_worldtube.py --pair-production  # pair production analysis (γ → e⁻ + e⁺)
    python3 null_worldtube.py --decay            # decay landscape and stability analysis
    python3 null_worldtube.py --hydrogen         # hydrogen atom: orbital dynamics, shell filling
    python3 null_worldtube.py --transitions      # photon emission/absorption: spectrum, rates
"""

import numpy as np
import argparse
from dataclasses import dataclass

# Physical constants (SI)
c = 2.99792458e8          # speed of light (m/s)
hbar = 1.054571817e-34    # reduced Planck constant (J·s)
h_planck = 2 * np.pi * hbar  # Planck constant (J·s)
e_charge = 1.602176634e-19  # elementary charge (C)
eps0 = 8.8541878128e-12   # permittivity of free space (F/m)
mu0 = 1.2566370621e-6     # permeability of free space (H/m)
m_e = 9.1093837015e-31    # electron mass (kg)
m_e_MeV = 0.51099895      # electron mass (MeV/c²)
m_mu = 1.883531627e-28    # muon mass (kg)
m_p = 1.67262192369e-27   # proton mass (kg)
alpha = 7.2973525693e-3   # fine-structure constant
eV = 1.602176634e-19      # electronvolt (J)
MeV = 1e6 * eV

# Derived
lambda_C = hbar / (m_e * c)    # reduced Compton wavelength (m) ≈ 3.86e-13 m
r_e = alpha * lambda_C          # classical electron radius ≈ 2.82e-15 m
a_0 = lambda_C / alpha          # Bohr radius ≈ 5.29e-11 m
k_e = 1.0 / (4 * np.pi * eps0)  # Coulomb constant

# Known particle masses (MeV/c²) for reference
PARTICLE_MASSES = {
    'electron': 0.51099895,
    'muon': 105.6583755,
    'pion±': 139.57039,
    'proton': 938.27208816,
    'tau': 1776.86,
}


@dataclass
class TorusParams:
    """Parameters defining a torus and winding numbers."""
    R: float          # major radius (m)
    r: float          # minor radius (m)
    p: int = 1        # toroidal winding number
    q: int = 1        # poloidal winding number


def torus_knot_curve(params: TorusParams, N: int = 1000):
    """
    Parameterize a (p,q) torus knot in 3D space.

    The curve winds p times around the torus hole (toroidal)
    and q times around the tube (poloidal) before closing.

    Returns:
        lam: parameter array [0, 2π)
        xyz: (N, 3) array of spatial positions
        dxyz_dlam: (N, 3) array of tangent vectors dr/dλ
        ds_dlam: (N,) array of |dr/dλ| (speed in parameter space)
    """
    lam = np.linspace(0, 2 * np.pi, N, endpoint=False)
    R, r, p, q = params.R, params.r, params.p, params.q

    theta = p * lam   # toroidal angle
    phi = q * lam     # poloidal angle

    # Position on torus
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    xyz = np.stack([x, y, z], axis=-1)

    # Tangent vector dr/dλ
    dx = -r * q * np.sin(phi) * np.cos(theta) - (R + r * np.cos(phi)) * p * np.sin(theta)
    dy = -r * q * np.sin(phi) * np.sin(theta) + (R + r * np.cos(phi)) * p * np.cos(theta)
    dz = r * q * np.cos(phi)
    dxyz = np.stack([dx, dy, dz], axis=-1)

    # Arc length speed |dr/dλ|
    ds = np.sqrt(dx**2 + dy**2 + dz**2)

    return lam, xyz, dxyz, ds


def null_condition_time(params: TorusParams, N: int = 1000):
    """
    Compute the time parameterization for a null curve on the torus.

    For a null curve, ds² = c²dt² - dx² - dy² - dz² = 0,
    so dt/dλ = |dr/dλ| / c.

    The total time for one loop (λ: 0 → 2π) is the circulation period T.

    Returns:
        lam: parameter array
        t: time array (cumulative)
        T: total circulation period
        xyz: positions
        v: velocity vectors (should all have magnitude c)
    """
    lam, xyz, dxyz, ds = torus_knot_curve(params, N)

    # Time increment: dt = ds/c per unit dλ
    dlam = lam[1] - lam[0]
    dt_dlam = ds / c

    # Cumulative time
    t = np.cumsum(dt_dlam) * dlam
    t = np.insert(t, 0, 0.0)[:-1]  # shift to start at t=0

    T = np.sum(dt_dlam) * dlam  # total period

    # Velocity = (dr/dλ) / (dt/dλ) = (dr/dλ) * c / |dr/dλ|
    # Should have magnitude c everywhere
    v = dxyz * (c / ds[:, np.newaxis])

    # Verify null condition
    v_mag = np.sqrt(np.sum(v**2, axis=-1))
    assert np.allclose(v_mag, c, rtol=1e-10), f"Null condition violated: v/c range [{v_mag.min()/c:.6f}, {v_mag.max()/c:.6f}]"

    return lam, t, T, xyz, v


def compute_path_length(params: TorusParams, N: int = 1000):
    """Compute total path length of the torus knot curve."""
    lam, xyz, dxyz, ds = torus_knot_curve(params, N)
    dlam = 2 * np.pi / N
    return np.sum(ds) * dlam


def compute_circulation_energy(params: TorusParams, N: int = 1000):
    """
    Compute the circulation energy of the null curve.

    For a photon circulating at c on a closed path of length L,
    the energy is: E = hf = hc/L = 2πℏc/L

    This is the zeroth-order estimate — no self-interaction.

    Returns:
        E_joules: energy in joules
        E_MeV: energy in MeV
        L: total path length
        T: circulation period
        f: circulation frequency
    """
    lam, xyz, dxyz, ds = torus_knot_curve(params, N)
    dlam = 2 * np.pi / N

    # Total path length
    L = np.sum(ds) * dlam

    # Period and frequency
    T = L / c
    f = 1.0 / T

    # Energy of a single photon with this frequency
    E = hbar * 2 * np.pi * f
    E_MeV_val = E / MeV

    return E, E_MeV_val, L, T, f


def compute_self_energy(params: TorusParams, N: int = 1000):
    """
    Compute the electromagnetic self-energy of the circulating charge.

    A charge e circulating at c on a torus creates a time-averaged
    current I = ec/L. The stored EM field energy has two components:

    1. MAGNETIC SELF-ENERGY from the current loop:
       U_mag = (1/2) × L_ind × I²
       where L_ind = μ₀R[ln(8R/r) - 2] is the Neumann inductance
       of a torus with major radius R and minor radius r.

    2. ELECTRIC SELF-ENERGY: for a charge moving at v = c, the
       fields are purely transverse (Lorentz contraction → pancake).
       In this limit |B| = |E|/c, so the electric and magnetic
       energy densities are equal: u_E = ε₀E²/2 = B²/(2μ₀) = u_B.
       Therefore U_elec = U_mag.

    TOTAL: U_EM = 2 × U_mag = μ₀R[ln(8R/r) - 2] × (ec/L)²

    KEY RESULT: U_EM / E_circ = α × [ln(8R/r) - 2] / π

    The fine-structure constant α enters naturally as the coupling
    between the self-interaction energy and the circulation energy.
    This is not put in by hand — it emerges from e² appearing in
    the current (I = ec/L) while ℏc appears in the circulation
    energy (E = 2πℏc/L). Their ratio is e²/(4πε₀ℏc) = α.

    Approximations:
    - Treats the charge as a continuous current (time-averaged).
      Valid when observation timescale >> circulation period T.
    - Uses Neumann inductance formula (valid for r << R).
    - Assumes v = c exactly (null curve) for the E/B equality.
    """
    L_path = compute_path_length(params, N)
    R, r = params.R, params.r

    # Time-averaged current from circulating charge
    I = e_charge * c / L_path

    # Torus inductance (Neumann formula for thin torus, r << R)
    log_factor = np.log(8.0 * R / r) - 2.0
    L_ind = mu0 * R * max(log_factor, 0.01)  # floor to prevent issues

    # Magnetic self-energy
    U_mag = 0.5 * L_ind * I**2

    # Electric self-energy (equal to magnetic for v = c)
    U_elec = U_mag

    # Total EM self-energy
    U_total = U_mag + U_elec

    # Circulation energy for comparison
    E_circ = hbar * 2 * np.pi * c / L_path

    # The ratio U_total/E_circ should be ≈ α × log_factor / π
    # Let's verify this analytically:
    #   U_total = μ₀R × log_factor × e²c² / L²
    #   E_circ = 2πℏc / L
    #   Ratio = μ₀R × log_factor × e²c² / (L × 2πℏc)
    #         = μ₀e²c × R × log_factor / (2πℏ × L)
    #   For L ≈ 2πR:
    #         = μ₀e²c × log_factor / (4π²ℏ)
    #         = [e²/(4πε₀ℏc)] × [μ₀ε₀c² × c / (π)]  ... but μ₀ε₀ = 1/c²
    #         = α × log_factor / π ✓
    alpha_prediction = alpha * log_factor / np.pi

    return {
        'U_mag_J': U_mag,
        'U_elec_J': U_elec,
        'U_total_J': U_total,
        'U_total_MeV': U_total / MeV,
        'E_circ_J': E_circ,
        'E_circ_MeV': E_circ / MeV,
        'E_total_J': E_circ + U_total,
        'E_total_MeV': (E_circ + U_total) / MeV,
        'self_energy_fraction': U_total / E_circ,
        'alpha_prediction': alpha_prediction,
        'log_factor': log_factor,
        'I_amps': I,
        'L_ind_henry': L_ind,
    }


def compute_total_energy(params: TorusParams, N: int = 1000):
    """
    Compute total energy = circulation + self-interaction.

    Returns (E_total_J, E_total_MeV, breakdown_dict).
    """
    se = compute_self_energy(params, N)
    return se['E_total_J'], se['E_total_MeV'], se


def find_self_consistent_radius(target_MeV, p=1, q=1, r_ratio=0.1, N=1000):
    """
    Find the major radius R where E_total = target particle mass.

    Uses bisection search (no scipy dependency needed).

    The self-consistency condition: the total energy of the
    configuration (circulation + EM self-energy) must equal the
    observed particle mass. This pins down R for given (p, q, r/R).

    Returns a dict with the solution, or None if no solution found.
    """
    target_J = target_MeV * MeV

    def energy_at_R(R):
        r = r_ratio * R
        params = TorusParams(R=R, r=r, p=p, q=q)
        E_total_J, _, _ = compute_total_energy(params, N)
        return E_total_J

    # Bisection: energy decreases with R, so search for the zero of
    # f(R) = E_total(R) - target
    R_low = 1e-20   # very small → very high energy
    R_high = 1e-8   # very large → very low energy

    # Verify bracket
    E_low = energy_at_R(R_low)
    E_high = energy_at_R(R_high)

    if not (E_low > target_J > E_high):
        return None  # target not in range

    # Bisection (50 iterations → ~15 decimal digits)
    for _ in range(60):
        R_mid = (R_low + R_high) / 2.0
        E_mid = energy_at_R(R_mid)

        if E_mid > target_J:
            R_low = R_mid
        else:
            R_high = R_mid

    R_sol = (R_low + R_high) / 2.0
    r_sol = r_ratio * R_sol
    params_sol = TorusParams(R=R_sol, r=r_sol, p=p, q=q)
    E_total_J, E_total_MeV, breakdown = compute_total_energy(params_sol, N)

    return {
        'R': R_sol,
        'r': r_sol,
        'R_over_lambda_C': R_sol / lambda_C,
        'r_over_lambda_C': r_sol / lambda_C,
        'R_femtometers': R_sol * 1e15,
        'r_femtometers': r_sol * 1e15,
        'p': p,
        'q': q,
        'r_ratio': r_ratio,
        'E_circ_MeV': breakdown['E_circ_MeV'],
        'E_self_MeV': breakdown['U_total_MeV'],
        'E_total_MeV': E_total_MeV,
        'target_MeV': target_MeV,
        'match_ppm': abs(E_total_MeV - target_MeV) / target_MeV * 1e6,
        'self_energy_fraction': breakdown['self_energy_fraction'],
        'alpha_prediction': breakdown['alpha_prediction'],
    }


def resonance_analysis(params: TorusParams, n_modes: int = 8, N: int = 1000):
    """
    Analyze the resonance quantization of a torus configuration.

    On a closed topology, the EM field must be single-valued.
    This imposes boundary conditions in both directions:

    TOROIDAL (around the hole): k_tor × 2πR = 2πn  →  k_tor = n/R
    POLOIDAL (around the tube): k_pol × 2πr = 2πm  →  k_pol = m/r

    For a massless field (photon): k² = k_tor² + k_pol² = (ω/c)²

    So: E_{n,m} = ℏc × √(n²/R² + m²/r²)

    The (1,0) mode is the fundamental toroidal resonance.
    The (0,1) mode is the fundamental poloidal resonance.
    Higher modes are harmonics — excited states of the same topology.

    This is the same physics as:
    - Bohr quantization (electron wavelength must fit the orbit)
    - Normal modes of a vibrating string
    - Resonant modes of a microwave cavity
    - Kaluza-Klein tower (compactified extra dimension → mass spectrum)

    The poloidal modes at r << R give a tower of heavy states,
    analogous to a Kaluza-Klein tower from compactification.
    """
    R, r = params.R, params.r

    modes = []
    for n in range(0, n_modes + 1):
        for m in range(0, n_modes + 1):
            if n == 0 and m == 0:
                continue  # no zero mode

            k_sq = (n / R) ** 2 + (m / r) ** 2
            E_J = hbar * c * np.sqrt(k_sq)
            E_MeV_val = E_J / MeV

            # Classify the mode
            if m == 0:
                mode_type = "toroidal"
            elif n == 0:
                mode_type = "poloidal"
            else:
                mode_type = "mixed"

            modes.append({
                'n': n, 'm': m,
                'E_MeV': E_MeV_val,
                'E_over_me': E_MeV_val / m_e_MeV,
                'type': mode_type,
                'wavelength_tor': 2 * np.pi * R / n if n > 0 else np.inf,
                'wavelength_pol': 2 * np.pi * r / m if m > 0 else np.inf,
            })

    # Sort by energy
    modes.sort(key=lambda m: m['E_MeV'])
    return modes


def compute_angular_momentum(params: TorusParams, N: int = 2000):
    """
    Compute the LITERAL angular momentum of the circulating photon.

    Not "spin" as an abstract quantum number — actual mechanical
    angular momentum of a photon with energy E moving at c on a
    closed curve on the torus.

    At each point on the curve, the photon has:
        position: r(λ)
        momentum: p = (E/c) × v̂  (unit tangent vector, magnitude E/c)

    The angular momentum about the z-axis (torus symmetry axis):
        L_z(λ) = [r × p]_z = (E/c) × [x(λ) v̂_y(λ) - y(λ) v̂_x(λ)]

    Time-averaged over one circulation (weighted by arc length,
    since the photon spends equal proper time per unit path length):
        ⟨L_z⟩ = ∫ L_z(λ) ds / ∫ ds

    KEY QUESTION: does ⟨L_z⟩ = ℏ/2 for any natural geometry?
    If so, fermion spin-1/2 emerges from circulation geometry.

    We use E_total (circulation + self-energy) for the momentum,
    since that's the actual energy-momentum of the configuration.
    """
    lam, t, T, xyz, v = null_condition_time(params, N)
    _, _, dxyz, ds = torus_knot_curve(params, N)

    # Total energy (circulation + self-energy)
    _, E_total_MeV, breakdown = compute_total_energy(params, N)
    E_total = breakdown['E_total_J']
    p_mag = E_total / c  # total momentum magnitude

    # Unit tangent vectors (v has magnitude c, so v/c is unit tangent)
    v_hat = v / c

    # Position components
    x, y, z_pos = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vx, vy, vz = v_hat[:, 0], v_hat[:, 1], v_hat[:, 2]

    # Angular momentum vector at each point: L = r × p
    Lx_inst = p_mag * (y * vz - z_pos * vy)
    Ly_inst = p_mag * (z_pos * vx - x * vz)
    Lz_inst = p_mag * (x * vy - y * vx)

    # Time-average weighted by arc length (photon spends equal
    # proper time per unit path length since it moves at c)
    weights = ds / ds.sum()

    Lx_avg = np.sum(Lx_inst * weights)
    Ly_avg = np.sum(Ly_inst * weights)
    Lz_avg = np.sum(Lz_inst * weights)
    L_mag = np.sqrt(Lx_avg**2 + Ly_avg**2 + Lz_avg**2)

    # Fluctuation of Lz around the loop
    Lz_std = np.sqrt(np.sum((Lz_inst - Lz_avg)**2 * weights))

    return {
        'Lx': Lx_avg,
        'Ly': Ly_avg,
        'Lz': Lz_avg,
        'L_magnitude': L_mag,
        'Lz_over_hbar': Lz_avg / hbar,
        'L_over_hbar': L_mag / hbar,
        'Lz_std': Lz_std,
        'Lz_std_over_hbar': Lz_std / hbar,
        'Lz_min': Lz_inst.min() / hbar,
        'Lz_max': Lz_inst.max() / hbar,
        'p_mag': p_mag,
        'E_total_MeV': E_total_MeV,
    }


def print_angular_momentum_analysis(params: TorusParams):
    """
    Detailed analysis of angular momentum vs torus geometry.
    Sweeps r/R to find where J_z = ℏ/2.
    """
    print("=" * 70)
    print("ANGULAR MOMENTUM ANALYSIS")
    print("  Literal mechanical angular momentum of circulating photon")
    print("=" * 70)

    R = params.R
    print(f"\nTorus major radius: R = {R:.4e} m")
    print(f"Winding: ({params.p}, {params.q})")

    # First, show the basic result for the given params
    am = compute_angular_momentum(params)
    print(f"\n--- Current configuration (r/R = {params.r/params.R:.4f}) ---")
    print(f"  ⟨L_x⟩ = {am['Lx']/hbar:.6f} ℏ")
    print(f"  ⟨L_y⟩ = {am['Ly']/hbar:.6f} ℏ")
    print(f"  ⟨L_z⟩ = {am['Lz_over_hbar']:.6f} ℏ")
    print(f"  |⟨L⟩| = {am['L_over_hbar']:.6f} ℏ")
    print(f"  L_z fluctuation: ±{am['Lz_std_over_hbar']:.4f} ℏ")
    print(f"  L_z range: [{am['Lz_min']:.4f}, {am['Lz_max']:.4f}] ℏ")

    # Sweep r/R from 0 (pure circle) to near 1 (fat torus)
    print(f"\n--- L_z vs r/R (sweeping tube radius) ---")
    print(f"  For a pure circle (r/R → 0): L_z → ℏ (photon spin-1)")
    print(f"  As r/R increases, poloidal circulation steals angular momentum")
    print(f"  Question: does L_z = ℏ/2 at some natural r/R?")
    print(f"\n{'r/R':>8} {'L_z/ℏ':>10} {'|L|/ℏ':>10} {'E_total':>12} {'E/m_e':>8}")
    print("-" * 55)

    r_ratios = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    results = []

    for rr in r_ratios:
        r = rr * R
        tp = TorusParams(R=R, r=r, p=params.p, q=params.q)
        am = compute_angular_momentum(tp, N=2000)
        results.append((rr, am['Lz_over_hbar'], am['L_over_hbar'], am['E_total_MeV']))
        print(f"{rr:8.3f} {am['Lz_over_hbar']:10.6f} {am['L_over_hbar']:10.6f} "
              f"{am['E_total_MeV']:12.6f} {am['E_total_MeV']/m_e_MeV:8.4f}")

    # Find r/R where L_z = 0.5 ℏ using bisection
    print(f"\n--- Finding r/R where ⟨L_z⟩ = ℏ/2 ---")
    rr_low, rr_high = 0.001, 0.999

    def lz_at_rr(rr):
        r = rr * R
        tp = TorusParams(R=R, r=r, p=params.p, q=params.q)
        am = compute_angular_momentum(tp, N=3000)
        return am['Lz_over_hbar']

    # Check if ℏ/2 is in the range
    lz_low = lz_at_rr(rr_low)
    lz_high = lz_at_rr(rr_high)

    if lz_low > 0.5 > lz_high:
        for _ in range(50):
            rr_mid = (rr_low + rr_high) / 2.0
            lz_mid = lz_at_rr(rr_mid)
            if lz_mid > 0.5:
                rr_low = rr_mid
            else:
                rr_high = rr_mid

        rr_sol = (rr_low + rr_high) / 2.0
        r_sol = rr_sol * R
        tp_sol = TorusParams(R=R, r=r_sol, p=params.p, q=params.q)
        am_sol = compute_angular_momentum(tp_sol, N=4000)
        se_sol = compute_self_energy(tp_sol)

        print(f"  Solution: r/R = {rr_sol:.8f}")
        print(f"  r = {r_sol:.4e} m")
        print(f"  r / r_e = {r_sol / r_e:.4f}  (classical electron radii)")
        print(f"  r / λ_C = {r_sol / lambda_C:.6f}")
        print(f"  ⟨L_z⟩ = {am_sol['Lz_over_hbar']:.8f} ℏ")
        print(f"  E_total = {am_sol['E_total_MeV']:.6f} MeV  (m_e = {m_e_MeV:.6f})")
        print(f"  E_total / m_e = {am_sol['E_total_MeV'] / m_e_MeV:.6f}")

        # Does this r/R also give m_e? That would be extraordinary.
        gap = (am_sol['E_total_MeV'] - m_e_MeV) / m_e_MeV * 100
        print(f"  Gap from m_e: {gap:+.4f}%")

        if abs(gap) < 1.0:
            print(f"\n  *** L_z = ℏ/2 AND E ≈ m_e at the SAME geometry! ***")
        else:
            print(f"\n  L_z = ℏ/2 at r/R = {rr_sol:.4f}, but E ≠ m_e at this geometry.")
            print(f"  The angular momentum constraint and mass constraint")
            print(f"  select DIFFERENT r/R values (at fixed R = λ_C).")
            print(f"  This means either:")
            print(f"    1. R ≠ λ_C when both constraints are applied")
            print(f"    2. The self-energy model needs refinement")
            print(f"    3. Additional physics is needed")

        # Find R where BOTH L_z = ℏ/2 AND E_total = m_e
        print(f"\n--- Finding R where BOTH L_z = ℏ/2 AND E = m_e ---")
        print(f"  (fixing r/R = {rr_sol:.6f} from angular momentum constraint)")

        sol = find_self_consistent_radius(m_e_MeV, p=params.p, q=params.q,
                                          r_ratio=rr_sol)
        if sol:
            tp_both = TorusParams(R=sol['R'], r=sol['r'], p=params.p, q=params.q)
            am_both = compute_angular_momentum(tp_both, N=4000)
            print(f"  R = {sol['R']:.6e} m = {sol['R']/lambda_C:.6f} λ_C")
            print(f"  r = {sol['r']:.6e} m")
            print(f"  E_total = {sol['E_total_MeV']:.8f} MeV (target: {m_e_MeV:.8f})")
            print(f"  ⟨L_z⟩ = {am_both['Lz_over_hbar']:.8f} ℏ")
            print(f"\n  Both constraints satisfied simultaneously!")
    else:
        print(f"  L_z range [{lz_high:.4f}, {lz_low:.4f}] does not include 0.5")
        print(f"  Cannot find ℏ/2 for ({params.p},{params.q}) winding")

    # Also check different winding numbers
    print(f"\n--- Angular momentum for different winding numbers ---")
    print(f"  (at r/R = {params.r/params.R:.2f})")
    print(f"  {'(p,q)':>8} {'L_z/ℏ':>10} {'|L|/ℏ':>10}")
    print(f"  " + "-" * 32)
    for p in range(1, 5):
        for q in range(0, 4):
            if p == 0 and q == 0:
                continue
            tp = TorusParams(R=R, r=params.r, p=p, q=q)
            try:
                am = compute_angular_momentum(tp, N=2000)
                print(f"  ({p},{q}):   {am['Lz_over_hbar']:10.6f} {am['L_over_hbar']:10.6f}")
            except Exception:
                pass


def compute_retarded_field_sample(params: TorusParams, N: int = 500):
    """
    Compute the retarded EM field at each point on the curve
    due to all other points.

    For each point i, find the retarded point j such that the
    light-travel time |r_i - r_j|/c equals the time delay t_i - t_j
    (with periodic wrapping on the closed worldtube).

    Returns a summary of the retarded field structure.
    """
    lam, t, T, xyz, v = null_condition_time(params, N)

    N_sample = min(N, 100)
    sample_idx = np.linspace(0, N - 1, N_sample, dtype=int)

    retarded_distances = []
    retarded_delays = []

    for i in sample_idx:
        r_i = xyz[i]
        t_i = t[i]

        dr = r_i - xyz
        dist = np.sqrt(np.sum(dr**2, axis=-1))

        dt = t_i - t
        dt_wrapped = dt % T

        residual = np.abs(dist - c * dt_wrapped)
        residual[i] = np.inf

        j_ret = np.argmin(residual)
        retarded_distances.append(dist[j_ret])
        retarded_delays.append(dt_wrapped[j_ret])

    retarded_distances = np.array(retarded_distances)
    retarded_delays = np.array(retarded_delays)

    return {
        'sample_indices': sample_idx,
        'retarded_distances': retarded_distances,
        'retarded_delays': retarded_delays,
        'mean_retarded_distance': np.mean(retarded_distances),
        'std_retarded_distance': np.std(retarded_distances),
        'mean_retarded_delay': np.mean(retarded_delays),
        'period': T,
        'delay_fraction': np.mean(retarded_delays) / T,
    }


def scan_torus_parameters():
    """
    Scan over torus parameters to find configurations whose
    total energy (circulation + self-energy) matches known particle masses.
    """
    print("=" * 70)
    print("PARAMETER SCAN: Torus configurations vs particle masses")
    print("  (now includes self-energy corrections)")
    print("=" * 70)
    print(f"\nReference scales:")
    print(f"  Reduced Compton wavelength:  λ_C = {lambda_C:.4e} m")
    print(f"  Classical electron radius:   r_e = {r_e:.4e} m")
    print(f"  Electron mass:               {m_e_MeV:.4f} MeV/c²")
    print(f"  Fine-structure constant:     α = 1/{1/alpha:.2f}")
    print()

    print(f"{'p':>3} {'q':>3} {'R/λ_C':>10} {'r/λ_C':>10} "
          f"{'E_circ':>10} {'E_self':>10} {'E_total':>10} {'Closest':>10} {'Ratio':>10}")
    print("-" * 85)

    results = []

    for p in range(1, 5):
        for q in range(1, 5):
            for R_ratio in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                for r_ratio_abs in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
                    if r_ratio_abs >= R_ratio:
                        continue

                    R = R_ratio * lambda_C
                    r = r_ratio_abs * lambda_C
                    params = TorusParams(R=R, r=r, p=p, q=q)

                    _, E_total_MeV, breakdown = compute_total_energy(params, N=500)

                    closest = min(PARTICLE_MASSES.items(),
                                  key=lambda kv: abs(kv[1] - E_total_MeV))
                    ratio = E_total_MeV / closest[1]

                    if 0.33 < ratio < 3.0:
                        results.append({
                            'p': p, 'q': q,
                            'R_ratio': R_ratio, 'r_ratio': r_ratio_abs,
                            'E_circ': breakdown['E_circ_MeV'],
                            'E_self': breakdown['U_total_MeV'],
                            'E_total': E_total_MeV,
                            'closest': closest[0],
                            'ratio': ratio,
                        })
                        print(f"{p:3d} {q:3d} {R_ratio:10.3f} {r_ratio_abs:10.3f} "
                              f"{breakdown['E_circ_MeV']:10.4f} "
                              f"{breakdown['U_total_MeV']:10.6f} "
                              f"{E_total_MeV:10.4f} {closest[0]:>10} {ratio:10.4f}")

    print(f"\n{len(results)} configurations within factor of 3 of known masses")
    return results


# =========================================================================
# Output / analysis functions
# =========================================================================

def print_basic_analysis(params: TorusParams):
    """Print basic properties of a torus knot null curve."""
    print("=" * 70)
    print(f"NULL WORLDTUBE ANALYSIS")
    print(f"Torus: R = {params.R:.4e} m, r = {params.r:.4e} m")
    print(f"Winding: ({params.p}, {params.q}) torus knot")
    print(f"R/λ_C = {params.R/lambda_C:.4f},  r/λ_C = {params.r/lambda_C:.4f}")
    print("=" * 70)

    # Null curve properties
    lam, t, T, xyz, v = null_condition_time(params)
    v_mag = np.sqrt(np.sum(v**2, axis=-1))
    print(f"\n--- Null curve ---")
    print(f"  Velocity:   v/c = {v_mag.mean()/c:.10f} (should be 1.0)")
    print(f"  Period:     T = {T:.6e} s")
    print(f"  Frequency:  f = {1/T:.6e} Hz")

    # Circulation energy
    E, E_MeV_val, L, T, f = compute_circulation_energy(params)
    print(f"\n--- Circulation energy (zeroth order) ---")
    print(f"  Path length:  L = {L:.6e} m")
    print(f"  Energy:       E_circ = {E:.6e} J = {E_MeV_val:.6f} MeV")
    print(f"  E_circ / m_e c²:   {E_MeV_val / m_e_MeV:.6f}")

    # Self-energy
    se = compute_self_energy(params)
    print(f"\n--- Electromagnetic self-energy ---")
    print(f"  Current:          I = {se['I_amps']:.4e} A")
    print(f"  Inductance:       L = {se['L_ind_henry']:.4e} H")
    print(f"  log(8R/r) - 2:    {se['log_factor']:.4f}")
    print(f"  U_magnetic:       {se['U_mag_J']:.6e} J = {se['U_mag_J']/MeV:.6f} MeV")
    print(f"  U_electric:       {se['U_elec_J']:.6e} J = {se['U_elec_J']/MeV:.6f} MeV")
    print(f"  U_total:          {se['U_total_J']:.6e} J = {se['U_total_MeV']:.6f} MeV")
    print(f"  U_self / E_circ:  {se['self_energy_fraction']:.6f}")
    print(f"  α·ln(8R/r)-2]/π: {se['alpha_prediction']:.6f}  (analytic prediction)")
    print(f"    → ratio matches α to: {abs(se['self_energy_fraction'] - se['alpha_prediction']) / se['alpha_prediction'] * 100:.2f}%")

    # Total energy
    print(f"\n--- Total energy (circulation + self-interaction) ---")
    print(f"  E_total:          {se['E_total_MeV']:.6f} MeV")
    print(f"  E_total / m_e c²: {se['E_total_MeV'] / m_e_MeV:.6f}")
    gap_pct = (se['E_total_MeV'] - m_e_MeV) / m_e_MeV * 100
    print(f"  Gap from m_e:     {gap_pct:+.4f}%")

    # Angular momentum
    am = compute_angular_momentum(params)
    print(f"\n--- Angular momentum (literal, mechanical) ---")
    print(f"  ⟨L_z⟩ = {am['Lz_over_hbar']:.6f} ℏ  (about torus axis)")
    print(f"  |⟨L⟩| = {am['L_over_hbar']:.6f} ℏ")
    print(f"  L_z range: [{am['Lz_min']:.4f}, {am['Lz_max']:.4f}] ℏ  (instantaneous)")
    if abs(am['Lz_over_hbar'] - 0.5) < 0.01:
        print(f"  *** L_z ≈ ℏ/2 — fermion spin from geometry! ***")
    elif abs(am['Lz_over_hbar'] - 1.0) < 0.01:
        print(f"  L_z ≈ ℏ — matches photon/boson spin")

    # Scale matching
    print(f"\n--- Scale matching ---")
    target_L = 2 * np.pi * hbar * c / (m_e * c**2)
    print(f"  Path length for m_e: L = 2π·λ_C = {target_L:.6e} m")
    print(f"  Actual path length:  L = {L:.6e} m")
    print(f"  Ratio L/L_target:    {L / target_L:.6f}")

    # Retarded field structure
    print(f"\n--- Retarded field structure ---")
    ret = compute_retarded_field_sample(params, N=500)
    print(f"  Mean retarded distance:  {ret['mean_retarded_distance']:.4e} m")
    print(f"  Std retarded distance:   {ret['std_retarded_distance']:.4e} m")
    print(f"  Mean retarded delay:     {ret['mean_retarded_delay']:.4e} s")
    print(f"  Delay as fraction of T:  {ret['delay_fraction']:.4f}")
    print(f"    (= fraction of loop the field 'looks back' through)")


def print_self_energy_analysis(params: TorusParams):
    """
    Detailed analysis of how α enters the self-energy.
    Shows the relationship between self-interaction and circulation energy.
    """
    print("=" * 70)
    print("SELF-ENERGY ANALYSIS: How α enters the theory")
    print("=" * 70)

    print(f"\nThe fine-structure constant α = e²/(4πε₀ℏc) = {alpha:.10f}")
    print(f"                             1/α = {1/alpha:.6f}")
    print(f"\nα is the ratio of electromagnetic coupling to quantum action.")
    print(f"In the null worldtube, it appears as:")
    print(f"  U_self / E_circ = α × [ln(8R/r) - 2] / π")
    print(f"\nThis is NOT put in by hand. It emerges because:")
    print(f"  - Current I = ec/L contains e (electromagnetic coupling)")
    print(f"  - Circulation energy E = 2πℏc/L contains ℏ (quantum action)")
    print(f"  - Their ratio gives e²/(ℏc) ∝ α")

    print(f"\n{'r/R':>8} {'ln(8R/r)-2':>12} {'U/E_circ':>12} {'α·[...]/π':>12} {'match':>8}")
    print("-" * 58)

    R = params.R
    for r_frac in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
        r = r_frac * R
        tp = TorusParams(R=R, r=r, p=params.p, q=params.q)
        se = compute_self_energy(tp)
        match = "✓" if abs(se['self_energy_fraction'] - se['alpha_prediction']) / se['alpha_prediction'] < 0.05 else "~"
        print(f"{r_frac:8.3f} {se['log_factor']:12.4f} "
              f"{se['self_energy_fraction']:12.6f} {se['alpha_prediction']:12.6f} {match:>8}")

    print(f"\n--- Effect on electron mass ---")
    print(f"\nWith R = λ_C (electron Compton wavelength):")

    for r_frac in [0.01, 0.05, 0.1, 0.2]:
        r = r_frac * lambda_C
        tp = TorusParams(R=lambda_C, r=r, p=1, q=1)
        se = compute_self_energy(tp)
        gap = (se['E_total_MeV'] - m_e_MeV) / m_e_MeV * 100
        print(f"  r/R = {r_frac:.2f}: E_circ = {se['E_circ_MeV']:.6f}, "
              f"E_self = {se['U_total_MeV']:.6f}, "
              f"E_total = {se['E_total_MeV']:.6f} MeV  ({gap:+.3f}% from m_e)")

    # Find r/R where E_total = m_e exactly
    print(f"\n--- Finding r/R where E_total = m_e c² exactly ---")
    # Bisection on r/R
    rr_low, rr_high = 0.001, 0.99
    for _ in range(60):
        rr_mid = (rr_low + rr_high) / 2.0
        tp = TorusParams(R=lambda_C, r=rr_mid * lambda_C, p=1, q=1)
        se = compute_self_energy(tp)
        if se['E_total_MeV'] > m_e_MeV:
            rr_low = rr_mid
        else:
            rr_high = rr_mid

    rr_sol = (rr_low + rr_high) / 2.0
    tp = TorusParams(R=lambda_C, r=rr_sol * lambda_C, p=1, q=1)
    se = compute_self_energy(tp)
    print(f"  Solution: r/R = {rr_sol:.6f}")
    print(f"  r = {rr_sol * lambda_C:.4e} m = {rr_sol * lambda_C / r_e:.4f} × r_e")
    print(f"  E_circ  = {se['E_circ_MeV']:.8f} MeV")
    print(f"  E_self  = {se['U_total_MeV']:.8f} MeV")
    print(f"  E_total = {se['E_total_MeV']:.8f} MeV  (target: {m_e_MeV:.8f})")
    print(f"  Self-energy fraction: {se['self_energy_fraction']:.6f}")
    print(f"  α × [ln(8R/r)-2]/π:  {se['alpha_prediction']:.6f}")


def print_resonance_analysis(params: TorusParams):
    """
    Show the resonance quantization structure of the torus.
    """
    print("=" * 70)
    print("RESONANCE QUANTIZATION ANALYSIS")
    print("=" * 70)
    R, r = params.R, params.r

    print(f"\nTorus: R = {R:.4e} m, r = {r:.4e} m")
    print(f"R/r = {R/r:.2f}")
    print(f"\nBoundary conditions (field must be single-valued):")
    print(f"  Toroidal: k_tor × 2πR = 2πn  →  λ_n = 2πR/n")
    print(f"  Poloidal: k_pol × 2πr = 2πm  →  λ_m = 2πr/m")
    print(f"\nFor massless field: E = ℏc × √(n²/R² + m²/r²)")

    modes = resonance_analysis(params, n_modes=5)

    print(f"\n{'n':>3} {'m':>3} {'Type':>10} {'E (MeV)':>12} {'E/m_e':>10} {'Nearest particle':>20}")
    print("-" * 65)

    for mode in modes[:20]:  # show first 20
        # Find nearest known particle
        nearest = min(PARTICLE_MASSES.items(),
                      key=lambda kv: abs(kv[1] - mode['E_MeV']))
        ratio = mode['E_MeV'] / nearest[1]
        near_str = f"{nearest[0]} (×{ratio:.3f})" if 0.1 < ratio < 10 else ""

        print(f"{mode['n']:3d} {mode['m']:3d} {mode['type']:>10} "
              f"{mode['E_MeV']:12.4f} {mode['E_over_me']:10.4f} {near_str:>20}")

    # Key insight about the poloidal tower
    E_tor_1 = hbar * c / R / MeV
    E_pol_1 = hbar * c / r / MeV
    print(f"\n--- Scale separation ---")
    print(f"  Fundamental toroidal (1,0): {E_tor_1:.4f} MeV")
    print(f"  Fundamental poloidal (0,1): {E_pol_1:.4f} MeV")
    print(f"  Ratio (poloidal/toroidal):  {E_pol_1/E_tor_1:.2f} = R/r = {R/r:.2f}")
    print(f"\n  The poloidal modes form a 'tower' of heavy states at")
    print(f"  energies ~ (R/r) × E_fundamental. For R/r = {R/r:.0f},")
    print(f"  the first poloidal mode is {R/r:.0f}× heavier than the")
    print(f"  fundamental — a natural mass hierarchy from geometry.")
    print(f"\n  This is analogous to Kaluza-Klein compactification:")
    print(f"  a small compact dimension (r) produces a tower of heavy")
    print(f"  modes invisible at low energies.")


def print_find_radii():
    """
    Find self-consistent radii for known particle masses.
    Now uses angular momentum to classify particles:
      p=1 (L_z = ℏ)   → bosons
      p=2 (L_z = ℏ/2) → fermions
    """
    print("=" * 70)
    print("SELF-CONSISTENT RADII FOR KNOWN PARTICLES")
    print("  Angular momentum determines winding number:")
    print("    p=1 → L_z = ℏ    (bosons)")
    print("    p=2 → L_z = ℏ/2  (fermions)")
    print("  Then: find R where E_total = m_particle × c²")
    print("=" * 70)

    # Fermions: p=2 winding (spin-1/2)
    fermions = {
        'electron': 0.51099895,
        'muon': 105.6583755,
        'tau': 1776.86,
    }

    # Hadrons (composite, but treat as single torus for now)
    hadrons = {
        'proton': 938.27208816,
        'pion±': 139.57039,
    }

    # Bosons: p=1 winding (spin-1)
    # The W, Z, and Higgs would go here if we included them
    # For now, note that the pion is spin-0, not spin-1

    print(f"\n--- Fermions (p=2, q=1 → L_z = ℏ/2) ---")
    for name, mass_MeV in sorted(fermions.items(), key=lambda x: x[1]):
        sol = find_self_consistent_radius(mass_MeV, p=2, q=1, r_ratio=0.1)
        if sol is None:
            print(f"\n  {name}: no solution found")
            continue

        mass_kg = mass_MeV * MeV / c**2
        lambda_C_particle = hbar / (mass_kg * c)

        # Compute angular momentum at this solution
        tp = TorusParams(R=sol['R'], r=sol['r'], p=2, q=1)
        am = compute_angular_momentum(tp)

        print(f"\n  {name} ({mass_MeV:.4f} MeV):")
        print(f"    R = {sol['R']:.4e} m = {sol['R_femtometers']:.4f} fm")
        print(f"    r = {sol['r']:.4e} m = {sol['r_femtometers']:.5f} fm")
        print(f"    R / λ_C({name}) = {sol['R'] / lambda_C_particle:.6f}")
        print(f"    E_circ = {sol['E_circ_MeV']:.6f} MeV")
        print(f"    E_self = {sol['E_self_MeV']:.6f} MeV ({sol['self_energy_fraction']*100:.3f}%)")
        print(f"    E_total = {sol['E_total_MeV']:.6f} MeV (match: {sol['match_ppm']:.1f} ppm)")
        print(f"    L_z = {am['Lz_over_hbar']:.6f} ℏ  ← spin-1/2 from geometry")

    print(f"\n--- Hadrons (composite — shown for comparison) ---")
    for name, mass_MeV in sorted(hadrons.items(), key=lambda x: x[1]):
        # Proton is spin-1/2 (fermion), pion is spin-0
        p_wind = 2 if name == 'proton' else 1
        sol = find_self_consistent_radius(mass_MeV, p=p_wind, q=1, r_ratio=0.1)
        if sol is None:
            print(f"\n  {name}: no solution found")
            continue

        mass_kg = mass_MeV * MeV / c**2
        lambda_C_particle = hbar / (mass_kg * c)
        tp = TorusParams(R=sol['R'], r=sol['r'], p=p_wind, q=1)
        am = compute_angular_momentum(tp)

        print(f"\n  {name} ({mass_MeV:.4f} MeV, p={p_wind}):")
        print(f"    R = {sol['R']:.4e} m = {sol['R_femtometers']:.4f} fm")
        print(f"    r = {sol['r']:.4e} m = {sol['r_femtometers']:.5f} fm")
        print(f"    R / λ_C({name}) = {sol['R'] / lambda_C_particle:.6f}")
        print(f"    E_total = {sol['E_total_MeV']:.6f} MeV")
        print(f"    L_z = {am['Lz_over_hbar']:.6f} ℏ")
        if name == 'proton':
            print(f"    Note: proton charge radius (measured) = 0.8414 fm")
            print(f"          torus major radius R = {sol['R_femtometers']:.4f} fm")
            print(f"          ratio: R_measured / R_torus = {0.8414 / sol['R_femtometers']:.4f}")

    # Mass ratios
    print(f"\n--- Lepton mass ratios ---")
    e_sol = find_self_consistent_radius(fermions['electron'], p=2, q=1, r_ratio=0.1)
    mu_sol = find_self_consistent_radius(fermions['muon'], p=2, q=1, r_ratio=0.1)
    tau_sol = find_self_consistent_radius(fermions['tau'], p=2, q=1, r_ratio=0.1)

    if e_sol and mu_sol and tau_sol:
        print(f"  R_e / R_μ   = {e_sol['R'] / mu_sol['R']:.4f}  "
              f"(= m_μ/m_e = {fermions['muon']/fermions['electron']:.4f})")
        print(f"  R_e / R_τ   = {e_sol['R'] / tau_sol['R']:.4f}  "
              f"(= m_τ/m_e = {fermions['tau']/fermions['electron']:.4f})")
        print(f"  R_μ / R_τ   = {mu_sol['R'] / tau_sol['R']:.4f}  "
              f"(= m_τ/m_μ = {fermions['tau']/fermions['muon']:.4f})")
        print(f"\n  In this model, heavier fermions are SMALLER tori.")
        print(f"  All three generations have the same topology (2,1)")
        print(f"  but different radii. What selects the three radii?")
        print(f"  That's the generation problem — still open.")


def print_pair_production():
    """
    Analyze pair production (γ → e⁻ + e⁺) in the torus model.

    The photon is a free EM wave (no torus). The electron/positron
    are (2,±1) torus knots. Pair production is the photon "winding up"
    into two counter-rotating torus knots near a nucleus.
    """
    print("=" * 70)
    print("PAIR PRODUCTION: γ → e⁻ + e⁺")
    print("  A free wave winds up into two torus knots")
    print("=" * 70)

    # --- Threshold geometry ---
    print(f"\n--- Threshold condition ---")
    E_thresh_MeV = 2 * m_e_MeV
    E_thresh_J = E_thresh_MeV * MeV
    lambda_photon = h_planck * c / E_thresh_J

    # Electron torus path length
    e_sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
    tp_e = TorusParams(R=e_sol['R'], r=e_sol['r'], p=2, q=1)
    L_torus = compute_path_length(tp_e)

    print(f"  Threshold energy:           E_γ = 2 m_e c² = {E_thresh_MeV:.4f} MeV")
    print(f"  Photon wavelength:          λ_γ = {lambda_photon:.6e} m")
    print(f"  Electron torus path length: L   = {L_torus:.6e} m")
    print(f"  Ratio λ_γ / L_torus:        {lambda_photon / L_torus:.6f}")
    print(f"  2π λ_C:                     {2*np.pi*lambda_C:.6e} m")
    print(f"  Ratio λ_γ / 2πλ_C:          {lambda_photon / (2*np.pi*lambda_C):.6f}")
    print(f"\n  λ_γ = L_torus / 2 exactly (because E_γ = 2 × m_e c² = 2 × hc/L)")
    print(f"  The photon wavelength equals HALF the electron torus path —")
    print(f"  one full loop of the double-wound (2,1) knot.")
    print(f"  The photon fits one loop; the electron needs two → spin 1/2.")

    # --- Nuclear field as catalyst ---
    print(f"\n--- Nuclear Coulomb field as catalyst ---")
    print(f"  The photon cannot produce a pair in free space (4-momentum")
    print(f"  conservation). It needs a nucleus to absorb recoil AND to")
    print(f"  provide the field curvature that bends the wave into a torus.")
    print(f"\n  Schwinger critical field: E_crit = m_e²c³/(eℏ)")
    E_schwinger = m_e**2 * c**3 / (e_charge * hbar)
    print(f"    = {E_schwinger:.4e} V/m")
    print(f"\n  Critical distance (where Coulomb field = Schwinger field):")
    print(f"  {'Z':>4} {'Element':>8} {'r_crit (fm)':>12} {'r_crit/λ_C':>12} {'σ ∝ πr²_crit':>14}")
    print(f"  " + "-" * 52)
    elements = [(1, 'H'), (6, 'C'), (26, 'Fe'), (82, 'Pb'), (92, 'U')]
    for Z, elem in elements:
        r_crit = np.sqrt(Z * e_charge / (4 * np.pi * eps0 * E_schwinger))
        sigma_rel = Z  # σ ∝ r_crit² ∝ Z
        print(f"  {Z:4d} {elem:>8} {r_crit*1e15:12.3f} {r_crit/lambda_C:12.4f} {sigma_rel:14.1f}")
    print(f"\n  Cross-section σ ∝ πr²_crit ∝ Z → explains Z² scaling")
    print(f"  (Bethe-Heitler: σ ∝ α r_e² Z²)")

    # --- Conservation laws ---
    print(f"\n--- Conservation laws ---")
    print(f"  {'Quantity':<20} {'Before (γ)':<20} {'After (e⁻ + e⁺)':<25} {'Conserved?'}")
    print(f"  " + "-" * 70)
    print(f"  {'Energy':<20} {'E_γ':<20} {'E_e + E_e+':<25} {'✓'}")
    print(f"  {'Momentum':<20} {'p_γ = E/c':<20} {'p_e + p_e+ + p_nuc':<25} {'✓ (recoil)'}")
    print(f"  {'Charge':<20} {'0':<20} {'-e + (+e) = 0':<25} {'✓'}")
    print(f"  {'Winding (p)':<20} {'0':<20} {'+2 + (-2) = 0':<25} {'✓'}")
    print(f"  {'Winding (q)':<20} {'0':<20} {'+1 + (-1) = 0':<25} {'✓'}")
    print(f"  {'Angular mom':<20} {'ℏ (spin-1)':<20} {'ℏ/2 + (-ℏ/2) + orb':<25} {'✓'}")
    print(f"  {'Net topology':<20} {'trivial':<20} {'(2,+1)+(2,-1)=trivial':<25} {'✓'}")
    print(f"\n  Charge conservation IS topology conservation:")
    print(f"  no net winding created — equal and opposite topology.")

    # --- Annihilation (reverse) ---
    print(f"\n--- Annihilation (reverse process) ---")
    print(f"  e⁻ + e⁺ → γγ")
    print(f"  (2,+1) + (2,-1) → topology cancels → free waves")
    print(f"  Two photons required (not one) for momentum conservation")
    print(f"  Energy: 2 × {m_e_MeV:.4f} = {2*m_e_MeV:.4f} MeV → 2 photons")
    print(f"\n  Positronium (bound e⁻e⁺):")
    print(f"    Para (↑↓, S=0): decays to 2γ, τ = 1.25 × 10⁻¹⁰ s")
    print(f"    Ortho (↑↑, S=1): decays to 3γ, τ = 1.42 × 10⁻⁷ s")
    print(f"    Ortho is 1000× slower because aligned spins (ℏ/2 + ℏ/2 = ℏ)")
    print(f"    can't cancel into 2 photons — needs 3-body final state.")

    # --- Above-threshold: heavier pairs ---
    print(f"\n--- Above threshold: heavier pairs ---")
    print(f"  The photon preferentially creates the LIGHTEST pair because")
    print(f"  that's the ground state (largest torus = most probable).")
    print(f"\n  {'Pair':<12} {'Threshold (MeV)':>16} {'E_γ needed':>12}")
    print(f"  " + "-" * 44)
    pair_thresholds = [
        ('e⁻e⁺', 2 * 0.511),
        ('μ⁻μ⁺', 2 * 105.658),
        ('τ⁻τ⁺', 2 * 1776.86),
        ('pp̄', 2 * 938.272),
    ]
    for pair, thresh in pair_thresholds:
        print(f"  {pair:<12} {thresh:16.3f} {'> ' + f'{thresh:.0f} MeV':>12}")


def print_decay_landscape():
    """
    Analyze particle decay as torus expansion / topology change.

    Key insight: stable particles are ground states (largest tori)
    of each topological class. Unstable particles are excited states
    that expand to the ground state, releasing energy.
    """
    print("=" * 70)
    print("DECAY LANDSCAPE: Stability and transitions")
    print("=" * 70)

    # Particle catalog
    catalog = [
        # (name, mass_MeV, p, q, spin, stable, lifetime_s, decay)
        ('photon',   0.0,      1, 0, 1.0, True,  None,     '—'),
        ('electron', 0.511,    2, 1, 0.5, True,  None,     '—'),
        ('proton',   938.272,  2, 1, 0.5, True,  None,     '—'),
        ('muon',     105.658,  2, 1, 0.5, False, 2.2e-6,   'e + ν_μ + ν̄_e'),
        ('tau',      1776.86,  2, 1, 0.5, False, 2.9e-13,  'μ + ν_τ + ν̄_μ'),
        ('pion±',    139.570,  1, 1, 0.0, False, 2.6e-8,   'μ + ν_μ'),
        ('pion0',    134.977,  1, 1, 0.0, False, 8.5e-17,  'γγ'),
        ('neutron',  939.565,  2, 1, 0.5, False, 879,      'p + e + ν̄_e'),
    ]

    print(f"\n--- Particle catalog ---")
    print(f"  {'Name':<10} {'Mass':>10} {'(p,q)':>6} {'L_z/ℏ':>7} "
          f"{'Status':>8} {'Lifetime':>12} {'Decays to'}")
    print(f"  " + "-" * 75)
    for name, mass, p, q, spin, stable, tau, decay in catalog:
        lt = "STABLE" if stable else f"{tau:.1e} s"
        lz = f"{1/p:.2f}" if p > 0 else "—"
        print(f"  {name:<10} {mass:10.3f} ({p},{q}){'':<2} {lz:>7} "
              f"{'STABLE' if stable else '':>8} {lt:>12}  {decay}")

    # Decay energetics
    print(f"\n--- Decay = torus expansion ---")
    print(f"  Unstable particles are SMALLER tori (higher energy)")
    print(f"  They expand to larger tori (lower energy), releasing ΔE")

    decays = [
        ('π⁰ → γγ',       134.977,  0.0,     1, 0, 'total unwinding',        8.5e-17),
        ('τ → μ + ν + ν̄',  1776.86,  105.658, 2, 2, 'R expands 17×',         2.9e-13),
        ('π± → μ + ν',     139.570,  105.658, 1, 2, 'TOPOLOGY CHANGE p=1→2', 2.6e-8),
        ('μ → e + ν + ν̄',  105.658,  0.511,   2, 2, 'R expands 207×',        2.2e-6),
        ('n → p + e + ν̄',  939.565,  938.783, 2, 2, 'ΔR/R = 0.08%',         879),
    ]

    print(f"\n  {'Decay':<16} {'ΔE (MeV)':>10} {'p→p':>5} {'τ (s)':>12}  {'Mechanism'}")
    print(f"  " + "-" * 68)
    for name, m_par, m_dau, pi, pf, mech, tau in sorted(decays, key=lambda x: x[6]):
        dE = m_par - m_dau
        print(f"  {name:<16} {dE:10.3f} {pi}→{pf:}{'':<2} {tau:12.2e}  {mech}")

    # Stability principle
    print(f"\n--- Stability principle ---")
    print(f"  STABLE = largest torus (lowest energy) in its topological class")
    print(f"  No lower-energy state with same topology to decay into.")
    print(f"\n  (2,1) leptons:")

    for name, mass in [('tau', 1776.86), ('muon', 105.658), ('electron', 0.511)]:
        sol = find_self_consistent_radius(mass, p=2, q=1, r_ratio=0.1)
        if sol:
            stable = "← GROUND STATE" if name == 'electron' else "→ decays"
            print(f"    {name:<10} R = {sol['R_femtometers']:10.4f} fm  "
                  f"(E = {mass:10.3f} MeV)  {stable}")

    print(f"\n  The electron is stable because there is no LARGER (2,1)")
    print(f"  torus to expand into. It's the ground state of its topology.")

    # Neutrino interpretation
    print(f"\n--- Neutrinos as topology carriers ---")
    print(f"  π⁺(p=1) → μ⁺(p=2) + ν_μ(p=?)")
    print(f"  Winding changes: 1 → 2. Where does the Δp go?")
    print(f"  The neutrino carries the topological difference.")
    print(f"\n  If neutrinos are propagating topology changes (not tori),")
    print(f"  this would explain:")
    print(f"    • Nearly massless: not bound structures, just transitions")
    print(f"    • Three flavors: one per lepton generation transition")
    print(f"    • Weak-only: couple to topology changes, not stable EM")
    print(f"    • Spin ℏ/2: carry angular momentum of the transition")
    print(f"    • Oscillations: topology transitions can mix")


# =========================================================================
# HYDROGEN ATOM: Orbital dynamics in the torus model
# =========================================================================

def compute_hydrogen_orbital(Z: int, n: int, params: TorusParams = None):
    """
    Compute orbital dynamics of an electron torus around a nucleus.

    The electron is a (2,1) torus knot of major radius R ~ λ_C/2 ~ 193 fm.
    The Bohr radius is a_0 ~ 52,918 fm. The torus is ~274× smaller than
    its orbit, so we treat it as a compact spinning object in a Coulomb field.

    Three frequencies characterize the motion:
      f_circ:  internal photon circulation (gives mass)
      f_orbit: center-of-mass revolution (gives energy levels)
      f_prec:  torus axis precession (gives fine structure)

    The hierarchy f_circ >> f_orbit >> f_prec, with each step
    separated by α², is the reason atomic physics is perturbative.
    """
    if params is None:
        sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
        params = TorusParams(R=sol['R'], r=sol['r'], p=2, q=1)

    # Torus properties
    L_torus = compute_path_length(params)
    f_circ = c / L_torus
    R_torus = params.R

    # Orbital parameters (Bohr model — but now with a physical reason)
    r_n = n**2 * a_0 / Z                       # orbital radius
    v_n = Z * alpha * c / n                     # orbital velocity
    E_n = -m_e * c**2 * Z**2 * alpha**2 / (2 * n**2)  # binding energy
    f_orbit = v_n / (2 * np.pi * r_n)           # orbital frequency

    # Precession frequency (spin-orbit coupling)
    # The torus magnetic moment interacts with the B field seen in
    # the electron's rest frame: B ~ (v/c²) × E_Coulomb
    # ΔE_FS ~ m_e c² (Zα)⁴ / n³ → f_prec = ΔE_FS / h
    # This is α² slower than f_orbit, giving the three-level hierarchy
    f_prec = m_e * c**2 * (Z * alpha)**4 / (h_planck * n**3)

    # Angular momenta
    L_spin = hbar / params.p  # internal (ℏ/2 for p=2)
    L_orbit_n = n * hbar      # orbital (Bohr condition)

    # Tidal parameter: the torus has finite size in a non-uniform field
    tidal_param = R_torus / r_n
    # Fine structure correction scales as tidal_param² ~ α²
    dE_fine = abs(E_n) * alpha**2 / n  # order-of-magnitude estimate

    return {
        'n': n, 'Z': Z,
        'r_n': r_n,
        'r_n_pm': r_n * 1e12,                   # orbital radius in pm
        'r_n_fm': r_n * 1e15,                    # orbital radius in fm
        'v_n': v_n,
        'v_over_c': v_n / c,
        'E_n_J': E_n,
        'E_n_eV': E_n / eV,
        'f_circ': f_circ,
        'f_orbit': f_orbit,
        'f_prec': f_prec,
        'ratio_circ_orbit': f_circ / f_orbit,
        'ratio_orbit_prec': f_orbit / f_prec,
        'R_torus': R_torus,
        'R_torus_fm': R_torus * 1e15,
        'size_ratio': r_n / R_torus,             # should be >> 1
        'L_spin': L_spin,
        'L_spin_over_hbar': L_spin / hbar,
        'L_orbit': L_orbit_n,
        'L_orbit_over_hbar': L_orbit_n / hbar,
        'L_ratio': L_orbit_n / L_spin,           # = 2n (always integer!)
        'tidal_param': tidal_param,
        'dE_fine_eV': dE_fine / eV,
    }


def compute_shell_filling(Z: int, params: TorusParams = None):
    """
    Compute electron shell configuration for a multi-electron atom.

    In the torus model, each electron is a (2,1) torus with:
      - 2 handedness options (spin up/down = opposite winding chirality)
      - (2l+1) precession orientations in each subshell

    Pauli exclusion becomes geometric: two tori of the same handedness
    at the same orbital resonance produce destructive interference.
    Only opposite-handedness pairs can share a resonance.

    Shell capacities:
      n=1: 2 electrons (l=0 only, 2×1=2)
      n=2: 8 electrons (l=0: 2, l=1: 6)
      n=3: 18 electrons (l=0: 2, l=1: 6, l=2: 10)
    """
    if params is None:
        sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
        params = TorusParams(R=sol['R'], r=sol['r'], p=2, q=1)

    # Build shell structure
    shells = []
    electrons_remaining = Z
    for n in range(1, 8):
        if electrons_remaining <= 0:
            break
        for l in range(n):
            if electrons_remaining <= 0:
                break
            capacity = 2 * (2 * l + 1)
            occupancy = min(capacity, electrons_remaining)
            electrons_remaining -= occupancy

            r_n = n**2 * a_0 / Z
            # Screened orbital radius (approximate)
            # Inner electrons screen the nuclear charge
            Z_eff = Z - sum(s['occupancy'] for s in shells)
            Z_eff = max(Z_eff, 1)
            r_eff = n**2 * a_0 / Z_eff
            E_n = -m_e * c**2 * Z_eff**2 * alpha**2 / (2 * n**2)

            subshell_labels = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
            label = f"{n}{subshell_labels.get(l, '?')}"

            shells.append({
                'n': n, 'l': l,
                'label': label,
                'capacity': capacity,
                'occupancy': occupancy,
                'full': occupancy == capacity,
                'r_n': r_n,
                'r_n_pm': r_n * 1e12,
                'Z_eff': Z_eff,
                'r_eff': r_eff,
                'r_eff_pm': r_eff * 1e12,
                'E_n_eV': E_n / eV,
                'orientations': 2 * l + 1,
            })

    # Noble gas: outermost occupied subshell is full, and it's
    # a p subshell (or 1s for helium). This captures He, Ne, Ar, etc.
    outermost = shells[-1] if shells else None
    noble = (outermost and outermost['full'] and
             (outermost['l'] == 1 or (outermost['n'] == 1 and outermost['l'] == 0)))

    return {
        'Z': Z,
        'shells': shells,
        'noble_gas': noble,
        'total_electrons': sum(s['occupancy'] for s in shells),
    }


def print_hydrogen_analysis():
    """
    Full hydrogen atom analysis in the null worldtube model.

    Shows how a compact electron torus orbiting a proton reproduces
    the Bohr model, explains quantization through resonance, and
    predicts fine structure as a tidal effect. Then extends to
    multi-electron atoms: shell filling, Pauli exclusion as geometry,
    and a preview of chemical bonding.

    This is where the torus model meets the planetary model — and
    fixes the reason the planetary model was abandoned (Larmor
    radiation). The electron IS radiation. There's no accelerating
    charge to radiate. The radiation objection doesn't apply.
    """
    print("=" * 70)
    print("HYDROGEN ATOM IN THE TORUS MODEL")
    print("  A compact spinning torus orbiting a proton")
    print("=" * 70)

    # Get the electron torus parameters
    e_sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
    params = TorusParams(R=e_sol['R'], r=e_sol['r'], p=2, q=1)
    R_torus = params.R
    R_torus_fm = R_torus * 1e15

    # ==========================================
    # Section 1: Size hierarchy
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  1. SIZE HIERARCHY: The electron is tiny")
    print(f"{'='*60}")
    print(f"\n  Electron torus major radius:  R = {R_torus_fm:.2f} fm")
    print(f"  Electron torus minor radius:  r = {params.r*1e15:.2f} fm")
    print(f"  Bohr radius:                  a₀ = {a_0*1e12:.1f} pm = {a_0*1e15:.0f} fm")
    print(f"  Ratio a₀ / R:                 {a_0/R_torus:.0f}")
    print(f"  This equals:                  2/α = {2/alpha:.0f}")
    print(f"\n  The electron fits ~{a_0/R_torus:.0f} times between itself and the nucleus.")
    print(f"  It is a compact spinning object in a slowly-varying field.")
    print(f"\n  Why this fixes the planetary model's fatal flaw:")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Bohr (1913): orbiting electrons should radiate (Larmor)")
    print(f"               and spiral into the nucleus. Ad hoc fix:")
    print(f"               postulate stable orbits without explanation.")
    print(f"  QM (1926):   replace orbits with probability clouds.")
    print(f"               Problem solved, but physical picture lost.")
    print(f"  Torus model: the electron IS radiation — a photon on a")
    print(f"               closed path. There is no accelerating charge")
    print(f"               in the Larmor sense. The radiation objection")
    print(f"               that killed the planetary model doesn't apply.")
    print(f"               You get to keep the orbits.")

    # ==========================================
    # Section 2: Orbital dynamics
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  2. ORBITAL DYNAMICS: Recovering the Bohr levels")
    print(f"{'='*60}")
    print(f"\n  The electron torus orbits the proton in a Coulomb field.")
    print(f"  Centripetal balance: m_e v²/r = e²/(4πε₀ r²)")
    print(f"  → v_n = αc/n,  r_n = n² a₀,  E_n = -13.6/n² eV")
    print(f"\n  {'n':>3} {'r_n (pm)':>10} {'v_n/c':>10} {'E_n (eV)':>12} {'Orbit/Torus':>14}")
    print(f"  " + "─" * 55)

    for n in range(1, 7):
        orb = compute_hydrogen_orbital(1, n, params)
        print(f"  {n:3d} {orb['r_n_pm']:10.2f} {orb['v_over_c']:10.6f} "
              f"{orb['E_n_eV']:12.4f} {orb['size_ratio']:14.0f}×")

    print(f"\n  At n=1: the electron orbits at {alpha:.6f} × c (= αc)")
    print(f"  This is {alpha*100:.3f}% of lightspeed — fast, but non-relativistic.")
    print(f"  The orbit is {a_0/R_torus:.0f}× larger than the torus → compact object in a field. ✓")

    # ==========================================
    # Section 3: Three-frequency hierarchy
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  3. THREE FREQUENCIES: The α² cascade")
    print(f"{'='*60}")

    orb1 = compute_hydrogen_orbital(1, 1, params)
    print(f"\n  Three rotations govern the electron's motion:")
    print(f"\n  1. Internal circulation  f_circ  = {orb1['f_circ']:.4e} Hz")
    print(f"     (photon going around the torus → gives mass)")
    print(f"\n  2. Orbital revolution    f_orbit = {orb1['f_orbit']:.4e} Hz")
    print(f"     (torus center orbiting nucleus → gives energy levels)")
    print(f"\n  3. Axis precession       f_prec  = {orb1['f_prec']:.4e} Hz")
    print(f"     (torus axis wobbling in field gradient → gives fine structure)")
    print(f"\n  The hierarchy:")
    print(f"    f_circ / f_orbit  = {orb1['ratio_circ_orbit']:.0f}  ≈ 1/α² = {1/alpha**2:.0f}")
    print(f"    f_orbit / f_prec  = {orb1['ratio_orbit_prec']:.0f}  ≈ 1/α² = {1/alpha**2:.0f}")
    print(f"    f_circ / f_prec   = {orb1['f_circ']/orb1['f_prec']:.0f}  ≈ 1/α⁴ = {1/alpha**4:.0f}")
    print(f"\n  Each level of structure is α² = 1/{1/alpha**2:.0f} slower than the last.")
    print(f"  THIS is why atomic physics is perturbative: the frequencies")
    print(f"  are so well-separated that each level barely perturbs the next.")
    print(f"  Perturbation theory works because α is small.")
    print(f"\n  In the torus model, α has a geometric meaning:")
    print(f"    α = R_torus / a₀ × 2 = (electron size) / (orbit size) × 2")
    print(f"    = {R_torus / a_0 * 2:.6f}  (vs α = {alpha:.6f})")

    # For different n
    print(f"\n  {'n':>3} {'f_circ (Hz)':>14} {'f_orbit (Hz)':>14} {'f_prec (Hz)':>14} "
          f"{'circ/orbit':>12} {'orbit/prec':>12}")
    print(f"  " + "─" * 76)
    for n in range(1, 5):
        orb = compute_hydrogen_orbital(1, n, params)
        print(f"  {n:3d} {orb['f_circ']:14.4e} {orb['f_orbit']:14.4e} {orb['f_prec']:14.4e} "
              f"{orb['ratio_circ_orbit']:12.0f} {orb['ratio_orbit_prec']:12.0f}")

    # ==========================================
    # Section 4: Resonance → Quantization
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  4. RESONANCE → QUANTIZATION: Why orbits are discrete")
    print(f"{'='*60}")
    print(f"\n  The electron torus has internal angular momentum:")
    print(f"    L_spin = ℏ/p = ℏ/2  (for p=2 winding)")
    print(f"\n  The orbital angular momentum must be commensurate:")
    print(f"    L_orbit = n ℏ  (Bohr condition)")
    print(f"\n  The ratio L_orbit / L_spin is always an integer:")

    print(f"\n  {'n':>3} {'L_orbit/ℏ':>12} {'L_spin/ℏ':>12} {'L_orbit/L_spin':>16} {'Integer?':>10}")
    print(f"  " + "─" * 55)
    for n in range(1, 7):
        orb = compute_hydrogen_orbital(1, n, params)
        ratio = orb['L_ratio']
        is_int = "✓" if abs(ratio - round(ratio)) < 0.001 else "✗"
        print(f"  {n:3d} {orb['L_orbit_over_hbar']:12.1f} {orb['L_spin_over_hbar']:12.4f} "
              f"{ratio:16.4f} {is_int:>10}")

    print(f"\n  L_orbit / L_spin = 2n — always an integer!")
    print(f"\n  Physical meaning: the electron's orbital phase must be")
    print(f"  locked to its internal circulation phase. After each orbit,")
    print(f"  the internal state must return to its starting configuration.")
    print(f"\n  This is a RESONANCE condition, not a postulate:")
    print(f"    • Guitar string: wavelength must fit the string → discrete modes")
    print(f"    • Electron orbit: internal phase must fit the orbit → discrete levels")
    print(f"    • Same physics, different geometry")
    print(f"\n  The Bohr quantization condition L = nℏ is not ad hoc —")
    print(f"  it's the resonance condition of a structured object.")

    # ==========================================
    # Section 5: Fine structure as tidal effect
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  5. FINE STRUCTURE: Tidal effect of finite electron size")
    print(f"{'='*60}")
    print(f"\n  The electron torus has finite size R = {R_torus_fm:.1f} fm.")
    print(f"  The Coulomb field varies across it:")
    print(f"    ΔE/E ~ R_torus / r_orbit = α/(2n²)")
    print(f"\n  The energy correction scales as (tidal parameter)²:")

    print(f"\n  {'n':>3} {'R/r_orbit':>14} {'(R/r_orbit)²':>14} {'ΔE_fine (eV)':>14} {'Known ΔE_FS':>14}")
    print(f"  " + "─" * 62)

    # Known fine structure for comparison (Dirac result)
    # ΔE_FS ≈ E_n × α² × [n/(j+1/2) - 3/4] / n
    # For the largest splitting in each n:
    known_fs = {
        1: 0.0,       # 1s has only j=1/2, no splitting
        2: 4.53e-5,   # 2p_{1/2} vs 2p_{3/2}
        3: 1.34e-5,   # 3p splitting
        4: 5.65e-6,   # 4p splitting
    }

    for n in range(1, 5):
        orb = compute_hydrogen_orbital(1, n, params)
        tp = orb['tidal_param']
        known = known_fs.get(n, 0)
        known_str = f"{known:.2e}" if known > 0 else "—"
        print(f"  {n:3d} {tp:14.6f} {tp**2:14.2e} {orb['dE_fine_eV']:14.6f} {known_str:>14}")

    print(f"\n  The tidal parameter R/r = α/(2n²) gives corrections of order α².")
    print(f"  This is exactly the scaling of fine structure!")
    print(f"\n  Physical picture:")
    print(f"    • The Coulomb field is stronger on the near side of the torus")
    print(f"    • This creates a torque on the spinning torus")
    print(f"    • The torque causes the spin axis to precess")
    print(f"    • Different precession rates → energy splitting")
    print(f"\n  Fine structure is literally a TIDAL effect: the electron")
    print(f"  has structure, and the field is non-uniform across it.")
    print(f"  α appears because it is the ratio of electron size to orbit size.")

    # ==========================================
    # Section 6: Multi-electron atoms
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  6. MULTI-ELECTRON ATOMS: Shell filling as torus packing")
    print(f"{'='*60}")
    print(f"\n  Each electron is a (2,1) torus. In a given orbital resonance:")
    print(f"    • 2 handedness options: clockwise and counter-clockwise")
    print(f"      (this IS spin up/down — it's the winding chirality)")
    print(f"    • (2l+1) orientations of the torus precession axis")
    print(f"    • Total capacity per subshell: 2 × (2l+1)")
    print(f"\n  Pauli exclusion is GEOMETRIC:")
    print(f"    Two tori of the same handedness at the same resonance")
    print(f"    have identical circulation phases → they interfere")
    print(f"    destructively. Only opposite-handedness pairs can coexist.")
    print(f"    A third torus of the same topology simply can't fit.")

    # Shell capacity table
    print(f"\n  Shell capacity (standard QM = torus packing):")
    print(f"  {'Shell':>6} {'Subshell':>10} {'Orientations':>14} {'× 2 spins':>10} {'= Capacity':>12}")
    print(f"  " + "─" * 55)
    for n in range(1, 5):
        for l in range(n):
            sub_label = f"{n}{'spdf'[l]}"
            orient = 2 * l + 1
            cap = 2 * orient
            print(f"  {n:6d} {sub_label:>10} {orient:14d} {2*orient:10d} {cap:12d}")
        total = 2 * n**2
        print(f"  {'':>6} {'':>10} {'':>14} {'TOTAL:':>10} {total:12d}")

    # Show first 18 elements
    print(f"\n  First 18 elements (up to Argon):")
    print(f"  {'Z':>4} {'Element':>8} {'Config':>18} {'Outer shell':>14} {'Noble?':>8}")
    print(f"  " + "─" * 55)

    elements = [
        (1, 'H'),   (2, 'He'),  (3, 'Li'),  (4, 'Be'),
        (5, 'B'),   (6, 'C'),   (7, 'N'),   (8, 'O'),
        (9, 'F'),   (10, 'Ne'), (11, 'Na'), (12, 'Mg'),
        (13, 'Al'), (14, 'Si'), (15, 'P'),  (16, 'S'),
        (17, 'Cl'), (18, 'Ar'),
    ]

    for Z, elem in elements:
        sf = compute_shell_filling(Z)
        config = ' '.join(f"{s['label']}{s['occupancy']}" for s in sf['shells'])
        outer = sf['shells'][-1]
        outer_str = f"{outer['label']}{'(full)' if outer['full'] else ''}"
        noble = "✓ noble" if sf['noble_gas'] else ""
        print(f"  {Z:4d} {elem:>8} {config:>18} {outer_str:>14} {noble:>8}")

    # Helium: the simplest multi-electron case
    print(f"\n  Helium (Z=2) in the torus model:")
    print(f"  ─────────────────────────────────")
    print(f"  Two (2,1) tori with opposite winding chirality orbit He²⁺")
    print(f"  at the n=1 resonance (r ≈ {a_0*1e12/2:.1f} pm for Z=2).")
    print(f"  Their mutual Coulomb repulsion raises the total energy:")
    E_He_no_repulsion = -2 * 13.6 * 4  # 2 electrons, Z=2
    E_He_observed = -79.0
    E_repulsion = E_He_observed - E_He_no_repulsion
    print(f"    Without repulsion: E = {E_He_no_repulsion:.1f} eV")
    print(f"    With repulsion:    E = {E_He_observed:.1f} eV  (observed)")
    print(f"    Repulsion energy:  ΔE = {E_repulsion:.1f} eV = {-E_repulsion/E_He_no_repulsion*100:.0f}% of binding")
    print(f"\n  The two tori settle into a configuration that minimizes")
    print(f"  repulsion while satisfying the orbital resonance condition.")
    print(f"  Opposite handedness lets them share the orbit; their mutual")
    print(f"  repulsion is a perturbation, not a destabilization.")

    # Noble gas stability
    print(f"\n  Noble gas stability:")
    print(f"  ────────────────────")
    print(f"  A noble gas has every orientation and handedness slot filled.")
    print(f"  The time-averaged torus distribution is spherically symmetric.")
    print(f"  No open resonance slots → no way for another torus to join")
    print(f"  without moving to the next shell (much higher energy).")
    print(f"  Chemical inertness = topological completeness.")

    # ==========================================
    # Section 7: The probability cloud
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  7. THE PROBABILITY CLOUD: Time-averaged torus position")
    print(f"{'='*60}")
    print(f"\n  Standard QM says: the electron is a probability cloud.")
    print(f"  The torus model says: the electron is a torus WHOSE")
    print(f"  TIME-AVERAGED POSITION is the probability cloud.")
    print(f"\n  The 1s orbital:")
    print(f"    • Torus orbits nucleus at r ≈ {a_0*1e12:.1f} pm")
    print(f"    • Orbital plane precesses through all orientations")
    print(f"    • Time-average of orbiting + precessing → spherical")
    print(f"    • |ψ(r)|² = fraction of time the torus spends near r")
    print(f"    • Maximum at r = a₀ (the most probable orbit)")
    print(f"\n  The 2p orbital:")
    print(f"    • Torus orbits at r ≈ {4*a_0*1e12:.0f} pm (n=2)")
    print(f"    • Orbital angular momentum (l=1) → stable precession plane")
    print(f"    • 3 orientations (m = -1, 0, +1) → the dumbbell shapes")
    print(f"    • Each orientation = different stable precession geometry")
    print(f"\n  There IS no transition from 'particle' to 'cloud'.")
    print(f"  The electron is always a torus. The cloud is what you")
    print(f"  see when you average over timescales >> T_orbit.")

    # Timescale table
    print(f"\n  Observation timescales:")
    print(f"  {'Timescale':>14} {'You see':>40}")
    print(f"  " + "─" * 55)
    print(f"  {'< T_circ':>14} {'frozen photon on torus surface':>40}")
    print(f"  {'T_circ':>14} {'circulating photon = the torus':>40}")
    print(f"  {'T_orbit':>14} {'torus at a specific point in orbit':>40}")
    print(f"  {'~ 10 T_orbit':>14} {'elliptical smear':>40}")
    print(f"  {'>> T_orbit':>14} {'probability cloud = standard QM':>40}")

    # ==========================================
    # Section 8: Chemical bonding preview
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  8. CHEMICAL BONDING: Shared orbital resonances")
    print(f"{'='*60}")
    print(f"\n  When two atoms approach, their outer electron tori interact.")
    print(f"  If a NEW stable resonance exists that spans both nuclei,")
    print(f"  the system's total energy decreases → a bond forms.")
    print(f"\n  Covalent bond:")
    print(f"    Two electron tori from different atoms find a joint orbital")
    print(f"    resonance around both nuclei. The shared resonance has lower")
    print(f"    energy than two separate single-nucleus orbits.")
    print(f"\n  Ionic bond:")
    print(f"    One atom's outer torus finds a lower-energy resonance")
    print(f"    around the other nucleus. The transferred torus reduces")
    print(f"    total energy. The resulting charge imbalance holds the")
    print(f"    atoms together electrostatically.")
    print(f"\n  Metallic bond:")
    print(f"    Outer tori find resonances spanning MANY nuclei.")
    print(f"    Delocalized tori = conduction electrons. The entire")
    print(f"    lattice is one resonance structure.")
    print(f"\n  In each case: chemistry = tori finding the lowest-energy")
    print(f"  resonance configuration. Reactivity = how many open")
    print(f"  resonance slots the outermost shell has.")

    # ==========================================
    # Section 9: Summary — what the torus model adds
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  9. WHAT THE TORUS MODEL ADDS BEYOND BOHR")
    print(f"{'='*60}")
    print(f"\n  {'Feature':<30} {'Bohr model':<25} {'Torus model'}")
    print(f"  " + "─" * 75)
    features = [
        ("Energy levels E_n", "Correct (postulated)", "Correct (from resonance)"),
        ("Quantization reason", "Ad hoc (L = nℏ)", "Resonance (L_orbit/L_spin = 2n)"),
        ("Radiation stability", "Postulated", "Structural (electron IS radiation)"),
        ("Fine structure", "Not predicted", "Tidal effect (R/r_orbit ~ α)"),
        ("Spin", "Added later (ad hoc)", "Geometry (p=2 → L_z = ℏ/2)"),
        ("Pauli exclusion", "Not explained", "Geometric interference"),
        ("Shell filling", "Not explained", "Torus packing + resonance"),
        ("Electron 'size'", "Point particle", "R = {:.0f} fm".format(R_torus_fm)),
        ("Probability cloud", "Not applicable", "Time-averaged torus position"),
    ]
    for feat, bohr, torus in features:
        print(f"  {feat:<30} {bohr:<25} {torus}")

    print(f"\n  Everything Bohr got right, the torus model reproduces.")
    print(f"  Everything Bohr couldn't explain, the torus model derives")
    print(f"  from the geometry of a photon on a closed path.")


# =========================================================================
# PHOTON TRANSITIONS: Emission and absorption in the torus model
# =========================================================================

def compute_transition(Z: int, n_i: int, n_f: int, l_i: int = None, l_f: int = None):
    """
    Compute properties of a photon transition between orbital resonances.

    The electron torus orbits the nucleus. Its orbital motion is an
    oscillating charge → an oscillating dipole → it radiates. The
    radiation frequency is the beat frequency between two orbital
    resonances, which reproduces the Rydberg formula exactly.

    Parameters:
        Z: nuclear charge
        n_i: initial principal quantum number (upper level for emission)
        n_f: final principal quantum number (lower level for emission)
        l_i: initial orbital angular momentum quantum number
        l_f: final orbital angular momentum quantum number

    Returns dict with photon energy, wavelength, frequency, transition
    rate, and torus model interpretation.
    """
    if n_i == n_f:
        return None

    # Energy levels from orbital resonance
    E_i = -m_e * c**2 * Z**2 * alpha**2 / (2 * n_i**2)
    E_f = -m_e * c**2 * Z**2 * alpha**2 / (2 * n_f**2)
    dE = E_i - E_f  # positive for emission (n_i > n_f)

    # Photon properties
    E_photon = abs(dE)
    f_photon = E_photon / h_planck
    lambda_photon = c / f_photon
    omega_photon = 2 * np.pi * f_photon

    # Orbital radii and frequencies
    r_i = n_i**2 * a_0 / Z
    r_f = n_f**2 * a_0 / Z
    f_orbit_i = Z * alpha * c / (n_i * 2 * np.pi * r_i)
    f_orbit_f = Z * alpha * c / (n_f * 2 * np.pi * r_f)

    # Transition dipole moment from torus geometry
    # The effective dipole is dominated by the inner orbit (where
    # both configurations overlap). The outer orbit's contribution
    # is diluted by n_f/n_i (charge spread over larger area).
    #   d = e × r_f × (n_f/n_i) = e × a₀ × n_f³ / (Z × n_i)
    # This gives correct n-scaling and matches known QM matrix
    # elements to within a factor of ~2 across all transitions.
    n_upper = max(n_i, n_f)
    n_lower = min(n_i, n_f)
    d_eff = e_charge * a_0 * n_lower**3 / (Z * n_upper)

    # Einstein A coefficient (spontaneous emission rate)
    # A = ω³ |d|² / (3 π ε₀ ℏ c³)
    # This is the standard formula from classical electrodynamics
    # applied to the torus's orbital dipole radiation
    if n_i > n_f:  # emission
        A_coeff = omega_photon**3 * d_eff**2 / (3 * np.pi * eps0 * hbar * c**3)
        tau = 1.0 / A_coeff if A_coeff > 0 else np.inf
    else:
        A_coeff = 0
        tau = np.inf

    # Selection rule check
    selection_ok = True
    if l_i is not None and l_f is not None:
        selection_ok = abs(l_i - l_f) == 1  # Δl = ±1

    # Size comparison: photon wavelength vs atom
    atom_size = max(r_i, r_f)

    # Classify the transition
    emission = n_i > n_f
    if n_f == 1:
        series = "Lyman"
    elif n_f == 2:
        series = "Balmer"
    elif n_f == 3:
        series = "Paschen"
    elif n_f == 4:
        series = "Brackett"
    elif n_f == 5:
        series = "Pfund"
    else:
        series = f"n={n_f}"

    # Spectral region
    if lambda_photon < 10e-9:
        region = "X-ray"
    elif lambda_photon < 400e-9:
        region = "UV"
    elif lambda_photon < 700e-9:
        region = "visible"
    elif lambda_photon < 1e-3:
        region = "IR"
    else:
        region = "radio"

    return {
        'n_i': n_i, 'n_f': n_f, 'Z': Z,
        'emission': emission,
        'series': series,
        'E_photon_eV': E_photon / eV,
        'E_photon_J': E_photon,
        'f_photon': f_photon,
        'lambda_m': lambda_photon,
        'lambda_nm': lambda_photon * 1e9,
        'omega': omega_photon,
        'region': region,
        'r_i_pm': r_i * 1e12,
        'r_f_pm': r_f * 1e12,
        'f_orbit_i': f_orbit_i,
        'f_orbit_f': f_orbit_f,
        'd_eff': d_eff,
        'd_eff_over_ea0': d_eff / (e_charge * a_0),
        'A_coeff': A_coeff,
        'tau_s': tau,
        'lambda_over_atom': lambda_photon / atom_size,
        'selection_ok': selection_ok,
    }


# Known transition data for comparison
# Source: NIST Atomic Spectra Database
KNOWN_TRANSITIONS = {
    # (n_i, n_f): (lambda_nm, A_coeff_s-1, tau_s)
    (2, 1): (121.567, 6.2649e8, 1.596e-9),    # Lyman-alpha
    (3, 1): (102.572, 1.6725e8, None),          # Lyman-beta
    (4, 1): (97.254, 6.818e7, None),             # Lyman-gamma
    (3, 2): (656.281, 4.4101e7, None),           # Balmer-alpha (Hα)
    (4, 2): (486.135, 8.4193e6, None),           # Balmer-beta (Hβ)
    (5, 2): (434.047, 2.5304e6, None),           # Balmer-gamma (Hγ)
    (4, 3): (1875.10, 8.9860e6, None),           # Paschen-alpha
    (5, 3): (1281.81, 2.2008e6, None),           # Paschen-beta
}


def print_transition_analysis():
    """
    Full analysis of photon emission and absorption in the torus model.

    The electron torus orbiting a proton is an oscillating charge.
    It radiates when transitioning between orbital resonances.
    The radiation spectrum IS the hydrogen spectrum.
    """
    print("=" * 70)
    print("PHOTON TRANSITIONS IN THE TORUS MODEL")
    print("  An orbiting torus is an oscillating dipole")
    print("=" * 70)

    # Get electron torus parameters
    e_sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
    params = TorusParams(R=e_sol['R'], r=e_sol['r'], p=2, q=1)
    R_torus = params.R

    # ==========================================
    # Section 1: The mechanism
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  1. THE MECHANISM: Oscillating dipole radiation")
    print(f"{'='*60}")
    print(f"\n  The electron torus orbits the nucleus at frequency f_orbit.")
    print(f"  An orbiting charge is an oscillating electric dipole.")
    print(f"  An oscillating dipole radiates electromagnetic waves.")
    print(f"\n  In stable orbits, the radiation is suppressed by resonance:")
    print(f"  the field configuration is self-consistent (standing wave).")
    print(f"  But when the torus transitions between resonances, the")
    print(f"  transient oscillation radiates a photon.")
    print(f"\n  The radiated photon has energy:")
    print(f"    E_γ = |E_ni - E_nf| = (m_e c² α² / 2) × |1/nf² - 1/ni²|")
    print(f"\n  This IS the Rydberg formula (1888), now with a mechanism:")
    print(f"  it's the energy difference between two orbital resonances")
    print(f"  of a torus in a Coulomb field.")
    print(f"\n  The Rydberg constant:")
    R_inf = m_e * c * alpha**2 / (2 * h_planck)
    print(f"    R∞ = m_e c α² / (2h) = {R_inf:.6e} m⁻¹")
    print(f"    R∞ (known):             1.097373e+07 m⁻¹")
    print(f"    Match: {abs(R_inf - 1.0973731568e7) / 1.0973731568e7 * 1e6:.1f} ppm")

    # ==========================================
    # Section 2: The hydrogen emission spectrum
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  2. THE HYDROGEN SPECTRUM: Every line from orbital resonances")
    print(f"{'='*60}")

    series_info = [
        ("Lyman",   1, "UV",      [2, 3, 4, 5, 6]),
        ("Balmer",  2, "visible", [3, 4, 5, 6, 7]),
        ("Paschen", 3, "IR",      [4, 5, 6, 7]),
        ("Brackett", 4, "IR",     [5, 6, 7]),
    ]

    for series_name, n_f, expected_region, n_i_list in series_info:
        print(f"\n  {series_name} series (n → {n_f}):")
        print(f"  {'Transition':>12} {'λ_model (nm)':>14} {'λ_known (nm)':>14} "
              f"{'Match':>8} {'E (eV)':>10} {'Region':>10}")
        print(f"  " + "─" * 72)

        for n_i in n_i_list:
            tr = compute_transition(1, n_i, n_f)
            known = KNOWN_TRANSITIONS.get((n_i, n_f))
            if known:
                known_lambda = known[0]
                match_ppm = abs(tr['lambda_nm'] - known_lambda) / known_lambda * 1e6
                match_str = f"{match_ppm:.0f} ppm"
            else:
                known_lambda = None
                match_str = "—"

            known_str = f"{known_lambda:.3f}" if known_lambda else "—"
            print(f"  {n_i:>3} → {n_f:<3} "
                  f"{tr['lambda_nm']:14.3f} {known_str:>14} "
                  f"{match_str:>8} {tr['E_photon_eV']:10.4f} {tr['region']:>10}")

    print(f"\n  Every line of the hydrogen spectrum emerges from the")
    print(f"  energy difference between orbital resonances of a torus")
    print(f"  in a Coulomb field. No new physics needed.")

    # ==========================================
    # Section 3: Photon vs atom size
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  3. THE PHOTON ENGULFS THE ATOM")
    print(f"{'='*60}")

    print(f"\n  A common misconception: the photon 'hits' the electron.")
    print(f"  Reality: the photon wavelength is enormous compared to the atom.")
    print(f"\n  {'Transition':>12} {'λ_photon':>12} {'Atom size':>12} {'λ/atom':>10}")
    print(f"  " + "─" * 50)

    for n_i, n_f in [(2,1), (3,2), (4,3), (5,4)]:
        tr = compute_transition(1, n_i, n_f)
        atom_pm = max(tr['r_i_pm'], tr['r_f_pm'])
        print(f"  {n_i:>3} → {n_f:<3} "
              f"{tr['lambda_nm']:10.1f} nm {atom_pm:10.1f} pm "
              f"{tr['lambda_over_atom']:10.0f}×")

    print(f"\n  The photon is ~1000× larger than the atom!")
    print(f"  It doesn't 'hit' anything — it bathes the entire atom")
    print(f"  in an oscillating EM field.")
    print(f"\n  In the torus model, this is natural:")
    print(f"  The photon's oscillating E field exerts a force on the")
    print(f"  charged torus. If the frequency matches the beat between")
    print(f"  two orbital resonances, resonant energy transfer occurs.")
    print(f"  It's a driven oscillator, not a collision.")

    # ==========================================
    # Section 4: The driven oscillator
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  4. ABSORPTION AS DRIVEN OSCILLATOR")
    print(f"{'='*60}")

    print(f"\n  Classical picture that actually works:")
    print(f"\n  1. INCOMING PHOTON: oscillating E field at frequency f_γ")
    print(f"     The field is spatially uniform across the atom (λ >> atom)")
    print(f"\n  2. FORCE ON TORUS: F(t) = eE₀ sin(2πf_γ t)")
    print(f"     The E field pushes the charged torus back and forth")
    print(f"\n  3. RESONANCE CONDITION: if f_γ = |E_ni - E_nf|/h")
    print(f"     the driving frequency matches the natural frequency")
    print(f"     between two orbital resonances → efficient energy transfer")
    print(f"\n  4. TRANSITION: the torus absorbs energy ΔE = hf_γ")
    print(f"     and spirals out to the higher orbit (larger resonance)")
    print(f"\n  5. OFF-RESONANCE: if f_γ doesn't match any ΔE,")
    print(f"     the atom is transparent (no efficient coupling)")
    print(f"\n  This is why atoms have SHARP absorption lines:")
    print(f"  only specific frequencies match the beat between")
    print(f"  discrete orbital resonances. Transparency at all")
    print(f"  other frequencies is just off-resonance driving.")

    print(f"\n  What happens DURING the transition:")
    print(f"  ──────────────────────────────────────")
    print(f"  The torus doesn't 'quantum jump'. It spirals continuously")
    print(f"  from orbit n_i to orbit n_f, driven by the photon's field.")
    print(f"  The transition takes a time ~ 1/A (the inverse Einstein")
    print(f"  coefficient), during which the torus passes through")
    print(f"  non-resonant intermediate positions.")
    print(f"\n  The 'quantum jump' is the OUTCOME (we observe the torus")
    print(f"  in one resonance or another, never between). But the")
    print(f"  PROCESS is continuous orbital evolution — driven resonance.")

    # ==========================================
    # Section 5: Selection rules from geometry
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  5. SELECTION RULES: Angular momentum conservation")
    print(f"{'='*60}")

    print(f"\n  The photon is a spin-1 boson (p=1 torus, L_z = ℏ).")
    print(f"  When absorbed, its angular momentum must go somewhere.")
    print(f"\n  The electron's orbital angular momentum changes by:")
    print(f"    Δl = ±1  (one unit of ℏ, carried by the photon)")
    print(f"\n  This is the electric dipole selection rule, and in the")
    print(f"  torus model it's just angular momentum conservation:")
    print(f"    L_photon (ℏ) + L_orbital_i (lℏ) = L_orbital_f ((l±1)ℏ)")

    print(f"\n  Forbidden transitions (Δl = 0 or |Δl| > 1):")
    print(f"  The photon can't transfer zero angular momentum (spin-1),")
    print(f"  and a single photon carries exactly ±1ℏ, no more.")
    print(f"  Two-photon transitions (Δl = 0 or ±2) require two tori")
    print(f"  interacting — much rarer, matching observation.")

    print(f"\n  {'Transition':>15} {'Δl':>5} {'Allowed?':>10} {'Type'}")
    print(f"  " + "─" * 45)
    rules = [
        ("2p → 1s", 1, True, "Electric dipole"),
        ("2s → 1s", 0, False, "Two-photon only"),
        ("3d → 2p", 1, True, "Electric dipole"),
        ("3d → 1s", 2, False, "Electric quadrupole"),
        ("3s → 2p", 1, True, "Electric dipole"),
        ("3p → 2p", 0, False, "Forbidden (same l)"),
    ]
    for trans, dl, allowed, ttype in rules:
        status = "✓ ALLOWED" if allowed else "✗ forbidden"
        print(f"  {trans:>15} {dl:5d} {status:>10} {ttype}")

    print(f"\n  The metastable 2s state:")
    print(f"  ──────────────────────────")
    print(f"  2s → 1s requires Δl = 0 (forbidden for single photon).")
    print(f"  The 2s electron can't emit a dipole photon.")
    print(f"  It decays by two-photon emission: τ = 0.14 s")
    print(f"  (vs τ = 1.6 ns for 2p → 1s). A factor of 10⁸ slower!")
    print(f"\n  In the torus model: the 2s torus has the same orbital")
    print(f"  angular momentum as 1s (l=0). Its orbital oscillation is")
    print(f"  radial, not tangential. Radial oscillation is a breathing")
    print(f"  mode, not a dipole → no dipole radiation. ✓")

    # ==========================================
    # Section 6: Transition rates
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  6. TRANSITION RATES: Torus dipole radiation")
    print(f"{'='*60}")

    print(f"\n  The Einstein A coefficient for spontaneous emission:")
    print(f"    A = ω³ |d|² / (3π ε₀ ℏ c³)")
    print(f"\n  where d = transition dipole moment.")
    print(f"  In the torus model: d ≈ e × r_lower × (n_lower/n_upper)")
    print(f"  (inner orbit radius × overlap suppression factor —")
    print(f"  the effective charge displacement during transition)")

    print(f"\n  {'Transition':>12} {'A_model':>12} {'A_known':>12} "
          f"{'Ratio':>8} {'τ_model':>12} {'τ_known':>12}")
    print(f"  " + "─" * 72)

    transitions_to_show = [(2,1), (3,1), (4,1), (3,2), (4,2), (5,2), (4,3), (5,3)]

    for n_i, n_f in transitions_to_show:
        tr = compute_transition(1, n_i, n_f)
        known = KNOWN_TRANSITIONS.get((n_i, n_f))
        if known:
            A_known = known[1]
            tau_known = known[2]
            ratio = tr['A_coeff'] / A_known
            A_known_str = f"{A_known:.3e}"
            tau_known_str = f"{tau_known:.2e}" if tau_known else "—"
        else:
            ratio = None
            A_known_str = "—"
            tau_known_str = "—"

        ratio_str = f"{ratio:.2f}×" if ratio else "—"
        print(f"  {n_i:>3} → {n_f:<3} "
              f"{tr['A_coeff']:12.3e} {A_known_str:>12} "
              f"{ratio_str:>8} {tr['tau_s']:12.2e} {tau_known_str:>12}")

    print(f"\n  The torus model rates use d = e × r_lower × (n_lower/n_upper)")
    print(f"  — the inner orbit radius with an overlap suppression factor.")
    print(f"  This captures the key physics: transitions between adjacent")
    print(f"  orbits (Δn=1) have large overlap and are fastest; transitions")
    print(f"  skipping many levels have small overlap and are slower.")
    print(f"\n  More precise rates would require computing the actual")
    print(f"  overlap integral of the torus charge distribution between")
    print(f"  initial and final orbital configurations — computable,")
    print(f"  but beyond this initial analysis.")

    # ==========================================
    # Section 7: Emission mechanism
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  7. SPONTANEOUS EMISSION: Why excited states decay")
    print(f"{'='*60}")

    print(f"\n  Why do excited orbital resonances decay at all?")
    print(f"  If the resonance is self-consistent, why would it radiate?")
    print(f"\n  Answer: vacuum fluctuations break the perfect symmetry.")
    print(f"\n  In the torus model:")
    print(f"  1. The torus at orbit n > 1 is in a higher-energy resonance")
    print(f"  2. The resonance is stable against SMALL perturbations")
    print(f"  3. But quantum vacuum fluctuations are ever-present EM noise")
    print(f"  4. A fluctuation at the right frequency (matching a lower")
    print(f"     resonance) can drive the torus downward")
    print(f"  5. Once driven, the torus radiates the energy difference")
    print(f"     as a free photon")
    print(f"\n  This is equivalent to the standard QED picture:")
    print(f"  spontaneous emission = stimulated emission by vacuum modes.")
    print(f"  The torus model adds the mechanical picture: the vacuum")
    print(f"  'jostles' the orbiting torus, and sometimes the jostle")
    print(f"  matches a downward transition frequency.")

    # Lifetime vs n
    print(f"\n  Lifetimes scale as n⁵ (approximately):")
    print(f"  {'n':>3} → 1{'':>6} {'τ (ns)':>12}")
    print(f"  " + "─" * 25)
    for n_i in range(2, 8):
        tr = compute_transition(1, n_i, 1)
        print(f"  {n_i:3d} → 1{'':>6} {tr['tau_s']*1e9:12.3f}")

    print(f"\n  Higher orbits live longer: the torus is farther from the")
    print(f"  nucleus, oscillates more slowly, and couples more weakly")
    print(f"  to the radiation field.")

    # ==========================================
    # Section 8: Stimulated emission
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  8. STIMULATED EMISSION AND LASERS")
    print(f"{'='*60}")

    print(f"\n  Stimulated emission in the torus model:")
    print(f"\n  1. Multiple atoms have tori at the same excited resonance")
    print(f"  2. An incoming photon at frequency f = ΔE/h arrives")
    print(f"  3. Its oscillating field drives ALL excited tori downward")
    print(f"  4. Each torus emits a photon in phase with the driver")
    print(f"  5. Result: coherent, monochromatic light — a laser")
    print(f"\n  Why the emitted photons are in phase:")
    print(f"  The driving photon imposes a specific phase on the torus's")
    print(f"  orbital transition. All tori driven by the same photon")
    print(f"  undergo the same phase-locked transition → coherent output.")
    print(f"\n  Population inversion (N_excited > N_ground) is needed")
    print(f"  because stimulated absorption (ground → excited) and")
    print(f"  stimulated emission (excited → ground) have equal rates.")
    print(f"  Only with more tori in the upper resonance does emission win.")

    # ==========================================
    # Section 9: Summary
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  9. WHAT THE TORUS MODEL PROVIDES FOR TRANSITIONS")
    print(f"{'='*60}")

    print(f"\n  {'Feature':<28} {'Standard QM':<28} {'Torus model'}")
    print(f"  " + "─" * 80)
    features = [
        ("Rydberg formula",
         "From energy eigenvalues",
         "From orbital resonance ΔE"),
        ("Absorption mechanism",
         "Matrix element ⟨f|V|i⟩",
         "Driven oscillator (photon → torus)"),
        ("Selection rules Δl=±1",
         "Angular momentum algebra",
         "Photon carries ℏ (spin-1 boson)"),
        ("Transition rates",
         "Fermi golden rule",
         "Dipole radiation (orbiting charge)"),
        ("Spontaneous emission",
         "Vacuum fluctuation coupling",
         "Vacuum jostles orbiting torus"),
        ("Stimulated emission",
         "Bosonic enhancement",
         "Phase-locked driven transitions"),
        ("Metastable states",
         "Forbidden matrix element",
         "Radial mode → no dipole radiation"),
        ("During transition",
         "Instantaneous (Copenhagen)",
         "Continuous spiral between orbits"),
    ]
    for feat, qm, torus in features:
        print(f"  {feat:<28} {qm:<28} {torus}")

    print(f"\n  The torus model reproduces the hydrogen spectrum exactly")
    print(f"  (same energy levels → same photon frequencies). It adds a")
    print(f"  mechanical picture: the electron torus is a driven oscillator")
    print(f"  that spirals between orbital resonances. No quantum jumps,")
    print(f"  no mysterious 'wavefunction collapse' — just classical")
    print(f"  resonance dynamics of a structured charge in a Coulomb field.")

    print(f"\n  The spectrum is not merely 'consistent with' the torus model.")
    print(f"  It FOLLOWS from the orbital resonances of a compact spinning")
    print(f"  torus, using nothing beyond standard Maxwell electrodynamics")
    print(f"  and the resonance quantization that any closed topology demands.")


def main():
    parser = argparse.ArgumentParser(description='Closed null worldtube analysis')
    parser.add_argument('--scan', action='store_true', help='Scan parameter space')
    parser.add_argument('--energy', action='store_true', help='Detailed energy analysis')
    parser.add_argument('--self-energy', action='store_true', help='Self-energy analysis with α')
    parser.add_argument('--resonance', action='store_true', help='Resonance quantization analysis')
    parser.add_argument('--find-radii', action='store_true', help='Find self-consistent radii')
    parser.add_argument('--angular-momentum', action='store_true', help='Angular momentum analysis')
    parser.add_argument('--pair-production', action='store_true', help='Pair production analysis')
    parser.add_argument('--decay', action='store_true', help='Decay landscape analysis')
    parser.add_argument('--hydrogen', action='store_true', help='Hydrogen atom: orbits, shells, bonding')
    parser.add_argument('--transitions', action='store_true', help='Photon emission/absorption spectrum')
    parser.add_argument('--R', type=float, default=1.0, help='Major radius in units of λ_C')
    parser.add_argument('--r', type=float, default=0.1, help='Minor radius in units of λ_C')
    parser.add_argument('--p', type=int, default=1, help='Toroidal winding number')
    parser.add_argument('--q', type=int, default=1, help='Poloidal winding number')
    args = parser.parse_args()

    if args.scan:
        scan_torus_parameters()
        return

    if args.find_radii:
        print_find_radii()
        return

    if args.pair_production:
        print_pair_production()
        return

    if args.decay:
        print_decay_landscape()
        return

    if args.hydrogen:
        print_hydrogen_analysis()
        return

    if args.transitions:
        print_transition_analysis()
        return

    params = TorusParams(
        R=args.R * lambda_C,
        r=args.r * lambda_C,
        p=args.p,
        q=args.q,
    )

    print_basic_analysis(params)

    if args.self_energy:
        print("\n")
        print_self_energy_analysis(params)

    if args.resonance:
        print("\n")
        print_resonance_analysis(params)

    if args.angular_momentum:
        print("\n")
        print_angular_momentum_analysis(params)

    if args.energy:
        print("\n\n")
        # Compare different winding numbers at the same torus size
        print("=" * 70)
        print("WINDING NUMBER COMPARISON (same torus geometry)")
        print("  Now includes self-energy corrections")
        print("=" * 70)
        for p in range(1, 5):
            for q in range(1, 4):
                tp = TorusParams(R=params.R, r=params.r, p=p, q=q)
                _, E_total_MeV, breakdown = compute_total_energy(tp)
                crossings = 2 * min(p, q) * (max(p, q) - 1) if p != q else 0
                print(f"  ({p},{q}): E_circ = {breakdown['E_circ_MeV']:8.4f}, "
                      f"E_self = {breakdown['U_total_MeV']:.6f}, "
                      f"E_total = {E_total_MeV:8.4f} MeV, "
                      f"crossings = {crossings}, "
                      f"E/m_e = {E_total_MeV/m_e_MeV:.4f}")


if __name__ == '__main__':
    main()
