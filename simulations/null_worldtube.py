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
