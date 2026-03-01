#!/usr/bin/env python3
"""
Nonlinear vacuum permittivity profile for Coulomb Z=92.

Tests whether the Euler-Heisenberg nonlinear permittivity transition
sets the minor radius (little r) of the NWT torus. Pure analytical —
no numerical grids, just evaluating closed-form expressions.

Key question: does the transition width of ε_eff(r) scale as α λ_C = r_e?

Usage:
    python3 simulations/nonlinear_permittivity.py
"""

import numpy as np

# Constants
c = 2.99792458e8
hbar = 1.054571817e-34
e_charge = 1.602176634e-19
eps0 = 8.8541878128e-12
m_e = 9.1093837015e-31
alpha = 7.2973525693e-3
eV = 1.602176634e-19
MeV = 1e6 * eV
lambda_C = hbar / (m_e * c)
r_e = alpha * lambda_C
a_0 = lambda_C / alpha
k_e = 1.0 / (4 * np.pi * eps0)
E_S = m_e**2 * c**3 / (e_charge * hbar)

Z = 92

print("=" * 70)
print("  NONLINEAR VACUUM PERMITTIVITY — COULOMB Z=92")
print("  Does the EH permittivity transition set little r?")
print("=" * 70)
print()

# Reference scales
print("  Reference scales:")
print(f"    lambda_C  = {lambda_C:.4e} m")
print(f"    r_e       = {r_e:.4e} m  = {r_e/lambda_C:.6f} lambda_C")
print(f"    alpha     = {alpha:.6f}")
print(f"    Z*alpha   = {Z*alpha:.4f}")
print(f"    E_S       = {E_S:.3e} V/m")
print()

# ──────────────────────────────────────────────────────────────
# Section 1: Permittivity profile ε_eff(r)
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  1. PERMITTIVITY PROFILE ε_eff(r)")
print("─" * 70)
print()
print("  EH nonlinear correction to vacuum permittivity:")
print()
print("    ε_eff = ε₀ [1 + (8α²/45)(E/E_S)² + ...]")
print()
print("  For Coulomb E = k_e Ze/r²:")
print()
print("    δε/ε₀ = (8α²/45)(Zα)²(λ_C/r)⁴")
print()

# Evaluate at a range of radii
r_over_lC = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20,
                       0.30, 0.50, 0.57, 0.80, 1.0, 2.0, 5.0])

print(f"    {'r/λ_C':>8s}  {'r/r_e':>8s}  {'E/E_S':>10s}  {'δε/ε₀':>12s}  {'ε_eff/ε₀':>10s}")
print(f"    {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*10}")

for x in r_over_lC:
    r = x * lambda_C
    E_over_ES = Z * alpha * (1.0 / x)**2
    delta_eps = (8.0 * alpha**2 / 45.0) * E_over_ES**2
    eps_ratio = 1.0 + delta_eps
    r_over_re = x / alpha  # r/r_e = (r/lambda_C) / alpha
    print(f"    {x:8.3f}  {r_over_re:8.1f}  {E_over_ES:10.4f}  {delta_eps:12.4e}  {eps_ratio:10.6f}")

print()

# ──────────────────────────────────────────────────────────────
# Section 2: Where does the nonlinear correction matter?
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  2. TRANSITION REGION — WHERE δε/ε₀ TRANSITIONS")
print("─" * 70)
print()

# Solve for specific thresholds
# δε/ε₀ = (8α²/45)(Zα)⁴(λ_C/r)⁴ = threshold
# r/λ_C = [(8α²/45)(Zα)²]^(1/4) / threshold^(1/4)

prefactor = (8.0 * alpha**2 / 45.0) * (Z * alpha)**2
# δε/ε₀ = prefactor * (λ_C/r)⁴

print(f"  δε/ε₀ = {prefactor:.4e} × (λ_C/r)⁴")
print()

thresholds = [1.0, 0.1, 0.01, 0.001, 1e-4, 1e-6]
print(f"    {'δε/ε₀':>10s}  {'r/λ_C':>8s}  {'r/r_e':>8s}  {'Meaning'}")
print(f"    {'─'*10}  {'─'*8}  {'─'*8}  {'─'*30}")
for th in thresholds:
    r_lC = (prefactor / th)**0.25
    r_re = r_lC / alpha
    if th >= 0.1:
        meaning = "strongly nonlinear"
    elif th >= 0.01:
        meaning = "nonlinear corrections significant"
    elif th >= 0.001:
        meaning = "perturbative but measurable"
    elif th >= 1e-4:
        meaning = "weakly nonlinear"
    else:
        meaning = "effectively linear"
    print(f"    {th:10.1e}  {r_lC:8.4f}  {r_re:8.1f}  {meaning}")
print()

# ──────────────────────────────────────────────────────────────
# Section 3: Gradient of permittivity — transition width
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  3. PERMITTIVITY GRADIENT — TRANSITION WIDTH")
print("─" * 70)
print()
print("  The transition width is set by where d(δε/ε₀)/dr is steepest.")
print()
print("  d(δε/ε₀)/dr = -4 × prefactor × λ_C⁴ / r⁵ = -4(δε/ε₀)/r")
print()
print("  But we want the logarithmic gradient (fractional change per unit r):")
print()
print("    d ln(δε/ε₀) / d(r/λ_C) = -4/(r/λ_C)")
print()
print("  This is scale-free! The fractional change is always 4/(r/λ_C) per λ_C.")
print("  The permittivity profile is a pure power law — there's no natural")
print("  transition width from the EH correction alone.")
print()
print("  BUT: the EH series breaks down when δε/ε₀ ~ O(1), which is where")
print("  the full nonperturbative Schwinger physics takes over. So the")
print("  transition has two regimes:")
print()
print("    r > r_NL: perturbative EH (power law)")
print("    r < r_NL: nonperturbative (Schwinger pair creation)")
print()

r_NL = (prefactor)**0.25  # where δε/ε₀ = 1
print(f"  r_NL (where δε/ε₀ = 1) = {r_NL:.4f} λ_C = {r_NL/alpha:.1f} r_e")
print()

# ──────────────────────────────────────────────────────────────
# Section 4: The real transition — EH breakdown scale
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  4. THE REAL TRANSITION — EH BREAKDOWN SCALE")
print("─" * 70)
print()
print("  The EH Lagrangian is a WEAK-FIELD expansion in powers of (E/E_S)².")
print("  It converges only for E < E_S. The full Heisenberg-Euler result is")
print("  an integral over proper time, valid for all E, but the polynomial")
print("  expansion breaks down at E ~ E_S.")
print()
print("  For Coulomb Z=92, E = E_S at:")

r_ES = lambda_C * np.sqrt(Z * alpha)  # E/E_S = Zα(λ_C/r)² = 1 => r = λ_C√(Zα)
print(f"    r(E=E_S) = λ_C √(Zα) = {r_ES/lambda_C:.4f} λ_C = {r_ES/r_e:.1f} r_e")
print()

print("  Compare the key radii:")
print()
print(f"    {'Radius':30s}  {'r/λ_C':>8s}  {'r/r_e':>8s}")
print(f"    {'─'*30}  {'─'*8}  {'─'*8}")
print(f"    {'r_e (classical electron)':30s}  {alpha:8.4f}  {1.0:8.1f}")
print(f"    {'r(δε/ε₀ = 1) [EH breakdown]':30s}  {r_NL:8.4f}  {r_NL/alpha:8.1f}")
print(f"    {'r(E = E_S) [Schwinger onset]':30s}  {r_ES/lambda_C:8.4f}  {r_ES/r_e:8.1f}")
print(f"    {'r_peak (pair emergence)':30s}  {0.57:8.4f}  {0.57/alpha:8.1f}")
print(f"    {'λ_C':30s}  {1.0:8.4f}  {1.0/alpha:8.1f}")
print()

# ──────────────────────────────────────────────────────────────
# Section 5: Transition width estimate
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  5. TRANSITION WIDTH — THE MINOR RADIUS ESTIMATE")
print("─" * 70)
print()
print("  The vacuum permittivity has three regimes:")
print()
print("    I.   r >> r_NL: linear vacuum (δε/ε₀ << 1)")
print("    II.  r ~ r_NL:  nonlinear transition (δε/ε₀ ~ 1)")
print("    III. r << r_NL: Schwinger regime (EH expansion fails)")
print()
print("  The transition from I to III defines a 'skin depth' of the")
print("  nonlinear vacuum region. Since δε/ε₀ ~ r^(-4), the profile")
print("  is steep. The width over which δε/ε₀ goes from 0.1 to 10:")
print()

r_01 = (prefactor / 0.1)**0.25    # δε/ε₀ = 0.1
r_10 = (prefactor / 10.0)**0.25   # δε/ε₀ = 10
width = r_01 - r_10

print(f"    r(δε/ε₀ = 0.1)  = {r_01:.4f} λ_C")
print(f"    r(δε/ε₀ = 10)   = {r_10:.4f} λ_C")
print(f"    Transition width = {width:.4f} λ_C = {width/alpha:.1f} r_e")
print()

# Also check 0.01 to 100
r_001 = (prefactor / 0.01)**0.25
r_100 = (prefactor / 100.0)**0.25
width2 = r_001 - r_100

print(f"    r(δε/ε₀ = 0.01) = {r_001:.4f} λ_C")
print(f"    r(δε/ε₀ = 100)  = {r_100:.4f} λ_C")
print(f"    Transition width = {width2:.4f} λ_C = {width2/alpha:.1f} r_e")
print()

# The ratio of outer/inner for a factor of 100 in δε/ε₀
# (prefactor/0.1)^0.25 / (prefactor/10)^0.25 = (10/0.1)^0.25 = 100^0.25 = 3.16
print("  Scale ratio for 2 decades: (r_outer/r_inner) = 100^(1/4) = 3.16")
print("  Scale ratio for 4 decades: (r_outer/r_inner) = 10000^(1/4) = 10.0")
print()
print("  The r^(-4) power law means each decade of δε/ε₀ spans a factor")
print(f"  of 10^(1/4) = {10**0.25:.2f} in radius. The transition is STEEP.")
print()

# ──────────────────────────────────────────────────────────────
# Section 6: Connection to r_e
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  6. CONNECTION TO r_e")
print("─" * 70)
print()

# The EH breakdown radius
print(f"  EH breakdown radius: r_NL = {r_NL:.4f} λ_C")
print(f"  Classical electron radius: r_e = α λ_C = {alpha:.4f} λ_C")
print(f"  Ratio: r_NL / r_e = {r_NL/alpha:.2f}")
print()
print(f"  Analytically: r_NL = [(8α²/45)(Zα)²]^(1/4) λ_C")
print(f"              = (8/45)^(1/4) × α^(1/2) × (Zα)^(1/2) × λ_C")

analytic_prefactor = (8.0/45.0)**0.25
print(f"              = {analytic_prefactor:.4f} × α^(1/2) × (Zα)^(1/2) × λ_C")
print(f"              = {analytic_prefactor:.4f} × {alpha**0.5:.4f} × {(Z*alpha)**0.5:.4f} × λ_C")

r_NL_check = analytic_prefactor * alpha**0.5 * (Z*alpha)**0.5 * lambda_C
print(f"              = {r_NL_check/lambda_C:.4f} λ_C  ✓")
print()

# Express in terms of r_e
# r_NL / r_e = r_NL / (α λ_C) = (8/45)^(1/4) × (Zα)^(1/2) / α^(1/2)
#            = (8/45)^(1/4) × (Z)^(1/2) × α^(1/2) / α^(1/2) ... wait
# r_NL = (8/45)^(1/4) × α^(1/2) × (Zα)^(1/2) × λ_C
# r_e = α × λ_C
# r_NL / r_e = (8/45)^(1/4) × (Zα)^(1/2) / α^(1/2)
#            = (8/45)^(1/4) × Z^(1/2) × α^(1/2) / α^(1/2)
# Hmm let me just compute:
# r_NL / r_e = (8/45)^(1/4) × α^(1/2) × (Zα)^(1/2) / α
#            = (8/45)^(1/4) × (Zα)^(1/2) / α^(1/2)
#            = (8/45)^(1/4) × (Z)^(1/2)
ratio_analytic = analytic_prefactor * Z**0.5
print(f"  r_NL / r_e = (8/45)^(1/4) × Z^(1/2) = {analytic_prefactor:.3f} × {Z**0.5:.2f} = {ratio_analytic:.1f}")
print()
print(f"  So r_NL ~ {ratio_analytic:.0f} r_e for Z=92.")
print()
print(f"  The EH breakdown scale is NOT r_e itself — it's about an order")
print(f"  of magnitude larger. But it depends on Z: for a bare electron (Z=1),")
r_NL_Z1 = (8.0 * alpha**2 / 45.0 * alpha**2)**0.25
ratio_Z1 = analytic_prefactor * 1.0**0.5
print(f"  r_NL/r_e = {ratio_Z1:.2f} — almost exactly r_e!")
print()

# ──────────────────────────────────────────────────────────────
# Section 7: Self-consistent picture
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  7. SELF-CONSISTENT PICTURE")
print("─" * 70)
print()
print("  For the ELECTRON'S OWN FIELD (Z=1, self-field):")
print()

prefactor_Z1 = (8.0 * alpha**2 / 45.0) * alpha**2
r_NL_Z1_lC = prefactor_Z1**0.25
print(f"    δε/ε₀ = {prefactor_Z1:.4e} × (λ_C/r)⁴")
print(f"    r_NL(Z=1) = {r_NL_Z1_lC:.6f} λ_C")
print(f"    r_e       = {alpha:.6f} λ_C")
print(f"    Ratio     = {r_NL_Z1_lC/alpha:.3f}")
print()
print("  For Z=1, the EH breakdown radius is 0.63 r_e.")
print("  The nonlinear vacuum transition happens RIGHT AT r_e.")
print()
print("  This is the self-consistency argument:")
print("  The electron's own field makes the vacuum nonlinear at r ~ r_e.")
print("  If the torus minor radius is set by where the vacuum transitions")
print("  from linear to nonlinear response, then little r ~ r_e follows")
print("  from the EH Lagrangian alone — no free parameters.")
print()
print("  The Z=92 analysis gives r_NL ~ 6 r_e because the nuclear Coulomb")
print("  field is 92× stronger. But the ELECTRON doesn't know about Z.")
print("  Its own self-field sets its own structure at r_e.")
print()

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print("─" * 70)
print("  8. SUMMARY")
print("─" * 70)
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print(f"  │  EH breakdown (Z=92): r_NL = {r_NL:.4f} λ_C = {r_NL/alpha:.1f} r_e"
      f"          │")
print(f"  │  EH breakdown (Z=1):  r_NL = {r_NL_Z1_lC:.4f} λ_C = {r_NL_Z1_lC/alpha:.2f} r_e"
      f"       │")
print(f"  │  Transition width (2 dec): {width:.4f} λ_C = {width/alpha:.1f} r_e"
      f"          │")
print(f"  │  Power law: δε/ε₀ ~ r^(-4)  (steep transition)"
      f"              │")
print(f"  │                                                             │")
print(f"  │  KEY INSIGHT:                                               │")
print(f"  │  For Z=1 (electron self-field), the vacuum becomes          │")
print(f"  │  nonlinear at r ~ 0.63 r_e. The minor radius of the        │")
print(f"  │  torus is set by the electron's OWN nonlinear vacuum        │")
print(f"  │  footprint, not the external Coulomb source.                │")
print(f"  └──────────────────────────────────────────────────────────────┘")
print()
print("=" * 70)
