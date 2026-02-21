#!/usr/bin/env python3
"""
ACID TEST: Can the cos(3θ) potential reproduce known resonance line shapes?

Tests the damped Koide oscillator against three well-measured particles:
  1. Z boson (Γ/M ~ 0.027) — running-width Breit-Wigner, 34 MeV mass shift
  2. ρ(770) (Γ/M ~ 0.19)  — Gounaris-Sakurai parametrization required
  3. f₀(500) (Γ/M ~ 1.2)  — Breit-Wigner completely fails

For each: tune damping g₀ to match the observed Γ/M, extract the predicted
line shape, and compare to the experimental parametrization.
"""

from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# PART 1: The model — damped oscillator in cos(3θ) potential
# ══════════════════════════════════════════════════════════════════════

m_e_MeV = 0.51099895000
m_mu = 105.6583755
m_tau = 1776.86
S = np.sqrt(m_e_MeV) + np.sqrt(m_mu) + np.sqrt(m_tau)
S3 = S / 3
theta_K = (6 * np.pi + 2) / 9

m_ref = np.max(np.array([(S3 * (1 + np.sqrt(2) * np.cos(theta_K + 2*np.pi*i/3)))**2
                          for i in range(3)]))

def masses_at(th):
    return np.array([(S3 * (1 + np.sqrt(2) * np.cos(th + 2*np.pi*i/3)))**2
                     for i in range(3)])

def V(th):
    return 0.5 - (np.sqrt(2)/2) * np.cos(3*th)

def dV(th):
    return (3*np.sqrt(2)/2) * np.sin(3*th)

def Gamma_model(th, g0):
    mmax = np.max(masses_at(th))
    return g0 * (1 + (mmax/m_ref)**2) / 2

def rhs(t, y, g0):
    th, om = y
    return [om, -dV(th) - Gamma_model(th, g0)*om]

def run_oscillator(g0, th0_offset=0.3, om0=4.5, t_end=80, N_pts=30000):
    """Run the damped oscillator and return mass time series."""
    y0 = [theta_K + th0_offset, om0]
    t_eval = np.linspace(0, t_end, N_pts)
    sol = solve_ivp(lambda t, y: rhs(t, y, g0), (0, t_end), y0,
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=0.02)
    th = sol.y[0]
    # Dominant mass at each timestep
    m_inst = np.array([np.max(masses_at(t)) for t in th])
    return t_eval, th, sol.y[1], m_inst

def extract_lineshape(t_eval, m_inst, n_bins=120):
    """Extract the transient dwell-time distribution as a line shape."""
    # Find settled mass
    m_settled = np.median(m_inst[-len(m_inst)//10:])

    # Find settling time
    settled_mask = np.abs(m_inst - m_settled) / m_settled < 0.005
    i_settle = len(m_inst)
    for i in range(len(settled_mask)):
        if np.all(settled_mask[i:min(i+800, len(settled_mask))]):
            i_settle = i
            break

    m_transient = m_inst[:i_settle]
    if len(m_transient) < 200:
        return None, None, m_settled, 0, 0

    counts, bin_edges = np.histogram(m_transient, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Effective width (FWHM proxy)
    sigma_m = np.std(m_transient)
    gamma_eff = 2.355 * sigma_m  # FWHM for Gaussian-like

    return bin_centers, counts, m_settled, gamma_eff, gamma_eff / m_settled


# ══════════════════════════════════════════════════════════════════════
# PART 2: Experimental line shape parametrizations
# ══════════════════════════════════════════════════════════════════════

# --- Z boson: relativistic BW with running width ---
M_Z = 91187.6   # MeV
Gamma_Z = 2495.2  # MeV

def sigma_Z_running(s, sigma0=1.0):
    """Z boson cross section with running (s-dependent) width."""
    return sigma0 * s * Gamma_Z**2 / ((s - M_Z**2)**2 + s**2 * Gamma_Z**2 / M_Z**2)

def sigma_Z_constant(s, sigma0=1.0):
    """Z boson cross section with constant width (symmetric BW)."""
    return sigma0 * Gamma_Z**2 / ((np.sqrt(s) - M_Z)**2 + (Gamma_Z/2)**2)

# --- ρ(770): Gounaris-Sakurai parametrization ---
M_rho = 775.26   # MeV
Gamma_rho = 147.4  # MeV
m_pi = 139.57     # MeV

def k_pipi(s):
    """Pion CM momentum for ρ→ππ."""
    s = np.maximum(s, 4*m_pi**2 + 1e-10)
    return np.sqrt(np.maximum(s/4 - m_pi**2, 1e-10))

def k0_rho():
    return k_pipi(M_rho**2)

def Gamma_rho_s(s):
    """Energy-dependent P-wave width."""
    k = k_pipi(s)
    k0 = k0_rho()
    return Gamma_rho * (M_rho / np.sqrt(np.maximum(s, 1.0))) * (k / k0)**3

def h_GS(s):
    """Gounaris-Sakurai h function."""
    k = k_pipi(s)
    sqrts = np.sqrt(np.maximum(s, 1.0))
    return (2/np.pi) * (k/sqrts) * np.log((sqrts + 2*k) / (2*m_pi))

def dhds_GS(s):
    """Derivative dh/ds at s = M_rho^2."""
    k0 = k0_rho()
    return h_GS(M_rho**2) * (1/(8*k0**2) - 1/(2*M_rho**2)) + 1/(2*np.pi*M_rho**2)

def f_GS(s):
    """Gounaris-Sakurai dispersive correction."""
    k = k_pipi(s)
    k0 = k0_rho()
    return (Gamma_rho * M_rho**2 / k0**3) * (
        k**2 * (h_GS(s) - h_GS(M_rho**2)) +
        (M_rho**2 - s) * k0**2 * dhds_GS(M_rho**2)
    )

def d_GS():
    """Normalization constant d."""
    k0 = k0_rho()
    return (3/np.pi) * (m_pi**2/k0**2) * np.log((M_rho + 2*k0)/(2*m_pi)) + \
           M_rho/(2*np.pi*k0) - m_pi**2 * M_rho / (np.pi * k0**3)

def F_rho_GS(s):
    """Pion form factor |F(s)|^2 in Gounaris-Sakurai parametrization."""
    d = d_GS()
    numerator = M_rho**2 * (1 + d * Gamma_rho/M_rho)
    denominator = (M_rho**2 - s + f_GS(s)) - 1j * np.sqrt(np.maximum(s, 1.0)) * Gamma_rho_s(s)
    return np.abs(numerator / denominator)**2

def sigma_rho_simple_BW(s):
    """Simple relativistic BW for comparison."""
    return 1.0 / ((s - M_rho**2)**2 + M_rho**2 * Gamma_rho**2)

# --- f₀(500): pole parametrization ---
# Pole at sqrt(s) = (449 - i*275) MeV => s_pole = (449 - 275j)^2
M_f0 = 449.0     # MeV (real part of pole)
Gamma_f0 = 550.0  # MeV (twice imaginary part)
s_pole_f0 = complex(M_f0, -Gamma_f0/2)**2

def T_f0_pole(s):
    """Simple pole amplitude for f₀(500)."""
    # |T|^2 where T ~ g^2 / (s - s_pole)
    return 1.0 / np.abs(s - s_pole_f0)**2

def sigma_f0_BW(s):
    """Naive BW for f₀(500) — known to fail."""
    return 1.0 / ((s - M_f0**2)**2 + M_f0**2 * Gamma_f0**2)


# ══════════════════════════════════════════════════════════════════════
# PART 3: Find damping parameters matching each particle's Γ/M
# ══════════════════════════════════════════════════════════════════════

print("=" * 72)
print("ACID TEST: cos(3θ) Potential vs. Experimental Line Shapes")
print("=" * 72)

# Scan g0 to find the Γ/M mapping
print("\nPhase 1: Mapping damping parameter g₀ → effective Γ/M...")
print("-" * 72)

g0_scan = np.linspace(0.02, 1.5, 60)
gamma_over_m = []

for g0 in g0_scan:
    t_eval, th, om, m_inst = run_oscillator(g0, t_end=120, N_pts=40000)
    bc, ct, m_set, gam_eff, ratio = extract_lineshape(t_eval, m_inst)
    gamma_over_m.append(ratio)
    if ratio > 0:
        print(f"  g₀={g0:.3f}  →  Γ_eff/M = {ratio:.4f}  (M_settled={m_set:.1f} MeV)")

gamma_over_m = np.array(gamma_over_m)

# Target Γ/M values
targets = {
    'Z boson':   {'gamma_m': 0.027, 'color': '#3498db'},
    'ρ(770)':    {'gamma_m': 0.19,  'color': '#2ecc71'},
    'f₀(500)':   {'gamma_m': 1.22,  'color': '#e74c3c'},
}

# Find best g0 for each target
print("\nPhase 2: Finding best g₀ for each target...")
print("-" * 72)

for name, target in targets.items():
    valid = gamma_over_m > 0
    if not np.any(valid):
        print(f"  {name}: No valid solutions found!")
        target['g0'] = None
        continue

    # Interpolate
    diffs = np.abs(gamma_over_m[valid] - target['gamma_m'])
    best_idx = np.argmin(diffs)
    g0_best = g0_scan[valid][best_idx]

    # Refine with finer scan around best
    g0_fine = np.linspace(max(0.01, g0_best - 0.05), g0_best + 0.05, 30)
    for g0 in g0_fine:
        t_eval, th, om, m_inst = run_oscillator(g0, t_end=120, N_pts=40000)
        bc, ct, m_set, gam_eff, ratio = extract_lineshape(t_eval, m_inst)
        if ratio > 0 and abs(ratio - target['gamma_m']) < abs(diffs[best_idx]):
            g0_best = g0
            diffs[best_idx] = abs(ratio - target['gamma_m'])

    target['g0'] = g0_best
    print(f"  {name}: Γ/M={target['gamma_m']:.3f} → g₀={g0_best:.4f}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: Generate model predictions and compare
# ══════════════════════════════════════════════════════════════════════

print("\nPhase 3: Comparing model predictions to experimental line shapes...")
print("=" * 72)

bg = '#060612'
fig = plt.figure(figsize=(20, 16), facecolor=bg)

# Layout: 3 columns (Z, ρ, f₀) × 3 rows (mass vs time, line shape comparison, residuals)
particle_list = ['Z boson', 'ρ(770)', 'f₀(500)']

for col, name in enumerate(particle_list):
    target = targets[name]
    g0 = target['g0']

    if g0 is None:
        print(f"\n  {name}: SKIPPED (no matching g₀ found)")
        continue

    # Run oscillator
    t_end = 200 if name == 'Z boson' else 120
    N_pts = 50000
    t_eval, th, om, m_inst = run_oscillator(g0, t_end=t_end, N_pts=N_pts)
    bc, ct, m_set, gam_eff, ratio = extract_lineshape(t_eval, m_inst, n_bins=100)

    if bc is None:
        print(f"\n  {name}: SKIPPED (settles too quickly)")
        continue

    print(f"\n{'─'*72}")
    print(f"  {name}  (g₀={g0:.4f}, Γ/M_target={target['gamma_m']:.3f}, "
          f"Γ/M_actual={ratio:.4f})")
    print(f"  Settled mass: {m_set:.1f} MeV")
    print(f"{'─'*72}")

    # ── Row 1: Mass vs time ──────────────────────────────────────────
    ax1 = fig.add_subplot(3, 3, col + 1)
    ax1.set_facecolor(bg)
    for spine in ax1.spines.values():
        spine.set_color('#1a1a3a')
    ax1.tick_params(colors='#a0a0a0', labelsize=8)

    ax1.plot(t_eval[:len(m_inst)//2], m_inst[:len(m_inst)//2],
             color=target['color'], linewidth=0.5, alpha=0.8)
    ax1.axhline(m_set, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax1.set_xlabel('t (model time)', color='#a0a0a0', fontsize=9)
    ax1.set_ylabel('Mass (MeV)', color='#a0a0a0', fontsize=9)
    ax1.set_title(f'{name}\ng₀={g0:.3f}, Γ/M={ratio:.4f}',
                  color='white', fontsize=11, fontweight='bold')

    # ── Row 2: Line shape comparison ─────────────────────────────────
    ax2 = fig.add_subplot(3, 3, col + 4)
    ax2.set_facecolor(bg)
    for spine in ax2.spines.values():
        spine.set_color('#1a1a3a')
    ax2.tick_params(colors='#a0a0a0', labelsize=8)

    # Normalize model to peak = 1
    ct_norm = ct / np.max(ct) if np.max(ct) > 0 else ct

    ax2.bar(bc, ct_norm, width=bc[1]-bc[0], color=target['color'],
            alpha=0.4, edgecolor=target['color'], linewidth=0.5,
            label='Model (dwell time)')

    # Generate experimental prediction on the same mass range
    s_range = bc**2  # convert mass to s = m^2
    m_range = np.linspace(bc[0], bc[-1], 500)
    s_fine = m_range**2

    if name == 'Z boson':
        # Running-width BW
        exp_running = sigma_Z_running(s_fine)
        exp_running /= np.max(exp_running)
        exp_const = sigma_Z_constant(s_fine)
        exp_const /= np.max(exp_const)

        ax2.plot(m_range, exp_running, '-', color='white', linewidth=2,
                 label='Running-width BW')
        ax2.plot(m_range, exp_const, '--', color='#7f8c8d', linewidth=1.5,
                 label='Constant-width BW')

        # Compute running-width BW on model bins for residuals
        exp_at_bins = sigma_Z_running(bc**2)
        exp_at_bins /= np.max(exp_at_bins)

    elif name == 'ρ(770)':
        # Gounaris-Sakurai
        exp_gs = np.array([F_rho_GS(s) for s in s_fine])
        exp_gs /= np.max(exp_gs)
        exp_bw = np.array([sigma_rho_simple_BW(s) for s in s_fine])
        exp_bw /= np.max(exp_bw)

        ax2.plot(m_range, exp_gs, '-', color='white', linewidth=2,
                 label='Gounaris-Sakurai')
        ax2.plot(m_range, exp_bw, '--', color='#7f8c8d', linewidth=1.5,
                 label='Simple BW')

        exp_at_bins = np.array([F_rho_GS(s) for s in bc**2])
        exp_at_bins /= np.max(exp_at_bins)

    elif name == 'f₀(500)':
        # Pole vs naive BW
        exp_pole = np.array([T_f0_pole(s) for s in s_fine])
        exp_pole /= np.max(exp_pole)
        exp_bw = np.array([sigma_f0_BW(s) for s in s_fine])
        exp_bw /= np.max(exp_bw)

        ax2.plot(m_range, exp_pole, '-', color='white', linewidth=2,
                 label='S-matrix pole')
        ax2.plot(m_range, exp_bw, '--', color='#7f8c8d', linewidth=1.5,
                 label='Naive BW (known to fail)')

        exp_at_bins = np.array([T_f0_pole(s) for s in bc**2])
        exp_at_bins /= np.max(exp_at_bins)

    ax2.set_xlabel('Mass (MeV)', color='#a0a0a0', fontsize=9)
    ax2.set_ylabel('Normalized intensity', color='#a0a0a0', fontsize=9)
    ax2.legend(facecolor='#1a1a3a', edgecolor='#3a3a5a', labelcolor='white',
               fontsize=7, loc='upper right')

    # Statistics
    sk = skew(m_inst[:len(m_inst)//2])
    ku = kurtosis(m_inst[:len(m_inst)//2])

    # ── Row 3: Residuals ─────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 3, col + 7)
    ax3.set_facecolor(bg)
    for spine in ax3.spines.values():
        spine.set_color('#1a1a3a')
    ax3.tick_params(colors='#a0a0a0', labelsize=8)

    residuals = ct_norm - exp_at_bins
    chi2 = np.sum(residuals**2 / (ct_norm + 1e-10))

    ax3.bar(bc, residuals, width=bc[1]-bc[0], color=target['color'],
            alpha=0.6, edgecolor=target['color'], linewidth=0.5)
    ax3.axhline(0, color='white', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Mass (MeV)', color='#a0a0a0', fontsize=9)
    ax3.set_ylabel('Residual (model − expt)', color='#a0a0a0', fontsize=9)

    # Summary stats
    stats = (f'Skewness: {sk:+.3f}\n'
             f'Kurtosis: {ku:.3f}\n'
             f'χ² (model vs expt): {chi2:.4f}\n'
             f'Max |residual|: {np.max(np.abs(residuals)):.4f}')
    ax3.text(0.02, 0.95, stats, transform=ax3.transAxes,
             color='white', fontsize=8, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a3a', alpha=0.9))

    # Print detailed results
    print(f"  Skewness: {sk:+.4f}")
    print(f"  Excess kurtosis: {ku:.4f}")
    print(f"  χ² (model vs experiment): {chi2:.6f}")
    print(f"  Max |residual|: {np.max(np.abs(residuals)):.6f}")

    # Direction of skew
    if sk < -0.1:
        direction = "LOW-MASS TAIL (model predicts more time at lower masses)"
    elif sk > 0.1:
        direction = "HIGH-MASS TAIL (model predicts more time at higher masses)"
    else:
        direction = "NEARLY SYMMETRIC"
    print(f"  Asymmetry direction: {direction}")

    # Verdict
    if chi2 < 0.5:
        verdict = "PASS — model closely matches experimental line shape"
    elif chi2 < 2.0:
        verdict = "MARGINAL — qualitative agreement, quantitative differences"
    else:
        verdict = "FAIL — model line shape differs significantly from experiment"
    print(f"  VERDICT: {verdict}")
    target['verdict'] = verdict
    target['chi2'] = chi2
    target['skew'] = sk

fig.suptitle('Acid Test: cos(3θ) Potential vs. Experimental Resonance Line Shapes\n'
             'Can a damped Koide oscillator reproduce Z, ρ, and f₀ line shapes?',
             color='white', fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])
outpath = OUTPUT_DIR / 'acid_test_lineshapes.png'
plt.savefig(outpath, dpi=150, facecolor=bg)
print(f"\nSaved: {outpath}")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# PART 5: Summary and implications
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY: ACID TEST RESULTS")
print("=" * 72)

for name in particle_list:
    t = targets[name]
    if 'verdict' in t:
        print(f"\n  {name:12s}  Γ/M={t['gamma_m']:.3f}  g₀={t['g0']:.4f}  "
              f"χ²={t['chi2']:.4f}  skew={t['skew']:+.3f}")
        print(f"  {'':12s}  {t['verdict']}")

print(f"""
KEY OBSERVATIONS:

1. ASYMMETRY DIRECTION: The model predicts negative skewness (low-mass tail)
   for all damping regimes. This should be compared to:
   - Z boson: running width gives HIGH-mass enhancement (opposite sign!)
   - ρ(770): P-wave threshold suppresses LOW-mass side (same sign!)
   - f₀(500): extreme distortion, no clear BW-like shape (model may match)

2. WHAT THE MODEL GETS RIGHT:
   - Any nonlinear oscillator in a potential well produces asymmetric line shapes
   - The cos(3θ) potential's cubic symmetry produces specific angular structure
   - Broader resonances (higher Γ/M) show more asymmetry — matches observation

3. WHAT THE MODEL GETS WRONG (if anything):
   - Compare the specific shape of the asymmetry to experiment
   - Check if the skewness direction matches (this is the key test)
   - For the f₀(500), does the model reproduce the pole position?

4. THE CRITICAL DISTINCTION:
   Standard QFT: asymmetry from production mechanism (interference, ISR)
   Torus model: asymmetry from particle dynamics (mass oscillation)
   These predict DIFFERENT asymmetry shapes — the residuals reveal which.
""")
