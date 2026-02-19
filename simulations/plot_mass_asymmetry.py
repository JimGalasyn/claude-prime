#!/usr/bin/env python3
"""
Asymmetry analysis: does the damped-oscillator decay model predict
a skewed mass distribution?

If a newly-created particle rings in the cos(3θ) potential before
settling, it spends more time near the turning points of the
oscillation. This creates an asymmetric "dwell time" distribution
that IS the predicted resonance line shape.

We compare this to a standard symmetric Breit-Wigner.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Physics (same as anim_phase_decay.py) ────────────────────────────
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

def Gamma(th, g0):
    mmax = np.max(masses_at(th))
    return g0 * (1 + (mmax/m_ref)**2) / 2

def rhs(t, y, g0):
    th, om = y
    return [om, -dV(th) - Gamma(th, g0)*om]

# ── Breit-Wigner and skewed BW functions ─────────────────────────────
def breit_wigner(m, m0, gamma, A):
    """Standard symmetric Breit-Wigner."""
    return A * (gamma/2)**2 / ((m - m0)**2 + (gamma/2)**2)

def skewed_breit_wigner(m, m0, gamma, A, alpha):
    """Breit-Wigner with skewness parameter alpha."""
    bw = (gamma/2)**2 / ((m - m0)**2 + (gamma/2)**2)
    skew_factor = 1 + alpha * (m - m0) / gamma
    return A * bw * skew_factor

# ── Run ensembles at different creation energies ─────────────────────
print("=" * 70)
print("MASS ASYMMETRY ANALYSIS — Damped Oscillator Decay Model")
print("=" * 70)

# Settle point: find which well the trajectory settles into
# (determines which generation we're creating)
bg = '#060612'
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=bg)

# Three scenarios: different damping rates (proxy for different particles)
scenarios = [
    {'name': 'Light damping\n(long-lived resonance)', 'g0': 0.08,
     'y0': [theta_K + 0.25, 3.5], 't_end': 80},
    {'name': 'Moderate damping\n(typical resonance)', 'g0': 0.15,
     'y0': [theta_K + 0.3, 4.5], 't_end': 50},
    {'name': 'Heavy damping\n(broad resonance)', 'g0': 0.35,
     'y0': [theta_K + 0.4, 5.0], 't_end': 30},
]

print("\nScenario results:")
print("-" * 70)

for col, scen in enumerate(scenarios):
    g0 = scen['g0']
    y0 = scen['y0']
    t_end = scen['t_end']
    N_pts = 20000

    t_eval = np.linspace(0, t_end, N_pts)
    sol = solve_ivp(lambda t, y: rhs(t, y, g0), (0, t_end), y0,
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=0.01)
    th = sol.y[0]
    om = sol.y[1]

    # Compute instantaneous "dominant mass" at each timestep
    # (the largest mass from the Koide formula at angle θ)
    m_inst = np.array([np.max(masses_at(t)) for t in th])

    # Find the settled mass (last 10% of trajectory)
    m_settled = np.median(m_inst[-N_pts//10:])

    # ── Top row: mass vs time ────────────────────────────────────────
    ax = axes[0, col]
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color('#1a1a3a')
    ax.tick_params(colors='#a0a0a0')

    ax.plot(t_eval, m_inst, color='#ff6633', linewidth=0.8, alpha=0.8)
    ax.axhline(m_settled, color='#2ecc71', linewidth=1.5, linestyle='--',
               alpha=0.7, label=f'Settled: {m_settled:.1f} MeV')
    ax.set_xlabel('t (model time)', color='#a0a0a0', fontsize=10)
    ax.set_ylabel('Dominant mass (MeV)', color='#a0a0a0', fontsize=10)
    ax.set_title(scen['name'], color='white', fontsize=12, fontweight='bold')
    ax.legend(facecolor='#1a1a3a', edgecolor='#3a3a5a', labelcolor='white',
              fontsize=9)

    # ── Bottom row: mass distribution + BW fits ──────────────────────
    ax2 = axes[1, col]
    ax2.set_facecolor(bg)
    for spine in ax2.spines.values():
        spine.set_color('#1a1a3a')
    ax2.tick_params(colors='#a0a0a0')

    # Only use the transient portion (before settling)
    # Define "settled" as when mass is within 1% of final value
    settled_mask = np.abs(m_inst - m_settled) / m_settled < 0.01
    # Find first time it stays settled
    for i_settle in range(len(settled_mask)):
        if np.all(settled_mask[i_settle:min(i_settle+500, len(settled_mask))]):
            break
    else:
        i_settle = len(m_inst)

    m_transient = m_inst[:i_settle]
    dt = t_eval[1] - t_eval[0]

    if len(m_transient) < 100:
        ax2.text(0.5, 0.5, 'Settles too quickly\nfor analysis',
                 transform=ax2.transAxes, color='white', ha='center',
                 fontsize=12)
        continue

    # Histogram (dwell time distribution)
    n_bins = 80
    counts, bin_edges = np.histogram(m_transient, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax2.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0],
            color='#3498db', alpha=0.6, edgecolor='#5dade2', linewidth=0.5,
            label='Model dwell time')

    # Fit symmetric BW
    try:
        p0_bw = [m_settled, np.std(m_transient)*2, np.max(counts)]
        popt_bw, _ = curve_fit(breit_wigner, bin_centers, counts, p0=p0_bw,
                               maxfev=10000)
        m_fit = np.linspace(bin_centers[0], bin_centers[-1], 300)
        ax2.plot(m_fit, breit_wigner(m_fit, *popt_bw), '--',
                 color='#2ecc71', linewidth=2, label='Symmetric BW fit')

        # Residuals for BW
        resid_bw = counts - breit_wigner(bin_centers, *popt_bw)
        chi2_bw = np.sum(resid_bw**2 / (counts + 1e-10))
    except Exception:
        chi2_bw = np.inf

    # Fit skewed BW
    try:
        p0_skew = [m_settled, np.std(m_transient)*2, np.max(counts), 0.1]
        popt_skew, _ = curve_fit(skewed_breit_wigner, bin_centers, counts,
                                 p0=p0_skew, maxfev=10000)
        ax2.plot(m_fit, skewed_breit_wigner(m_fit, *popt_skew), '-',
                 color='#e74c3c', linewidth=2,
                 label=f'Skewed BW (α={popt_skew[3]:.3f})')

        resid_skew = counts - skewed_breit_wigner(bin_centers, *popt_skew)
        chi2_skew = np.sum(resid_skew**2 / (counts + 1e-10))
    except Exception:
        chi2_skew = np.inf
        popt_skew = [0, 0, 0, 0]

    ax2.set_xlabel('Mass (MeV)', color='#a0a0a0', fontsize=10)
    ax2.set_ylabel('Probability density', color='#a0a0a0', fontsize=10)
    ax2.legend(facecolor='#1a1a3a', edgecolor='#3a3a5a', labelcolor='white',
               fontsize=8, loc='upper right')

    # Statistics
    sk = skew(m_transient)
    ku = kurtosis(m_transient)

    stats_text = (f'Skewness: {sk:+.3f}\n'
                  f'Kurtosis: {ku:.3f}\n'
                  f'χ²(BW): {chi2_bw:.2f}\n'
                  f'χ²(skew): {chi2_skew:.2f}')
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes,
             color='white', fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a3a', alpha=0.9))

    # Print results
    print(f"\n{scen['name'].replace(chr(10), ' ')}  (γ₀={g0}):")
    print(f"  Settled mass: {m_settled:.1f} MeV")
    print(f"  Transient duration: {t_eval[i_settle]:.1f} model time "
          f"({i_settle} points)")
    print(f"  Mass range during transient: {m_transient.min():.1f} — "
          f"{m_transient.max():.1f} MeV")
    print(f"  Skewness: {sk:+.4f}")
    print(f"  Excess kurtosis: {ku:.4f}")
    print(f"  BW fit χ²: {chi2_bw:.3f}")
    print(f"  Skewed BW fit χ²: {chi2_skew:.3f}")
    if chi2_bw > 0:
        print(f"  χ² improvement: {(chi2_bw - chi2_skew)/chi2_bw * 100:.1f}%")
    print(f"  Skewness parameter α: {popt_skew[3]:+.4f}")

    # Highlight on mass-vs-time plot
    axes[0, col].axvline(t_eval[i_settle], color='#f1c40f', linewidth=1,
                          linestyle=':', alpha=0.5)
    axes[0, col].axvspan(0, t_eval[i_settle], alpha=0.08, color='#f1c40f')

fig.suptitle('Predicted Mass Asymmetry from Damped Koide Oscillator\n'
             'Does a newly-created particle ring asymmetrically before settling?',
             color='white', fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.93])
outpath = '/home/jim/repos/claude-prime/simulations/mass_asymmetry.png'
plt.savefig(outpath, dpi=150, facecolor=bg)
print(f"\nSaved: {outpath}")
plt.close()

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The damped oscillator in cos(3θ) predicts:
  1. Mass oscillation during transient (particle literally gets heavier/lighter)
  2. Asymmetric dwell-time distribution (NOT a symmetric Breit-Wigner)
  3. Skewness depends on damping rate (particle lifetime proxy)
  4. The skew parameter α is the MODEL'S PREDICTION for resonance asymmetry

Key question: does experimental data show this asymmetry?
  - Z boson line shape (LEP): most precisely measured resonance
  - τ mass near threshold (Belle/BaBar)
  - Top quark mass distribution (LHC)
""")
