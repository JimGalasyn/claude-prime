#!/usr/bin/env python3
"""
QUANTUM ACID TEST: Schrödinger equation in cos(3θ) potential.

Instead of a classical damped oscillator, we solve the quantum problem:
  H = -(1/2M) d²/dθ² + V(θ)
  V(θ) = V₀ [1/2 - (√2/2) cos(3θ)]

The spectral function A(E) = -Im[G(E)]/π is the CORRECT line shape
prediction, not the classical dwell-time distribution.

Optimized for speed: N=64 grid (plenty for a smooth periodic potential),
targeted parameter search instead of brute-force sweep.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# PART 1: Quantum Hamiltonian in cos(3θ) potential
# ══════════════════════════════════════════════════════════════════════

N_GRID = 64  # sufficient for smooth cos(3θ)

def build_hamiltonian(M_eff, V0, gamma=0.0, N=N_GRID):
    """Build H on a ring with periodic BCs. Returns H, theta, V."""
    dtheta = 2 * np.pi / N
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    V = V0 * (0.5 - (np.sqrt(2)/2) * np.cos(3*theta))

    # Kinetic: second derivative via circulant finite differences
    diag_main = np.full(N, 2.0 / (2*M_eff*dtheta**2))
    diag_off = np.full(N-1, -1.0 / (2*M_eff*dtheta**2))
    T = np.diag(diag_main) - np.diag(diag_off, 1) - np.diag(diag_off, -1)
    # Periodic BCs
    T[0, -1] = -1.0 / (2*M_eff*dtheta**2)
    T[-1, 0] = -1.0 / (2*M_eff*dtheta**2)

    H = T + np.diag(V)

    if gamma > 0:
        V_norm = (V - V.min()) / (V.max() - V.min() + 1e-30)
        W = gamma * (0.3 + 0.7 * V_norm)
        H = H.astype(complex) - 1j * np.diag(W)

    return H, theta, V


def get_resonances(M_eff, V0, gamma):
    """Get complex eigenvalues → resonance (mass, width, Γ/M) list."""
    H, _, _ = build_hamiltonian(M_eff, V0, gamma)
    evals = linalg.eigvals(H) if np.iscomplexobj(H) else linalg.eigvalsh(H).astype(complex)
    evals = evals[np.argsort(np.real(evals))]
    results = []
    for E in evals:
        m = np.real(E)
        w = -2*np.imag(E)
        if m > 0 and w > 0:
            results.append((m, w, w/m))
    return results


def spectral_function(H, E_range, well_idx):
    """Local spectral function A(E,θ₀) via eigendecomposition."""
    evals, evecs = linalg.eig(H) if np.iscomplexobj(H) else (
        linalg.eigh(H)[0].astype(complex), linalg.eigh(H)[1])
    eps = 5e-4
    weights = np.abs(evecs[well_idx, :])**2
    A = np.zeros(len(E_range))
    for j, E in enumerate(E_range):
        A[j] = -np.imag(np.sum(weights / (E - evals + 1j*eps))) / np.pi
    return A


# ══════════════════════════════════════════════════════════════════════
# PART 2: BW fitting functions
# ══════════════════════════════════════════════════════════════════════

def breit_wigner(E, E0, gamma, A=1.0):
    return A * (gamma/2)**2 / ((E - E0)**2 + (gamma/2)**2)

def breit_wigner_skewed(E, E0, gamma, A, alpha):
    bw = (gamma/2)**2 / ((E - E0)**2 + (gamma/2)**2)
    return A * bw * (1 + alpha * (E - E0) / gamma)


# ══════════════════════════════════════════════════════════════════════
# PART 3: Smart parameter search for each target Γ/M
# ══════════════════════════════════════════════════════════════════════

print("=" * 72)
print("QUANTUM ACID TEST: Schrödinger Equation in cos(3θ)")
print("=" * 72)

V0 = 1.0
target_gm = {
    'Z boson': 2495.2/91187.6,   # 0.0274
    'ρ(770)': 147.4/775.26,      # 0.190
    'f₀(500)': 550.0/449.0,      # 1.225
}
colors = {'Z boson': '#3498db', 'ρ(770)': '#2ecc71', 'f₀(500)': '#e74c3c'}

print("\nPhase 1: Targeted parameter search (M_eff, γ) for each particle...")
print("-" * 72)

best_params = {}

for name, tgm in target_gm.items():
    print(f"\n  {name}: target Γ/M = {tgm:.4f}")

    def objective(log_params):
        lM, lg = log_params
        M_eff = 10**lM
        gamma = 10**lg
        try:
            res = get_resonances(M_eff, V0, gamma)
            if len(res) == 0:
                return 100.0
            # Find resonance closest to target Γ/M
            diffs = [abs(r[2] - tgm) for r in res[:8]]
            return min(diffs)
        except Exception:
            return 100.0

    # Coarse grid search first (10×10 = 100 evals, fast at N=64)
    best_val = np.inf
    best_lM, best_lg = 1.0, -2.0
    for lM in np.linspace(-0.5, 2.5, 10):
        for lg in np.linspace(-4, 0.5, 10):
            v = objective([lM, lg])
            if v < best_val:
                best_val = v
                best_lM, best_lg = lM, lg

    # Refine with Nelder-Mead
    result = minimize(objective, [best_lM, best_lg], method='Nelder-Mead',
                      options={'xatol': 0.01, 'fatol': 1e-5, 'maxiter': 100})
    M_eff = 10**result.x[0]
    gamma = 10**result.x[1]

    res = get_resonances(M_eff, V0, gamma)
    if len(res) == 0:
        print(f"    NO RESONANCES FOUND")
        best_params[name] = None
        continue

    # Pick the resonance closest to target
    best_res = min(res[:8], key=lambda r: abs(r[2] - tgm))
    print(f"    Found: M_eff={M_eff:.4f}, γ={gamma:.6f}")
    print(f"    Pole: E={best_res[0]:.6f}, Γ={best_res[1]:.6f}, Γ/M={best_res[2]:.4f}")
    best_params[name] = {'M_eff': M_eff, 'gamma': gamma, 'res': best_res}


# ══════════════════════════════════════════════════════════════════════
# PART 4: Compute spectral line shapes and compare
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("Phase 2: Spectral line shapes")
print("=" * 72)

bg = '#060612'
fig = plt.figure(figsize=(20, 14), facecolor=bg)

# Top row: band structure + potential + poles
# Middle row: spectral functions for Z, ρ, f₀
# Bottom row: residuals from BW fit

# ── Band structure ───────────────────────────────────────────────────
ax_bands = fig.add_subplot(3, 4, 1)
ax_bands.set_facecolor(bg)
for spine in ax_bands.spines.values(): spine.set_color('#1a1a3a')
ax_bands.tick_params(colors='#a0a0a0', labelsize=7)

for M_eff in [0.5, 1, 2, 5, 10, 20, 50, 100, 200]:
    H, _, _ = build_hamiltonian(M_eff, V0)
    evals = linalg.eigvalsh(H)[:12]
    ax_bands.plot([M_eff]*len(evals), evals, 'o', color='#3498db',
                  markersize=3, alpha=0.7)
ax_bands.set_xscale('log')
ax_bands.set_xlabel('M_eff', color='#a0a0a0', fontsize=9)
ax_bands.set_ylabel('Energy', color='#a0a0a0', fontsize=9)
ax_bands.set_title('Band structure\n(closed system)', color='white',
                    fontsize=10, fontweight='bold')

# ── Potential + wavefunctions ────────────────────────────────────────
ax_wf = fig.add_subplot(3, 4, 2)
ax_wf.set_facecolor(bg)
for spine in ax_wf.spines.values(): spine.set_color('#1a1a3a')
ax_wf.tick_params(colors='#a0a0a0', labelsize=7)

H_show, theta_show, V_show = build_hamiltonian(20.0, V0)
evals_show, evecs_show = linalg.eigh(H_show)
ax_wf.plot(theta_show, V_show, color='#3498db', linewidth=2)

wf_colors = ['#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22']
for n in range(6):
    psi = evecs_show[:, n]
    psi_norm = psi / np.max(np.abs(psi)) * 0.12
    ax_wf.plot(theta_show, evals_show[n] + psi_norm,
               color=wf_colors[n], linewidth=1, alpha=0.8)
    ax_wf.axhline(evals_show[n], color=wf_colors[n], linewidth=0.5,
                   alpha=0.3, linestyle='--')
ax_wf.set_xlabel('θ', color='#a0a0a0', fontsize=9)
ax_wf.set_ylabel('Energy', color='#a0a0a0', fontsize=9)
ax_wf.set_title('Wavefunctions (M=20)\nFirst 6 states', color='white',
                 fontsize=10, fontweight='bold')

# ── Poles in complex plane ───────────────────────────────────────────
for pidx, (pname, pcolor) in enumerate([('ρ(770)', '#2ecc71'), ('f₀(500)', '#e74c3c')]):
    ax_p = fig.add_subplot(3, 4, 3 + pidx)
    ax_p.set_facecolor(bg)
    for spine in ax_p.spines.values(): spine.set_color('#1a1a3a')
    ax_p.tick_params(colors='#a0a0a0', labelsize=7)

    p = best_params.get(pname)
    if p is not None:
        H_p, _, V_p = build_hamiltonian(p['M_eff'], V0, p['gamma'])
        ev = linalg.eigvals(H_p)
        ev = ev[np.argsort(np.real(ev))][:20]
        ax_p.scatter(np.real(ev), -np.imag(ev), s=30, c=pcolor,
                     edgecolors='white', linewidths=0.5, zorder=5)
        ax_p.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        # Highlight the matched resonance
        res = p['res']
        ax_p.scatter([res[0]], [res[1]/2], s=100, marker='*',
                     c='#f1c40f', edgecolors='white', linewidths=1, zorder=10)
    ax_p.set_xlabel('Re(E)', color='#a0a0a0', fontsize=9)
    ax_p.set_ylabel('-Im(E) = Γ/2', color='#a0a0a0', fontsize=9)
    ax_p.set_title(f'S-matrix poles ({pname})', color='white',
                    fontsize=10, fontweight='bold')

# ── Spectral functions + comparison ──────────────────────────────────
particle_list = ['Z boson', 'ρ(770)', 'f₀(500)']

for col, name in enumerate(particle_list):
    p = best_params.get(name)
    if p is None:
        continue

    M_eff, gamma, res = p['M_eff'], p['gamma'], p['res']
    E0, Gamma0, gm = res

    print(f"\n{'─'*72}")
    print(f"  {name}: M_eff={M_eff:.4f}, γ={gamma:.6f}")
    print(f"  Pole: E={E0:.6f}, Γ={Gamma0:.6f}, Γ/M={gm:.4f}")

    H, theta, V_arr = build_hamiltonian(M_eff, V0, gamma)
    well_idx = np.argmin(V_arr)

    # Energy range around the resonance
    hw = max(5*Gamma0, 0.3*E0)
    E_range = np.linspace(max(E0 - hw, V_arr.min() - 0.05), E0 + hw, 500)

    A = spectral_function(H, E_range, well_idx)
    A_max = np.max(A)
    A_norm = A / A_max if A_max > 0 else A

    # ── Middle row: spectral function ────────────────────────────────
    ax = fig.add_subplot(3, 3, col + 4)
    ax.set_facecolor(bg)
    for spine in ax.spines.values(): spine.set_color('#1a1a3a')
    ax.tick_params(colors='#a0a0a0', labelsize=7)

    ax.fill_between(E_range, A_norm, alpha=0.3, color=colors[name])
    ax.plot(E_range, A_norm, color=colors[name], linewidth=2,
            label='Quantum A(E)')

    # Fit symmetric BW
    try:
        popt_bw, _ = curve_fit(breit_wigner, E_range, A_norm,
                               p0=[E0, Gamma0, 1.0], maxfev=5000)
        bw_fit = breit_wigner(E_range, *popt_bw)
        bw_norm = bw_fit / np.max(bw_fit) if np.max(bw_fit) > 0 else bw_fit
        ax.plot(E_range, bw_norm, '--', color='#7f8c8d', linewidth=1.5,
                label=f'Symmetric BW')
        resid_bw = np.sum((A_norm - bw_norm)**2)
    except Exception:
        resid_bw = np.inf
        bw_norm = np.zeros_like(A_norm)

    # Fit skewed BW
    try:
        popt_sk, _ = curve_fit(breit_wigner_skewed, E_range, A_norm,
                               p0=[E0, Gamma0, 1.0, 0.0], maxfev=5000)
        sk_fit = breit_wigner_skewed(E_range, *popt_sk)
        sk_norm = sk_fit / np.max(sk_fit) if np.max(sk_fit) > 0 else sk_fit
        ax.plot(E_range, sk_norm, '-', color='white', linewidth=1.5,
                label=f'Skewed BW (α={popt_sk[3]:.4f})')
        resid_sk = np.sum((A_norm - sk_norm)**2)
    except Exception:
        resid_sk = np.inf
        popt_sk = [0, 0, 0, 0]
        sk_norm = np.zeros_like(A_norm)

    ax.set_xlabel('Energy (model units)', color='#a0a0a0', fontsize=9)
    ax.set_ylabel('A(E) normalized', color='#a0a0a0', fontsize=9)
    ax.set_title(f'{name}\nΓ/M={gm:.4f} (target {target_gm[name]:.4f})',
                 color='white', fontsize=10, fontweight='bold')
    ax.legend(facecolor='#1a1a3a', edgecolor='#3a3a5a', labelcolor='white',
              fontsize=7, loc='upper right')

    # Spectral moments
    if np.sum(A_norm) > 0:
        prob = A_norm / np.sum(A_norm)
        E_mean = np.sum(E_range * prob)
        E_std = np.sqrt(np.sum((E_range - E_mean)**2 * prob))
        E_skew = np.sum(((E_range - E_mean)/(E_std+1e-30))**3 * prob) if E_std > 0 else 0
    else:
        E_skew = 0

    improvement = (resid_bw - resid_sk)/resid_bw*100 if resid_bw > 0 and resid_bw < np.inf else 0

    stats = (f'Skewness: {E_skew:+.4f}\n'
             f'BW resid: {resid_bw:.4f}\n'
             f'Skew resid: {resid_sk:.4f}\n'
             f'Improvement: {improvement:.1f}%\n'
             f'α: {popt_sk[3]:+.4f}')
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, color='white',
            fontsize=7, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a3a', alpha=0.9))

    print(f"  Spectral skewness: {E_skew:+.6f}")
    print(f"  BW residual: {resid_bw:.6f}")
    print(f"  Skewed BW residual: {resid_sk:.6f}")
    print(f"  Skew improvement: {improvement:.1f}%")
    print(f"  Skew parameter α: {popt_sk[3]:+.6f}")

    # Direction check
    if name == 'Z boson':
        exp_sign = "positive (running width → high-E tail)"
    elif name == 'ρ(770)':
        exp_sign = "negative (P-wave threshold → low-E suppression)"
    else:
        exp_sign = "complex (BW inapplicable)"

    model_sign = "negative" if E_skew < -0.01 else ("positive" if E_skew > 0.01 else "symmetric")
    print(f"  Model asymmetry: {model_sign}")
    print(f"  Experiment expects: {exp_sign}")

    if abs(E_skew) > 0.01:
        verdict = "ASYMMETRY DETECTED"
    else:
        verdict = "NEARLY SYMMETRIC"

    if name == 'ρ(770)' and E_skew < -0.01:
        verdict += " — direction MATCHES experiment"
    elif name == 'Z boson' and E_skew > 0.01:
        verdict += " — direction MATCHES experiment"
    elif name == 'Z boson' and E_skew < -0.01:
        verdict += " — direction OPPOSES experiment"
    elif name == 'ρ(770)' and E_skew > 0.01:
        verdict += " — direction OPPOSES experiment"

    print(f"  VERDICT: {verdict}")

    # ── Bottom row: residuals ────────────────────────────────────────
    ax3 = fig.add_subplot(3, 3, col + 7)
    ax3.set_facecolor(bg)
    for spine in ax3.spines.values(): spine.set_color('#1a1a3a')
    ax3.tick_params(colors='#a0a0a0', labelsize=7)

    residuals = A_norm - bw_norm
    ax3.fill_between(E_range, residuals, alpha=0.4, color=colors[name])
    ax3.plot(E_range, residuals, color=colors[name], linewidth=1)
    ax3.axhline(0, color='white', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Energy', color='#a0a0a0', fontsize=9)
    ax3.set_ylabel('A(E) − BW', color='#a0a0a0', fontsize=9)
    ax3.set_title(f'{name}: residual from symmetric BW',
                  color='white', fontsize=10, fontweight='bold')

fig.suptitle('Quantum Acid Test: Spectral Function of cos(3θ) Potential\n'
             'Does the quantum propagator produce the right line shape asymmetry?',
             color='white', fontsize=14, fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath = '/home/jim/repos/claude-prime/simulations/quantum_koide_lineshapes.png'
plt.savefig(outpath, dpi=150, facecolor=bg)
print(f"\nSaved: {outpath}")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("QUANTUM ACID TEST: FINAL SUMMARY")
print("=" * 72)
for name in particle_list:
    p = best_params.get(name)
    if p is None:
        print(f"\n  {name}: NO MATCH")
    else:
        res = p['res']
        print(f"\n  {name}:")
        print(f"    Target Γ/M = {target_gm[name]:.4f}")
        print(f"    Achieved Γ/M = {res[2]:.4f}")
        print(f"    Parameters: M_eff={p['M_eff']:.4f}, γ={p['gamma']:.6f}")
