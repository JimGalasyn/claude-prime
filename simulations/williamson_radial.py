#!/usr/bin/env python3
"""
Williamson Radial Model: Bound States from Extended Maxwell
============================================================

Principled derivation of the radial eigenvalue problem:

1. The extended Maxwell equations with φ⁴ self-interaction give
   the pivot field P an effective mass: m_eff = √(2λ)·v

2. In the far field (away from the torus), P behaves as a massive
   scalar field. The dispersion relation is ω² = k² + m², which
   means P excitations can be BOUND (unlike massless photons).

3. A "nucleus" (another toroidal resonance) creates a Coulomb
   potential V(r) = -αZ/r. A massive scalar in this potential
   satisfies the Klein-Gordon-Coulomb equation:

       d²u/dr² + [E_eff - V_eff(r)] u = 0

   where u = r·P, E_eff = ω² - m², and
   V_eff(r) = -2ωαZ/r + (l(l+1) - α²Z²)/r²

4. This is EXACTLY the hydrogen radial equation with relativistic
   corrections. Bound states give the Sommerfeld fine structure:

       ω_nl = m / √(1 + (αZ)²/(n - δ_l)²)

   where δ_l = l + ½ - √((l+½)² - (αZ)²)

5. The non-relativistic limit reproduces E_n = -mα²Z²/(2n²),
   i.e., the Bohr hydrogen spectrum.

The key prediction: quantized energy levels emerge from the same
field equations that produce the toroidal electron resonance.
Resonances all the way down.

Usage:
    python3 williamson_radial.py --mode eigenvalues
    python3 williamson_radial.py --mode wavefunctions --save
    python3 williamson_radial.py --mode fine_structure --save
    python3 williamson_radial.py --mode alpha_scan --save
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import argparse


# =============================================================================
# Physics: Klein-Gordon-Coulomb eigenvalue problem
# =============================================================================

def kg_coulomb_analytical(n, l, m, alpha, Z=1):
    """Exact Klein-Gordon-Coulomb eigenvalue (Sommerfeld formula).

    Args:
        n: Principal quantum number (n >= l + 1)
        l: Angular momentum quantum number
        m: Field mass (m_eff from φ⁴)
        alpha: Fine structure constant (coupling strength)
        Z: Nuclear charge

    Returns:
        ω: Bound state energy (ω < m for bound states)
    """
    aZ = alpha * Z
    # Relativistic correction to angular momentum
    j_eff = l + 0.5  # using l+1/2 form for KG (no spin)
    gamma = np.sqrt(j_eff**2 - aZ**2)

    # Radial quantum number
    n_r = n - l - 1  # n_r = 0, 1, 2, ...

    # Sommerfeld formula
    denom = np.sqrt(1 + aZ**2 / (n_r + gamma)**2)
    omega = m / denom

    return omega


def kg_coulomb_nonrelativistic(n, m, alpha, Z=1):
    """Non-relativistic limit: Bohr spectrum.

    E_n = m - m·(αZ)²/(2n²)
    Binding energy = m·(αZ)²/(2n²)
    """
    return m * (1 - (alpha * Z)**2 / (2 * n**2))


def binding_energy(n, l, m, alpha, Z=1):
    """Binding energy = m - ω (positive for bound states)."""
    omega = kg_coulomb_analytical(n, l, m, alpha, Z)
    return m - omega


# =============================================================================
# Numerical solver: matrix method on radial grid
# =============================================================================

class RadialSolver:
    """Solve the radial KG-Coulomb eigenvalue problem numerically.

    The radial equation (in natural units, ℏ = c = 1):

        -d²u/dr² + V_eff(r)·u = (ω² - m²)·u    [NOT ω·u]

    where u = r·R(r) and V_eff(r) = l(l+1)/r² - 2ωαZ/r

    Note: this is a nonlinear eigenvalue problem because V_eff
    depends on ω through the 2ωαZ/r term. We handle this by:
    (a) First solving the Schrödinger-like approximation (ω → m in V_eff)
    (b) Then iterating with the correct ω

    For weak coupling (α << 1), the Schrödinger approximation
    is excellent and iteration converges in 2-3 steps.
    """

    def __init__(self, m=1.0, alpha=1/137.036, Z=1, r_max=None, n_grid=2000):
        self.m = m
        self.alpha = alpha
        self.Z = Z
        self.n_grid = n_grid

        # r_max should be large enough to contain the wavefunctions
        # Bohr radius: a₀ = 1/(m·α·Z) in natural units
        self.a_bohr = 1.0 / (m * alpha * Z)
        if r_max is None:
            r_max = 50 * self.a_bohr  # capture up to n~5 states

        self.r_max = r_max
        self.dr = r_max / n_grid

        # Radial grid (avoid r=0 singularity)
        self.r = np.linspace(self.dr, r_max, n_grid)

    def _build_hamiltonian(self, l, omega_approx=None):
        """Build the discretized Hamiltonian matrix.

        H = -d²/dr² + V_eff(r)

        For the KG equation: H·u = (ω² - m²)·u
        The eigenvalue is ε = ω² - m², so ω = √(m² + ε).
        For bound states, ε < 0.

        Uses omega_approx for the ω-dependent Coulomb term.
        If None, uses the Schrödinger approximation (ω → m).
        """
        r = self.r
        dr = self.dr
        N = self.n_grid

        if omega_approx is None:
            omega_approx = self.m

        # Effective potential
        centrifugal = l * (l + 1) / r**2
        coulomb = -2 * omega_approx * self.alpha * self.Z / r

        V = centrifugal + coulomb

        # Kinetic energy: -d²/dr² discretized as tridiagonal matrix
        # -u''(r) ≈ (-u_{i-1} + 2u_i - u_{i+1}) / dr²
        diag = 2.0 / dr**2 + V
        off_diag = -np.ones(N - 1) / dr**2

        return diag, off_diag

    def solve(self, l=0, n_states=5, iterate=True, max_iter=10, tol=1e-10):
        """Find bound state energies and wavefunctions.

        Args:
            l: Angular momentum quantum number
            n_states: Number of states to find
            iterate: Whether to iterate for self-consistent ω
            max_iter: Maximum iterations
            tol: Convergence tolerance on ω

        Returns:
            dict with 'omega' (energies), 'binding' (binding energies),
            'u' (radial wavefunctions), 'r' (grid)
        """
        # Initial solve with Schrödinger approximation
        diag, off_diag = self._build_hamiltonian(l)

        # Find lowest eigenvalues (most negative = most bound)
        eigenvalues, eigenvectors = eigh_tridiagonal(
            diag, off_diag, select='i', select_range=(0, n_states - 1)
        )

        # ε = ω² - m², so ω = √(m² + ε)
        omegas = np.sqrt(self.m**2 + eigenvalues)

        if iterate:
            # Self-consistent iteration: update ω in V_eff
            for iteration in range(max_iter):
                omegas_old = omegas.copy()

                for i in range(n_states):
                    diag_i, off_diag_i = self._build_hamiltonian(
                        l, omega_approx=omegas[i]
                    )
                    ev, _ = eigh_tridiagonal(
                        diag_i, off_diag_i, select='i',
                        select_range=(i, i)
                    )
                    omegas[i] = np.sqrt(self.m**2 + ev[0])

                if np.max(np.abs(omegas - omegas_old)) < tol:
                    break

            # Final solve with converged ω values for wavefunctions
            # Use average ω for the matrix (good enough for wavefunctions)
            diag, off_diag = self._build_hamiltonian(l, omega_approx=np.mean(omegas))
            eigenvalues, eigenvectors = eigh_tridiagonal(
                diag, off_diag, select='i', select_range=(0, n_states - 1)
            )

        # Normalize wavefunctions
        for i in range(eigenvectors.shape[1]):
            norm = np.sqrt(np.sum(eigenvectors[:, i]**2 * self.dr))
            eigenvectors[:, i] /= norm

        binding = self.m - omegas

        return {
            'omega': omegas,
            'binding': binding,
            'eigenvalues': eigenvalues,
            'u': eigenvectors,
            'r': self.r,
            'l': l,
        }


# =============================================================================
# Connection to torus simulation parameters
# =============================================================================

def torus_to_radial_params(lam=1.0, v=0.5, R=3.0, r=1.0):
    """Extract radial model parameters from torus simulation.

    From the φ⁴ self-interaction: V(P) = λ/4·(P² - v²)²
    Expanding around P = v (vacuum): V ≈ λv²·δP² + ...
    Effective mass: m_eff² = 2λv²  →  m_eff = √(2λ)·v

    The "charge" of the toroidal resonance comes from the
    topological winding of the pivot field. For a (2,1) mode
    on a torus, the effective coupling is related to the
    ratio of EM↔P energy transfer.

    Returns dict with m_eff and related quantities.
    """
    m_eff = np.sqrt(2 * lam) * v

    # Compton wavelength: λ_C = 1/m_eff (in natural units)
    lambda_compton = 1.0 / m_eff if m_eff > 0 else np.inf

    # For the actual electron: m_e = 511 keV, α = 1/137
    # Bohr radius: a₀ = 1/(m_e·α) = 137/m_e
    # If our m_eff = m_e, then a₀ = 137/m_eff

    return {
        'm_eff': m_eff,
        'lambda_compton': lambda_compton,
        'lambda': lam,
        'v': v,
        'R': R,
        'r': r,
    }


# =============================================================================
# Visualization and analysis
# =============================================================================

def run_eigenvalues(alpha=1/137.036, m=1.0, Z=1, n_max=5, save=False):
    """Compute and display eigenvalues, compare with analytical."""
    print("=" * 64)
    print("Klein-Gordon-Coulomb Bound States")
    print(f"m = {m:.4f}, α = {alpha:.6f} (1/α = {1/alpha:.1f}), Z = {Z}")
    print(f"Bohr radius: a₀ = {1/(m*alpha*Z):.2f}")
    print("=" * 64)
    print()

    solver = RadialSolver(m=m, alpha=alpha, Z=Z)

    all_results = {}
    for l in range(min(n_max, 4)):
        n_states = n_max - l
        if n_states <= 0:
            continue

        result = solver.solve(l=l, n_states=n_states)
        all_results[l] = result

    # Display results
    print(f"{'n':>3s} {'l':>3s} {'ω (numerical)':>16s} {'ω (analytical)':>16s} "
          f"{'ω (Bohr)':>16s} {'E_bind (num)':>14s} {'rel. err':>10s}")
    print("-" * 90)

    for l, result in all_results.items():
        for i, omega_num in enumerate(result['omega']):
            n = i + l + 1  # principal quantum number

            omega_exact = kg_coulomb_analytical(n, l, m, alpha, Z)
            omega_bohr = kg_coulomb_nonrelativistic(n, m, alpha, Z)
            e_bind = m - omega_num

            rel_err = abs(omega_num - omega_exact) / omega_exact

            print(f"{n:3d} {l:3d} {omega_num:16.10f} {omega_exact:16.10f} "
                  f"{omega_bohr:16.10f} {e_bind:14.4e} {rel_err:10.2e}")

    # Fine structure test
    print()
    print("--- Fine Structure (l-dependence at fixed n) ---")
    header = "  n   l  l'       dw/w    dw/w (exact)"
    print(header)
    print("-" * 42)

    for n in range(2, min(n_max + 1, 5)):
        for l in range(n - 1):
            if l + 1 < n and l in all_results and l + 1 in all_results:
                # Get the right state index
                i_l = n - l - 1
                i_l1 = n - (l + 1) - 1

                if i_l < len(all_results[l]['omega']) and i_l1 < len(all_results[l + 1]['omega']):
                    omega_l = all_results[l]['omega'][i_l]
                    omega_l1 = all_results[l + 1]['omega'][i_l1]
                    split = (omega_l1 - omega_l) / omega_l

                    omega_exact_l = kg_coulomb_analytical(n, l, m, alpha, Z)
                    omega_exact_l1 = kg_coulomb_analytical(n, l + 1, m, alpha, Z)
                    split_exact = (omega_exact_l1 - omega_exact_l) / omega_exact_l

                    print(f"{n:3d} {l:3d} {l+1:3d} {split:14.4e} {split_exact:14.4e}")

    if save:
        _plot_energy_levels(all_results, m, alpha, Z, n_max)


def run_wavefunctions(alpha=1/137.036, m=1.0, Z=1, save=False):
    """Plot radial wavefunctions for hydrogen-like bound states."""
    solver = RadialSolver(m=m, alpha=alpha, Z=Z)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Radial Wavefunctions: KG-Coulomb (α = 1/{1/alpha:.0f})",
                 fontsize=14)

    states = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
    a0 = 1.0 / (m * alpha * Z)

    for ax, (n, l) in zip(axes.flat, states):
        n_states = n - l
        result = solver.solve(l=l, n_states=n_states)

        idx = n - l - 1
        u = result['u'][:, idx]
        r = result['r']

        # Radial probability: |R(r)|² r² = |u(r)|²
        prob = u**2

        ax.plot(r / a0, u, 'b-', linewidth=1.5, label=f'u_{n}{l}(r)')
        ax.fill_between(r / a0, 0, prob / np.max(prob) * np.max(np.abs(u)),
                        alpha=0.2, color='blue')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlim(0, min(2 * n**2, r[-1] / a0))
        ax.set_xlabel('r / a₀')
        ax.set_ylabel('u(r)')
        ax.set_title(f'n={n}, l={l}  (nodes={n-l-1})')
        ax.legend(fontsize=9)

    plt.tight_layout()

    if save:
        path = '/home/jim/repos/claude-prime/simulations/radial_wavefunctions.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def run_fine_structure(m=1.0, Z=1, save=False):
    """Show how fine structure splitting depends on α.

    The KG-Coulomb equation predicts l-dependent energy levels
    (unlike Schrödinger hydrogen where E depends only on n).
    This l-splitting is the fine structure.
    """
    print("=" * 64)
    print("Fine Structure: α-dependence of l-splitting")
    print("=" * 64)
    print()

    alphas = np.logspace(-3, -0.5, 30)  # α from 0.001 to ~0.3

    # Track splitting for n=2 (l=0 vs l=1)
    splits_exact = []
    splits_numerical = []

    for alpha in alphas:
        # Analytical
        w_20 = kg_coulomb_analytical(2, 0, m, alpha, Z)
        w_21 = kg_coulomb_analytical(2, 1, m, alpha, Z)
        splits_exact.append((w_21 - w_20) / m)

        # Numerical
        solver = RadialSolver(m=m, alpha=alpha, Z=Z, n_grid=3000)
        try:
            r0 = solver.solve(l=0, n_states=2)
            r1 = solver.solve(l=1, n_states=1)
            splits_numerical.append((r1['omega'][0] - r0['omega'][1]) / m)
        except Exception:
            splits_numerical.append(np.nan)

    splits_exact = np.array(splits_exact)
    splits_numerical = np.array(splits_numerical)

    # Expected scaling: Δω ∝ α⁴ (leading relativistic correction)
    alpha_ref = 1 / 137
    scale = splits_exact[np.argmin(np.abs(alphas - alpha_ref))]
    alpha4_fit = scale * (alphas / alpha_ref)**4

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(alphas, np.abs(splits_exact), 'b-', linewidth=2, label='Analytical (Sommerfeld)')
    valid = ~np.isnan(splits_numerical)
    ax.loglog(alphas[valid], np.abs(splits_numerical[valid]), 'ro', markersize=4,
              label='Numerical')
    ax.loglog(alphas, np.abs(alpha4_fit), 'k--', linewidth=1, label='∝ α⁴')
    ax.axvline(x=1/137, color='gold', linestyle=':', linewidth=2, label='α = 1/137')
    ax.set_xlabel('α (coupling strength)')
    ax.set_ylabel('|Δω₂₁ - Δω₂₀| / m')
    ax.set_title('Fine Structure Splitting (n=2)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy level diagram
    ax = axes[1]
    alpha_show = 1 / 10  # exaggerated for visibility
    levels = {}
    for n in range(1, 5):
        for l in range(n):
            w = kg_coulomb_analytical(n, l, m, alpha_show, Z)
            e_bind = (m - w) / m
            levels[(n, l)] = e_bind

    # Plot levels
    for (n, l), e in levels.items():
        x_left = n - 0.3
        x_right = n + 0.3
        color = plt.cm.tab10(l)
        ax.plot([x_left, x_right], [-e, -e], color=color, linewidth=2)
        ax.text(x_right + 0.05, -e, f'l={l}', fontsize=8, va='center', color=color)

    ax.set_xlabel('n (principal quantum number)')
    ax.set_ylabel('E_bind / m')
    ax.set_title(f'Energy Levels (α = 1/{1/alpha_show:.0f}, exaggerated)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = '/home/jim/repos/claude-prime/simulations/radial_fine_structure.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def run_alpha_scan(m=1.0, Z=1, save=False):
    """Scan coupling strength α to find what produces physical spectra.

    In the Williamson model, α emerges from the geometry of the
    toroidal resonance. What value of α does our torus simulation
    predict? We compare the energy ratios from the torus (EM↔P
    coupling strength) to the α that produces hydrogen-like spectra.
    """
    print("=" * 64)
    print("Alpha Scan: From Torus Coupling to Fine Structure Constant")
    print("=" * 64)
    print()

    # From our torus simulation:
    # (2,1) mode at R/r=3.0: EM ratio = 0.675 → ~32.5% to P
    # This gives a coupling strength related to α
    em_ratio_torus = 0.675  # from our simulation
    p_fraction = 1 - em_ratio_torus  # fraction of energy in P

    print(f"From torus simulation: EM→P fraction = {p_fraction:.3f}")
    print()

    # Scan α and compute properties
    alphas = np.logspace(-3, 0, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Williamson Model: α Scan", fontsize=14)

    # 1. Binding energy of ground state
    ax = axes[0, 0]
    e_bind = np.array([binding_energy(1, 0, m, a, Z) / m for a in alphas])
    ax.loglog(alphas, e_bind, 'b-', linewidth=2)
    ax.axvline(x=1/137, color='gold', linestyle=':', linewidth=2, label='α = 1/137')
    ax.axhline(y=13.6 / 511000, color='red', linestyle='--', label='13.6 eV / 511 keV')
    ax.set_xlabel('α')
    ax.set_ylabel('E_bind(1s) / m')
    ax.set_title('Ground State Binding Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Bohr radius in units of Compton wavelength
    ax = axes[0, 1]
    a_bohr = 1.0 / (m * alphas * Z)
    lambda_c = 1.0 / m
    ax.loglog(alphas, a_bohr / lambda_c, 'g-', linewidth=2)
    ax.axvline(x=1/137, color='gold', linestyle=':', linewidth=2, label='α = 1/137')
    ax.axhline(y=137, color='red', linestyle='--', label='a₀/λ_C = 137')
    ax.set_xlabel('α')
    ax.set_ylabel('a₀ / λ_C')
    ax.set_title('Bohr Radius / Compton Wavelength')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Spectrum ratios E(n=2)/E(n=1) — should be 1/4
    ax = axes[1, 0]
    ratios = []
    for a in alphas:
        e1 = binding_energy(1, 0, m, a, Z)
        e2 = binding_energy(2, 0, m, a, Z)
        ratios.append(e2 / e1 if e1 > 0 else 0)
    ratios = np.array(ratios)

    ax.semilogx(alphas, ratios, 'b-', linewidth=2, label='E(2s)/E(1s)')
    ax.axhline(y=0.25, color='red', linestyle='--', label='Bohr: 1/4')
    ax.axvline(x=1/137, color='gold', linestyle=':', linewidth=2)
    ax.set_xlabel('α')
    ax.set_ylabel('E(n=2) / E(n=1)')
    ax.set_title('Spectrum Ratio (deviation = relativistic correction)')
    ax.legend()
    ax.set_ylim(0.15, 0.30)
    ax.grid(True, alpha=0.3)

    # 4. Fine structure splitting
    ax = axes[1, 1]
    fs_split = []
    for a in alphas:
        try:
            w20 = kg_coulomb_analytical(2, 0, m, a, Z)
            w21 = kg_coulomb_analytical(2, 1, m, a, Z)
            fs_split.append(abs(w21 - w20) / m)
        except (ValueError, RuntimeWarning):
            fs_split.append(np.nan)
    fs_split = np.array(fs_split)

    valid = ~np.isnan(fs_split) & (fs_split > 0)
    ax.loglog(alphas[valid], fs_split[valid], 'r-', linewidth=2)
    ax.axvline(x=1/137, color='gold', linestyle=':', linewidth=2, label='α = 1/137')
    ax.set_xlabel('α')
    ax.set_ylabel('|Δω(2s-2p)| / m')
    ax.set_title('Fine Structure Splitting')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = '/home/jim/repos/claude-prime/simulations/radial_alpha_scan.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def _plot_energy_levels(all_results, m, alpha, Z, n_max):
    """Plot energy level diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Numerical vs analytical
    ax = axes[0]
    for l, result in all_results.items():
        for i, omega in enumerate(result['omega']):
            n = i + l + 1
            e_bind_num = (m - omega) / (m * (alpha * Z)**2 / 2)  # in Rydbergs
            e_bind_exact = (m - kg_coulomb_analytical(n, l, m, alpha, Z)) / (m * (alpha * Z)**2 / 2)

            color = plt.cm.tab10(l)
            ax.plot(n - 0.15, -1 / e_bind_num if e_bind_num != 0 else 0,
                    'o', color=color, markersize=8)
            ax.plot(n + 0.15, -1 / e_bind_exact if e_bind_exact != 0 else 0,
                    's', color=color, markersize=6, markerfacecolor='none')

    ax.set_xlabel('n')
    ax.set_ylabel('-1/E_bind (Rydberg units)')
    ax.set_title(f'Energy Levels (α = 1/{1/alpha:.0f})')
    ax.legend(['Numerical', 'Analytical'], loc='lower right')
    ax.grid(True, alpha=0.3)

    # Binding energies
    ax = axes[1]
    ns = []
    bindings_num = []
    bindings_exact = []
    labels = []

    for l, result in all_results.items():
        for i, omega in enumerate(result['omega']):
            n = i + l + 1
            ns.append(n)
            bindings_num.append(m - omega)
            bindings_exact.append(m - kg_coulomb_analytical(n, l, m, alpha, Z))
            labels.append(f'({n},{l})')

    ax.semilogy(range(len(ns)), bindings_num, 'bo-', label='Numerical')
    ax.semilogy(range(len(ns)), bindings_exact, 'rs--', label='Analytical')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Binding Energy')
    ax.set_title('Numerical vs Analytical')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = '/home/jim/repos/claude-prime/simulations/radial_eigenvalues.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Williamson Radial Model")
    parser.add_argument('--mode', choices=['eigenvalues', 'wavefunctions',
                        'fine_structure', 'alpha_scan'],
                        default='eigenvalues')
    parser.add_argument('--alpha', type=float, default=1/137.036,
                        help='Fine structure constant')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Effective mass (from φ⁴: m = √(2λ)·v)')
    parser.add_argument('--Z', type=int, default=1, help='Nuclear charge')
    parser.add_argument('--n-max', type=int, default=5,
                        help='Maximum principal quantum number')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.mode == 'eigenvalues':
        run_eigenvalues(alpha=args.alpha, m=args.mass, Z=args.Z,
                        n_max=args.n_max, save=args.save)
    elif args.mode == 'wavefunctions':
        run_wavefunctions(alpha=args.alpha, m=args.mass, Z=args.Z,
                          save=args.save)
    elif args.mode == 'fine_structure':
        run_fine_structure(m=args.mass, Z=args.Z, save=args.save)
    elif args.mode == 'alpha_scan':
        run_alpha_scan(m=args.mass, Z=args.Z, save=args.save)


if __name__ == '__main__':
    main()
