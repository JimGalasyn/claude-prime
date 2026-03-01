#!/usr/bin/env python3
"""
Settling Radiation Spectrum — Comparison with Experimental Data
================================================================

Generates a two-panel figure comparing NWT settling radiation predictions
with the anomalous soft photon excess measured by DELPHI, WA83, WA91,
WA102, and ALICE.

Top panel:  Stick spectrum of Kelvin wave mode energies and powers
            for electron, pion, and proton (hadronic scenario).
Bottom:     Excess ratio (observed/IB) — data vs NWT prediction.

Usage:
    python3 plot_settling_spectrum.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from null_worldtube import (
    find_self_consistent_radius, compute_settling_modes,
    compute_settling_spectrum, alpha, hbar, c, MeV, eV,
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# --- Style: dark background matching existing diagrams ---
plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#0f0f0f',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'axes.edgecolor': '#404040',
    'grid.color': '#1a1a1a',
    'grid.alpha': 0.15,
    'font.family': 'monospace',
    'font.size': 11,
})

# Colors
C_ELECTRON = '#42A5F5'
C_PION     = '#FFA726'
C_PROTON   = '#EF5350'
C_THRESH   = '#66BB6A'
C_HADRON   = '#FF7043'
C_IB       = '#78909C'
C_TEXT     = '#e0e0e0'


def compute_all():
    """Compute settling modes for electron, pion, proton."""
    particles = [
        ('electron', 0.51099895, C_ELECTRON),
        ('pion',     139.57039,  C_PION),
        ('proton',   938.27209,  C_PROTON),
    ]
    results = {}
    for name, mass, color in particles:
        sol = find_self_consistent_radius(mass, p=2, q=1, r_ratio=alpha)
        if sol is None:
            continue
        R, r = sol['R'], sol['r']
        sm = compute_settling_modes(R, r, 2, 1, N_modes=8)
        sp_thr = compute_settling_spectrum(R, r, 2, 1, N_modes=8,
                                           scenario='threshold')
        sp_had = compute_settling_spectrum(R, r, 2, 1, N_modes=8,
                                           scenario='hadronic')
        results[name] = {
            'sol': sol, 'modes': sm, 'color': color, 'mass': mass,
            'threshold': sp_thr, 'hadronic': sp_had,
        }
    return results


def make_figure(results):
    """Create the two-panel comparison figure."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5),
                                    gridspec_kw={'height_ratios': [1.15, 1],
                                                 'hspace': 0.35})

    # ================================================================
    # Panel 1: Kelvin wave mode spectrum (stick + envelope)
    # ================================================================

    # Soft photon experimental window
    ax1.axvspan(0.01, 80, alpha=0.06, color='#FDD835', zorder=0)
    ax1.text(1.5, 0.94, 'experimental soft photon window\n(pT < 80 MeV)',
             transform=ax1.get_xaxis_transform(), fontsize=8,
             color='#FDD835', alpha=0.55, ha='center', va='top')

    y_offsets = {'electron': 0, 'pion': 1, 'proton': 2}
    bar_height = 0.7

    for name in ['electron', 'pion', 'proton']:
        if name not in results:
            continue
        d = results[name]
        modes = d['modes']['modes']
        sp_had = d['hadronic']
        color = d['color']
        y_base = y_offsets[name]

        # Amplitude array (hadronic scenario)
        A = sp_had['amplitudes']

        # Compute mode power: P_n * A_n^2 (Watts) — use log scale
        powers = []
        energies = []
        for i, m in enumerate(modes[:6]):
            P_mode = m['P_unit'] * A[i]**2
            energies.append(m['E_MeV'])
            powers.append(P_mode)

        # Normalize power to max for this particle (for bar heights)
        powers = np.array(powers)
        p_max = powers.max()
        if p_max <= 0:
            continue
        p_norm = powers / p_max

        # Draw sticks (vertical bars from baseline)
        for i, (E, pn) in enumerate(zip(energies, p_norm)):
            h = pn * bar_height
            ax1.bar(E, h, bottom=y_base, width=E * 0.15,
                    color=color, alpha=0.85 - 0.08 * i, zorder=3,
                    edgecolor=color, linewidth=0.5)

            # Mode label on tallest bars
            if i < 3:
                ax1.text(E, y_base + h + 0.04, f'n={i+1}',
                         fontsize=7, color=color, alpha=0.7,
                         ha='center', va='bottom')

        # Particle label
        E1 = energies[0]
        label_E = E1 * 0.15
        ax1.text(label_E, y_base + 0.35, f'{name}',
                 fontsize=10, fontweight='bold', color=color,
                 ha='right', va='center', alpha=0.9)
        ax1.text(label_E, y_base + 0.12,
                 f'E₁ = {E1*1000:.0f} keV' if E1 < 1 else f'E₁ = {E1:.1f} MeV',
                 fontsize=7.5, color=color, ha='right', va='center', alpha=0.6)

    ax1.set_xscale('log')
    ax1.set_xlim(0.008, 5000)
    ax1.set_ylim(-0.15, 3.1)
    ax1.set_yticks([])
    ax1.set_xlabel('Photon energy (MeV)', fontsize=11)
    ax1.set_title('Settling Radiation — Kelvin Wave Mode Energies (hadronic scenario)',
                   fontsize=12.5, fontweight='bold', pad=10)

    # Horizontal baselines
    for name, yb in y_offsets.items():
        ax1.axhline(yb, color='#333333', linewidth=0.5, zorder=1)

    # Energy scale reference lines
    for E_ref, label in [(0.511, 'm$_e$'), (139.6, 'm$_{\\pi}$'),
                          (938.3, 'm$_p$')]:
        ax1.axvline(E_ref, color='#333333', linewidth=0.6, linestyle=':',
                    zorder=0)
        ax1.text(E_ref, 3.0, label, fontsize=7, color='#555555',
                 ha='center', va='bottom')

    # ================================================================
    # Panel 2: Excess ratio — NWT prediction vs experimental data
    # ================================================================

    # Experimental data
    # (name, pT_center, pT_half_width, ratio, err_lo, err_hi, year, color)
    experiments = [
        ('WA102',  5,    5,    4.5,  0.5,  0.5,  '2002', '#4FC3F7'),
        ('WA83',   30,   30,   5.5,  1.5,  1.5,  '1994', '#81C784'),
        ('WA91',   30,   25,   4.5,  1.5,  1.5,  '1996', '#AED581'),
        ('DELPHI', 40,   40,   3.4,  0.8,  0.8,  '2006', '#FFD54F'),
        ('ALICE',  200,  200,  3.25, 1.75, 1.75, '2012', '#EF9A9A'),
    ]

    # Offset WA83/WA91 vertically slightly so they don't overlap
    y_nudge = {'WA83': 0.15, 'WA91': -0.15}

    for exp in experiments:
        name, pt_c, pt_hw, ratio, err_lo, err_hi, year, color = exp
        nudge = y_nudge.get(name, 0)

        # Shaded band for pT range
        ax2.fill_between([max(pt_c - pt_hw, 0.3), pt_c + pt_hw],
                         ratio - err_lo + nudge, ratio + err_hi + nudge,
                         alpha=0.12, color=color, zorder=1)

        # Data point with error bars
        ax2.errorbar(pt_c, ratio + nudge,
                     yerr=[[err_lo], [err_hi]],
                     xerr=[[min(pt_hw, pt_c - 0.3)], [pt_hw]],
                     fmt='s', color=color, markersize=7, capsize=3,
                     linewidth=1.5, markeredgecolor='white',
                     markeredgewidth=0.5, zorder=5,
                     label=f'{name} ({year})')

    # NWT prediction band
    nwt_ratio = (0.12 / 0.05)**2  # = 5.76
    ax2.axhspan(nwt_ratio * 0.85, nwt_ratio * 1.15,
                alpha=0.18, color=C_HADRON, zorder=2)
    ax2.axhline(nwt_ratio, color=C_HADRON, linewidth=2.2,
                linestyle='-', zorder=3,
                label=f'NWT settling: (A_had/A_thr)² = {nwt_ratio:.1f}x')

    # IB baseline
    ax2.axhline(1.0, color=C_IB, linewidth=1.2, linestyle='--',
                alpha=0.5, zorder=2, label='Inner bremsstrahlung = 1')

    ax2.set_xscale('log')
    ax2.set_xlim(0.3, 800)
    ax2.set_ylim(0, 9)
    ax2.set_xlabel('Photon pT or E cutoff (MeV)', fontsize=11)
    ax2.set_ylabel('Observed / IB prediction', fontsize=11)
    ax2.set_title('Anomalous Soft Photon Excess — Experiments vs NWT Prediction',
                   fontsize=12.5, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.4,
               edgecolor='#404040', ncol=2, handlelength=1.8)
    ax2.grid(True, alpha=0.12)

    # Mechanism annotation
    ax2.text(0.025, 0.92,
             'Kelvin wave settling of\n'
             'deformed hadronic vortices.\n'
             'Ratio depends only on\n'
             'amplitude ratio A_had/A_thr.',
             transform=ax2.transAxes, fontsize=7.5, color=C_TEXT,
             alpha=0.45, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#151515',
                       edgecolor='#2a2a2a', alpha=0.8))

    # Save
    outpath = os.path.join(OUTDIR, 'nwt_settling_spectrum.png')
    fig.savefig(outpath, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved: {outpath}")
    plt.close(fig)
    return outpath


if __name__ == '__main__':
    print("Computing settling radiation spectra...")
    results = compute_all()
    print("Generating figure...")
    outpath = make_figure(results)
    print(f"Done. Output: {outpath}")
