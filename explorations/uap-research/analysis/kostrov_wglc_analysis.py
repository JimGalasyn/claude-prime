#!/usr/bin/env python3
"""
Kostrov Hypothesis Test: RMOB Meteor Activity vs WGLC Global Lightning
======================================================================
The definitive test using global lightning stroke density (WGLC, 2010-2024)
cross-correlated with RMOB radio meteor activity (2000-2020).

Overlap period: 2010-2020 (11 years, ~4000 days).

This is the stronger test vs the NOAA StormEvents analysis because:
1. Global coverage (not US-only)
2. Actual stroke density (not discrete event reports)
3. Continuous sensor data (WWLLN network)

Author: Claude (Opus 4.6) for Jim Galasyn
Date: 2026-02-20
"""

import os
import csv
import json
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from pathlib import Path

# ============================================================================
# DATA LOADING
# ============================================================================

def load_wglc_daily_global(nc_path: str, end_year: int = 2020) -> dict:
    """
    Load WGLC NetCDF and compute global daily lightning stroke total.

    Processes in yearly chunks to manage memory (~1.1 GB per year).

    Returns dict mapping date -> global_stroke_density_sum.
    """
    import xarray as xr

    print(f"  Opening: {nc_path}", flush=True)
    ds = xr.open_dataset(nc_path)

    # Get time range
    times = ds.time.values
    print(f"  Full range: {str(times[0])[:10]} to {str(times[-1])[:10]}", flush=True)
    print(f"  Processing up to {end_year}...", flush=True)

    daily_totals = {}

    # Process year by year to manage memory
    for year in range(2010, end_year + 1):
        print(f"    {year}...", end=' ', flush=True)
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"

        # Select one year of density data
        density = ds['density'].sel(time=slice(year_start, year_end))

        # Sum over all grid cells for each day (global total)
        # Use nansum to handle missing values
        global_daily = density.sum(dim=['lat', 'lon']).values

        # Get corresponding dates
        year_times = density.time.values

        count = 0
        for t, val in zip(year_times, global_daily):
            d = np.datetime64(t, 'D')
            py_date = date.fromisoformat(str(d))
            if not np.isnan(val) and val > 0:
                daily_totals[py_date] = float(val)
                count += 1

        print(f"{count} days", flush=True)

        # Free memory
        del density, global_daily, year_times

    ds.close()
    return daily_totals


def load_rmob_timeseries(csv_path: str) -> dict:
    """Load RMOB daily meteor activity index."""
    timeseries = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = date.fromisoformat(row['date'])
            timeseries[d] = {
                'activity_index': float(row['activity_index']),
                'n_observers': int(row['n_observers']),
            }
    return timeseries


# ============================================================================
# ANALYSIS
# ============================================================================

def build_aligned_series(meteor_data: dict, lightning_data: dict,
                         min_observers: int = 5) -> tuple:
    """Build aligned daily arrays for cross-correlation."""
    common_dates = sorted(set(meteor_data.keys()) & set(lightning_data.keys()))

    dates, meteor, lightning = [], [], []
    for d in common_dates:
        if meteor_data[d]['n_observers'] >= min_observers:
            dates.append(d)
            meteor.append(meteor_data[d]['activity_index'])
            lightning.append(lightning_data[d])

    return np.array(dates), np.array(meteor), np.array(lightning)


def cross_correlate(x: np.ndarray, y: np.ndarray, max_lag: int = 60) -> tuple:
    """Normalized cross-correlation at lags 0..max_lag (x leads y)."""
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

    n = len(x_norm)
    lags = np.arange(0, max_lag + 1)
    correlations = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            correlations[i] = np.mean(x_norm * y_norm)
        else:
            correlations[i] = np.mean(x_norm[:-lag] * y_norm[lag:])

    return lags, correlations


def bootstrap_significance(x: np.ndarray, y: np.ndarray, max_lag: int = 60,
                           n_bootstrap: int = 1000, seed: int = 42) -> tuple:
    """Block bootstrap significance testing (30-day blocks)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    block_size = 30

    bootstrap_maxima = []
    for _ in range(n_bootstrap):
        n_blocks = n // block_size + 1
        block_indices = rng.permutation(n_blocks)
        shuffled_x = np.concatenate([
            x[i * block_size:(i + 1) * block_size]
            for i in block_indices
        ])[:n]
        _, corr = cross_correlate(shuffled_x, y, max_lag)
        bootstrap_maxima.append(np.max(np.abs(corr)))

    return np.percentile(bootstrap_maxima, 95), np.percentile(bootstrap_maxima, 99)


def seasonal_decompose(dates: np.ndarray, values: np.ndarray,
                       period: int = 366) -> tuple:
    """Remove seasonal cycle, return (seasonal, anomaly)."""
    doy = np.array([d.timetuple().tm_yday for d in dates])

    seasonal = np.zeros(period + 1)
    counts = np.zeros(period + 1)
    for i, d in enumerate(doy):
        if d <= period:
            seasonal[d] += values[i]
            counts[d] += 1

    mask = counts > 0
    seasonal[mask] /= counts[mask]

    # 15-day smooth
    kernel = np.ones(15) / 15
    seasonal_smooth = np.convolve(seasonal, kernel, mode='same')

    seasonal_vals = np.array([seasonal_smooth[d] for d in doy])
    anomaly = values - seasonal_vals

    return seasonal_vals, anomaly


def scan_lag_windows(lags, corr, corr_anom, p95, p95_anom):
    """Scan different lag windows for significant correlations."""
    windows = [
        ('Kostrov (10-15d)', 10, 15),
        ('Extended (8-20d)', 8, 20),
        ('Short (3-7d)', 3, 7),
        ('Medium (20-30d)', 20, 30),
        ('Long (30-45d)', 30, 45),
    ]
    results = []
    for name, lo, hi in windows:
        mask = (lags >= lo) & (lags <= hi)
        raw_max = np.max(corr[mask])
        raw_lag = lags[mask][np.argmax(corr[mask])]
        anom_max = np.max(corr_anom[mask])
        anom_lag = lags[mask][np.argmax(corr_anom[mask])]
        results.append({
            'window': name,
            'raw_max': float(raw_max),
            'raw_lag': int(raw_lag),
            'raw_sig': bool(raw_max > p95),
            'anom_max': float(anom_max),
            'anom_lag': int(anom_lag),
            'anom_sig': bool(anom_max > p95_anom),
        })
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(dates, meteor, lightning, lags, corr,
                 lags_anom, corr_anom, p95, p99, p95_anom, p99_anom,
                 output_dir):
    """Generate comprehensive analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: Raw cross-correlation
    ax = axes[0, 0]
    colors = ['green' if 10 <= l <= 15 else 'steelblue' for l in lags]
    ax.bar(lags, corr, width=0.8, color=colors, alpha=0.7)
    ax.axhline(p95, color='orange', linestyle='--', label=f'95% ({p95:.4f})')
    ax.axhline(p99, color='red', linestyle='--', label=f'99% ({p99:.4f})')
    ax.axhline(-p95, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(-p99, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(10, 15, alpha=0.1, color='green', label='Kostrov window')
    ax.set_xlabel('Lag (days, meteor leads lightning)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Raw: RMOB Meteor Activity → WGLC Global Lightning')
    ax.legend(fontsize=7)

    # 1b: Anomaly cross-correlation
    ax = axes[0, 1]
    colors_a = ['green' if 10 <= l <= 15 else 'darkred' for l in lags_anom]
    ax.bar(lags_anom, corr_anom, width=0.8, color=colors_a, alpha=0.7)
    ax.axhline(p95_anom, color='orange', linestyle='--', label=f'95% ({p95_anom:.4f})')
    ax.axhline(p99_anom, color='red', linestyle='--', label=f'99% ({p99_anom:.4f})')
    ax.axhline(-p95_anom, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(-p99_anom, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(10, 15, alpha=0.1, color='green', label='Kostrov window')
    ax.set_xlabel('Lag (days, meteor leads lightning)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Anomaly: Seasonal Cycle Removed')
    ax.legend(fontsize=7)

    # 1c: Time series overlay (2015)
    ax = axes[1, 0]
    rep_year = 2015
    mask = np.array([d.year == rep_year for d in dates])
    if np.sum(mask) > 30:
        rep_dates = dates[mask]
        rep_meteor = meteor[mask]
        rep_lightning = lightning[mask]

        ax2 = ax.twinx()
        ax.plot(rep_dates, rep_meteor, color='steelblue', linewidth=1,
                alpha=0.5, label='Meteor activity (raw)')

        # 7-day smooth
        if len(rep_meteor) > 7:
            k7 = np.ones(7) / 7
            sm_m = np.convolve(rep_meteor, k7, mode='valid')
            sm_l = np.convolve(rep_lightning, k7, mode='valid')
            sm_d = rep_dates[3:3 + len(sm_m)]
            ax.plot(sm_d, sm_m, color='darkblue', linewidth=2, label='Meteor (7d avg)')
            ax2.plot(sm_d, sm_l, color='red', linewidth=2, label='Lightning (7d avg)')

        ax.set_ylabel('Meteor Activity Index', color='steelblue')
        ax2.set_ylabel('Global Lightning Strokes/day', color='red')
        ax.set_title(f'{rep_year} — Meteor Activity vs Global Lightning')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # 1d: Annual profiles
    ax = axes[1, 1]
    doy_m = defaultdict(list)
    doy_l = defaultdict(list)
    for i, d in enumerate(dates):
        doy = d.timetuple().tm_yday
        doy_m[doy].append(meteor[i])
        doy_l[doy].append(lightning[i])

    doys = sorted(set(doy_m.keys()) & set(doy_l.keys()))
    mean_m = [np.mean(doy_m[d]) for d in doys]
    mean_l = [np.mean(doy_l[d]) for d in doys]

    k15 = np.ones(15) / 15
    if len(mean_m) > 15:
        sm_m = np.convolve(mean_m, k15, mode='valid')
        sm_l = np.convolve(mean_l, k15, mode='valid')
        sm_doys = doys[7:7 + len(sm_m)]

        ax2 = ax.twinx()
        ax.plot(sm_doys, sm_m, color='darkblue', linewidth=2, label='Meteor activity')

        # Show storms both actual and shifted
        shifted_doys = [d - 12 for d in sm_doys]
        ax2.plot(sm_doys, sm_l, color='red', linewidth=1, alpha=0.4,
                 label='Lightning (actual)')
        ax2.plot(shifted_doys, sm_l, color='red', linewidth=2, linestyle='--',
                 label='Lightning (shifted -12d)')

        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Meteor Activity Index', color='darkblue')
        ax2.set_ylabel('Global Stroke Density', color='red')
        ax.set_title('Annual Profiles: Meteor vs Global Lightning')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'kostrov_wglc_cross_correlation.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    os.makedirs(output_dir, exist_ok=True)
    rmob_csv = os.path.join(output_dir, 'rmob_daily_meteor_flux.csv')
    wglc_path = str(repo_root / "data" / "wglc" / "wglc_timeseries_30m_daily.nc")

    print("=" * 70, flush=True)
    print("Kostrov Hypothesis: RMOB Meteors vs WGLC Global Lightning", flush=True)
    print("=" * 70, flush=True)

    # Load RMOB
    print(f"\nLoading RMOB meteor activity...", flush=True)
    meteor_data = load_rmob_timeseries(rmob_csv)
    print(f"  {len(meteor_data)} days loaded", flush=True)

    # Load WGLC
    print(f"\nLoading WGLC global lightning density...", flush=True)
    lightning_data = load_wglc_daily_global(wglc_path, end_year=2020)
    print(f"  {len(lightning_data)} days loaded", flush=True)

    # Align
    print(f"\nAligning time series...", flush=True)
    dates, meteor, lightning = build_aligned_series(meteor_data, lightning_data,
                                                     min_observers=5)
    print(f"  Aligned: {len(dates)} days", flush=True)
    print(f"  Range: {dates[0]} to {dates[-1]}", flush=True)
    print(f"  Mean meteor activity: {np.mean(meteor):.3f}", flush=True)
    print(f"  Mean global strokes/day: {np.mean(lightning):.0f}", flush=True)
    print(f"  Std global strokes/day: {np.std(lightning):.0f}", flush=True)

    if len(dates) < 365:
        print("ERROR: Insufficient data.", flush=True)
        sys.exit(1)

    # Raw cross-correlation
    max_lag = 60
    print(f"\nRaw cross-correlation (lags 0-{max_lag}d)...", flush=True)
    lags, corr = cross_correlate(meteor, lightning, max_lag)
    peak_lag = lags[np.argmax(corr)]
    peak_corr = np.max(corr)
    print(f"  Peak: r={peak_corr:.4f} at lag {peak_lag}d", flush=True)

    # Negative correlations too
    neg_peak_lag = lags[np.argmin(corr)]
    neg_peak_corr = np.min(corr)
    print(f"  Most negative: r={neg_peak_corr:.4f} at lag {neg_peak_lag}d", flush=True)

    # Bootstrap
    print(f"\nBootstrap significance (1000 iterations)...", flush=True)
    p95, p99 = bootstrap_significance(meteor, lightning, max_lag)
    print(f"  95%: {p95:.4f},  99%: {p99:.4f}", flush=True)
    print(f"  Peak {'SIGNIFICANT' if peak_corr > p95 else 'not significant'} at 95%",
          flush=True)

    # Seasonal decomposition
    print(f"\nSeasonal decomposition...", flush=True)
    _, meteor_anom = seasonal_decompose(dates, meteor)
    _, lightning_anom = seasonal_decompose(dates, lightning)

    lags_anom, corr_anom = cross_correlate(meteor_anom, lightning_anom, max_lag)
    peak_lag_anom = lags_anom[np.argmax(corr_anom)]
    peak_corr_anom = np.max(corr_anom)
    print(f"  Anomaly peak: r={peak_corr_anom:.4f} at lag {peak_lag_anom}d", flush=True)

    print(f"\nAnomaly bootstrap...", flush=True)
    p95_anom, p99_anom = bootstrap_significance(meteor_anom, lightning_anom, max_lag)
    print(f"  95%: {p95_anom:.4f},  99%: {p99_anom:.4f}", flush=True)
    print(f"  Peak {'SIGNIFICANT' if peak_corr_anom > p95_anom else 'not significant'} at 95%",
          flush=True)

    # Scan all lag windows
    print(f"\nLag window scan:", flush=True)
    windows = scan_lag_windows(lags, corr, corr_anom, p95, p95_anom)
    print(f"  {'Window':<22} {'Raw r':>8} {'Lag':>4} {'Sig':>4}  {'Anom r':>8} {'Lag':>4} {'Sig':>4}",
          flush=True)
    print(f"  {'-'*60}", flush=True)
    for w in windows:
        print(f"  {w['window']:<22} {w['raw_max']:>8.4f} {w['raw_lag']:>4}d "
              f"{'*' if w['raw_sig'] else ' ':>4}  {w['anom_max']:>8.4f} {w['anom_lag']:>4}d "
              f"{'*' if w['anom_sig'] else ' ':>4}", flush=True)

    # Plot
    print(f"\nGenerating plots...", flush=True)
    plot_results(dates, meteor, lightning, lags, corr,
                 lags_anom, corr_anom, p95, p99, p95_anom, p99_anom,
                 output_dir)

    # Save results
    results = {
        'data': {
            'date_range': f"{dates[0]} to {dates[-1]}",
            'n_days': len(dates),
            'mean_meteor_activity': float(np.mean(meteor)),
            'mean_global_strokes_per_day': float(np.mean(lightning)),
        },
        'raw_correlation': {
            'peak_lag': int(peak_lag),
            'peak_value': float(peak_corr),
            'negative_peak_lag': int(neg_peak_lag),
            'negative_peak_value': float(neg_peak_corr),
            'p95_threshold': float(p95),
            'p99_threshold': float(p99),
            'significant_95': bool(peak_corr > p95),
        },
        'anomaly_correlation': {
            'peak_lag': int(peak_lag_anom),
            'peak_value': float(peak_corr_anom),
            'p95_threshold': float(p95_anom),
            'p99_threshold': float(p99_anom),
            'significant_95': bool(peak_corr_anom > p95_anom),
        },
        'lag_windows': windows,
        'all_lags_raw': {int(l): float(c) for l, c in zip(lags, corr)},
        'all_lags_anomaly': {int(l): float(c) for l, c in zip(lags_anom, corr_anom)},
    }

    results_path = os.path.join(output_dir, 'kostrov_wglc_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}", flush=True)

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS SUMMARY — WGLC Global Lightning", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Data: {len(dates)} days, {dates[0]} to {dates[-1]}", flush=True)
    print(f"  Global mean: {np.mean(lightning):.0f} strokes/day", flush=True)
    print(f"", flush=True)
    print(f"  Raw correlation:", flush=True)
    print(f"    Peak: r={peak_corr:.4f} at {peak_lag}d "
          f"({'*' if peak_corr > p95 else 'ns'})", flush=True)
    print(f"    95% threshold: {p95:.4f}", flush=True)
    print(f"", flush=True)
    print(f"  Anomaly correlation:", flush=True)
    print(f"    Peak: r={peak_corr_anom:.4f} at {peak_lag_anom}d "
          f"({'*' if peak_corr_anom > p95_anom else 'ns'})", flush=True)
    print(f"    95% threshold: {p95_anom:.4f}", flush=True)
    print(f"", flush=True)
    print(f"  Kostrov window (10-15d):", flush=True)
    kw = [w for w in windows if 'Kostrov' in w['window']][0]
    print(f"    Raw: r={kw['raw_max']:.4f} at {kw['raw_lag']}d "
          f"({'SUPPORTS' if kw['raw_sig'] else 'does not support'})", flush=True)
    print(f"    Anomaly: r={kw['anom_max']:.4f} at {kw['anom_lag']}d "
          f"({'SUPPORTS' if kw['anom_sig'] else 'does not support'})", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == '__main__':
    main()
