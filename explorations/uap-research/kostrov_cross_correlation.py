#!/usr/bin/env python3
"""
Kostrov Hypothesis Cross-Correlation Test
==========================================
Tests whether meteor shower dust, settling through the atmosphere with a
~12-day delay (Kostrov 2020), correlates with thunderstorm activity.

Data sources:
  - RMOB daily meteor activity index (from rmob_parser.py)
  - NOAA StormEvents daily thunderstorm counts (US, 2000-2020)
  - [Future] WGLC global lightning stroke density (2010-2024)

Method:
  1. Parse NOAA StormEvents into daily thunderstorm event counts
  2. Load RMOB daily meteor activity index
  3. Cross-correlate at lags 0-60 days
  4. Test significance via bootstrap/shuffle
  5. Compare observed lag against Kostrov's predicted 10-15 days

Author: Claude (Opus 4.6) for Jim Galasyn
Date: 2026-02-20
"""

import os
import csv
import gzip
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from pathlib import Path

# ============================================================================
# NOAA STORMEVENTS PARSER
# ============================================================================

# Event types associated with thunderstorm activity
THUNDER_EVENTS = {
    'Thunderstorm Wind',
    'Lightning',
    'Hail',
    'Tornado',
    'Funnel Cloud',
}

# Broader convective events
CONVECTIVE_EVENTS = THUNDER_EVENTS | {
    'Flash Flood',
    'Heavy Rain',
}


def parse_stormevents_date(date_str: str) -> date:
    """Parse StormEvents date format like '31-DEC-00 06:00:00' or '01-JAN-10 12:00:00'."""
    try:
        dt = datetime.strptime(date_str.strip().strip('"'), '%d-%b-%y %H:%M:%S')
        # Fix Y2K: years 00-49 -> 2000-2049, 50-99 -> 1950-1999
        if dt.year < 100:
            if dt.year < 50:
                dt = dt.replace(year=dt.year + 2000)
            else:
                dt = dt.replace(year=dt.year + 1900)
        return dt.date()
    except (ValueError, AttributeError):
        return None


def parse_stormevents_file(filepath: str, event_filter: set = None) -> dict:
    """
    Parse a single NOAA StormEvents CSV (gzipped) into daily event counts.

    Args:
        filepath: Path to .csv.gz file
        event_filter: Set of EVENT_TYPE strings to include (None = all)

    Returns:
        dict mapping date -> count of matching events
    """
    if event_filter is None:
        event_filter = THUNDER_EVENTS

    daily_counts = defaultdict(int)

    open_fn = gzip.open if filepath.endswith('.gz') else open
    try:
        with open_fn(filepath, 'rt', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_type = row.get('EVENT_TYPE', '').strip().strip('"')
                if event_type not in event_filter:
                    continue

                date_str = row.get('BEGIN_DATE_TIME', '')
                d = parse_stormevents_date(date_str)
                if d is not None:
                    daily_counts[d] += 1
    except Exception as e:
        print(f"  Warning: Error reading {filepath}: {e}", flush=True)

    return dict(daily_counts)


def load_all_stormevents(data_dir: str, event_filter: set = None) -> dict:
    """Load all StormEvents files and merge into single daily time series."""
    all_counts = defaultdict(int)

    gz_files = sorted(Path(data_dir).glob('*.csv.gz'))
    print(f"  Found {len(gz_files)} StormEvents files", flush=True)

    for fp in gz_files:
        print(f"  Parsing: {fp.name}", flush=True)
        counts = parse_stormevents_file(str(fp), event_filter)
        for d, n in counts.items():
            all_counts[d] += n

    return dict(all_counts)


# ============================================================================
# RMOB DATA LOADER
# ============================================================================

def load_rmob_timeseries(csv_path: str) -> dict:
    """Load the RMOB daily meteor activity index from CSV."""
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
# CROSS-CORRELATION ANALYSIS
# ============================================================================

def build_aligned_series(meteor_data: dict, storm_data: dict,
                         min_observers: int = 5) -> tuple:
    """
    Build aligned daily arrays for cross-correlation.

    Only includes dates present in both datasets where RMOB has
    sufficient observer coverage.

    Returns (dates, meteor_activity, storm_counts) as aligned arrays.
    """
    common_dates = sorted(
        set(meteor_data.keys()) & set(storm_data.keys())
    )

    dates = []
    meteor = []
    storms = []

    for d in common_dates:
        if meteor_data[d]['n_observers'] >= min_observers:
            dates.append(d)
            meteor.append(meteor_data[d]['activity_index'])
            storms.append(storm_data[d])

    return np.array(dates), np.array(meteor), np.array(storms)


def cross_correlate(x: np.ndarray, y: np.ndarray, max_lag: int = 60) -> tuple:
    """
    Compute normalized cross-correlation between x and y at lags 0..max_lag.

    Positive lag means x leads y (meteor activity precedes storm activity).

    Returns (lags, correlations).
    """
    # Detrend and normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

    n = len(x_norm)
    lags = np.arange(0, max_lag + 1)
    correlations = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            correlations[i] = np.mean(x_norm * y_norm)
        else:
            # x leads y by 'lag' days: correlate x[:-lag] with y[lag:]
            correlations[i] = np.mean(x_norm[:-lag] * y_norm[lag:])

    return lags, correlations


def bootstrap_significance(x: np.ndarray, y: np.ndarray, max_lag: int = 60,
                           n_bootstrap: int = 1000, seed: int = 42) -> tuple:
    """
    Estimate significance of cross-correlation via block bootstrap.

    Shuffles the meteor time series in ~30-day blocks (preserving
    autocorrelation structure) and recomputes cross-correlation.

    Returns (p95_upper, p99_upper) significance thresholds.
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    block_size = 30  # ~1 month blocks

    bootstrap_maxima = []

    for _ in range(n_bootstrap):
        # Block shuffle x
        n_blocks = n // block_size + 1
        block_indices = rng.permutation(n_blocks)
        shuffled_x = np.concatenate([
            x[i * block_size:(i + 1) * block_size]
            for i in block_indices
        ])[:n]

        _, corr = cross_correlate(shuffled_x, y, max_lag)
        bootstrap_maxima.append(np.max(np.abs(corr)))

    p95 = np.percentile(bootstrap_maxima, 95)
    p99 = np.percentile(bootstrap_maxima, 99)

    return p95, p99


def seasonal_decompose(dates: np.ndarray, values: np.ndarray,
                       period: int = 365) -> tuple:
    """
    Remove seasonal cycle to isolate anomalies.

    Returns (seasonal_component, anomaly).
    """
    doy = np.array([d.timetuple().tm_yday for d in dates])

    # Compute mean for each day-of-year
    seasonal = np.zeros(period + 1)
    counts = np.zeros(period + 1)
    for i, d in enumerate(doy):
        if d <= period:
            seasonal[d] += values[i]
            counts[d] += 1

    # Average
    mask = counts > 0
    seasonal[mask] /= counts[mask]

    # Smooth the seasonal cycle (15-day window)
    kernel = np.ones(15) / 15
    seasonal_smooth = np.convolve(seasonal, kernel, mode='same')

    # Extract seasonal component for each date
    seasonal_vals = np.array([seasonal_smooth[d] for d in doy])
    anomaly = values - seasonal_vals

    return seasonal_vals, anomaly


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(dates, meteor, storms, lags, corr,
                 lags_anom, corr_anom, p95, p99,
                 p95_anom, p99_anom, output_dir):
    """Generate comprehensive analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # ---- Plot 1: Cross-correlation results ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: Raw cross-correlation
    ax = axes[0, 0]
    ax.bar(lags, corr, width=0.8, color='steelblue', alpha=0.7)
    ax.axhline(p95, color='orange', linestyle='--', label=f'95% significance ({p95:.4f})')
    ax.axhline(p99, color='red', linestyle='--', label=f'99% significance ({p99:.4f})')
    ax.axhline(-p95, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(-p99, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(10, 15, alpha=0.15, color='green', label='Kostrov predicted lag')
    ax.set_xlabel('Lag (days, meteor leads storms)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Raw Cross-Correlation: Meteor Activity → Thunderstorms')
    ax.legend(fontsize=8)

    # 1b: Anomaly cross-correlation (seasonal removed)
    ax = axes[0, 1]
    ax.bar(lags_anom, corr_anom, width=0.8, color='darkred', alpha=0.7)
    ax.axhline(p95_anom, color='orange', linestyle='--',
               label=f'95% significance ({p95_anom:.4f})')
    ax.axhline(p99_anom, color='red', linestyle='--',
               label=f'99% significance ({p99_anom:.4f})')
    ax.axhline(-p95_anom, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(-p99_anom, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(10, 15, alpha=0.15, color='green', label='Kostrov predicted lag')
    ax.set_xlabel('Lag (days, meteor leads storms)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Anomaly Cross-Correlation (Seasonal Cycle Removed)')
    ax.legend(fontsize=8)

    # 1c: Time series overlay (one representative year)
    ax = axes[1, 0]
    rep_year = 2015
    mask = np.array([d.year == rep_year for d in dates])
    if np.sum(mask) > 30:
        rep_dates = dates[mask]
        rep_meteor = meteor[mask]
        rep_storms = storms[mask]

        ax2 = ax.twinx()
        ax.plot(rep_dates, rep_meteor, color='steelblue', linewidth=1,
                label='Meteor activity', alpha=0.7)

        # 7-day smoothed storms
        if len(rep_storms) > 7:
            kernel7 = np.ones(7) / 7
            sm_storms = np.convolve(rep_storms, kernel7, mode='valid')
            sm_dates = rep_dates[3:3 + len(sm_storms)]
            ax2.plot(sm_dates, sm_storms, color='red', linewidth=1.5,
                     label='Thunderstorms (7d avg)')

        ax.set_ylabel('Meteor Activity Index', color='steelblue')
        ax2.set_ylabel('Daily Storm Events', color='red')
        ax.set_title(f'{rep_year} — Meteor Activity vs Thunderstorm Events')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # 1d: Annual profiles overlaid
    ax = axes[1, 1]
    # Meteor annual profile
    doy_meteor = defaultdict(list)
    doy_storms = defaultdict(list)
    for i, d in enumerate(dates):
        doy = d.timetuple().tm_yday
        doy_meteor[doy].append(meteor[i])
        doy_storms[doy].append(storms[i])

    doys = sorted(set(doy_meteor.keys()) & set(doy_storms.keys()))
    mean_meteor = [np.mean(doy_meteor[d]) for d in doys]
    mean_storms = [np.mean(doy_storms[d]) for d in doys]

    # Smooth both
    kernel15 = np.ones(15) / 15
    if len(mean_meteor) > 15:
        sm_meteor = np.convolve(mean_meteor, kernel15, mode='valid')
        sm_storms = np.convolve(mean_storms, kernel15, mode='valid')
        sm_doys = doys[7:7 + len(sm_meteor)]

        ax2 = ax.twinx()
        ax.plot(sm_doys, sm_meteor, color='steelblue', linewidth=2,
                label='Meteor activity')

        # Shift storms back by 12 days for comparison
        shifted_doys = [d - 12 for d in sm_doys]
        ax2.plot(shifted_doys, sm_storms, color='red', linewidth=2,
                 linestyle='--', label='Storms (shifted -12 days)')
        ax2.plot(sm_doys, sm_storms, color='red', linewidth=1,
                 alpha=0.3, label='Storms (actual)')

        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Meteor Activity Index', color='steelblue')
        ax2.set_ylabel('Daily Storm Events', color='red')
        ax.set_title('Annual Profiles: Meteor vs Storms (red dashed = storms shifted -12d)')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kostrov_cross_correlation.png'), dpi=150)
    plt.close()
    print(f"  Saved: kostrov_cross_correlation.png", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys

    output_dir = os.path.dirname(os.path.abspath(__file__))
    rmob_csv = os.path.join(output_dir, 'rmob_daily_meteor_flux.csv')
    noaa_dir = '/tmp/noaa_stormevents'

    print("=" * 70, flush=True)
    print("Kostrov Hypothesis: Meteor-Thunderstorm Cross-Correlation", flush=True)
    print("=" * 70, flush=True)

    # Load RMOB data
    print(f"\nLoading RMOB meteor activity index...", flush=True)
    meteor_data = load_rmob_timeseries(rmob_csv)
    print(f"  Loaded {len(meteor_data)} days of meteor data", flush=True)

    # Load NOAA StormEvents
    print(f"\nLoading NOAA StormEvents (thunderstorm-related)...", flush=True)
    storm_data = load_all_stormevents(noaa_dir, THUNDER_EVENTS)
    print(f"  Loaded {len(storm_data)} days of storm data", flush=True)
    total_events = sum(storm_data.values())
    print(f"  Total thunderstorm events: {total_events:,}", flush=True)

    # Align series
    print(f"\nAligning time series...", flush=True)
    dates, meteor, storms = build_aligned_series(meteor_data, storm_data,
                                                  min_observers=5)
    print(f"  Aligned days: {len(dates)}", flush=True)
    print(f"  Date range: {dates[0]} to {dates[-1]}", flush=True)
    print(f"  Mean meteor activity: {np.mean(meteor):.3f}", flush=True)
    print(f"  Mean daily storm events: {np.mean(storms):.1f}", flush=True)

    if len(dates) < 365:
        print("ERROR: Insufficient overlapping data for meaningful analysis.", flush=True)
        sys.exit(1)

    # Cross-correlation (raw)
    max_lag = 60
    print(f"\nComputing raw cross-correlation (lags 0-{max_lag} days)...", flush=True)
    lags, corr = cross_correlate(meteor, storms, max_lag)

    peak_lag = lags[np.argmax(corr)]
    peak_corr = np.max(corr)
    print(f"  Peak correlation: {peak_corr:.4f} at lag {peak_lag} days", flush=True)

    # Significance testing
    print(f"\nBootstrap significance testing (1000 iterations)...", flush=True)
    p95, p99 = bootstrap_significance(meteor, storms, max_lag)
    print(f"  95% significance threshold: {p95:.4f}", flush=True)
    print(f"  99% significance threshold: {p99:.4f}", flush=True)

    significant = peak_corr > p95
    print(f"  Peak is {'SIGNIFICANT' if significant else 'not significant'} at 95% level",
          flush=True)

    # Check Kostrov window specifically
    kostrov_window = corr[(lags >= 10) & (lags <= 15)]
    kostrov_max = np.max(kostrov_window)
    kostrov_lag = lags[(lags >= 10) & (lags <= 15)][np.argmax(kostrov_window)]
    print(f"\n  Kostrov window (10-15 days):", flush=True)
    print(f"    Max correlation: {kostrov_max:.4f} at lag {kostrov_lag} days", flush=True)
    print(f"    Significant: {'YES' if kostrov_max > p95 else 'NO'}", flush=True)

    # Seasonal decomposition
    print(f"\nRemoving seasonal cycle...", flush=True)
    meteor_seasonal, meteor_anomaly = seasonal_decompose(dates, meteor)
    storm_seasonal, storm_anomaly = seasonal_decompose(dates, storms)

    # Cross-correlation on anomalies
    print(f"Computing anomaly cross-correlation...", flush=True)
    lags_anom, corr_anom = cross_correlate(meteor_anomaly, storm_anomaly, max_lag)

    peak_lag_anom = lags_anom[np.argmax(corr_anom)]
    peak_corr_anom = np.max(corr_anom)
    print(f"  Peak anomaly correlation: {peak_corr_anom:.4f} at lag {peak_lag_anom} days",
          flush=True)

    # Anomaly significance
    print(f"Bootstrap significance for anomalies...", flush=True)
    p95_anom, p99_anom = bootstrap_significance(meteor_anomaly, storm_anomaly, max_lag)
    print(f"  95% threshold: {p95_anom:.4f}", flush=True)
    print(f"  99% threshold: {p99_anom:.4f}", flush=True)

    anom_significant = peak_corr_anom > p95_anom
    print(f"  Peak is {'SIGNIFICANT' if anom_significant else 'not significant'} at 95% level",
          flush=True)

    kostrov_anom = corr_anom[(lags_anom >= 10) & (lags_anom <= 15)]
    kostrov_anom_max = np.max(kostrov_anom)
    kostrov_anom_lag = lags_anom[(lags_anom >= 10) & (lags_anom <= 15)][np.argmax(kostrov_anom)]
    print(f"\n  Kostrov window (anomaly):", flush=True)
    print(f"    Max correlation: {kostrov_anom_max:.4f} at lag {kostrov_anom_lag} days", flush=True)
    print(f"    Significant: {'YES' if kostrov_anom_max > p95_anom else 'NO'}", flush=True)

    # Plot
    print(f"\nGenerating plots...", flush=True)
    plot_results(dates, meteor, storms, lags, corr,
                 lags_anom, corr_anom, p95, p99,
                 p95_anom, p99_anom, output_dir)

    # Save results
    results = {
        'date_range': f"{dates[0]} to {dates[-1]}",
        'n_days': len(dates),
        'raw_correlation': {
            'peak_lag': int(peak_lag),
            'peak_value': float(peak_corr),
            'p95_threshold': float(p95),
            'p99_threshold': float(p99),
            'significant_95': bool(significant),
            'kostrov_window_max': float(kostrov_max),
            'kostrov_window_lag': int(kostrov_lag),
            'kostrov_significant': bool(kostrov_max > p95),
        },
        'anomaly_correlation': {
            'peak_lag': int(peak_lag_anom),
            'peak_value': float(peak_corr_anom),
            'p95_threshold': float(p95_anom),
            'p99_threshold': float(p99_anom),
            'significant_95': bool(anom_significant),
            'kostrov_window_max': float(kostrov_anom_max),
            'kostrov_window_lag': int(kostrov_anom_lag),
            'kostrov_significant': bool(kostrov_anom_max > p95_anom),
        },
        'all_lags_raw': {int(l): float(c) for l, c in zip(lags, corr)},
        'all_lags_anomaly': {int(l): float(c) for l, c in zip(lags_anom, corr_anom)},
    }

    import json
    results_path = os.path.join(output_dir, 'kostrov_cross_correlation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}", flush=True)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Data: {len(dates)} days ({dates[0]} to {dates[-1]})", flush=True)
    print(f"  Raw correlation peak: r={peak_corr:.4f} at lag {peak_lag}d "
          f"({'*' if significant else 'ns'})", flush=True)
    print(f"  Anomaly correlation peak: r={peak_corr_anom:.4f} at lag {peak_lag_anom}d "
          f"({'*' if anom_significant else 'ns'})", flush=True)
    print(f"  Kostrov window (10-15d):", flush=True)
    print(f"    Raw: r={kostrov_max:.4f} at {kostrov_lag}d "
          f"({'SUPPORTS' if kostrov_max > p95 else 'does not support'} hypothesis)", flush=True)
    print(f"    Anomaly: r={kostrov_anom_max:.4f} at {kostrov_anom_lag}d "
          f"({'SUPPORTS' if kostrov_anom_max > p95_anom else 'does not support'} hypothesis)",
          flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == '__main__':
    main()
