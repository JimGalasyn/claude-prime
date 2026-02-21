#!/usr/bin/env python3
"""
RMOB (Radio Meteor Observation Bulletin) Parser
================================================
Parses the RMOB survey data (2000-2020) into a unified daily meteor flux
time series. Designed for cross-correlation with lightning data to test
Kostrov's hypothesis that meteor stream dust modulates the global electric
circuit with a ~12-day settling delay.

Data source: RMOB survey files from multiple radio meteor observers worldwide.
Each file contains hourly meteor echo counts for one observer for one month.

Output: Daily aggregate meteor flux CSV + diagnostic plots.

Author: Claude (Opus 4.6) for Jim Galasyn
Date: 2026-02-20
"""

import os
import re
import csv
import json
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# RMOB FILE PARSER
# ============================================================================

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'june': 6, 'july': 7, 'august': 8, 'september': 9,
    'october': 10, 'november': 11, 'december': 12,
}


@dataclass
class ObserverInfo:
    """Metadata for an RMOB observer."""
    name: str = ""
    country: str = ""
    city: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    frequency_mhz: Optional[float] = None
    antenna: str = ""
    method: str = ""


@dataclass
class RMOBFile:
    """Parsed contents of a single RMOB file."""
    filepath: str
    year: int
    month: int
    observer: ObserverInfo
    # hourly_counts[day][hour] = count (None if missing)
    hourly_counts: dict = field(default_factory=dict)

    @property
    def daily_totals(self) -> dict:
        """Sum hourly counts for each day, handling missing data."""
        totals = {}
        for day, hours in self.hourly_counts.items():
            valid = [v for v in hours.values() if v is not None]
            if valid:
                n_valid = len(valid)
                n_total = 24
                raw_sum = sum(valid)
                # Scale up to estimate full 24h if partial coverage
                if n_valid >= 6:  # Need at least 6 hours for reasonable estimate
                    totals[day] = {
                        'raw_sum': raw_sum,
                        'valid_hours': n_valid,
                        'estimated_24h': raw_sum * (n_total / n_valid),
                        'date': date(self.year, self.month, day),
                    }
        return totals


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from RMOB filename like 'Camps_012010rmob.TXT'."""
    # Pattern: digits before 'rmob' — last 4 are year, preceding 2 are month
    m = re.search(r'(\d{2})(\d{4})rmob', filename, re.IGNORECASE)
    if m:
        return int(m.group(2))
    return None


def extract_month_from_filename(filename: str) -> Optional[int]:
    """Extract month from RMOB filename."""
    m = re.search(r'(\d{2})(\d{4})rmob', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_metadata(lines: list) -> ObserverInfo:
    """Parse observer metadata from the bottom of an RMOB file."""
    info = ObserverInfo()
    for line in lines:
        line = line.strip()
        if line.startswith('[Observer]'):
            info.name = line.split(']', 1)[1].strip()
        elif line.startswith('[Country]'):
            info.country = line.split(']', 1)[1].strip()
        elif line.startswith('[City]'):
            info.city = line.split(']', 1)[1].strip()
        elif line.startswith('[Latitude GMAP]'):
            try:
                info.latitude = float(line.split(']', 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('[Longitude GMAP]'):
            try:
                info.longitude = float(line.split(']', 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('[Frequencies]'):
            freq_str = line.split(']', 1)[1].strip()
            # Convert to MHz
            try:
                if 'khz' in freq_str.lower() or 'kHz' in freq_str:
                    val = float(re.search(r'[\d.]+', freq_str).group())
                    info.frequency_mhz = val / 1000.0
                elif 'mhz' in freq_str.lower() or 'MHz' in freq_str:
                    val = float(re.search(r'[\d.]+', freq_str).group())
                    info.frequency_mhz = val
                else:
                    val = float(re.search(r'[\d.]+', freq_str).group())
                    # Heuristic: if > 1000, likely kHz
                    info.frequency_mhz = val / 1000.0 if val > 1000 else val
            except (ValueError, AttributeError):
                pass
        elif line.startswith('[Antenna]'):
            info.antenna = line.split(']', 1)[1].strip()
        elif line.startswith('[Observing Method]'):
            info.method = line.split(']', 1)[1].strip()
    return info


def parse_rmob_file(filepath: str) -> Optional[RMOBFile]:
    """Parse a single RMOB file into structured data."""
    filename = os.path.basename(filepath)
    year = extract_year_from_filename(filename)
    month_from_file = extract_month_from_filename(filename)

    if year is None:
        return None

    try:
        # Try multiple encodings
        content = None
        for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            return None
    except (IOError, OSError):
        return None

    # Find the header line to determine month
    month = None
    header_idx = None
    for i, line in enumerate(content):
        line_stripped = line.strip().lower()
        for mname, mnum in MONTH_MAP.items():
            if line_stripped.startswith(mname + '|') or line_stripped.startswith(mname + ' |'):
                month = mnum
                header_idx = i
                break
        if month is not None:
            break

    if month is None:
        # Fall back to filename
        month = month_from_file
        # Try to find header by looking for "00h"
        for i, line in enumerate(content):
            if '00h' in line and '23h' in line:
                header_idx = i
                break

    if month is None or header_idx is None:
        return None

    # Parse data rows
    hourly_counts = {}
    metadata_lines = []
    in_data = True

    for i in range(header_idx + 1, len(content)):
        line = content[i].strip()
        if not line:
            continue

        # Check if this is a metadata line
        if line.startswith('['):
            in_data = False
            metadata_lines.append(line)
            continue

        if not in_data:
            metadata_lines.append(line)
            continue

        # Try to parse as data row
        parts = line.split('|')
        if len(parts) < 3:
            continue

        # First part should be day number
        try:
            day = int(parts[0].strip())
        except ValueError:
            # Might be metadata without brackets
            if any(c.isalpha() for c in parts[0].strip()):
                in_data = False
                metadata_lines.append(line)
            continue

        if day < 1 or day > 31:
            continue

        # Validate day for this month/year
        try:
            date(year, month, day)
        except ValueError:
            continue

        # Parse hourly values
        hours = {}
        for h in range(24):
            col_idx = h + 1
            if col_idx < len(parts):
                val_str = parts[col_idx].strip()
                if val_str == '???' or val_str == '' or val_str == '?':
                    hours[h] = None
                else:
                    try:
                        hours[h] = int(val_str)
                    except ValueError:
                        try:
                            hours[h] = int(float(val_str))
                        except ValueError:
                            hours[h] = None
            else:
                hours[h] = None

        hourly_counts[day] = hours

    observer = parse_metadata(metadata_lines)

    return RMOBFile(
        filepath=filepath,
        year=year,
        month=month,
        observer=observer,
        hourly_counts=hourly_counts,
    )


# ============================================================================
# DATA AGGREGATION
# ============================================================================

def find_rmob_files(base_dir: str) -> list:
    """Recursively find all RMOB files."""
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.lower().endswith('rmob.txt'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def build_daily_timeseries(parsed_files: list) -> dict:
    """
    Aggregate all parsed RMOB files into a daily time series.

    Strategy: normalize each observer's daily count by their own monthly
    median, then average the normalized activity indices across observers.
    This handles the huge variation in equipment sensitivity.

    Also produces a raw median time series for absolute scale reference.

    Returns dict keyed by date with aggregation stats.
    """
    # Step 1: Compute each observer's monthly baseline
    # Key: (observer_name, year, month) -> list of daily 24h-estimated counts
    observer_monthly = defaultdict(list)
    # Key: date -> list of (observer_name, estimated_24h)
    date_raw = defaultdict(list)

    for rmob in parsed_files:
        obs_key = rmob.observer.name or rmob.filepath
        dailies = rmob.daily_totals
        for day, info in dailies.items():
            d = info['date']
            val = info['estimated_24h']
            if val < 0 or val > 100000:  # Sanity filter
                continue
            observer_monthly[(obs_key, d.year, d.month)].append(val)
            date_raw[d].append((obs_key, val))

    # Step 2: Compute monthly medians per observer
    observer_baselines = {}
    for key, vals in observer_monthly.items():
        median_val = np.median(vals)
        if median_val > 0:
            observer_baselines[key] = median_val

    # Step 3: Build normalized time series
    date_normalized = defaultdict(list)
    date_absolute = defaultdict(list)

    for d, obs_vals in date_raw.items():
        for obs_name, val in obs_vals:
            key = (obs_name, d.year, d.month)
            if key in observer_baselines:
                baseline = observer_baselines[key]
                normalized = val / baseline  # 1.0 = typical day for this observer
                date_normalized[d].append(normalized)
                date_absolute[d].append(val)

    # Step 4: Aggregate
    timeseries = {}
    for d in sorted(date_normalized.keys()):
        norm_vals = np.array(date_normalized[d])
        abs_vals = np.array(date_absolute[d])

        # Use IQR-based outlier rejection on normalized values
        if len(norm_vals) >= 5:
            q1, q3 = np.percentile(norm_vals, [25, 75])
            iqr = q3 - q1
            mask = (norm_vals >= q1 - 3 * iqr) & (norm_vals <= q3 + 3 * iqr)
            norm_clean = norm_vals[mask]
            abs_clean = abs_vals[mask]
        else:
            norm_clean = norm_vals
            abs_clean = abs_vals

        if len(norm_clean) == 0:
            continue

        timeseries[d] = {
            'date': d,
            'activity_index': float(np.mean(norm_clean)),  # 1.0 = baseline
            'activity_median': float(np.median(norm_clean)),
            'activity_std': float(np.std(norm_clean)) if len(norm_clean) > 1 else 0.0,
            'raw_median': float(np.median(abs_clean)),
            'raw_mean': float(np.mean(abs_clean)),
            'n_observers': len(norm_clean),
            'n_rejected': int(len(norm_vals) - len(norm_clean)),
        }

    return timeseries


# ============================================================================
# METEOR SHOWER CATALOG
# ============================================================================

MAJOR_SHOWERS = [
    {'name': 'Quadrantids', 'peak_doy': 3, 'duration_days': 4, 'zhr': 120},
    {'name': 'Lyrids', 'peak_doy': 112, 'duration_days': 3, 'zhr': 18},
    {'name': 'eta Aquariids', 'peak_doy': 126, 'duration_days': 10, 'zhr': 50},
    {'name': 'delta Aquariids', 'peak_doy': 210, 'duration_days': 15, 'zhr': 20},
    {'name': 'Perseids', 'peak_doy': 224, 'duration_days': 10, 'zhr': 100},
    {'name': 'Orionids', 'peak_doy': 294, 'duration_days': 7, 'zhr': 20},
    {'name': 'Leonids', 'peak_doy': 321, 'duration_days': 4, 'zhr': 15},
    {'name': 'Geminids', 'peak_doy': 347, 'duration_days': 6, 'zhr': 150},
    {'name': 'Ursids', 'peak_doy': 356, 'duration_days': 3, 'zhr': 10},
]


def get_shower_dates(year: int) -> list:
    """Get shower peak dates for a given year."""
    results = []
    for s in MAJOR_SHOWERS:
        try:
            peak = date(year, 1, 1) + timedelta(days=s['peak_doy'] - 1)
            results.append({
                'name': s['name'],
                'peak': peak,
                'start': peak - timedelta(days=s['duration_days'] // 2),
                'end': peak + timedelta(days=s['duration_days'] // 2),
                'zhr': s['zhr'],
            })
        except ValueError:
            pass
    return results


# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================

def compute_annual_profile(timeseries: dict) -> dict:
    """Compute mean meteor activity index by day-of-year across all years."""
    doy_values = defaultdict(list)
    for d, info in timeseries.items():
        doy = d.timetuple().tm_yday
        doy_values[doy].append(info['activity_index'])

    profile = {}
    for doy in sorted(doy_values.keys()):
        vals = doy_values[doy]
        profile[doy] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'n_years': len(vals),
        }
    return profile


def compute_kostrov_delayed_profile(annual_profile: dict, delay_days: int = 12) -> dict:
    """Shift the annual meteor profile by delay_days to predict weather effects."""
    delayed = {}
    for doy, info in annual_profile.items():
        new_doy = (doy + delay_days - 1) % 365 + 1
        delayed[new_doy] = info.copy()
    return delayed


def plot_results(timeseries: dict, annual_profile: dict, output_dir: str):
    """Generate diagnostic and analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    # ---- Plot 1: Full time series overview ----
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    dates = sorted(timeseries.keys())
    activity = [timeseries[d]['activity_index'] for d in dates]
    n_obs = [timeseries[d]['n_observers'] for d in dates]

    # 1a: Normalized activity index
    ax = axes[0]
    ax.plot(dates, activity, linewidth=0.3, color='steelblue', alpha=0.5)
    # 30-day rolling average
    if len(activity) > 30:
        kernel = np.ones(30) / 30
        smoothed = np.convolve(activity, kernel, mode='valid')
        smooth_dates = dates[15:15 + len(smoothed)]
        ax.plot(smooth_dates, smoothed, linewidth=1.5, color='darkblue',
                label='30-day rolling mean')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    ax.set_ylabel('Activity Index (1.0 = monthly baseline)')
    ax.set_title('RMOB Daily Meteor Activity (2000-2020) — Normalized Multi-Observer Index')
    ax.legend(loc='upper right')
    ax.set_xlim(dates[0], dates[-1])

    # 1b: Number of observers
    ax = axes[1]
    ax.fill_between(dates, n_obs, color='orange', alpha=0.5)
    ax.set_ylabel('# Contributing observers')
    ax.set_title('Observer Coverage')
    ax.set_xlim(dates[0], dates[-1])

    # 1c: One representative year with shower annotations
    ax = axes[2]
    # Use 2015 as representative (good coverage)
    rep_year = 2015
    rep_dates = [d for d in dates if d.year == rep_year]
    rep_activity = [timeseries[d]['activity_index'] for d in rep_dates]

    if rep_dates:
        ax.plot(rep_dates, rep_activity, linewidth=0.5, color='steelblue', alpha=0.7)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
        # 7-day smoothing for single year
        if len(rep_activity) > 7:
            kernel7 = np.ones(7) / 7
            sm7 = np.convolve(rep_activity, kernel7, mode='valid')
            sm7_dates = rep_dates[3:3 + len(sm7)]
            ax.plot(sm7_dates, sm7, linewidth=2, color='darkblue')

        # Annotate showers
        showers = get_shower_dates(rep_year)
        colors = plt.cm.Set2(np.linspace(0, 1, len(showers)))
        for shower, color in zip(showers, colors):
            ax.axvspan(shower['start'], shower['end'], alpha=0.15, color=color)
            ax.annotate(shower['name'],
                       xy=(shower['peak'], ax.get_ylim()[1] * 0.9),
                       fontsize=7, rotation=45, ha='left', va='top',
                       color='darkred')

        # Add Kostrov delay markers (12 days after each shower peak)
        for shower in showers:
            delayed = shower['peak'] + timedelta(days=12)
            if date(rep_year, 1, 1) <= delayed <= date(rep_year, 12, 31):
                ax.axvline(delayed, color='red', linestyle=':', alpha=0.5, linewidth=1)

        ax.set_ylabel('Activity Index')
        ax.set_title(f'{rep_year} Detail — Showers (shaded) + Kostrov 12-day delay (red dotted)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmob_timeseries_overview.png'), dpi=150)
    plt.close()
    print(f"  Saved: rmob_timeseries_overview.png")

    # ---- Plot 2: Annual meteor profile (day-of-year average) ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    doys = sorted(annual_profile.keys())
    prof_means = [annual_profile[d]['mean'] for d in doys]
    prof_stds = [annual_profile[d]['std'] for d in doys]

    # 2a: Mean annual profile with shower labels
    ax = axes[0]
    ax.fill_between(doys,
                    [m - s for m, s in zip(prof_means, prof_stds)],
                    [m + s for m, s in zip(prof_means, prof_stds)],
                    alpha=0.2, color='steelblue')
    ax.plot(doys, prof_means, linewidth=1.5, color='darkblue')

    # Smooth version
    if len(prof_means) > 15:
        kernel15 = np.ones(15) / 15
        sm = np.convolve(prof_means, kernel15, mode='valid')
        sm_doys = doys[7:7 + len(sm)]
        ax.plot(sm_doys, sm, linewidth=2.5, color='red', label='15-day smoothed')

    for shower in MAJOR_SHOWERS:
        ax.axvline(shower['peak_doy'], color='green', linestyle='--', alpha=0.4)
        ax.annotate(shower['name'], xy=(shower['peak_doy'], max(prof_means) * 0.95),
                   fontsize=7, rotation=90, ha='right', va='top', color='darkgreen')

    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Activity Index (1.0 = baseline)')
    ax.set_title('Mean Annual Meteor Activity Profile (2000-2020 average)')
    ax.legend()

    # 2b: Same profile, delayed by 12 days (Kostrov prediction)
    ax = axes[1]
    delayed = compute_kostrov_delayed_profile(annual_profile, delay_days=12)
    del_doys = sorted(delayed.keys())
    del_means = [delayed[d]['mean'] for d in del_doys]

    ax.plot(doys, prof_means, linewidth=1, color='steelblue', alpha=0.5,
            label='Meteor flux (actual)')
    ax.plot(del_doys, del_means, linewidth=2, color='red',
            label='Meteor flux shifted +12 days (Kostrov delay)')

    # Mark predicted weather-effect windows
    for shower in MAJOR_SHOWERS:
        effect_doy = (shower['peak_doy'] + 12 - 1) % 365 + 1
        ax.axvline(effect_doy, color='red', linestyle=':', alpha=0.5)
        ax.annotate(f"{shower['name']}+12d",
                   xy=(effect_doy, max(prof_means) * 0.85),
                   fontsize=6, rotation=90, ha='right', va='top', color='darkred')

    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Activity Index')
    ax.set_title('Kostrov Prediction: Dust Settling Delay of 12 Days')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmob_annual_profile.png'), dpi=150)
    plt.close()
    print(f"  Saved: rmob_annual_profile.png")

    # ---- Plot 3: Observer statistics ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 3a: Observers per year
    ax = axes[0, 0]
    year_observers = defaultdict(set)
    for d, info in timeseries.items():
        year_observers[d.year].add(d.month)  # rough proxy
    # Actually count from parsed files
    years = sorted(set(d.year for d in dates))
    obs_per_year = []
    for y in years:
        n = max(timeseries[d]['n_observers'] for d in dates if d.year == y)
        obs_per_year.append(n)
    ax.bar(years, obs_per_year, color='steelblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Peak # observers in any day')
    ax.set_title('Observer Network Growth')

    # 3b: Distribution of activity index
    ax = axes[0, 1]
    all_activity = [timeseries[d]['activity_index'] for d in dates]
    ax.hist(all_activity, bins=100, color='steelblue', alpha=0.7, edgecolor='none',
            range=(0, 5))
    ax.set_xlabel('Daily Activity Index')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Daily Meteor Activity')
    ax.axvline(np.median(all_activity), color='red', linestyle='--',
               label=f'Median: {np.median(all_activity):.2f}')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline (1.0)')
    ax.legend()

    # 3c: Hourly profile (averaged across all data)
    ax = axes[1, 0]
    # This would need the raw hourly data — use annual profile as proxy
    ax.text(0.5, 0.5, 'Cross-correlation\nwith lightning data\n(Stage 2)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, color='gray', style='italic')
    ax.set_title('Reserved: Lightning Cross-Correlation')

    # 3d: Year-over-year shower strength
    ax = axes[1, 1]
    # Track Perseids and Geminids across years
    for shower_name, color, doy_center, half_width in [
        ('Perseids', 'blue', 224, 7),
        ('Geminids', 'red', 347, 5),
        ('Quadrantids', 'green', 3, 3),
    ]:
        yearly_peaks = []
        yearly_years = []
        for y in years:
            year_vals = []
            for doy_offset in range(-half_width, half_width + 1):
                try:
                    target_doy = doy_center + doy_offset
                    if target_doy <= 0:
                        target_doy += 365
                    target_date = date(y, 1, 1) + timedelta(days=target_doy - 1)
                    if target_date in timeseries:
                        year_vals.append(timeseries[target_date]['activity_index'])
                except ValueError:
                    pass
            if year_vals:
                yearly_peaks.append(max(year_vals))
                yearly_years.append(y)
        if yearly_years:
            ax.plot(yearly_years, yearly_peaks, 'o-', color=color,
                    label=shower_name, markersize=4, linewidth=1)

    ax.set_xlabel('Year')
    ax.set_ylabel('Peak Activity Index during shower')
    ax.set_title('Major Shower Intensity by Year')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmob_observer_stats.png'), dpi=150)
    plt.close()
    print(f"  Saved: rmob_observer_stats.png")


def save_csv(timeseries: dict, output_path: str):
    """Save the daily time series to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'year', 'month', 'day', 'doy',
            'activity_index', 'activity_median', 'activity_std',
            'raw_median', 'raw_mean', 'n_observers', 'n_rejected'
        ])
        for d in sorted(timeseries.keys()):
            info = timeseries[d]
            writer.writerow([
                d.isoformat(),
                d.year, d.month, d.day,
                d.timetuple().tm_yday,
                f"{info['activity_index']:.4f}",
                f"{info['activity_median']:.4f}",
                f"{info['activity_std']:.4f}",
                f"{info['raw_median']:.1f}",
                f"{info['raw_mean']:.1f}",
                info['n_observers'],
                info['n_rejected'],
            ])
    print(f"  Saved: {output_path}")


def save_summary(timeseries: dict, parsed_files: list, output_path: str):
    """Save analysis summary as JSON."""
    dates = sorted(timeseries.keys())

    # Unique observers
    observers = set()
    countries = set()
    for rmob in parsed_files:
        if rmob.observer.name:
            observers.add(rmob.observer.name)
        if rmob.observer.country:
            countries.add(rmob.observer.country)

    summary = {
        'date_range': {
            'start': dates[0].isoformat(),
            'end': dates[-1].isoformat(),
            'total_days': len(dates),
        },
        'observers': {
            'total_unique': len(observers),
            'countries': sorted(countries),
        },
        'files_parsed': len(parsed_files),
        'daily_stats': {
            'mean_activity_index': float(np.mean([timeseries[d]['activity_index'] for d in dates])),
            'median_activity_index': float(np.median([timeseries[d]['activity_index'] for d in dates])),
            'mean_n_observers': float(np.mean([timeseries[d]['n_observers'] for d in dates])),
        },
        'shower_detection': {},
    }

    # Check if major showers are detectable
    annual = compute_annual_profile(timeseries)
    baseline = np.median([annual[doy]['mean'] for doy in annual])
    for shower in MAJOR_SHOWERS:
        peak_doy = shower['peak_doy']
        window = [doy for doy in range(peak_doy - 3, peak_doy + 4) if doy in annual]
        if window:
            peak_val = max(annual[doy]['mean'] for doy in window)
            snr = peak_val / baseline if baseline > 0 else 0
            summary['shower_detection'][shower['name']] = {
                'expected_zhr': shower['zhr'],
                'observed_peak_doy_mean': float(peak_val),
                'baseline': float(baseline),
                'signal_to_noise': float(snr),
                'detected': bool(snr > 1.5),
            }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    base_dir = str(repo_root / "data" / "rmob")
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("RMOB Radio Meteor Data Parser", flush=True)
    print("=" * 70, flush=True)

    # Find all files
    print(f"\nScanning: {base_dir}", flush=True)
    all_files = find_rmob_files(base_dir)
    print(f"  Found {len(all_files)} RMOB files", flush=True)

    # Parse
    print(f"\nParsing files...", flush=True)
    parsed = []
    errors = 0
    for i, fp in enumerate(all_files):
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(all_files)}", flush=True)
        result = parse_rmob_file(fp)
        if result is not None:
            parsed.append(result)
        else:
            errors += 1

    print(f"  Successfully parsed: {len(parsed)}", flush=True)
    print(f"  Failed to parse: {errors}", flush=True)

    # Build time series
    print(f"\nBuilding daily time series...")
    timeseries = build_daily_timeseries(parsed)
    print(f"  Total days with data: {len(timeseries)}")

    if not timeseries:
        print("ERROR: No data produced. Check file paths and formats.")
        sys.exit(1)

    dates = sorted(timeseries.keys())
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    # Save CSV
    print(f"\nSaving outputs to: {output_dir}")
    csv_path = os.path.join(output_dir, 'rmob_daily_meteor_flux.csv')
    save_csv(timeseries, csv_path)

    # Save summary
    summary_path = os.path.join(output_dir, 'rmob_analysis_summary.json')
    save_summary(timeseries, parsed, summary_path)

    # Compute annual profile
    print(f"\nComputing annual meteor profile...")
    annual = compute_annual_profile(timeseries)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_results(timeseries, annual, output_dir)

    print(f"\n{'=' * 70}")
    print("Done! Key outputs:")
    print(f"  - rmob_daily_meteor_flux.csv     (daily time series)")
    print(f"  - rmob_analysis_summary.json     (statistics & shower detection)")
    print(f"  - rmob_timeseries_overview.png   (21-year overview)")
    print(f"  - rmob_annual_profile.png        (mean annual cycle + Kostrov delay)")
    print(f"  - rmob_observer_stats.png        (network & shower diagnostics)")
    print(f"\nNext step: Download WGLC lightning data from Zenodo for cross-correlation")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
