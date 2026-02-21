"""
UAP-Meteor Temporal Correlation Analysis
=========================================
Extends the UK MoD Condign Report's finding (r=0.62 in 1996) across 20 years
of NUFORC sighting data and IMO Visual Meteor Database records.

Hypothesis: If UAP are meteor-ablated dusty plasma structures, sighting rates
should correlate with meteor activity, especially during major showers.

Data sources:
- NUFORC: All-Nuforc-Records.csv (~97k records, 2000-2021)
- IMO VMDB: Rate CSV files (2000-2021, observer session meteor counts)
- NOAA StormEvents: Lightning/thunderstorm data (2000-2020)

Output: Publication-quality figures + statistical tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plasma-consistent NUFORC shapes (from our earlier analysis)
PLASMA_SHAPES = {
    'fireball', 'sphere', 'circle', 'oval', 'egg', 'teardrop',
    'light', 'flash', 'orb', 'round', 'star'
}

FORMATION_SHAPES = {
    'triangle', 'formation', 'diamond', 'chevron', 'v-shaped'
}

# Major meteor showers with approximate peak dates (day of year)
MAJOR_SHOWERS = {
    'Quadrantids': (3, 4),       # Jan 3-4
    'Lyrids': (112, 113),        # Apr 22-23
    'Eta Aquariids': (126, 127), # May 6-7
    'Perseids': (224, 225),      # Aug 12-13
    'Orionids': (294, 295),      # Oct 21-22
    'Leonids': (321, 322),       # Nov 17-18
    'Geminids': (347, 348),      # Dec 13-14
}

# ============================================================
# 1. Load and parse NUFORC data
# ============================================================
def load_nuforc():
    """Load NUFORC records and extract dates."""
    print("Loading NUFORC data...")
    df = pd.read_csv(DATA_DIR / "nuforc" / "All-Nuforc-Records.csv",
                     encoding='latin-1', on_bad_lines='skip')

    # Parse dates - format is "5/19/21 20:15"
    def parse_nuforc_date(s):
        try:
            dt = pd.to_datetime(s, format='mixed', dayfirst=False)
            # Fix 2-digit years: NUFORC uses MM/DD/YY format
            # Records span ~1930s to 2021
            return dt
        except:
            return pd.NaT

    df['parsed_date'] = pd.to_datetime(df['EventDate'], format='mixed',
                                        dayfirst=False, errors='coerce')
    df = df.dropna(subset=['parsed_date'])

    # Normalize shape names
    df['shape_lower'] = df['Shape'].str.lower().str.strip()

    # Flag plasma-consistent and formation shapes
    df['is_plasma'] = df['shape_lower'].isin(PLASMA_SHAPES)
    df['is_formation'] = df['shape_lower'].isin(FORMATION_SHAPES)
    df['is_any_plasma'] = df['is_plasma'] | df['is_formation']

    # Filter to 2000-2021 to match IMO data
    df = df[(df['parsed_date'] >= '2000-01-01') &
            (df['parsed_date'] < '2022-01-01')]

    print(f"  Loaded {len(df)} NUFORC records (2000-2021)")
    print(f"  Plasma-consistent: {df['is_plasma'].sum()}")
    print(f"  Formations: {df['is_formation'].sum()}")

    return df

# ============================================================
# 2. Load and parse IMO meteor rate data
# ============================================================
def load_imo_rates():
    """Load all IMO VMDB rate files and aggregate to daily counts."""
    print("Loading IMO meteor rate data...")

    rate_files = sorted(DATA_DIR.glob("imo/Rate-IMO-VMDB-*.csv"))
    all_rates = []

    for f in rate_files:
        if 'aggregation' in f.name or 'scrubbed' in f.name:
            continue
        print(f"  Reading {f.name}...")
        try:
            df = pd.read_csv(f, sep=';', encoding='utf-8-sig')
            all_rates.append(df)
        except Exception as e:
            print(f"  Warning: {e}")

    if not all_rates:
        print("  No rate files loaded!")
        return pd.DataFrame()

    rates = pd.concat(all_rates, ignore_index=True)

    # Parse start dates
    rates['parsed_date'] = pd.to_datetime(rates['Start Date'], errors='coerce')
    rates = rates.dropna(subset=['parsed_date'])
    rates['date'] = rates['parsed_date'].dt.date

    # Aggregate: sum of meteor counts per day across all observers/showers
    daily = rates.groupby('date').agg(
        total_meteors=('Number', 'sum'),
        n_sessions=('Obs Session ID', 'nunique'),
        n_observers=('User ID', 'nunique')
    ).reset_index()

    daily['date'] = pd.to_datetime(daily['date'])

    # Compute rate per observer-hour (normalize for observing effort)
    # Teff is effective observing time in hours
    rates['Teff_float'] = pd.to_numeric(rates['Teff'], errors='coerce')
    teff_daily = rates.groupby(rates['parsed_date'].dt.date)['Teff_float'].sum().reset_index()
    teff_daily.columns = ['date', 'total_teff']
    teff_daily['date'] = pd.to_datetime(teff_daily['date'])

    daily = daily.merge(teff_daily, on='date', how='left')
    daily['rate_per_hour'] = daily['total_meteors'] / daily['total_teff'].replace(0, np.nan)

    print(f"  Total days with meteor data: {len(daily)}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Total meteors: {daily['total_meteors'].sum():,.0f}")

    return daily

# ============================================================
# 3. Load IMO daily aggregation (pre-processed by Jim)
# ============================================================
def load_imo_daily_agg():
    """Load Jim's pre-aggregated daily meteor counts for 2017-2021."""
    print("Loading IMO daily aggregation (2017-2021)...")
    f = BASE_PATH / "IMO" / "Rate-IMO-VMDB-2017-5-30-2021-5-30 daily aggregation.csv"
    df = pd.read_csv(f)
    df.columns = ['date', 'count']
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded {len(df)} daily records")
    return df

# ============================================================
# 4. Aggregate to weekly/monthly bins
# ============================================================
def aggregate_timeseries(nuforc_df, meteor_daily, freq='W'):
    """Aggregate NUFORC and meteor data to common time bins."""

    # NUFORC: count sightings per bin
    nuforc_ts = nuforc_df.set_index('parsed_date').resample(freq).agg(
        total_sightings=('Shape', 'count'),
        plasma_sightings=('is_plasma', 'sum'),
        formation_sightings=('is_formation', 'sum'),
        any_plasma_sightings=('is_any_plasma', 'sum')
    )

    # Meteor: sum counts per bin
    meteor_ts = meteor_daily.set_index('date').resample(freq).agg(
        total_meteors=('total_meteors', 'sum'),
        n_sessions=('n_sessions', 'sum'),
        rate_per_hour=('rate_per_hour', 'mean')
    )

    # Merge on date
    combined = nuforc_ts.join(meteor_ts, how='inner')
    combined = combined.dropna()

    print(f"  Combined {freq} bins: {len(combined)} periods")

    return combined

# ============================================================
# 5. Correlation analysis with lag
# ============================================================
def compute_correlations(combined, max_lag_weeks=8):
    """Compute Pearson & Spearman correlations at various lags."""

    results = []

    for target_col in ['total_sightings', 'plasma_sightings', 'formation_sightings', 'any_plasma_sightings']:
        for lag in range(-max_lag_weeks, max_lag_weeks + 1):
            # Shift meteor data by lag weeks
            shifted_meteors = combined['total_meteors'].shift(lag)
            valid = combined[[target_col]].join(shifted_meteors.rename('meteors_shifted')).dropna()

            if len(valid) < 20:
                continue

            r_pearson, p_pearson = stats.pearsonr(valid[target_col], valid['meteors_shifted'])
            r_spearman, p_spearman = stats.spearmanr(valid[target_col], valid['meteors_shifted'])

            results.append({
                'uap_type': target_col,
                'lag_weeks': lag,
                'r_pearson': r_pearson,
                'p_pearson': p_pearson,
                'r_spearman': r_spearman,
                'p_spearman': p_spearman,
                'n': len(valid)
            })

    return pd.DataFrame(results)

# ============================================================
# 6. Shower-specific analysis
# ============================================================
def shower_analysis(nuforc_df):
    """Check if UAP sighting rates spike around major meteor showers."""

    nuforc_df = nuforc_df.copy()
    nuforc_df['doy'] = nuforc_df['parsed_date'].dt.dayofyear
    nuforc_df['year'] = nuforc_df['parsed_date'].dt.year

    results = []

    for shower_name, (peak_start, peak_end) in MAJOR_SHOWERS.items():
        # Define shower window: peak ± 7 days
        window_start = (peak_start - 7) % 366
        window_end = (peak_end + 7) % 366

        if window_start < window_end:
            in_window = (nuforc_df['doy'] >= window_start) & (nuforc_df['doy'] <= window_end)
        else:  # wraps around year boundary
            in_window = (nuforc_df['doy'] >= window_start) | (nuforc_df['doy'] <= window_end)

        window_days = 15  # approximate window size
        non_window_days = 365 - window_days

        # Sightings per day during shower vs rest of year
        shower_sightings = nuforc_df[in_window]
        non_shower_sightings = nuforc_df[~in_window]

        n_years = nuforc_df['year'].nunique()

        shower_rate = len(shower_sightings) / (window_days * n_years)
        baseline_rate = len(non_shower_sightings) / (non_window_days * n_years)

        # Plasma-consistent only
        shower_plasma_rate = shower_sightings['is_any_plasma'].sum() / (window_days * n_years)
        baseline_plasma_rate = non_shower_sightings['is_any_plasma'].sum() / (non_window_days * n_years)

        ratio = shower_rate / max(baseline_rate, 0.01)
        plasma_ratio = shower_plasma_rate / max(baseline_plasma_rate, 0.01)

        results.append({
            'shower': shower_name,
            'peak_doy': peak_start,
            'shower_sightings': len(shower_sightings),
            'shower_rate_per_day': shower_rate,
            'baseline_rate_per_day': baseline_rate,
            'ratio': ratio,
            'shower_plasma': shower_sightings['is_any_plasma'].sum(),
            'plasma_ratio': plasma_ratio
        })

    return pd.DataFrame(results)

# ============================================================
# 7. Annual cycle analysis (seasonal decomposition)
# ============================================================
def annual_cycle_analysis(nuforc_df, meteor_daily):
    """Compare average annual patterns of UAP sightings vs meteor rates."""

    # NUFORC: average sightings by week-of-year
    nuforc_df = nuforc_df.copy()
    nuforc_df['week'] = nuforc_df['parsed_date'].dt.isocalendar().week.astype(int)
    nuforc_df['year'] = nuforc_df['parsed_date'].dt.year

    weekly_annual = nuforc_df.groupby('week').agg(
        mean_total=('Shape', lambda x: len(x) / nuforc_df['year'].nunique()),
        mean_plasma=('is_any_plasma', lambda x: x.sum() / nuforc_df['year'].nunique())
    )

    # Meteor: average counts by week-of-year
    meteor_daily = meteor_daily.copy()
    meteor_daily['week'] = meteor_daily['date'].dt.isocalendar().week.astype(int)
    meteor_daily['year'] = meteor_daily['date'].dt.year

    meteor_weekly = meteor_daily.groupby('week').agg(
        mean_meteors=('total_meteors', lambda x: x.sum() / meteor_daily['year'].nunique())
    )

    return weekly_annual, meteor_weekly

# ============================================================
# 8. NOAA storm correlation
# ============================================================
def load_storms():
    """Load NOAA StormEvents and extract lightning/thunderstorm events."""
    print("Loading NOAA StormEvents...")

    storm_files = sorted(BASE_PATH.glob("NOAA/StormEvents/StormEvents_details-*.csv"))
    all_storms = []

    for f in storm_files:
        if f.suffix != '.csv':
            continue
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            # Filter to lightning/thunderstorm events
            lightning_types = ['Lightning', 'Thunderstorm Wind', 'Hail', 'Funnel Cloud', 'Tornado']
            df = df[df['EVENT_TYPE'].isin(lightning_types)]
            all_storms.append(df[['BEGIN_DATE_TIME', 'EVENT_TYPE', 'STATE', 'BEGIN_LAT', 'BEGIN_LON']])
        except Exception as e:
            print(f"  Warning reading {f.name}: {e}")

    storms = pd.concat(all_storms, ignore_index=True)
    storms['date'] = pd.to_datetime(storms['BEGIN_DATE_TIME'], format='mixed', errors='coerce')
    storms = storms.dropna(subset=['date'])

    print(f"  Loaded {len(storms)} atmospheric electrical events")
    return storms

# ============================================================
# 9. Publication figures
# ============================================================
def plot_correlation_figures(combined_weekly, combined_monthly,
                            corr_weekly, corr_monthly,
                            shower_results, weekly_annual, meteor_weekly):
    """Generate publication-quality figures."""

    fig = plt.figure(figsize=(16, 20))

    # ---- Panel A: Time series comparison (monthly) ----
    ax1 = fig.add_subplot(4, 2, 1)
    ax1b = ax1.twinx()

    ax1.plot(combined_monthly.index, combined_monthly['any_plasma_sightings'],
             'C0-', alpha=0.7, linewidth=0.8, label='UAP sightings (plasma-consistent)')
    ax1b.plot(combined_monthly.index, combined_monthly['total_meteors'],
              'C1-', alpha=0.7, linewidth=0.8, label='Meteor counts')

    ax1.set_ylabel('UAP Sightings / month', color='C0')
    ax1b.set_ylabel('Meteor Counts / month', color='C1')
    ax1.set_title('A. Monthly UAP Sightings vs Meteor Activity (2000–2021)')
    ax1.xaxis.set_major_locator(mdates.YearLocator(4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

    # ---- Panel B: Scatter plot (monthly) ----
    ax2 = fig.add_subplot(4, 2, 2)

    x = combined_monthly['total_meteors']
    y = combined_monthly['any_plasma_sightings']
    ax2.scatter(x, y, alpha=0.4, s=15, c='C0', edgecolors='none')

    # Fit line
    mask = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5)
    ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {mask.sum()}',
             transform=ax2.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel('Monthly Meteor Counts')
    ax2.set_ylabel('Monthly Plasma-Consistent UAP Sightings')
    ax2.set_title('B. UAP–Meteor Correlation (Monthly)')

    # ---- Panel C: Lag correlation (weekly) ----
    ax3 = fig.add_subplot(4, 2, 3)

    plasma_corr = corr_weekly[corr_weekly['uap_type'] == 'any_plasma_sightings']
    total_corr = corr_weekly[corr_weekly['uap_type'] == 'total_sightings']

    ax3.plot(plasma_corr['lag_weeks'], plasma_corr['r_pearson'],
             'C0-o', markersize=3, label='Plasma-consistent')
    ax3.plot(total_corr['lag_weeks'], total_corr['r_pearson'],
             'C2-s', markersize=3, alpha=0.6, label='All sightings')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    # Significance threshold (approximate, n varies)
    n_approx = plasma_corr['n'].median()
    sig_threshold = 1.96 / np.sqrt(n_approx)
    ax3.axhline(y=sig_threshold, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=-sig_threshold, color='red', linestyle=':', linewidth=0.5, alpha=0.5)

    ax3.set_xlabel('Lag (weeks, positive = UAP lags meteors)')
    ax3.set_ylabel('Pearson r')
    ax3.set_title('C. Cross-Correlation: UAP Sightings vs Meteor Rate')
    ax3.legend(fontsize=8)

    # ---- Panel D: Spearman lag correlation ----
    ax4 = fig.add_subplot(4, 2, 4)

    ax4.plot(plasma_corr['lag_weeks'], plasma_corr['r_spearman'],
             'C0-o', markersize=3, label='Plasma-consistent')
    ax4.plot(total_corr['lag_weeks'], total_corr['r_spearman'],
             'C2-s', markersize=3, alpha=0.6, label='All sightings')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    ax4.set_xlabel('Lag (weeks)')
    ax4.set_ylabel('Spearman ρ')
    ax4.set_title('D. Rank Cross-Correlation (Robust to Outliers)')
    ax4.legend(fontsize=8)

    # ---- Panel E: Annual cycle comparison ----
    ax5 = fig.add_subplot(4, 2, 5)
    ax5b = ax5.twinx()

    weeks = weekly_annual.index
    ax5.bar(weeks - 0.2, weekly_annual['mean_plasma'], width=0.4,
            color='C0', alpha=0.6, label='UAP (plasma-consistent)')
    ax5b.bar(weeks + 0.2, meteor_weekly.reindex(weeks)['mean_meteors'].fillna(0),
             width=0.4, color='C1', alpha=0.6, label='Meteors')

    # Mark major showers
    for name, (peak, _) in MAJOR_SHOWERS.items():
        week_num = peak // 7 + 1
        if week_num <= 52:
            ax5.axvline(x=week_num, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
            ax5.text(week_num, ax5.get_ylim()[1] * 0.95, name[:3],
                    fontsize=5, ha='center', color='red', rotation=90)

    ax5.set_xlabel('Week of Year')
    ax5.set_ylabel('Mean UAP Sightings / week', color='C0')
    ax5b.set_ylabel('Mean Meteor Count / week', color='C1')
    ax5.set_title('E. Average Annual Cycle: UAP vs Meteor Activity')
    ax5.set_xlim(0, 53)

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

    # ---- Panel F: Shower enhancement ratios ----
    ax6 = fig.add_subplot(4, 2, 6)

    x_pos = range(len(shower_results))
    bars = ax6.bar(x_pos, shower_results['plasma_ratio'], color='C0', alpha=0.7)
    ax6.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline rate')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(shower_results['shower'], rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Sighting Rate Ratio\n(shower window / baseline)')
    ax6.set_title('F. Plasma-Consistent UAP Enhancement During Meteor Showers')
    ax6.legend(fontsize=8)

    # Add counts on bars
    for i, (bar, count) in enumerate(zip(bars, shower_results['shower_plasma'])):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={count:.0f}', ha='center', fontsize=6)

    # ---- Panel G: Year-by-year correlation ----
    ax7 = fig.add_subplot(4, 2, 7)

    yearly_corrs = []
    for year in combined_monthly.index.year.unique():
        year_data = combined_monthly[combined_monthly.index.year == year]
        if len(year_data) >= 6:
            r, p = stats.pearsonr(year_data['total_meteors'],
                                  year_data['any_plasma_sightings'])
            yearly_corrs.append({'year': year, 'r': r, 'p': p, 'n': len(year_data)})

    yearly_df = pd.DataFrame(yearly_corrs)
    colors = ['C0' if p < 0.05 else 'gray' for p in yearly_df['p']]
    ax7.bar(yearly_df['year'], yearly_df['r'], color=colors, alpha=0.7)
    ax7.axhline(y=0, color='black', linewidth=0.5)
    ax7.axhline(y=0.62, color='red', linestyle='--', linewidth=1,
                label='MoD Condign (1996): r=0.62')
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Pearson r')
    ax7.set_title('G. Year-by-Year UAP–Meteor Correlation\n(blue: p<0.05, gray: not significant)')
    ax7.legend(fontsize=8)

    # ---- Panel H: Shape-specific correlations ----
    ax8 = fig.add_subplot(4, 2, 8)

    shape_corrs = []
    for shape_type in ['total_sightings', 'plasma_sightings', 'formation_sightings', 'any_plasma_sightings']:
        peak_corr = corr_monthly[corr_monthly['uap_type'] == shape_type]
        if len(peak_corr) > 0:
            best = peak_corr.loc[peak_corr['r_pearson'].abs().idxmax()]
            shape_corrs.append({
                'type': shape_type.replace('_sightings', '').replace('_', ' ').title(),
                'best_r': best['r_pearson'],
                'best_lag': best['lag_weeks'],
                'p': best['p_pearson']
            })

    shape_df = pd.DataFrame(shape_corrs)
    colors = ['C0' if p < 0.05 else 'gray' for p in shape_df['p']]
    bars = ax8.barh(shape_df['type'], shape_df['best_r'], color=colors, alpha=0.7)
    ax8.axvline(x=0, color='black', linewidth=0.5)
    ax8.set_xlabel('Peak Pearson r')
    ax8.set_title('H. Peak Correlation by UAP Shape Category\n(monthly, best lag)')

    for i, row in shape_df.iterrows():
        ax8.text(row['best_r'] + 0.01 * np.sign(row['best_r']), i,
                f"lag={row['best_lag']:.0f}w, p={row['p']:.3f}",
                fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'uap_meteor_correlation.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'uap_meteor_correlation.pdf', bbox_inches='tight')
    print(f"\nFigures saved to {OUTPUT_DIR}")
    plt.close()

# ============================================================
# 10. Statistical summary table
# ============================================================
def print_summary(corr_weekly, corr_monthly, shower_results, combined_monthly):
    """Print publication-ready summary statistics."""

    print("\n" + "="*70)
    print("UAP–METEOR CORRELATION ANALYSIS: SUMMARY STATISTICS")
    print("="*70)

    print("\n--- Monthly Correlations (lag=0) ---")
    for utype in ['total_sightings', 'plasma_sightings', 'formation_sightings', 'any_plasma_sightings']:
        row = corr_monthly[(corr_monthly['uap_type'] == utype) & (corr_monthly['lag_weeks'] == 0)]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"  {utype:30s}  Pearson r={row['r_pearson']:+.4f} (p={row['p_pearson']:.2e})  "
                  f"Spearman ρ={row['r_spearman']:+.4f} (p={row['p_spearman']:.2e})  n={row['n']:.0f}")

    print("\n--- Peak Monthly Correlations (best lag) ---")
    for utype in ['total_sightings', 'plasma_sightings', 'formation_sightings', 'any_plasma_sightings']:
        sub = corr_monthly[corr_monthly['uap_type'] == utype]
        if len(sub) > 0:
            best = sub.loc[sub['r_pearson'].abs().idxmax()]
            print(f"  {utype:30s}  r={best['r_pearson']:+.4f} at lag={best['lag_weeks']:.0f} weeks  "
                  f"(p={best['p_pearson']:.2e})")

    print("\n--- Weekly Correlations (lag=0) ---")
    for utype in ['total_sightings', 'any_plasma_sightings']:
        row = corr_weekly[(corr_weekly['uap_type'] == utype) & (corr_weekly['lag_weeks'] == 0)]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"  {utype:30s}  Pearson r={row['r_pearson']:+.4f} (p={row['p_pearson']:.2e})  "
                  f"Spearman ρ={row['r_spearman']:+.4f} (p={row['p_spearman']:.2e})")

    print("\n--- Meteor Shower Enhancement (plasma-consistent UAP) ---")
    print(f"  {'Shower':20s} {'n sightings':>12s} {'Rate ratio':>12s}")
    for _, row in shower_results.iterrows():
        print(f"  {row['shower']:20s} {row['shower_plasma']:12.0f} {row['plasma_ratio']:12.2f}x")

    # MoD comparison
    print("\n--- Comparison with UK MoD Condign Report (1996) ---")
    print(f"  MoD finding:  r = 0.62 (one year, UK only)")

    zero_lag = corr_monthly[(corr_monthly['uap_type'] == 'any_plasma_sightings') &
                            (corr_monthly['lag_weeks'] == 0)]
    if len(zero_lag) > 0:
        r = zero_lag.iloc[0]['r_pearson']
        n = zero_lag.iloc[0]['n']
        print(f"  This analysis: r = {r:.4f} (n={n:.0f} months, 2000-2021, global NUFORC + IMO)")

    # Save table
    table_data = []
    for utype in ['total_sightings', 'plasma_sightings', 'formation_sightings', 'any_plasma_sightings']:
        row0 = corr_monthly[(corr_monthly['uap_type'] == utype) & (corr_monthly['lag_weeks'] == 0)]
        sub = corr_monthly[corr_monthly['uap_type'] == utype]
        if len(row0) > 0 and len(sub) > 0:
            best = sub.loc[sub['r_pearson'].abs().idxmax()]
            table_data.append({
                'UAP Category': utype.replace('_sightings', ''),
                'r (lag=0)': f"{row0.iloc[0]['r_pearson']:.4f}",
                'p (lag=0)': f"{row0.iloc[0]['p_pearson']:.2e}",
                'ρ (lag=0)': f"{row0.iloc[0]['r_spearman']:.4f}",
                'Best r': f"{best['r_pearson']:.4f}",
                'Best lag (weeks)': f"{best['lag_weeks']:.0f}",
                'n': f"{row0.iloc[0]['n']:.0f}"
            })

    table_df = pd.DataFrame(table_data)
    table_df.to_csv(OUTPUT_DIR / 'correlation_table.csv', index=False)
    print(f"\nTable saved to {OUTPUT_DIR / 'correlation_table.csv'}")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("UAP–Meteor Temporal Correlation Analysis")
    print("Extending MoD Condign Report (1996) to 2000–2021")
    print("="*60)

    # Load data
    nuforc = load_nuforc()
    meteor_daily = load_imo_rates()

    # Aggregate
    print("\nAggregating to weekly bins...")
    combined_weekly = aggregate_timeseries(nuforc, meteor_daily, freq='W')

    print("Aggregating to monthly bins...")
    combined_monthly = aggregate_timeseries(nuforc, meteor_daily, freq='ME')

    # Correlations
    print("\nComputing weekly cross-correlations...")
    corr_weekly = compute_correlations(combined_weekly, max_lag_weeks=12)

    print("Computing monthly cross-correlations...")
    corr_monthly = compute_correlations(combined_monthly, max_lag_weeks=6)

    # Shower analysis
    print("\nAnalyzing meteor shower enhancement...")
    shower_results = shower_analysis(nuforc)

    # Annual cycle
    print("\nComputing annual cycles...")
    weekly_annual, meteor_weekly = annual_cycle_analysis(nuforc, meteor_daily)

    # Figures
    print("\nGenerating figures...")
    plot_correlation_figures(combined_weekly, combined_monthly,
                           corr_weekly, corr_monthly,
                           shower_results, weekly_annual, meteor_weekly)

    # Summary
    print_summary(corr_weekly, corr_monthly, shower_results, combined_monthly)

    print("\nDone!")
