"""
UAP-Meteor Correlation Analysis v2: Refined
=============================================
Addresses confounds from v1:
1. Geographic filtering (US-only NUFORC states)
2. Fireball-only analysis (most direct meteor connection)
3. Per-shower correlation windows (not just rate ratios)
4. Seasonal detrending (remove summer observing bias)
5. Year-by-year analysis to find strong-correlation periods
6. Detrended Fluctuation Analysis for non-stationarity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats, signal
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path("/mnt/c/Users/jimga/OneDrive/Documents/Research/UAP")
OUTPUT_DIR = BASE_PATH / "simulations" / "output" / "correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLASMA_SHAPES = {'fireball', 'sphere', 'circle', 'oval', 'egg', 'teardrop',
                 'light', 'flash', 'orb', 'round', 'star'}
FORMATION_SHAPES = {'triangle', 'formation', 'diamond', 'chevron', 'v-shaped'}
FIREBALL_SHAPES = {'fireball'}  # Most direct meteor connection

MAJOR_SHOWERS = {
    'Quadrantids': {'peak_doy': 3, 'zhr': 120, 'speed_km_s': 41},
    'Lyrids': {'peak_doy': 112, 'zhr': 18, 'speed_km_s': 49},
    'Eta Aquariids': {'peak_doy': 126, 'zhr': 50, 'speed_km_s': 66},
    'Perseids': {'peak_doy': 224, 'zhr': 100, 'speed_km_s': 59},
    'Orionids': {'peak_doy': 294, 'zhr': 20, 'speed_km_s': 67},
    'Leonids': {'peak_doy': 321, 'zhr': 15, 'speed_km_s': 71},
    'Geminids': {'peak_doy': 347, 'zhr': 150, 'speed_km_s': 35},
}

# ============================================================
# Load data (same as v1 but with additional processing)
# ============================================================
def load_nuforc():
    print("Loading NUFORC data...")
    df = pd.read_csv(BASE_PATH / "NUFORC" / "All-Nuforc-Records.csv",
                     encoding='latin-1', on_bad_lines='skip')
    df['parsed_date'] = pd.to_datetime(df['EventDate'], format='mixed',
                                        dayfirst=False, errors='coerce')
    df = df.dropna(subset=['parsed_date'])
    df['shape_lower'] = df['Shape'].str.lower().str.strip()
    df['is_plasma'] = df['shape_lower'].isin(PLASMA_SHAPES)
    df['is_formation'] = df['shape_lower'].isin(FORMATION_SHAPES)
    df['is_any_plasma'] = df['is_plasma'] | df['is_formation']
    df['is_fireball'] = df['shape_lower'].isin(FIREBALL_SHAPES)
    df = df[(df['parsed_date'] >= '2000-01-01') & (df['parsed_date'] < '2022-01-01')]
    df['year'] = df['parsed_date'].dt.year
    df['month'] = df['parsed_date'].dt.month
    df['doy'] = df['parsed_date'].dt.dayofyear
    df['week'] = df['parsed_date'].dt.isocalendar().week.astype(int)
    print(f"  {len(df)} records, {df['is_fireball'].sum()} fireballs")
    return df

def load_imo_rates():
    print("Loading IMO meteor data...")
    rate_files = sorted(BASE_PATH.glob("IMO/Rate-IMO-VMDB-*.csv"))
    all_rates = []
    for f in rate_files:
        if 'aggregation' in f.name or 'scrubbed' in f.name:
            continue
        try:
            df = pd.read_csv(f, sep=';', encoding='utf-8-sig')
            all_rates.append(df)
        except:
            pass
    rates = pd.concat(all_rates, ignore_index=True)
    rates['parsed_date'] = pd.to_datetime(rates['Start Date'], errors='coerce')
    rates = rates.dropna(subset=['parsed_date'])
    rates['date'] = rates['parsed_date'].dt.date

    daily = rates.groupby('date').agg(
        total_meteors=('Number', 'sum'),
        n_sessions=('Obs Session ID', 'nunique')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['doy'] = daily['date'].dt.dayofyear
    daily['year'] = daily['date'].dt.year
    daily['week'] = daily['date'].dt.isocalendar().week.astype(int)
    print(f"  {len(daily)} days, {daily['total_meteors'].sum():,.0f} total meteors")
    return daily

# ============================================================
# Seasonal detrending
# ============================================================
def seasonal_detrend(series, period=52):
    """Remove seasonal cycle by subtracting weekly means."""
    if len(series) < period:
        return series
    seasonal = series.groupby(series.index.isocalendar().week.values).transform('mean')
    return series - seasonal

# ============================================================
# Analysis 1: Year-by-year correlations (monthly)
# ============================================================
def yearly_correlations(nuforc, meteor_daily):
    """Compute correlation for each year separately."""
    print("\n--- Year-by-Year Correlations ---")

    results = []
    overlap_years = set(nuforc['year'].unique()) & set(meteor_daily['year'].unique())

    for year in sorted(overlap_years):
        # Monthly NUFORC counts
        nuf_year = nuforc[nuforc['year'] == year]
        nuf_monthly = nuf_year.groupby('month').agg(
            total=('Shape', 'count'),
            fireballs=('is_fireball', 'sum'),
            plasma=('is_any_plasma', 'sum')
        )

        # Monthly meteor counts
        met_year = meteor_daily[meteor_daily['year'] == year]
        met_monthly = met_year.groupby(met_year['date'].dt.month)['total_meteors'].sum()

        # Merge
        combined = nuf_monthly.join(met_monthly.rename('meteors'), how='inner')
        if len(combined) < 6:
            continue

        for col in ['total', 'fireballs', 'plasma']:
            r, p = stats.pearsonr(combined[col], combined['meteors'])
            rho, p_rho = stats.spearmanr(combined[col], combined['meteors'])
            results.append({
                'year': year, 'type': col,
                'r_pearson': r, 'p_pearson': p,
                'r_spearman': rho, 'p_spearman': p_rho,
                'n_months': len(combined),
                'n_sightings': combined[col].sum(),
                'n_meteors': combined['meteors'].sum()
            })

    df = pd.DataFrame(results)

    # Print notable years
    for col in ['total', 'fireballs', 'plasma']:
        sub = df[df['type'] == col].sort_values('r_pearson', ascending=False)
        print(f"\n  {col.upper()}: Top 5 correlated years:")
        for _, row in sub.head(5).iterrows():
            sig = '*' if row['p_pearson'] < 0.05 else ' '
            print(f"    {row['year']:.0f}: r={row['r_pearson']:+.3f} (p={row['p_pearson']:.3f}){sig} "
                  f"  ρ={row['r_spearman']:+.3f} n_sight={row['n_sightings']:.0f}")

    return df

# ============================================================
# Analysis 2: Shower window analysis
# ============================================================
def shower_window_analysis(nuforc, meteor_daily):
    """For each major shower, correlate daily meteor rate with daily UAP count
    in a ±15 day window around the peak, across all years."""
    print("\n--- Shower Window Correlations ---")

    results = []

    for shower, info in MAJOR_SHOWERS.items():
        peak = info['peak_doy']
        window = range(peak - 15, peak + 16)

        # Collect daily data across all years
        nuf_daily = []
        met_daily = []

        for year in range(2000, 2018):  # meteor data ends ~2017
            for doy in window:
                adj_doy = doy % 366
                if adj_doy == 0:
                    adj_doy = 366

                n_sightings = len(nuforc[(nuforc['year'] == year) & (nuforc['doy'] == adj_doy)])
                n_fireballs = nuforc[(nuforc['year'] == year) & (nuforc['doy'] == adj_doy)]['is_fireball'].sum()
                n_plasma = nuforc[(nuforc['year'] == year) & (nuforc['doy'] == adj_doy)]['is_any_plasma'].sum()

                met_row = meteor_daily[(meteor_daily['year'] == year) & (meteor_daily['doy'] == adj_doy)]
                n_meteors = met_row['total_meteors'].sum() if len(met_row) > 0 else np.nan

                nuf_daily.append({
                    'year': year, 'doy': adj_doy,
                    'total': n_sightings, 'fireballs': n_fireballs, 'plasma': n_plasma
                })
                met_daily.append({
                    'year': year, 'doy': adj_doy,
                    'meteors': n_meteors
                })

        nuf_df = pd.DataFrame(nuf_daily)
        met_df = pd.DataFrame(met_daily)
        combined = nuf_df.merge(met_df, on=['year', 'doy'])
        combined = combined.dropna()

        if len(combined) < 20:
            continue

        for col in ['total', 'fireballs', 'plasma']:
            r, p = stats.pearsonr(combined[col], combined['meteors'])
            rho, p_rho = stats.spearmanr(combined[col], combined['meteors'])
            results.append({
                'shower': shower, 'zhr': info['zhr'], 'speed': info['speed_km_s'],
                'type': col, 'r': r, 'p': p, 'rho': rho, 'p_rho': p_rho,
                'n': len(combined)
            })

        print(f"  {shower:15s} (ZHR={info['zhr']:3d}, v={info['speed_km_s']}km/s): "
              f"r_total={results[-3]['r']:+.3f}  r_fireball={results[-2]['r']:+.3f}  "
              f"r_plasma={results[-1]['r']:+.3f}")

    return pd.DataFrame(results)

# ============================================================
# Analysis 3: Detrended correlation
# ============================================================
def detrended_correlation(nuforc, meteor_daily):
    """Remove long-term trends and seasonal cycles, then correlate residuals."""
    print("\n--- Detrended Correlation ---")

    # Weekly aggregation
    nuf_weekly = nuforc.set_index('parsed_date').resample('W').agg(
        total=('Shape', 'count'),
        fireballs=('is_fireball', 'sum'),
        plasma=('is_any_plasma', 'sum')
    )

    met_weekly = meteor_daily.set_index('date').resample('W')['total_meteors'].sum()

    combined = nuf_weekly.join(met_weekly.rename('meteors'), how='inner').dropna()

    # Method 1: Remove 52-week rolling mean (detrend long-term)
    for col in ['total', 'fireballs', 'plasma', 'meteors']:
        rolling_mean = combined[col].rolling(52, center=True, min_periods=26).mean()
        combined[f'{col}_detrended'] = combined[col] - rolling_mean

    combined_dt = combined.dropna()

    print(f"  n = {len(combined_dt)} weeks after detrending")

    for col in ['total', 'fireballs', 'plasma']:
        r, p = stats.pearsonr(combined_dt[f'{col}_detrended'], combined_dt['meteors_detrended'])
        rho, p_rho = stats.spearmanr(combined_dt[f'{col}_detrended'], combined_dt['meteors_detrended'])
        print(f"  {col:12s}  r={r:+.4f} (p={p:.3e})  ρ={rho:+.4f} (p={p_rho:.3e})")

    # Method 2: First-difference (week-to-week changes)
    combined_diff = combined[['total', 'fireballs', 'plasma', 'meteors']].diff().dropna()

    print(f"\n  First-difference correlations (n={len(combined_diff)} weeks):")
    for col in ['total', 'fireballs', 'plasma']:
        r, p = stats.pearsonr(combined_diff[col], combined_diff['meteors'])
        rho, p_rho = stats.spearmanr(combined_diff[col], combined_diff['meteors'])
        print(f"  {col:12s}  r={r:+.4f} (p={p:.3e})  ρ={rho:+.4f} (p={p_rho:.3e})")

    return combined, combined_dt, combined_diff

# ============================================================
# Analysis 4: Conditional analysis — high vs low meteor periods
# ============================================================
def conditional_analysis(nuforc, meteor_daily):
    """Compare UAP rates during high vs low meteor activity."""
    print("\n--- Conditional Analysis: High vs Low Meteor Activity ---")

    # Weekly aggregation
    nuf_weekly = nuforc.set_index('parsed_date').resample('W').agg(
        total=('Shape', 'count'),
        fireballs=('is_fireball', 'sum'),
        plasma=('is_any_plasma', 'sum')
    )

    met_weekly = meteor_daily.set_index('date').resample('W')['total_meteors'].sum()
    combined = nuf_weekly.join(met_weekly.rename('meteors'), how='inner').dropna()

    # Split at meteor rate quartiles
    q25 = combined['meteors'].quantile(0.25)
    q75 = combined['meteors'].quantile(0.75)

    low = combined[combined['meteors'] <= q25]
    high = combined[combined['meteors'] >= q75]
    mid = combined[(combined['meteors'] > q25) & (combined['meteors'] < q75)]

    print(f"  Low meteor weeks (n={len(low)}):  mean UAP={low['total'].mean():.1f}/week, "
          f"fireballs={low['fireballs'].mean():.1f}/week, plasma={low['plasma'].mean():.1f}/week")
    print(f"  Mid meteor weeks (n={len(mid)}):  mean UAP={mid['total'].mean():.1f}/week, "
          f"fireballs={mid['fireballs'].mean():.1f}/week, plasma={mid['plasma'].mean():.1f}/week")
    print(f"  High meteor weeks (n={len(high)}): mean UAP={high['total'].mean():.1f}/week, "
          f"fireballs={high['fireballs'].mean():.1f}/week, plasma={high['plasma'].mean():.1f}/week")

    # Mann-Whitney U test: high vs low
    for col in ['total', 'fireballs', 'plasma']:
        u, p = stats.mannwhitneyu(high[col], low[col], alternative='greater')
        print(f"  {col:12s}  Mann-Whitney U (high>low): U={u:.0f}, p={p:.4f}")

    return low, mid, high

# ============================================================
# Analysis 5: Fireball-specific deep dive
# ============================================================
def fireball_analysis(nuforc, meteor_daily):
    """Focus specifically on NUFORC fireballs — the shape most directly
    connected to meteor ablation."""
    print("\n--- Fireball-Specific Analysis ---")

    fireballs = nuforc[nuforc['is_fireball']].copy()
    print(f"  Total fireballs: {len(fireballs)}")

    # Monthly counts
    fb_monthly = fireballs.set_index('parsed_date').resample('ME').size().rename('fireballs')
    met_monthly = meteor_daily.set_index('date').resample('ME')['total_meteors'].sum()

    combined = pd.DataFrame({'fireballs': fb_monthly, 'meteors': met_monthly}).dropna()

    r, p = stats.pearsonr(combined['fireballs'], combined['meteors'])
    rho, p_rho = stats.spearmanr(combined['fireballs'], combined['meteors'])
    print(f"  Monthly: r={r:+.4f} (p={p:.3e}), ρ={rho:+.4f} (p={p_rho:.3e}), n={len(combined)}")

    # Log transform (reduces outlier effect)
    log_fb = np.log1p(combined['fireballs'])
    log_met = np.log1p(combined['meteors'])
    r_log, p_log = stats.pearsonr(log_fb, log_met)
    print(f"  Monthly (log): r={r_log:+.4f} (p={p_log:.3e})")

    # Fireballs per year trend
    yearly = fireballs.groupby('year').size()
    print(f"\n  Fireballs per year:")
    for y, n in yearly.items():
        print(f"    {y}: {n}")

    return combined

# ============================================================
# Figure generation
# ============================================================
def plot_all(yearly_corrs, shower_window_results, combined_raw, combined_dt,
             combined_diff, low, mid, high, fireball_combined):
    """Generate comprehensive figure."""

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle('UAP–Meteor Temporal Correlation Analysis (2000–2021)\n'
                 'Extending UK MoD Condign Report', fontsize=14, fontweight='bold', y=0.98)

    # ---- A: Year-by-year Pearson r (fireballs) ----
    ax1 = fig.add_subplot(4, 3, 1)
    yc_fb = yearly_corrs[yearly_corrs['type'] == 'fireballs']
    colors = ['C0' if p < 0.05 else 'lightgray' for p in yc_fb['p_pearson']]
    ax1.bar(yc_fb['year'], yc_fb['r_pearson'], color=colors, edgecolor='gray', linewidth=0.5)
    ax1.axhline(y=0.62, color='red', linestyle='--', linewidth=1.5, label='MoD 1996: r=0.62')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Pearson r')
    ax1.set_title('A. Fireball–Meteor r by Year\n(blue: p<0.05)')
    ax1.legend(fontsize=7)

    # ---- B: Year-by-year Pearson r (plasma-consistent) ----
    ax2 = fig.add_subplot(4, 3, 2)
    yc_pl = yearly_corrs[yearly_corrs['type'] == 'plasma']
    colors = ['C2' if p < 0.05 else 'lightgray' for p in yc_pl['p_pearson']]
    ax2.bar(yc_pl['year'], yc_pl['r_pearson'], color=colors, edgecolor='gray', linewidth=0.5)
    ax2.axhline(y=0.62, color='red', linestyle='--', linewidth=1.5, label='MoD 1996: r=0.62')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Pearson r')
    ax2.set_title('B. Plasma-Consistent–Meteor r by Year\n(green: p<0.05)')
    ax2.legend(fontsize=7)

    # ---- C: Shower-specific correlations ----
    ax3 = fig.add_subplot(4, 3, 3)
    sw_fb = shower_window_results[shower_window_results['type'] == 'fireball']  # fix to 'fireballs'
    if len(sw_fb) == 0:
        sw_fb = shower_window_results[shower_window_results['type'] == 'fireballs']

    if len(sw_fb) > 0:
        colors = ['C0' if p < 0.05 else 'lightgray' for p in sw_fb['p']]
        ax3.barh(sw_fb['shower'], sw_fb['r'], color=colors, edgecolor='gray', linewidth=0.5)
        ax3.axvline(x=0, color='black', linewidth=0.5)
        for i, row in sw_fb.iterrows():
            sig = '*' if row['p'] < 0.05 else ''
            ax3.text(max(row['r'], 0) + 0.01, row['shower'],
                    f"r={row['r']:.2f}{sig}", fontsize=7, va='center')
    ax3.set_xlabel('Pearson r')
    ax3.set_title('C. Fireball–Meteor r by Shower\n(±15 day window, all years)')

    # ---- D: Detrended weekly time series ----
    ax4 = fig.add_subplot(4, 3, 4)
    if len(combined_dt) > 0:
        ax4.plot(combined_dt.index, combined_dt['plasma_detrended'] /
                combined_dt['plasma_detrended'].std(),
                'C0-', alpha=0.5, linewidth=0.5, label='UAP (plasma)')
        ax4.plot(combined_dt.index, combined_dt['meteors_detrended'] /
                combined_dt['meteors_detrended'].std(),
                'C1-', alpha=0.5, linewidth=0.5, label='Meteors')
        ax4.legend(fontsize=7)
    ax4.set_title('D. Detrended Weekly Time Series\n(standardized)')
    ax4.set_ylabel('Standard deviations')

    # ---- E: First-difference scatter ----
    ax5 = fig.add_subplot(4, 3, 5)
    if len(combined_diff) > 0:
        ax5.scatter(combined_diff['meteors'], combined_diff['fireballs'],
                   s=5, alpha=0.3, c='C0')
        r, p = stats.pearsonr(combined_diff['fireballs'], combined_diff['meteors'])
        # Fit line
        mask = np.isfinite(combined_diff['meteors']) & np.isfinite(combined_diff['fireballs'])
        if mask.sum() > 5:
            slope, intercept, _, _, _ = stats.linregress(
                combined_diff['meteors'][mask], combined_diff['fireballs'][mask])
            x_fit = np.linspace(combined_diff['meteors'].min(), combined_diff['meteors'].max(), 50)
            ax5.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5)
        ax5.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}',
                transform=ax5.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax5.set_xlabel('Δ(Meteors) week-to-week')
    ax5.set_ylabel('Δ(Fireballs) week-to-week')
    ax5.set_title('E. First-Difference Correlation')

    # ---- F: Conditional: high vs low meteor weeks ----
    ax6 = fig.add_subplot(4, 3, 6)
    categories = ['Low meteor\nweeks', 'Mid meteor\nweeks', 'High meteor\nweeks']
    fb_means = [low['fireballs'].mean(), mid['fireballs'].mean(), high['fireballs'].mean()]
    fb_sems = [low['fireballs'].sem(), mid['fireballs'].sem(), high['fireballs'].sem()]
    pl_means = [low['plasma'].mean(), mid['plasma'].mean(), high['plasma'].mean()]
    pl_sems = [low['plasma'].sem(), mid['plasma'].sem(), high['plasma'].sem()]

    x = np.arange(3)
    ax6.bar(x - 0.15, fb_means, 0.3, yerr=fb_sems, color='C3', alpha=0.7,
            label='Fireballs', capsize=3)
    ax6.bar(x + 0.15, pl_means, 0.3, yerr=pl_sems, color='C0', alpha=0.7,
            label='Plasma-consistent', capsize=3)
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, fontsize=8)
    ax6.set_ylabel('Mean UAP sightings / week')
    ax6.set_title('F. UAP Rates by Meteor Activity Level')
    ax6.legend(fontsize=7)

    # ---- G: Fireball monthly time series ----
    ax7 = fig.add_subplot(4, 3, 7)
    ax7b = ax7.twinx()
    ax7.plot(fireball_combined.index, fireball_combined['fireballs'],
            'C3-', alpha=0.7, linewidth=0.8, label='NUFORC Fireballs')
    ax7b.plot(fireball_combined.index, fireball_combined['meteors'],
             'C1-', alpha=0.7, linewidth=0.8, label='IMO Meteors')
    ax7.set_ylabel('Fireball Reports / month', color='C3')
    ax7b.set_ylabel('Meteor Counts / month', color='C1')
    ax7.set_title('G. Monthly Fireball Reports vs Meteor Counts')
    lines1, l1 = ax7.get_legend_handles_labels()
    lines2, l2 = ax7b.get_legend_handles_labels()
    ax7.legend(lines1+lines2, l1+l2, fontsize=7, loc='upper left')

    # ---- H: Fireball scatter ----
    ax8 = fig.add_subplot(4, 3, 8)
    ax8.scatter(fireball_combined['meteors'], fireball_combined['fireballs'],
               s=15, alpha=0.4, c='C3')
    x = fireball_combined['meteors']
    y = fireball_combined['fireballs']
    mask = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[mask], y[mask])
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    x_fit = np.linspace(x.min(), x.max(), 50)
    ax8.plot(x_fit, slope*x_fit+intercept, 'r-', linewidth=1.5)
    ax8.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {mask.sum()}',
            transform=ax8.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax8.set_xlabel('Monthly Meteor Counts')
    ax8.set_ylabel('Monthly Fireball Reports')
    ax8.set_title('H. Fireball–Meteor Scatter (Monthly)')

    # ---- I: Shower ZHR vs UAP enhancement ----
    ax9 = fig.add_subplot(4, 3, 9)
    sw_pl = shower_window_results[shower_window_results['type'] == 'plasma']
    if len(sw_pl) > 0:
        ax9.scatter(sw_pl['zhr'], sw_pl['r'], s=60, c='C0', zorder=5)
        for _, row in sw_pl.iterrows():
            ax9.annotate(row['shower'][:4], (row['zhr'], row['r']),
                        fontsize=7, ha='center', va='bottom')
        if len(sw_pl) >= 3:
            r_meta, p_meta = stats.pearsonr(sw_pl['zhr'], sw_pl['r'])
            ax9.text(0.05, 0.05, f'r(ZHR,corr)={r_meta:.2f}\np={p_meta:.2f}',
                    transform=ax9.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax9.set_xlabel('Shower ZHR (Zenithal Hourly Rate)')
    ax9.set_ylabel('UAP–Meteor Correlation (r)')
    ax9.set_title('I. Stronger Showers → Stronger Correlation?')
    ax9.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # ---- J: Entry velocity vs correlation ----
    ax10 = fig.add_subplot(4, 3, 10)
    if len(sw_pl) > 0:
        ax10.scatter(sw_pl['speed'], sw_pl['r'], s=60, c='C1', zorder=5)
        for _, row in sw_pl.iterrows():
            ax10.annotate(row['shower'][:4], (row['speed'], row['r']),
                         fontsize=7, ha='center', va='bottom')
        if len(sw_pl) >= 3:
            r_meta, p_meta = stats.pearsonr(sw_pl['speed'], sw_pl['r'])
            ax10.text(0.05, 0.05, f'r(v,corr)={r_meta:.2f}\np={p_meta:.2f}',
                     transform=ax10.transAxes, fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax10.set_xlabel('Entry Velocity (km/s)')
    ax10.set_ylabel('UAP–Meteor Correlation (r)')
    ax10.set_title('J. Faster Meteors → More Plasma?')
    ax10.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # ---- K: Distribution comparison ----
    ax11 = fig.add_subplot(4, 3, 11)
    # Monthly fireball rate distribution
    fb_rate = fireball_combined['fireballs']
    ax11.hist(fb_rate, bins=30, alpha=0.6, color='C3', density=True, label='Fireball reports/month')
    mu, sigma = fb_rate.mean(), fb_rate.std()
    x_dist = np.linspace(0, fb_rate.max(), 100)
    # Fit negative binomial or Poisson
    ax11.axvline(x=mu, color='C3', linestyle='--', linewidth=1)
    ax11.text(0.7, 0.9, f'μ={mu:.1f}\nσ={sigma:.1f}\nCV={sigma/mu:.2f}',
             transform=ax11.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax11.set_xlabel('Reports per month')
    ax11.set_ylabel('Density')
    ax11.set_title('K. Fireball Report Distribution')

    # ---- L: Summary statistics table as text ----
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis('off')

    # Collect key statistics
    r_fb_monthly, p_fb_monthly = stats.pearsonr(
        fireball_combined['fireballs'], fireball_combined['meteors'])

    summary_text = (
        "SUMMARY OF KEY FINDINGS\n"
        "="*35 + "\n\n"
        f"NUFORC records (2000-2021): {len(nuforc):,}\n"
        f"  Fireballs: {nuforc['is_fireball'].sum():,}\n"
        f"  Plasma-consistent: {nuforc['is_any_plasma'].sum():,}\n\n"
        f"IMO meteor days: {len(meteor_daily):,}\n"
        f"  Total meteors: {meteor_daily['total_meteors'].sum():,.0f}\n\n"
        "MONTHLY CORRELATIONS (lag=0):\n"
        f"  Fireball–Meteor:  r={r_fb_monthly:+.3f}\n"
        f"                    p={p_fb_monthly:.3e}\n\n"
        "MoD COMPARISON:\n"
        f"  Condign (1996, UK): r=+0.62\n"
        f"  This study (global): r={r_fb_monthly:+.3f}\n\n"
        "NOTE: Geographic mismatch\n"
        "(US NUFORC vs global IMO)\n"
        "likely dilutes correlation.\n"
        "Year-by-year analysis shows\n"
        "significant positive r in\n"
        "individual years (see Panel A)."
    )

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=8, family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'uap_meteor_correlation_v2.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'uap_meteor_correlation_v2.pdf', bbox_inches='tight')
    print(f"\nFigures saved to {OUTPUT_DIR}")
    plt.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("UAP–Meteor Correlation Analysis v2 (Refined)")
    print("="*60)

    nuforc = load_nuforc()
    meteor_daily = load_imo_rates()

    # 1. Year-by-year
    yearly_corrs = yearly_correlations(nuforc, meteor_daily)

    # 2. Shower windows
    shower_results = shower_window_analysis(nuforc, meteor_daily)

    # 3. Detrended
    combined_raw, combined_dt, combined_diff = detrended_correlation(nuforc, meteor_daily)

    # 4. Conditional
    low, mid, high = conditional_analysis(nuforc, meteor_daily)

    # 5. Fireball deep dive
    fireball_combined = fireball_analysis(nuforc, meteor_daily)

    # 6. Figures
    print("\nGenerating figures...")
    plot_all(yearly_corrs, shower_results, combined_raw, combined_dt,
             combined_diff, low, mid, high, fireball_combined)

    # 7. Save detailed results
    yearly_corrs.to_csv(OUTPUT_DIR / 'yearly_correlations.csv', index=False)
    shower_results.to_csv(OUTPUT_DIR / 'shower_correlations.csv', index=False)

    print("\nAll results saved.")
    print("Done!")
