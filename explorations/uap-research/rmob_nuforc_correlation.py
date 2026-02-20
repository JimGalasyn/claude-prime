"""
RMOB Radio Meteor Data × NUFORC UAP Correlation Analysis
=========================================================
Radio Meteor Observation Bulletin (RMOB) provides hourly meteor counts
from dozens of stations worldwide. Key advantages over IMO visual data:
  1. 24/7 detection (independent of weather/daylight)
  2. Hourly resolution → daily aggregation
  3. Multiple independent observers per day
  4. Geographic coordinates for each station

We compute daily median meteor rate across all active stations, then
correlate with NUFORC daily UAP report counts.

Uses the NUFORC All-Records dataset and RMOB text files (2000-2020).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import calendar
import re
from scipy import stats

# ============================================================
# Paths
# ============================================================
RMOB_DIR = Path("/tmp/rmob_data")
NUFORC_FILE = Path("/mnt/c/Users/jimga/OneDrive/Documents/Research/UAP/NUFORC/All-Nuforc-Records.csv")
OUTPUT_DIR = Path("/home/jim/repos/claude-prime/explorations/uap-research/output/rmob_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Parse a single RMOB file
# ============================================================
def parse_rmob_file(filepath):
    """Parse an RMOB text file. Returns dict with metadata and daily counts.

    Format: month header row, then day rows with hourly counts.
    Missing data marked as ???.
    Metadata in [Key]Value lines at bottom.
    """
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except Exception:
        return None

    # Extract filename info: Observer_MMYYYY rmob.TXT
    fname = filepath.stem.lower()

    # Parse month from header or filename
    month_names = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                   'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

    month = None
    year = None

    # Try to get month/year from filename pattern: name_MMYYYYrmob
    m = re.search(r'(\d{2})(\d{4})rmob', fname)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))

    # Parse header line for month name
    if lines and '|' in lines[0]:
        header = lines[0].strip().lower()
        for mname, mnum in month_names.items():
            if header.startswith(mname):
                month = mnum
                break

    if month is None or year is None:
        return None

    # Parse metadata
    metadata = {}
    for line in lines:
        if line.strip().startswith('['):
            m = re.match(r'\[([^\]]+)\](.*)', line.strip())
            if m:
                metadata[m.group(1).lower().strip()] = m.group(2).strip()

    lat = None
    lon = None
    if 'latitude gmap' in metadata:
        try:
            lat = float(metadata['latitude gmap'])
        except ValueError:
            pass
    if 'longitude gmap' in metadata:
        try:
            lon = float(metadata['longitude gmap'])
        except ValueError:
            pass

    observer = metadata.get('observer', fname.split('_')[0])
    country = metadata.get('country', '').strip()

    # Parse daily counts (skip header line)
    daily_totals = {}
    for line in lines[1:]:
        if line.strip().startswith('['):
            break
        if '|' not in line:
            continue

        parts = [p.strip() for p in line.strip().split('|')]
        if len(parts) < 3:
            continue

        try:
            day = int(parts[0])
        except ValueError:
            continue

        if day < 1 or day > 31:
            continue

        # Sum hourly counts, skipping ???
        hourly = []
        for p in parts[1:]:
            p = p.strip()
            if p and p != '???' and p != '':
                try:
                    hourly.append(int(p))
                except ValueError:
                    pass

        if len(hourly) >= 12:  # require at least 12 hours of data
            # Scale to full 24 hours if partial
            daily_total = sum(hourly) * 24 / len(hourly)
            try:
                date = datetime(year, month, day)
                daily_totals[date] = daily_total
            except ValueError:
                pass  # invalid date (e.g., Feb 30)

    if not daily_totals:
        return None

    return {
        'observer': observer,
        'country': country,
        'lat': lat,
        'lon': lon,
        'year': year,
        'month': month,
        'daily_totals': daily_totals,
    }


# ============================================================
# Parse all RMOB data
# ============================================================
print("Parsing RMOB radio meteor data...")
print("="*60)

all_records = []
file_count = 0
parse_errors = 0

for year_dir in sorted(RMOB_DIR.iterdir()):
    if not year_dir.is_dir():
        continue
    # Handle nested directory structure
    search_dirs = [year_dir]
    for sub in year_dir.iterdir():
        if sub.is_dir():
            search_dirs.append(sub)

    for sdir in search_dirs:
        for fpath in sdir.glob('*rmob*'):
            if fpath.suffix.lower() in ('.txt',):
                file_count += 1
                result = parse_rmob_file(fpath)
                if result:
                    all_records.append(result)
                else:
                    parse_errors += 1

print(f"  Parsed {len(all_records)} observer-months from {file_count} files")
print(f"  Parse errors: {parse_errors}")

# Aggregate to daily meteor index
# For each day: take the MEDIAN across all observers (robust to outliers)
daily_data = {}  # date → list of observer totals

for rec in all_records:
    for date, count in rec['daily_totals'].items():
        if date not in daily_data:
            daily_data[date] = []
        daily_data[date].append(count)

# Build dataframe
dates = sorted(daily_data.keys())
df_rmob = pd.DataFrame({
    'date': dates,
    'meteor_median': [np.median(daily_data[d]) for d in dates],
    'meteor_mean': [np.mean(daily_data[d]) for d in dates],
    'n_observers': [len(daily_data[d]) for d in dates],
})
df_rmob['date'] = pd.to_datetime(df_rmob['date'])
df_rmob = df_rmob.set_index('date')

print(f"\n  Daily meteor index: {len(df_rmob)} days ({df_rmob.index.min().date()} to {df_rmob.index.max().date()})")
print(f"  Observers per day: median={df_rmob['n_observers'].median():.0f}, "
      f"max={df_rmob['n_observers'].max():.0f}")

# ============================================================
# Load NUFORC data
# ============================================================
print(f"\nLoading NUFORC data...")

nuforc = pd.read_csv(NUFORC_FILE, low_memory=False)
print(f"  Total records: {len(nuforc)}")

# Parse dates
nuforc['date'] = pd.to_datetime(nuforc['EventDate'], format='mixed', errors='coerce')
nuforc = nuforc.dropna(subset=['date'])
nuforc['date_only'] = nuforc['date'].dt.date

# Daily counts
nuforc_daily = nuforc.groupby('date_only').size().reset_index(name='uap_count')
nuforc_daily['date'] = pd.to_datetime(nuforc_daily['date_only'])
nuforc_daily = nuforc_daily.set_index('date')

print(f"  Daily UAP counts: {len(nuforc_daily)} days")

# ============================================================
# Merge datasets
# ============================================================
df = df_rmob.join(nuforc_daily[['uap_count']], how='inner')
df = df.dropna()

# Filter to dates with sufficient observer coverage
df_good = df[df['n_observers'] >= 5].copy()

print(f"\n  Merged dataset: {len(df)} days")
print(f"  With ≥5 observers: {len(df_good)} days")
print(f"  Date range: {df_good.index.min().date()} to {df_good.index.max().date()}")

# Add derived columns
df_good['year'] = df_good.index.year
df_good['month'] = df_good.index.month
df_good['day_of_year'] = df_good.index.dayofyear

# ============================================================
# Detrending: remove long-term trends and seasonal cycles
# ============================================================
print("\nDetrending...")

# Monthly means for seasonal detrending
for col in ['meteor_median', 'uap_count']:
    monthly_mean = df_good.groupby('month')[col].transform('mean')
    df_good[f'{col}_deseason'] = df_good[col] / monthly_mean

# Year-by-year means for long-term detrending
for col in ['meteor_median', 'uap_count']:
    yearly_mean = df_good.groupby('year')[col].transform('mean')
    df_good[f'{col}_detrend'] = df_good[f'{col}_deseason'] / yearly_mean * yearly_mean.mean()

# ============================================================
# Correlation analysis
# ============================================================
print("\n" + "="*60)
print("CORRELATION RESULTS")
print("="*60)

# 1. Raw daily correlation
r_raw, p_raw = stats.pearsonr(df_good['meteor_median'], df_good['uap_count'])
rho_raw, p_rho_raw = stats.spearmanr(df_good['meteor_median'], df_good['uap_count'])
print(f"\n1. Raw daily correlation (N={len(df_good)}):")
print(f"   Pearson  r = {r_raw:+.4f} (p = {p_raw:.2e})")
print(f"   Spearman ρ = {rho_raw:+.4f} (p = {p_rho_raw:.2e})")

# 2. Deseasonalized
r_ds, p_ds = stats.pearsonr(df_good['meteor_median_deseason'], df_good['uap_count_deseason'])
rho_ds, p_ds_rho = stats.spearmanr(df_good['meteor_median_deseason'], df_good['uap_count_deseason'])
print(f"\n2. Deseasonalized daily correlation:")
print(f"   Pearson  r = {r_ds:+.4f} (p = {p_ds:.2e})")
print(f"   Spearman ρ = {rho_ds:+.4f} (p = {p_ds_rho:.2e})")

# 3. Monthly aggregation
monthly = df_good.resample('ME').agg({
    'meteor_median': 'mean',
    'uap_count': 'sum',
    'n_observers': 'mean',
}).dropna()
monthly = monthly[monthly['n_observers'] >= 5]

r_monthly, p_monthly = stats.pearsonr(monthly['meteor_median'], monthly['uap_count'])
rho_monthly, p_monthly_rho = stats.spearmanr(monthly['meteor_median'], monthly['uap_count'])
print(f"\n3. Monthly aggregated (N={len(monthly)}):")
print(f"   Pearson  r = {r_monthly:+.4f} (p = {p_monthly:.2e})")
print(f"   Spearman ρ = {rho_monthly:+.4f} (p = {p_monthly_rho:.2e})")

# 4. Year-by-year correlation
print(f"\n4. Year-by-year monthly correlation:")
yearly_results = []
for yr in sorted(df_good['year'].unique()):
    yr_data = df_good[df_good['year'] == yr]
    yr_monthly = yr_data.resample('ME').agg({
        'meteor_median': 'mean',
        'uap_count': 'sum',
    }).dropna()
    if len(yr_monthly) >= 8:
        r, p = stats.pearsonr(yr_monthly['meteor_median'], yr_monthly['uap_count'])
        rho, p_rho = stats.spearmanr(yr_monthly['meteor_median'], yr_monthly['uap_count'])
        sig = '*' if p < 0.05 else ' '
        print(f"   {yr}: r={r:+.3f} (p={p:.3f}){sig}  ρ={rho:+.3f}  N_months={len(yr_monthly)}")
        yearly_results.append({'year': yr, 'r': r, 'p': p, 'rho': rho, 'n': len(yr_monthly)})

yearly_df = pd.DataFrame(yearly_results)
sig_years = yearly_df[yearly_df['p'] < 0.05]
print(f"\n   Significant years (p<0.05): {len(sig_years)} / {len(yearly_df)}")
if len(sig_years) > 0:
    print(f"   Mean r for significant years: {sig_years['r'].mean():+.3f}")

# 5. Lag analysis: NUFORC reports might lag meteors by 0-7 days
#    (transport time from mesosphere)
print(f"\n5. Lag analysis (deseasonalized):")
lags = range(-3, 15)
lag_results = []
for lag in lags:
    shifted = df_good['meteor_median_deseason'].shift(lag)
    mask = shifted.notna() & df_good['uap_count_deseason'].notna()
    if mask.sum() > 100:
        r, p = stats.pearsonr(shifted[mask], df_good.loc[mask, 'uap_count_deseason'])
        lag_results.append({'lag': lag, 'r': r, 'p': p})

lag_df = pd.DataFrame(lag_results)
best_lag = lag_df.loc[lag_df['r'].abs().idxmax()]
print(f"   Best lag: {int(best_lag['lag'])} days (r={best_lag['r']:+.4f}, p={best_lag['p']:.2e})")
for _, row in lag_df.iterrows():
    sig = '*' if row['p'] < 0.05 else ' '
    print(f"   lag={int(row['lag']):+2d}d: r={row['r']:+.4f} (p={row['p']:.2e}){sig}")

# 6. Shower-window analysis
print(f"\n6. Major shower windows:")
showers = {
    'Quadrantids':  (1, 1, 1, 5),    # Jan 1-5
    'Lyrids':       (4, 16, 4, 25),
    'Eta Aquariids': (4, 19, 5, 28),
    'Perseids':     (7, 17, 8, 24),
    'Orionids':     (10, 2, 11, 7),
    'Leonids':      (11, 6, 11, 30),
    'Geminids':     (12, 4, 12, 17),
}

for shower, (m1, d1, m2, d2) in showers.items():
    # Days in and outside shower window
    in_shower = df_good[
        ((df_good['month'] == m1) & (df_good.index.day >= d1)) |
        ((df_good['month'] == m2) & (df_good.index.day <= d2)) |
        ((df_good['month'] > m1) & (df_good['month'] < m2))
    ]
    out_shower = df_good.drop(in_shower.index)

    if len(in_shower) > 30 and len(out_shower) > 100:
        uap_in = in_shower['uap_count'].mean()
        uap_out = out_shower['uap_count'].mean()
        met_in = in_shower['meteor_median'].mean()
        met_out = out_shower['meteor_median'].mean()
        enhancement = uap_in / max(uap_out, 0.01)
        met_enh = met_in / max(met_out, 0.01)
        t, p = stats.ttest_ind(in_shower['uap_count'], out_shower['uap_count'])
        sig = '*' if p < 0.05 else ' '
        print(f"   {shower:15s}: UAP {enhancement:.2f}x (p={p:.3f}){sig}  "
              f"meteor {met_enh:.2f}x  N_in={len(in_shower)}")

# ============================================================
# Figures
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Radio Meteor (RMOB) × NUFORC UAP Report Correlation\n'
             f'Daily data, {df_good.index.min().date()} to {df_good.index.max().date()}, '
             f'N={len(df_good)} days',
             fontsize=14, fontweight='bold')

# (a) Time series: meteor rate
ax = axes[0, 0]
weekly_meteor = df_good['meteor_median'].resample('W').mean()
ax.plot(weekly_meteor.index, weekly_meteor.values, 'C0-', linewidth=0.8, alpha=0.7)
ax.set_ylabel('Median Meteor Rate\n(counts/day)')
ax.set_title('(a) RMOB Weekly Meteor Rate')
ax.set_xlim(df_good.index.min(), df_good.index.max())

# (b) Time series: UAP counts
ax = axes[0, 1]
weekly_uap = df_good['uap_count'].resample('W').sum()
ax.plot(weekly_uap.index, weekly_uap.values, 'C3-', linewidth=0.8, alpha=0.7)
ax.set_ylabel('Weekly UAP Reports')
ax.set_title('(b) NUFORC Weekly Report Count')
ax.set_xlim(df_good.index.min(), df_good.index.max())

# (c) Observer coverage over time
ax = axes[0, 2]
monthly_obs = df_good['n_observers'].resample('ME').mean()
ax.bar(monthly_obs.index, monthly_obs.values, width=25, color='C2', alpha=0.6)
ax.set_ylabel('Mean Observers/Day')
ax.set_title('(c) RMOB Observer Coverage')

# (d) Scatter: daily correlation
ax = axes[1, 0]
# Downsample for visibility
ax.scatter(df_good['meteor_median'], df_good['uap_count'],
          s=3, alpha=0.15, c='C0')
ax.set_xlabel('Daily Median Meteor Rate')
ax.set_ylabel('Daily UAP Reports')
ax.set_title(f'(d) Daily Scatter (r={r_raw:+.3f}, p={p_raw:.1e})')
# Add trend line
z = np.polyfit(df_good['meteor_median'], df_good['uap_count'], 1)
x_fit = np.linspace(df_good['meteor_median'].min(), df_good['meteor_median'].max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2, alpha=0.7)

# (e) Monthly scatter
ax = axes[1, 1]
ax.scatter(monthly['meteor_median'], monthly['uap_count'],
          s=30, alpha=0.5, c='C1', edgecolors='k', linewidth=0.3)
ax.set_xlabel('Monthly Mean Meteor Rate')
ax.set_ylabel('Monthly UAP Report Count')
ax.set_title(f'(e) Monthly Scatter (r={r_monthly:+.3f}, p={p_monthly:.1e})')
z = np.polyfit(monthly['meteor_median'], monthly['uap_count'], 1)
x_fit = np.linspace(monthly['meteor_median'].min(), monthly['meteor_median'].max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2, alpha=0.7)

# (f) Year-by-year correlation coefficients
ax = axes[1, 2]
if len(yearly_df) > 0:
    colors = ['C3' if p < 0.05 else 'C0' for p in yearly_df['p']]
    ax.bar(yearly_df['year'], yearly_df['r'], color=colors, alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Pearson r (monthly)')
    ax.set_title('(f) Year-by-Year Correlation\n(red = p<0.05)')

# (g) Lag correlation
ax = axes[2, 0]
if len(lag_df) > 0:
    colors_lag = ['C3' if p < 0.05 else 'C0' for p in lag_df['p']]
    ax.bar(lag_df['lag'], lag_df['r'], color=colors_lag, alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Lag (days): meteor leads UAP →')
    ax.set_ylabel('Pearson r (deseasonalized)')
    ax.set_title(f'(g) Lag Correlation\n(best: lag={int(best_lag["lag"])}d, r={best_lag["r"]:+.3f})')
    ax.axvspan(1, 7, alpha=0.1, color='orange', label='Transport window (1-7d)')
    ax.legend(fontsize=8)

# (h) Seasonal cycle comparison
ax = axes[2, 1]
monthly_season_met = df_good.groupby('month')['meteor_median'].mean()
monthly_season_uap = df_good.groupby('month')['uap_count'].mean()
ax2 = ax.twinx()
ax.bar(monthly_season_met.index - 0.2, monthly_season_met.values, 0.4,
      color='C0', alpha=0.6, label='Meteor rate')
ax2.bar(monthly_season_uap.index + 0.2, monthly_season_uap.values, 0.4,
       color='C3', alpha=0.6, label='UAP reports')
ax.set_xlabel('Month')
ax.set_ylabel('Mean Meteor Rate', color='C0')
ax2.set_ylabel('Mean UAP Reports/day', color='C3')
ax.set_title('(h) Seasonal Cycles')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# (i) Shower enhancement
ax = axes[2, 2]
shower_names = []
shower_enh = []
shower_met_enh = []
for shower, (m1, d1, m2, d2) in showers.items():
    in_shower = df_good[
        ((df_good['month'] == m1) & (df_good.index.day >= d1)) |
        ((df_good['month'] == m2) & (df_good.index.day <= d2)) |
        ((df_good['month'] > m1) & (df_good['month'] < m2))
    ]
    out_shower = df_good.drop(in_shower.index)
    if len(in_shower) > 30:
        shower_names.append(shower.replace(' ', '\n'))
        shower_enh.append(in_shower['uap_count'].mean() / max(out_shower['uap_count'].mean(), 0.01))
        shower_met_enh.append(in_shower['meteor_median'].mean() / max(out_shower['meteor_median'].mean(), 0.01))

x_pos = np.arange(len(shower_names))
ax.bar(x_pos - 0.2, shower_met_enh, 0.4, color='C0', alpha=0.7, label='Meteor enhancement')
ax.bar(x_pos + 0.2, shower_enh, 0.4, color='C3', alpha=0.7, label='UAP enhancement')
ax.axhline(y=1, color='gray', ls='--', lw=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(shower_names, fontsize=7)
ax.set_ylabel('Enhancement Factor')
ax.set_title('(i) Shower Window Enhancement')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rmob_nuforc_correlation.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'rmob_nuforc_correlation.pdf', bbox_inches='tight')
plt.close()
print("Figure saved: rmob_nuforc_correlation.png/pdf")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"RMOB data: {len(all_records)} observer-months, {len(df_rmob)} daily records")
print(f"NUFORC data: {len(nuforc)} reports")
print(f"Merged: {len(df_good)} days with ≥5 RMOB observers")
print(f"\nKey results:")
print(f"  Raw daily:        r={r_raw:+.4f} (p={p_raw:.2e})")
print(f"  Deseasonalized:   r={r_ds:+.4f} (p={p_ds:.2e})")
print(f"  Monthly:          r={r_monthly:+.4f} (p={p_monthly:.2e})")
if len(sig_years) > 0:
    print(f"  Significant years: {len(sig_years)}/{len(yearly_df)} with mean r={sig_years['r'].mean():+.3f}")
print(f"  Best lag:          {int(best_lag['lag'])} days (r={best_lag['r']:+.4f})")
print(f"\nFigures saved to {OUTPUT_DIR}")
print("Done!")
