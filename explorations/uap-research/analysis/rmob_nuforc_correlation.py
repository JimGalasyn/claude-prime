"""
RMOB Radio Meteor Data × NUFORC UAP Correlation Analysis
=========================================================
Radio Meteor Observation Bulletin (RMOB) provides hourly meteor counts
from dozens of stations worldwide. Key advantages over IMO visual data:
  1. 24/7 detection (independent of weather/daylight)
  2. Hourly resolution → daily aggregation
  3. Multiple independent observers per day
  4. Geographic coordinates for each station

Normalization strategy (per rmob_parser.py):
  - Each observer's daily count is normalized by their own monthly median
    to produce an activity index (1.0 = typical day for that observer).
  - This handles huge variation in equipment sensitivity across stations.
  - IQR-based outlier rejection filters noisy readings before aggregation.

Uses the NUFORC All-Records dataset and RMOB text files (2000-2020).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import calendar
import re
from scipy import stats

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RMOB_DIR = DATA_DIR / "rmob"
NUFORC_FILE = DATA_DIR / "nuforc" / "All-Nuforc-Records.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "rmob_correlation"
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

# ============================================================
# Per-observer normalization (adapted from rmob_parser.py)
# ============================================================
# Step 1: Collect per-observer daily counts keyed by (observer, year, month)
observer_monthly = defaultdict(list)  # (observer, year, month) → [daily counts]
date_raw = defaultdict(list)          # date → [(observer, count)]

for rec in all_records:
    obs_name = rec['observer']
    for dt, count in rec['daily_totals'].items():
        if count < 0 or count > 100000:  # sanity filter
            continue
        observer_monthly[(obs_name, dt.year, dt.month)].append(count)
        date_raw[dt].append((obs_name, count))

# Step 2: Compute monthly median baseline per observer
observer_baselines = {}
for key, vals in observer_monthly.items():
    median_val = np.median(vals)
    if median_val > 0:
        observer_baselines[key] = median_val

print(f"\n  Observer-month baselines computed: {len(observer_baselines)}")

# Step 3: Normalize each daily count by observer's own monthly baseline
date_normalized = defaultdict(list)
date_absolute = defaultdict(list)

for dt, obs_vals in date_raw.items():
    for obs_name, val in obs_vals:
        key = (obs_name, dt.year, dt.month)
        if key in observer_baselines:
            baseline = observer_baselines[key]
            normalized = val / baseline  # 1.0 = typical day for this observer
            date_normalized[dt].append(normalized)
            date_absolute[dt].append(val)

# Step 4: Aggregate with IQR-based outlier rejection
daily_rows = []
for dt in sorted(date_normalized.keys()):
    norm_vals = np.array(date_normalized[dt])
    abs_vals = np.array(date_absolute[dt])

    # IQR outlier rejection on normalized values
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

    daily_rows.append({
        'date': dt,
        'activity_index': float(np.mean(norm_clean)),
        'activity_median': float(np.median(norm_clean)),
        'meteor_median': float(np.median(abs_clean)),  # keep for backward compat
        'meteor_mean': float(np.mean(abs_clean)),
        'n_observers': len(norm_clean),
        'n_rejected': int(len(norm_vals) - len(norm_clean)),
    })

df_rmob = pd.DataFrame(daily_rows)
df_rmob['date'] = pd.to_datetime(df_rmob['date'])
df_rmob = df_rmob.set_index('date')

print(f"  Daily meteor index: {len(df_rmob)} days ({df_rmob.index.min().date()} to {df_rmob.index.max().date()})")
print(f"  Observers per day: median={df_rmob['n_observers'].median():.0f}, "
      f"max={df_rmob['n_observers'].max():.0f}")
print(f"  Mean activity index: {df_rmob['activity_index'].mean():.3f} (should be ~1.0)")
print(f"  Outliers rejected: {df_rmob['n_rejected'].sum()} total across all days")

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

# Primary metric is now activity_index (already normalized per-observer)
# Also deseason the raw median and UAP count for comparison
for col in ['activity_index', 'meteor_median', 'uap_count']:
    monthly_mean = df_good.groupby('month')[col].transform('mean')
    # Avoid division by zero
    monthly_mean = monthly_mean.replace(0, np.nan)
    df_good[f'{col}_deseason'] = df_good[col] / monthly_mean

# Year-by-year means for long-term detrending
for col in ['activity_index', 'meteor_median', 'uap_count']:
    yearly_mean = df_good.groupby('year')[col].transform('mean')
    yearly_mean = yearly_mean.replace(0, np.nan)
    deseason_col = f'{col}_deseason'
    df_good[f'{col}_detrend'] = df_good[deseason_col] / yearly_mean * yearly_mean.mean()

# ============================================================
# Correlation analysis
# ============================================================
print("\n" + "="*60)
print("CORRELATION RESULTS")
print("="*60)

# 1. Raw daily correlation — activity_index (normalized) vs UAP count
r_raw, p_raw = stats.pearsonr(df_good['activity_index'], df_good['uap_count'])
rho_raw, p_rho_raw = stats.spearmanr(df_good['activity_index'], df_good['uap_count'])
print(f"\n1. Raw daily correlation — activity_index (N={len(df_good)}):")
print(f"   Pearson  r = {r_raw:+.4f} (p = {p_raw:.2e})")
print(f"   Spearman ρ = {rho_raw:+.4f} (p = {p_rho_raw:.2e})")

# Also report raw median for comparison
r_raw_med, p_raw_med = stats.pearsonr(df_good['meteor_median'], df_good['uap_count'])
print(f"   (raw median for comparison: r = {r_raw_med:+.4f}, p = {p_raw_med:.2e})")

# 2. Deseasonalized
r_ds, p_ds = stats.pearsonr(df_good['activity_index_deseason'], df_good['uap_count_deseason'])
rho_ds, p_ds_rho = stats.spearmanr(df_good['activity_index_deseason'], df_good['uap_count_deseason'])
print(f"\n2. Deseasonalized daily correlation (activity_index):")
print(f"   Pearson  r = {r_ds:+.4f} (p = {p_ds:.2e})")
print(f"   Spearman ρ = {rho_ds:+.4f} (p = {p_ds_rho:.2e})")

# 3. Monthly aggregation
monthly = df_good.resample('ME').agg({
    'activity_index': 'mean',
    'meteor_median': 'mean',
    'uap_count': 'sum',
    'n_observers': 'mean',
}).dropna()
monthly = monthly[monthly['n_observers'] >= 5]

r_monthly, p_monthly = stats.pearsonr(monthly['activity_index'], monthly['uap_count'])
rho_monthly, p_monthly_rho = stats.spearmanr(monthly['activity_index'], monthly['uap_count'])
print(f"\n3. Monthly aggregated — activity_index (N={len(monthly)}):")
print(f"   Pearson  r = {r_monthly:+.4f} (p = {p_monthly:.2e})")
print(f"   Spearman ρ = {rho_monthly:+.4f} (p = {p_monthly_rho:.2e})")

# 4. Year-by-year correlation
print(f"\n4. Year-by-year monthly correlation (activity_index):")
yearly_results = []
for yr in sorted(df_good['year'].unique()):
    yr_data = df_good[df_good['year'] == yr]
    yr_monthly = yr_data.resample('ME').agg({
        'activity_index': 'mean',
        'meteor_median': 'mean',
        'uap_count': 'sum',
    }).dropna()
    if len(yr_monthly) >= 8:
        r, p = stats.pearsonr(yr_monthly['activity_index'], yr_monthly['uap_count'])
        rho, p_rho = stats.spearmanr(yr_monthly['activity_index'], yr_monthly['uap_count'])
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
print(f"\n5. Lag analysis (deseasonalized activity_index):")
lags = range(-3, 15)
lag_results = []
for lag in lags:
    shifted = df_good['activity_index_deseason'].shift(lag)
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
        act_in = in_shower['activity_index'].mean()
        act_out = out_shower['activity_index'].mean()
        enhancement = uap_in / max(uap_out, 0.01)
        act_enh = act_in / max(act_out, 0.01)
        t, p = stats.ttest_ind(in_shower['uap_count'], out_shower['uap_count'])
        sig = '*' if p < 0.05 else ' '
        print(f"   {shower:15s}: UAP {enhancement:.2f}x (p={p:.3f}){sig}  "
              f"activity {act_enh:.2f}x  N_in={len(in_shower)}")

# ============================================================
# 7. Composition-dependent analysis: entry velocity & parent body
# ============================================================
# Meteor showers with known physical properties from spectroscopy literature
# (Borovicka et al. 2005, Trigo-Rodriguez et al. 2003, Vojacek et al. 2019)
print(f"\n7. Composition-dependent shower analysis:")
print(f"   Testing: does entry velocity predict UAP enhancement?")
print(f"   (High-v cometary → nm-scale MSPs → dusty plasma)")
print(f"   (Low-v asteroidal → μm-scale melt droplets → no plasma)")

shower_properties = {
    'Quadrantids':   {'v_entry': 41.0, 'parent': '2003 EH1',       'type': 'extinct_comet', 'zhr': 120, 'density': 'medium'},
    'Lyrids':        {'v_entry': 49.0, 'parent': 'C/1861 G1',      'type': 'comet',         'zhr': 18,  'density': 'low'},
    'Eta Aquariids': {'v_entry': 65.4, 'parent': '1P/Halley',      'type': 'comet',         'zhr': 50,  'density': 'low'},
    'Perseids':      {'v_entry': 59.0, 'parent': '109P/Swift-Tuttle','type': 'comet',        'zhr': 100, 'density': 'low'},
    'Orionids':      {'v_entry': 66.0, 'parent': '1P/Halley',      'type': 'comet',         'zhr': 20,  'density': 'low'},
    'Leonids':       {'v_entry': 70.0, 'parent': '55P/Tempel-Tuttle','type': 'comet',        'zhr': 15,  'density': 'very_low'},
    'Geminids':      {'v_entry': 35.0, 'parent': '3200 Phaethon',  'type': 'asteroid',      'zhr': 150, 'density': 'high'},
}

# Use WITHIN-SEASON control: compare shower peak to adjacent ±30 days
# in the same season. This controls for outdoor activity while preserving
# the meteor-driven signal (unlike monthly deseasonalization which is
# too aggressive and removes both confound and signal simultaneously).
from datetime import timedelta

# Define shower peak dates using approximate day-of-year
shower_peaks = {
    'Quadrantids':   {'peak_doy': 3,   'half_width': 3},
    'Lyrids':        {'peak_doy': 112, 'half_width': 5},
    'Eta Aquariids': {'peak_doy': 126, 'half_width': 10},
    'Perseids':      {'peak_doy': 224, 'half_width': 10},
    'Orionids':      {'peak_doy': 294, 'half_width': 7},
    'Leonids':       {'peak_doy': 321, 'half_width': 5},
    'Geminids':      {'peak_doy': 347, 'half_width': 7},
}

shower_analysis = []
print(f"\n   Within-season control: shower peak vs ±30 day adjacent baseline")
print(f"   (same season = same outdoor activity, different meteor flux)\n")

for shower, (m1, d1, m2, d2) in showers.items():
    in_shower = df_good[
        ((df_good['month'] == m1) & (df_good.index.day >= d1)) |
        ((df_good['month'] == m2) & (df_good.index.day <= d2)) |
        ((df_good['month'] > m1) & (df_good['month'] < m2))
    ]

    # Within-season control: ±30 days around shower, excluding shower window
    peak_info = shower_peaks.get(shower, {})
    peak_doy = peak_info.get('peak_doy', 0)
    hw = peak_info.get('half_width', 5)

    # Build control window: 30 days before and after shower, same season
    control_mask = pd.Series(False, index=df_good.index)
    shower_mask = pd.Series(False, index=df_good.index)

    for yr in df_good['year'].unique():
        try:
            from datetime import date as dt_date
            peak_date = dt_date(yr, 1, 1) + timedelta(days=peak_doy - 1)
            shower_start = peak_date - timedelta(days=hw)
            shower_end = peak_date + timedelta(days=hw)
            control_start = shower_start - timedelta(days=30)
            control_end = shower_end + timedelta(days=30)

            yr_shower = (df_good.index.date >= shower_start) & (df_good.index.date <= shower_end)
            yr_control_before = (df_good.index.date >= control_start) & (df_good.index.date < shower_start)
            yr_control_after = (df_good.index.date > shower_end) & (df_good.index.date <= control_end)

            shower_mask = shower_mask | yr_shower
            control_mask = control_mask | yr_control_before | yr_control_after
        except (ValueError, OverflowError):
            pass

    in_peak = df_good[shower_mask]
    in_control = df_good[control_mask]

    if len(in_peak) > 20 and len(in_control) > 40:
        uap_peak = in_peak['uap_count'].mean()
        uap_control = in_control['uap_count'].mean()
        act_peak = in_peak['activity_index'].mean()
        act_control = in_control['activity_index'].mean()

        within_enh = uap_peak / max(uap_control, 0.01)
        act_enh = act_peak / max(act_control, 0.01)
        excess_within = within_enh / max(act_enh, 0.01)

        # Mann-Whitney U test (non-parametric, handles skewed distributions)
        u_stat, p_within = stats.mannwhitneyu(
            in_peak['uap_count'], in_control['uap_count'],
            alternative='two-sided'
        )

        # Also raw global enhancement for comparison
        out_shower = df_good.drop(in_shower.index)
        uap_enh_global = in_shower['uap_count'].mean() / max(out_shower['uap_count'].mean(), 0.01)
        act_enh_global = in_shower['activity_index'].mean() / max(out_shower['activity_index'].mean(), 0.01)

        props = shower_properties.get(shower, {})
        shower_analysis.append({
            'name': shower,
            'v_entry': props.get('v_entry', np.nan),
            'parent_type': props.get('type', 'unknown'),
            'parent': props.get('parent', ''),
            'zhr': props.get('zhr', 0),
            'uap_enh_raw': uap_enh_global,
            'uap_enh_within': within_enh,
            'act_enh': act_enh,
            'act_enh_global': act_enh_global,
            'excess_enh': excess_within,
            'p_within': p_within,
            'n_peak': len(in_peak),
            'n_control': len(in_control),
        })

shower_df = pd.DataFrame(shower_analysis)

print(f"\n   {'Shower':15s} {'Parent':20s} {'Type':14s} v(km/s)  UAP_raw  Within   Act_w   Excess  p_within  N_pk/ctrl")
print(f"   {'-'*125}")
for _, row in shower_df.sort_values('v_entry', ascending=False).iterrows():
    sig = '*' if row['p_within'] < 0.05 else ' '
    print(f"   {row['name']:15s} {row['parent']:20s} {row['parent_type']:14s} "
          f"{row['v_entry']:5.1f}    {row['uap_enh_raw']:.3f}    {row['uap_enh_within']:.3f}    "
          f"{row['act_enh']:.3f}   {row['excess_enh']:.3f}   {row['p_within']:.4f}{sig}  "
          f"{row['n_peak']}/{row['n_control']}")

# Correlation: entry velocity vs UAP enhancement
if len(shower_df) >= 4:
    r_vel, p_vel = stats.pearsonr(shower_df['v_entry'], shower_df['uap_enh_raw'])
    r_vel_within, p_vel_within = stats.pearsonr(shower_df['v_entry'], shower_df['uap_enh_within'])
    r_vel_excess, p_vel_excess = stats.pearsonr(shower_df['v_entry'], shower_df['excess_enh'])
    print(f"\n   Entry velocity vs UAP enhancement:")
    print(f"     Raw (global):     r={r_vel:+.3f} (p={p_vel:.3f})")
    print(f"     Within-season:    r={r_vel_within:+.3f} (p={p_vel_within:.3f})")
    print(f"     Excess (UAP/act): r={r_vel_excess:+.3f} (p={p_vel_excess:.3f})")

    # Cometary vs asteroidal/extinct
    cometary = shower_df[shower_df['parent_type'] == 'comet']
    non_cometary = shower_df[shower_df['parent_type'] != 'comet']
    if len(cometary) >= 2 and len(non_cometary) >= 1:
        print(f"\n   Cometary showers (N={len(cometary)}):  mean within-season enh = {cometary['uap_enh_within'].mean():.3f}, "
              f"mean excess = {cometary['excess_enh'].mean():.3f}")
        print(f"   Non-cometary (N={len(non_cometary)}):    mean within-season enh = {non_cometary['uap_enh_within'].mean():.3f}, "
              f"mean excess = {non_cometary['excess_enh'].mean():.3f}")

# ============================================================
# Figures
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Radio Meteor (RMOB) × NUFORC UAP Report Correlation\n'
             f'Daily data, {df_good.index.min().date()} to {df_good.index.max().date()}, '
             f'N={len(df_good)} days',
             fontsize=14, fontweight='bold')

# (a) Time series: normalized activity index
ax = axes[0, 0]
weekly_activity = df_good['activity_index'].resample('W').mean()
ax.plot(weekly_activity.index, weekly_activity.values, 'C0-', linewidth=0.8, alpha=0.7)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_ylabel('Activity Index\n(1.0 = baseline)')
ax.set_title('(a) RMOB Weekly Meteor Activity (normalized)')
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

# (d) Scatter: daily correlation (activity_index)
ax = axes[1, 0]
ax.scatter(df_good['activity_index'], df_good['uap_count'],
          s=3, alpha=0.15, c='C0')
ax.set_xlabel('Daily Activity Index (normalized)')
ax.set_ylabel('Daily UAP Reports')
ax.set_title(f'(d) Daily Scatter (r={r_raw:+.3f}, p={p_raw:.1e})')
# Add trend line
z = np.polyfit(df_good['activity_index'], df_good['uap_count'], 1)
x_fit = np.linspace(df_good['activity_index'].min(), df_good['activity_index'].max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2, alpha=0.7)

# (e) Monthly scatter (activity_index)
ax = axes[1, 1]
ax.scatter(monthly['activity_index'], monthly['uap_count'],
          s=30, alpha=0.5, c='C1', edgecolors='k', linewidth=0.3)
ax.set_xlabel('Monthly Mean Activity Index')
ax.set_ylabel('Monthly UAP Report Count')
ax.set_title(f'(e) Monthly Scatter (r={r_monthly:+.3f}, p={p_monthly:.1e})')
z = np.polyfit(monthly['activity_index'], monthly['uap_count'], 1)
x_fit = np.linspace(monthly['activity_index'].min(), monthly['activity_index'].max(), 100)
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
monthly_season_act = df_good.groupby('month')['activity_index'].mean()
monthly_season_uap = df_good.groupby('month')['uap_count'].mean()
ax2 = ax.twinx()
ax.bar(monthly_season_act.index - 0.2, monthly_season_act.values, 0.4,
      color='C0', alpha=0.6, label='Activity index')
ax2.bar(monthly_season_uap.index + 0.2, monthly_season_uap.values, 0.4,
       color='C3', alpha=0.6, label='UAP reports')
ax.set_xlabel('Month')
ax.set_ylabel('Mean Activity Index', color='C0')
ax2.set_ylabel('Mean UAP Reports/day', color='C3')
ax.set_title('(h) Seasonal Cycles')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# (i) Shower enhancement
ax = axes[2, 2]
shower_names = []
shower_enh = []
shower_act_enh = []
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
        shower_act_enh.append(in_shower['activity_index'].mean() / max(out_shower['activity_index'].mean(), 0.01))

x_pos = np.arange(len(shower_names))
ax.bar(x_pos - 0.2, shower_act_enh, 0.4, color='C0', alpha=0.7, label='Activity enhancement')
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
# Figure 2: Composition-dependent analysis
# ============================================================
if len(shower_df) >= 4:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Meteor Shower Composition & UAP Enhancement\n'
                 'Testing: entry velocity and parent body type predict nanoparticle production',
                 fontsize=13, fontweight='bold')

    # (a) UAP enhancement vs entry velocity
    ax = axes[0, 0]
    for _, row in shower_df.iterrows():
        color = 'C0' if row['parent_type'] == 'comet' else ('C2' if row['parent_type'] == 'extinct_comet' else 'C3')
        marker = 'o' if row['parent_type'] == 'comet' else ('s' if row['parent_type'] == 'extinct_comet' else 'D')
        ax.scatter(row['v_entry'], row['uap_enh_raw'], c=color, marker=marker,
                  s=max(40, row['zhr']), edgecolors='k', linewidth=0.5, zorder=5)
        ax.annotate(row['name'], (row['v_entry'], row['uap_enh_raw']),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    # Add trend line
    z = np.polyfit(shower_df['v_entry'], shower_df['uap_enh_raw'], 1)
    x_fit = np.linspace(30, 75, 50)
    ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=1.5, alpha=0.5)
    r_v = stats.pearsonr(shower_df['v_entry'], shower_df['uap_enh_raw'])
    ax.set_xlabel('Entry Velocity (km/s)')
    ax.set_ylabel('UAP Enhancement Factor')
    ax.set_title(f'(a) Entry Velocity vs UAP Enhancement\n(r={r_v[0]:+.3f}, p={r_v[1]:.3f})')
    # Legend for parent types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=8, label='Cometary'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='C2', markersize=8, label='Extinct comet'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='C3', markersize=8, label='Asteroidal'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # Annotate velocity regimes
    ax.axvspan(30, 40, alpha=0.05, color='red')
    ax.axvspan(55, 75, alpha=0.05, color='blue')
    ax.text(34, ax.get_ylim()[0] + 0.02, 'Melt\nspraying', fontsize=7, color='red', ha='center', style='italic')
    ax.text(63, ax.get_ylim()[0] + 0.02, 'Complete\nvaporization', fontsize=7, color='blue', ha='center', style='italic')

    # (b) Within-season UAP enhancement vs entry velocity
    ax = axes[0, 1]
    for _, row in shower_df.iterrows():
        color = 'C0' if row['parent_type'] == 'comet' else ('C2' if row['parent_type'] == 'extinct_comet' else 'C3')
        marker = 'o' if row['parent_type'] == 'comet' else ('s' if row['parent_type'] == 'extinct_comet' else 'D')
        edge = 'red' if row['p_within'] < 0.05 else 'k'
        lw = 1.5 if row['p_within'] < 0.05 else 0.5
        ax.scatter(row['v_entry'], row['uap_enh_within'], c=color, marker=marker,
                  s=max(40, row['zhr']), edgecolors=edge, linewidth=lw, zorder=5)
        ax.annotate(row['name'], (row['v_entry'], row['uap_enh_within']),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    z_ws = np.polyfit(shower_df['v_entry'], shower_df['uap_enh_within'], 1)
    ax.plot(x_fit, np.polyval(z_ws, x_fit), 'r-', linewidth=1.5, alpha=0.5)
    r_v_ws = stats.pearsonr(shower_df['v_entry'], shower_df['uap_enh_within'])
    ax.set_xlabel('Entry Velocity (km/s)')
    ax.set_ylabel('Within-Season UAP Enhancement')
    ax.set_title(f'(b) Within-season control (±30d baseline)\n(r={r_v_ws[0]:+.3f}, p={r_v_ws[1]:.3f})')

    # (c) Bar chart: raw vs within-season enhancement, sorted by velocity
    ax = axes[1, 0]
    sorted_df = shower_df.sort_values('v_entry', ascending=False)
    x_pos = np.arange(len(sorted_df))
    bar_colors = ['C0' if t == 'comet' else ('C2' if t == 'extinct_comet' else 'C3')
                  for t in sorted_df['parent_type']]

    bars1 = ax.bar(x_pos - 0.2, sorted_df['uap_enh_raw'], 0.35,
                   color=bar_colors, alpha=0.6, edgecolor='k', linewidth=0.3, label='Raw (global)')
    bars2 = ax.bar(x_pos + 0.2, sorted_df['uap_enh_within'], 0.35,
                   color=bar_colors, alpha=1.0, edgecolor='k', linewidth=0.3, label='Within-season (±30d)')
    # Mark significant within-season results
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        if row['p_within'] < 0.05:
            ax.text(i + 0.2, row['uap_enh_within'] + 0.01, '*', fontsize=14, ha='center', fontweight='bold', color='red')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    labels = [f"{row['name']}\n({row['v_entry']:.0f} km/s)" for _, row in sorted_df.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('UAP Enhancement Factor')
    ax.set_title('(c) Showers Sorted by Entry Velocity\n(fastest → slowest, * = p<0.05 within-season)')
    ax.legend(fontsize=8)

    # (d) Ablation physics schematic: velocity → particle size → plasma formation
    ax = axes[1, 1]
    # Plot excess enhancement (UAP/activity) vs kinetic energy proxy (v²)
    v_sq = shower_df['v_entry']**2
    ax.scatter(v_sq / 1000, shower_df['excess_enh'],
              s=100, c=['C0' if t == 'comet' else ('C2' if t == 'extinct_comet' else 'C3')
                        for t in shower_df['parent_type']],
              edgecolors='k', linewidth=0.5, zorder=5)
    for _, row in shower_df.iterrows():
        ax.annotate(row['name'], (row['v_entry']**2 / 1000, row['excess_enh']),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    z_ke = np.polyfit(v_sq / 1000, shower_df['excess_enh'], 1)
    x_ke = np.linspace(v_sq.min() / 1000, v_sq.max() / 1000, 50)
    ax.plot(x_ke, np.polyval(z_ke, x_ke), 'r-', linewidth=1.5, alpha=0.5)
    r_ke = stats.pearsonr(v_sq, shower_df['excess_enh'])
    ax.set_xlabel('Specific Kinetic Energy (v²/1000, km²/s²)')
    ax.set_ylabel('Excess UAP Enhancement\n(UAP enh / activity enh)')
    ax.set_title(f'(d) Kinetic Energy vs Excess UAP\n(r={r_ke[0]:+.3f}, p={r_ke[1]:.3f})')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shower_composition_analysis.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'shower_composition_analysis.pdf', bbox_inches='tight')
    plt.close()
    print("Figure saved: shower_composition_analysis.png/pdf")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"RMOB data: {len(all_records)} observer-months, {len(df_rmob)} daily records")
print(f"  Normalization: per-observer monthly median, IQR outlier rejection")
print(f"  Mean activity index: {df_rmob['activity_index'].mean():.3f} (expected ~1.0)")
print(f"NUFORC data: {len(nuforc)} reports")
print(f"Merged: {len(df_good)} days with ≥5 RMOB observers")
print(f"\nKey results (using normalized activity_index):")
print(f"  Raw daily:        r={r_raw:+.4f} (p={p_raw:.2e})")
print(f"  Deseasonalized:   r={r_ds:+.4f} (p={p_ds:.2e})")
print(f"  Monthly:          r={r_monthly:+.4f} (p={p_monthly:.2e})")
if len(sig_years) > 0:
    print(f"  Significant years: {len(sig_years)}/{len(yearly_df)} with mean r={sig_years['r'].mean():+.3f}")
print(f"  Best lag:          {int(best_lag['lag'])} days (r={best_lag['r']:+.4f})")
print(f"\nFigures saved to {OUTPUT_DIR}")
print("Done!")
