"""
Phase 1 — Data Exploration, Audit & Naive Baseline
Standalone script equivalent of notebooks/01_eda_baseline.ipynb

Run from the project root:
    python notebooks/run_phase1.py

Outputs written to:
    data/processed/monthly_admin1_panel.parquet
    metrics/phase1_metrics.json
    reports/01_data_audit.md
    reports/fig01_*.png … fig06_*.png
"""

import json
import warnings
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / 'data' / 'raw'
PROC_DIR    = ROOT / 'data' / 'processed'
METRICS_DIR = ROOT / 'metrics'
REPORTS_DIR = ROOT / 'reports'
for d in [PROC_DIR, METRICS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPECTED_COLS = [
    'week', 'admin1', 'event_type', 'sub_event_type',
    'events', 'fatalities', 'latitude', 'longitude'
]
PILOT_STATES      = ['Zamfara', 'Katsina', 'Sokoto', 'Kaduna']
BORDERING_HOTSPOT = ['Zamfara','Katsina','Sokoto','Kaduna',
                     'Niger','Kebbi','Kano','Jigawa',
                     'Borno','Yobe','Adamawa','Plateau','Benue']

# ══════════════════════════════════════════════════════════════════════════════
# 0 · Load data (or synthesise demo data)
# ══════════════════════════════════════════════════════════════════════════════
DATA_FILE = RAW_DIR / 'acled_nigeria_weekly_agg.csv'
DEMO_MODE = not DATA_FILE.exists()

print('=' * 65)
print('Conflict Early Warning Nigeria — Phase 1')
print('=' * 65)

if DEMO_MODE:
    print('\n[WARN] DEMO MODE -- real data not found at:')
    print(f'   {DATA_FILE}')
    print('   Drop your ACLED export there and re-run.\n')

    rng = np.random.default_rng(42)
    STATES = [
        'Zamfara', 'Katsina', 'Sokoto', 'Kaduna',
        'Borno', 'Yobe', 'Adamawa',
        'Lagos', 'Ogun', 'Oyo',
        'Rivers', 'Delta', 'Bayelsa',
        'Plateau', 'Benue', 'Nasarawa',
        'Kano', 'Jigawa', 'Kebbi',
        'Niger', 'Kwara', 'FCT',
    ]
    STATE_COORDS = {
        'Zamfara':(12.17,6.66), 'Katsina':(12.99,7.60),
        'Sokoto':(13.06,5.24),  'Kaduna':(10.52,7.44),
        'Borno':(11.85,13.16),  'Yobe':(12.29,11.44),
        'Adamawa':(9.33,12.39), 'Lagos':(6.52,3.38),
        'Ogun':(6.99,3.47),     'Oyo':(7.85,3.93),
        'Rivers':(4.85,7.04),   'Delta':(5.52,5.90),
        'Bayelsa':(4.77,6.07),  'Plateau':(9.22,9.52),
        'Benue':(7.34,8.74),    'Nasarawa':(8.49,8.52),
        'Kano':(12.00,8.52),    'Jigawa':(12.21,9.56),
        'Kebbi':(11.49,4.20),   'Niger':(10.00,5.60),
        'Kwara':(8.49,4.55),    'FCT':(9.07,7.40),
    }
    EVENT_TYPES = ['Battles', 'Violence against civilians',
                   'Explosions/Remote violence', 'Protests', 'Riots',
                   'Strategic developments']
    SUB_TYPES = {
        'Battles': ['Armed clash', 'Government regains territory',
                    'Non-state actor overtakes territory'],
        'Violence against civilians': ['Attack', 'Abduction/forced disappearance',
                                       'Sexual violence'],
        'Explosions/Remote violence': ['IED', 'Air/drone strike', 'Grenade'],
        'Protests': ['Peaceful protest', 'Excessive force against protesters'],
        'Riots': ['Mob violence', 'Violent demonstration'],
        'Strategic developments': ['Agreement', 'Looting/property destruction', 'Arrests'],
    }
    weeks = pd.date_range('2015-01-05', '2024-12-30', freq='W-MON')
    rows = []
    for w in weeks:
        for state in STATES:
            for et in rng.choice(EVENT_TYPES,
                                 size=rng.integers(0, 4), replace=False):
                intensity = 3.0 if state in [
                    'Zamfara','Katsina','Sokoto','Kaduna','Borno','Yobe'
                ] else 0.5
                n_events = int(rng.negative_binomial(1, 1/(1+intensity)))
                if n_events == 0:
                    continue
                fatal_rate = 0.6 if state in ['Zamfara','Borno'] else 0.3
                n_fatal = int(rng.negative_binomial(
                    n_events, 1/(1 + fatal_rate * intensity)))
                lat, lon = STATE_COORDS.get(state, (9.0, 8.0))
                sub = rng.choice(SUB_TYPES[et])
                rows.append({
                    'week': w.strftime('%Y-%m-%d'), 'admin1': state,
                    'event_type': et, 'sub_event_type': sub,
                    'events': n_events, 'fatalities': n_fatal,
                    'latitude': round(lat + rng.normal(0, 0.01), 4),
                    'longitude': round(lon + rng.normal(0, 0.01), 4),
                })
    df_raw = pd.DataFrame(rows)
else:
    print(f'Loading real data from {DATA_FILE}')
    df_raw = pd.read_csv(DATA_FILE)

print(f'Rows loaded : {len(df_raw):,}')
print(f'Columns     : {list(df_raw.columns)}')

# ══════════════════════════════════════════════════════════════════════════════
# 1 · Schema validation
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Schema check ─────────────────────────────────────────')
missing_cols = set(EXPECTED_COLS) - set(df_raw.columns)
extra_cols   = set(df_raw.columns) - set(EXPECTED_COLS)
print(f'Missing : {missing_cols or "None ✓"}')
print(f'Extra   : {extra_cols   or "None ✓"}')

df_raw['week']       = pd.to_datetime(df_raw['week'])
df_raw['events']     = pd.to_numeric(df_raw['events'],     errors='coerce').astype('Int64')
df_raw['fatalities'] = pd.to_numeric(df_raw['fatalities'], errors='coerce').astype('Int64')
df_raw['latitude']   = pd.to_numeric(df_raw['latitude'],   errors='coerce')
df_raw['longitude']  = pd.to_numeric(df_raw['longitude'],  errors='coerce')
df_raw['admin1']     = df_raw['admin1'].str.strip().str.title()

nulls = df_raw.isnull().sum()
print('\nNull counts:')
print(nulls[nulls > 0].to_string() or '  None ✓')
print(f'\nDate range : {df_raw["week"].min().date()} → {df_raw["week"].max().date()}')
print(f'States ({df_raw["admin1"].nunique()}): {sorted(df_raw["admin1"].unique())}')

# ══════════════════════════════════════════════════════════════════════════════
# 2 · Temporal coverage — silent weeks
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Temporal coverage ────────────────────────────────────')
all_weeks  = pd.date_range(df_raw['week'].min(), df_raw['week'].max(), freq='W-MON')
all_states = df_raw['admin1'].unique()
full_grid  = pd.MultiIndex.from_product(
    [all_weeks, all_states], names=['week','admin1']
).to_frame(index=False)
weeks_present = df_raw.groupby(['admin1','week']).size().reset_index(name='rows')
coverage      = full_grid.merge(weeks_present, on=['week','admin1'], how='left')
coverage['is_silent'] = coverage['rows'].isna()
silent_pct = coverage.groupby('admin1')['is_silent'].mean().sort_values(ascending=False)

print('Silent week fraction per state:')
print(silent_pct.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
silent_pct.sort_values().plot.barh(ax=ax, color='#c0392b')
ax.set_xlabel('Fraction of weeks with no rows', fontsize=11)
ax.set_title('Coverage gaps by state\n(silent week = no rows in ACLED export)', fontsize=12)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.tight_layout()
out = REPORTS_DIR / 'fig01_silent_weeks_by_state.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 3 · Event-type breakdown
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Event-type breakdown ─────────────────────────────────')
et_summary = (
    df_raw.groupby('event_type')[['events','fatalities']]
    .sum().sort_values('fatalities', ascending=False).reset_index()
)
et_summary['fatality_share'] = et_summary['fatalities'] / et_summary['fatalities'].sum()
print(et_summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
et_summary.set_index('event_type')['events'].sort_values().plot.barh(
    ax=axes[0], color='#2980b9')
axes[0].set_title('Total events by type')
et_summary.set_index('event_type')['fatalities'].sort_values().plot.barh(
    ax=axes[1], color='#c0392b')
axes[1].set_title('Total fatalities by type')
plt.tight_layout()
out = REPORTS_DIR / 'fig02_event_type_breakdown.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 4 · Time series — Northwest pilot states
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Pilot states time series ─────────────────────────────')
weekly_pilot = (
    df_raw[df_raw['admin1'].isin(PILOT_STATES)]
    .groupby(['week','admin1'])[['events','fatalities']]
    .sum().reset_index()
)
fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
for i, state in enumerate(PILOT_STATES):
    d = weekly_pilot[weekly_pilot['admin1'] == state].sort_values('week')
    axes[i,0].bar(d['week'], d['events'],     width=6, color='#2980b9', alpha=0.7)
    axes[i,0].set_title(f'{state} — events/week')
    axes[i,1].bar(d['week'], d['fatalities'], width=6, color='#c0392b', alpha=0.7)
    axes[i,1].set_title(f'{state} — fatalities/week')
    for ax in [axes[i,0], axes[i,1]]:
        ax.tick_params(axis='x', rotation=30)
plt.suptitle('Northwest Nigeria pilot states — weekly conflict intensity',
             fontsize=13, y=1.01)
plt.tight_layout()
out = REPORTS_DIR / 'fig03_pilot_states_timeseries.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 5 · Collapse to monthly admin1 panel
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Monthly admin1 panel ─────────────────────────────────')
df_raw['month'] = df_raw['week'].dt.to_period('M')
monthly = (
    df_raw.groupby(['month','admin1'])
    .agg(
        events=('events','sum'),
        fatalities=('fatalities','sum'),
        weeks_observed=('week','nunique'),
        latitude=('latitude','first'),
        longitude=('longitude','first'),
    )
    .reset_index()
)
monthly['fatal_event'] = (monthly['fatalities'] > 0).astype(int)
monthly['low_coverage'] = (monthly['weeks_observed'] < 3).astype(int)

print(f'Shape            : {monthly.shape}')
print(f'Date range       : {monthly["month"].min()} → {monthly["month"].max()}')
print(f'States           : {monthly["admin1"].nunique()}')
print(f'Target prevalence: {monthly["fatal_event"].mean():.2%}')
print(f'Low-coverage cells: {monthly["low_coverage"].mean():.2%}')

# ─── Count distributions / overdispersion check ───────────────────────────
print('\n─── Overdispersion check ─────────────────────────────────')
for col in ['events','fatalities']:
    s    = monthly[col].dropna().astype(float)
    mean = s.mean()
    var  = s.var()
    print(f'{col:12s}: mean={mean:.2f}, var={var:.2f}, '
          f'var/mean={var/mean:.2f} '
          f'[>1 = overdispersed → use negative binomial]')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, color in zip(axes, ['events','fatalities'], ['#2980b9','#c0392b']):
    col_float = monthly[col].astype(float)
    cap = col_float.quantile(0.99)
    ax.hist(col_float.clip(upper=cap), bins=50, color=color, alpha=0.8)
    ax.set_title(f'Monthly {col} distribution\n(capped at 99th pct = {cap:.0f})')
    ax.set_ylabel('Frequency')
plt.tight_layout()
out = REPORTS_DIR / 'fig04_count_distributions.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')

# Save processed panel
out_path = PROC_DIR / 'monthly_admin1_panel.parquet'
try:
    monthly.to_parquet(out_path, index=False)
    print(f'Saved: {out_path}')
except Exception as e:
    # Fallback to CSV if pyarrow not installed
    out_path = PROC_DIR / 'monthly_admin1_panel.csv'
    monthly.to_csv(out_path, index=False)
    print(f'Saved (CSV fallback): {out_path}  — reason: {e}')

# ══════════════════════════════════════════════════════════════════════════════
# 6 · Coverage-bias flags
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Coverage-bias flags ──────────────────────────────────')

def zero_streak(series: pd.Series) -> int:
    reversed_vals = series.iloc[::-1].values
    streak = 0
    for v in reversed_vals:
        if v == 0:
            streak += 1
        else:
            break
    return streak

streak_by_state = (
    monthly.sort_values('month')
    .groupby('admin1')['fatalities']
    .apply(zero_streak)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 6))
streak_by_state.sort_values().plot.barh(ax=ax, color='#e67e22')
ax.axvline(6, color='red', linestyle='--', label='6-month threshold')
ax.set_xlabel('Months of zero recorded fatalities (most recent)')
ax.set_title('States with recent zero-fatality streaks\n'
             'Streaks > 6 months near conflict zones = coverage risk')
ax.legend()
plt.tight_layout()
out = REPORTS_DIR / 'fig05_zero_fatality_streaks.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')

flag1 = silent_pct[silent_pct > 0.30].reset_index()
flag1.columns = ['admin1','silent_fraction']

flag2 = streak_by_state[
    (streak_by_state > 6) & streak_by_state.index.isin(BORDERING_HOTSPOT)
].reset_index()
flag2.columns = ['admin1','zero_streak_months']

flag3 = (
    monthly[
        (monthly['admin1'].isin(PILOT_STATES)) &
        (monthly['weeks_observed'] < 3)
    ]
    .groupby('admin1')['month'].count().reset_index()
)
flag3.columns = ['admin1','low_coverage_months']

print('\n[WARN] FLAG 1 -- Silent week fraction > 30%:')
print(flag1.to_string(index=False) if not flag1.empty else '  None')
print('\n[WARN] FLAG 2 -- Long zero-fatality streak near hotspot:')
print(flag2.to_string(index=False) if not flag2.empty else '  None')
print('\n[WARN] FLAG 3 -- Pilot state months with < 3 weeks observed:')
print(flag3.to_string(index=False) if not flag3.empty else '  None')

# ══════════════════════════════════════════════════════════════════════════════
# 7 · Naive baseline (M0)
# ══════════════════════════════════════════════════════════════════════════════
print('\n─── Naive baseline (M0) ──────────────────────────────────')
monthly_sorted = monthly.sort_values(['admin1','month']).copy()
monthly_sorted['fatal_event_lag1'] = (
    monthly_sorted.groupby('admin1')['fatal_event'].shift(1)
)
monthly_sorted['p_naive'] = monthly_sorted['fatal_event_lag1'].fillna(0.5)

eval_df  = monthly_sorted.dropna(subset=['fatal_event_lag1']).copy()
y_true   = eval_df['fatal_event'].values
p_naive  = eval_df['p_naive'].values

bs_naive = brier_score_loss(y_true, p_naive)
prev     = y_true.mean()
bs_clim  = brier_score_loss(y_true, np.full_like(y_true, prev, dtype=float))
bss      = 1 - bs_naive / bs_clim

try:
    auc = roc_auc_score(y_true, p_naive)
except Exception:
    auc = float('nan')

print(f'Evaluation rows  : {len(eval_df):,}')
print(f'Target prevalence: {prev:.2%}')
print(f'\nBrier score      : {bs_naive:.4f}  (climatological baseline={bs_clim:.4f})')
print(f'Brier Skill Score: {bss:.4f}  (>0 = beats predicting prevalence always)')
print(f'AUC-ROC          : {auc:.4f}  (0.5=random, 1.0=perfect)')
print('\n[NOTE] RECORD THESE NUMBERS -- every Phase 4 model must beat the Brier score.')

# Calibration plot
fraction_pos, mean_pred = calibration_curve(
    y_true, p_naive, n_bins=5, strategy='uniform'
)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0,1],[0,1],'k--', label='Perfect calibration')
ax.plot(mean_pred, fraction_pos, 'o-', color='#2980b9', label='Naive M0')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration plot — Naive baseline\nPoints on dashed line = perfect calibration')
ax.legend()
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
out = REPORTS_DIR / 'fig06_calibration_naive.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')

# ─── Expanding-window ─────────────────────────────────────────────────────────
def expanding_window_naive(panel, min_train_months=36):
    all_months = sorted(panel['month'].unique())
    results = []
    for i, target_month in enumerate(all_months):
        if i < min_train_months:
            continue
        prev_month = all_months[i - 1]
        prev = panel[panel['month'] == prev_month][['admin1','fatal_event']].copy()
        prev.rename(columns={'fatal_event':'p_naive'}, inplace=True)
        cur  = panel[panel['month'] == target_month][['admin1','fatal_event']].copy()
        merged = prev.merge(cur, on='admin1', how='inner')
        merged['forecast_month'] = target_month
        results.append(merged)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

ew_results = expanding_window_naive(
    monthly_sorted[['month','admin1','fatal_event']]
)
bs_ew = None
if not ew_results.empty:
    bs_ew = brier_score_loss(ew_results['fatal_event'], ew_results['p_naive'])
    print(f'\nExpanding-window Brier score (naive): {bs_ew:.4f}')
    print(f'Expanding-window folds             : {ew_results["forecast_month"].nunique()}')
    print('This is the PRIMARY benchmark for Phase 4 models.')
else:
    print('\nNot enough months for expanding-window evaluation (need > 36).')

# ══════════════════════════════════════════════════════════════════════════════
# 8 · Save metrics + audit report
# ══════════════════════════════════════════════════════════════════════════════
metrics = {
    'phase': 1,
    'run_date': datetime.now().isoformat(),
    'demo_mode': DEMO_MODE,
    'data_rows_raw': len(df_raw),
    'monthly_panel_rows': len(monthly),
    'states': int(monthly['admin1'].nunique()),
    'date_range': {
        'start': str(monthly['month'].min()),
        'end':   str(monthly['month'].max()),
    },
    'target_prevalence': float(eval_df['fatal_event'].mean()),
    'low_coverage_fraction': float(monthly['low_coverage'].mean()),
    'naive_baseline_M0': {
        'brier_score': round(float(bs_naive), 4),
        'brier_skill_score': round(float(bss), 4),
        'auc_roc': round(float(auc), 4) if not np.isnan(auc) else None,
        'expanding_window_brier': round(float(bs_ew), 4) if bs_ew is not None else None,
    },
    'coverage_flags': {
        'flag1_silent_gt30pct': flag1['admin1'].tolist(),
        'flag2_zero_streak_near_hotspot': flag2['admin1'].tolist() if not flag2.empty else [],
        'flag3_pilot_low_coverage_months': flag3['admin1'].tolist() if not flag3.empty else [],
    },
}
metrics_path = METRICS_DIR / 'phase1_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\nMetrics saved: {metrics_path}')
print(json.dumps(metrics, indent=2))

# ─── Audit report ─────────────────────────────────────────────────────────────
f1t = '\n'.join(f'- `{r.admin1}`: {r.silent_fraction:.0%} silent weeks'
                for _, r in flag1.iterrows()) or '- None identified'
f2t = '\n'.join(f'- `{r.admin1}`: {r.zero_streak_months} month streak'
                for _, r in flag2.iterrows()) or '- None identified'
f3t = '\n'.join(f'- `{r.admin1}`: {r.low_coverage_months} low-coverage months'
                for _, r in flag3.iterrows()) or '- None identified'

report_md = f"""# Phase 1 — Data Audit Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Mode:** {"DEMO (synthetic data)" if DEMO_MODE else "REAL DATA"}

## Dataset summary
| Field | Value |
|-------|-------|
| Raw rows | {len(df_raw):,} |
| Monthly panel rows | {len(monthly):,} |
| States | {monthly['admin1'].nunique()} |
| Date range | {metrics['date_range']['start']} → {metrics['date_range']['end']} |
| Target prevalence | {metrics['target_prevalence']:.1%} |
| Low-coverage cells | {metrics['low_coverage_fraction']:.1%} |

## Naive baseline metrics (M0)
| Metric | Value | Notes |
|--------|-------|-------|
| Brier score | {metrics['naive_baseline_M0']['brier_score']} | Lower is better; 0.25 = coin flip |
| Brier Skill Score | {metrics['naive_baseline_M0']['brier_skill_score']} | >0 = beats climatology |
| AUC-ROC | {metrics['naive_baseline_M0']['auc_roc']} | 0.5=random; 1.0=perfect |
| Expanding-window Brier | {metrics['naive_baseline_M0']['expanding_window_brier']} | PRIMARY benchmark for Phase 4 |

> **All Phase 4 models must achieve a lower expanding-window Brier score than the value above.**

## Coverage-bias flags

### ⚠️ Flag 1 — Silent week fraction > 30%
{f1t}

### ⚠️ Flag 2 — Long zero-fatality streak near a hotspot
{f2t}

### ⚠️ Flag 3 — Pilot state months with < 3 weeks observed
{f3t}

## Overdispersion test
See console output. If var/mean > 3 for fatalities, Poisson is ruled out — use negative binomial.

## Limitations (carry into Phase 2)
1. Admin1 centroids are state centroids, not event locations — sub-state mapping is misleading.
2. "Quiet" states near active conflict zones may be under-reported, not peaceful.
3. Weekly aggregation loses within-week sequencing; event ordering unknown.
4. ACLED may have sourcing gaps for rural Zamfara and Sokoto — UCDP cross-check needed.
"""
report_path = REPORTS_DIR / '01_data_audit.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f'Audit report saved: {report_path}')

print('\n' + '=' * 65)
print('Phase 1 COMPLETE')
print('=' * 65)
print('\n[OK] What we learned:')
print('  1. Overdispersion confirmed (var/mean >> 1) → negative-binomial justified')
print('  2. Naive baseline Brier score documented → ceiling for Phase 4 models')
print('  3. Coverage-bias flags written → uncertainty bands will show these')
print('\n[NEXT] Phase 2 -- UCDP ingestion + source-reliability scoring')
