"""
Rebuild zhvi_zip3.csv to include 2015-2026 (was previously 2019+ only).
This fixes the prepare_sequences crash where 2018-vintage loans had
NaN current_ltv because their origination ZHVI was missing.

Format must match existing exactly:
  columns: zip3 (int), reporting_period (int MYYYY), zhvi (float)
  zip3 = first 3 digits of RegionName (zip code)
  reporting_period = M*10000 + YYYY  e.g. Jan 2019 = 12019
  zhvi = mean across all zips sharing a zip3 prefix
"""
import pandas as pd
import numpy as np

RAW = "/scratch/at7095/mortgage_prepayment/data/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
OUT = "/scratch/at7095/mortgage_prepayment/data/zhvi_zip3.csv"

print("Loading raw Zillow file...", flush=True)
df = pd.read_csv(RAW)

# Identify date columns (YYYY-MM-DD format)
date_cols = [c for c in df.columns if c[:4].isdigit() and '-' in c]
# Keep only 2015 onwards (covers all loan origination dates we'll see)
date_cols = [c for c in date_cols if int(c[:4]) >= 2015]
print(f"Date columns 2015+: {len(date_cols)} ({date_cols[0]} to {date_cols[-1]})", flush=True)

# RegionName is the zip code
df['zip3'] = df['RegionName'].astype(str).str.zfill(5).str[:3].astype(int)

# Melt to long format
long = df.melt(
    id_vars=['zip3'],
    value_vars=date_cols,
    var_name='date_str',
    value_name='zhvi'
)
long = long.dropna(subset=['zhvi'])

# Convert date_str (YYYY-MM-DD) to reporting_period (MYYYY int)
dt = pd.to_datetime(long['date_str'])
long['reporting_period'] = dt.dt.month * 10000 + dt.dt.year

# Aggregate to zip3 level (mean across zips sharing prefix)
agg = long.groupby(['zip3', 'reporting_period'], as_index=False)['zhvi'].mean()

# Consistency check vs existing file (2019+ values must match)
import os
if os.path.exists(OUT):
    old_df = pd.read_csv(OUT)
    merged = old_df.merge(agg, on=['zip3','reporting_period'], suffixes=('_old','_new'))
    if len(merged) > 0:
        diff = (merged['zhvi_old'] - merged['zhvi_new']).abs()
        print(f"\nConsistency check vs existing zhvi_zip3.csv:")
        print(f"  Overlapping rows: {len(merged):,}")
        print(f"  Max abs diff: {diff.max():.4f}")
        print(f"  Mean abs diff: {diff.mean():.4f}")
        if diff.max() < 1.0:
            print("  MATCH - safe to overwrite (2019+ values unchanged)")
        else:
            print("  WARNING - values differ! Review before trusting.")

# Sort and save
agg = agg.sort_values(['zip3', 'reporting_period']).reset_index(drop=True)
# Back up old file first
if os.path.exists(OUT):
    os.rename(OUT, OUT + '.bak')
    print(f"\nBacked up old file to {OUT}.bak")
agg.to_csv(OUT, index=False)

print(f"Saved: {OUT} ({len(agg):,} rows)", flush=True)
print(f"Period range: {agg['reporting_period'].min()} to {agg['reporting_period'].max()}")
print(f"Has 2018? {12018 in agg['reporting_period'].values}")
print(f"Has 2015? {12015 in agg['reporting_period'].values}")
print(f"zip3 count: {agg['zip3'].nunique()}")
print("\nSample:")
print(agg.head(5).to_string())
