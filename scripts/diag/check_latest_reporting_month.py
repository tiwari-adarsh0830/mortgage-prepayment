"""
One-off check: latest monthly_reporting_period (converted MMYYYY->YYYYMM)
across the raw vintage files, both pre-2013 and modern eras. Needed to
confirm the raw data extends through Dec 2025 (so a cutoff_year=2024 build,
forecasting CY2025, has rows to score against). Not a permanent test.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prepare_sequences_multiobs_zbc import mmyyyy_to_yyyymm

BASE = '/scratch/at7095/mortgage_prepayment'


def latest_ym(path):
    col = pd.read_csv(path, sep='|', header=None, usecols=[2], names=['m'], dtype=str)
    col['m'] = pd.to_numeric(col['m'], errors='coerce')
    col = col.dropna()
    return int(col['m'].astype(int).apply(mmyyyy_to_yyyymm).max())


modern_files = sorted(glob.glob(os.path.join(BASE, 'data/raw/*.csv')))
pre2013_files = sorted(glob.glob(os.path.join(BASE, 'data_pre2013_raw/*.csv')))

print(f'{len(modern_files)} modern-era files, {len(pre2013_files)} pre-2013 files', flush=True)

overall_max = 0
for f in modern_files:
    ym = latest_ym(f)
    overall_max = max(overall_max, ym)
    print(f'{os.path.basename(f)}: max yyyymm={ym}', flush=True)

for f in pre2013_files[-3:]:
    ym = latest_ym(f)
    overall_max = max(overall_max, ym)
    print(f'{os.path.basename(f)} (pre2013 sample): max yyyymm={ym}', flush=True)

print(f'\nOVERALL latest reporting month seen: {overall_max}', flush=True)
