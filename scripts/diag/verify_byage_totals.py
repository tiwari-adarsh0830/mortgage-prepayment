"""
Validate the age-keyed aggregation against the existing baseline.

Summing UPB over all age buckets within each (coupon_bucket, yyyymm) cell must
reproduce outputs/realized_cpr_by_coupon_v6_upb.csv. If it does, the age split
is a pure partition and nothing was lost or double-counted. If it does not,
the re-run is wrong and must not feed the S-curve.

Run this BEFORE using the byage file for anything.
"""
import numpy as np, pandas as pd

base = pd.read_csv('outputs/realized_cpr_by_coupon_v6_upb.csv')
new  = pd.read_csv('outputs/realized_cpr_by_coupon_v6_upb_byage.csv')

agg = (new.groupby(['coupon_bucket', 'yyyymm'])
          .agg(upb_atrisk=('upb_atrisk', 'sum'),
               upb_prepay=('upb_prepay', 'sum'),
               n_atrisk=('n_atrisk', 'sum'),
               n_prepay=('n_prepay', 'sum'))
          .reset_index())

m = base.merge(agg, on=['coupon_bucket', 'yyyymm'],
               suffixes=('_base', '_new'), how='outer', indicator=True)

print("=== cell coverage ===")
print(m['_merge'].value_counts().to_string())

both = m[m['_merge'] == 'both'].copy()
print("\n=== totals reconciliation (cells present in both) ===")
for c in ['upb_atrisk', 'upb_prepay', 'n_atrisk', 'n_prepay']:
    a, b = both[c + '_base'], both[c + '_new']
    denom = a.abs().replace(0, np.nan)
    rel = ((b - a).abs() / denom).max()
    print("  %-12s max abs diff %18.2f   max rel diff %.3e" %
          (c, (b - a).abs().max(), rel if pd.notna(rel) else 0.0))

bad = both[(both['upb_atrisk_base'] - both['upb_atrisk_new']).abs() > 1.0]
print("\ncells with at-risk UPB off by more than 1.0: %d" % len(bad))
if len(bad):
    print(bad[['coupon_bucket', 'yyyymm',
               'upb_atrisk_base', 'upb_atrisk_new']].head(20).to_string(index=False))

print("\n=== seasoned share, coupons 2.5-6.5 ===")
t = new[new['implied_mbs_coupon'].between(2.5, 6.5)]
s = t.groupby(t['age_group'] >= 60)['upb_atrisk'].sum()
tot = s.sum()
for k, v in s.items():
    print("  age>60mo = %-5s : %6.2f%% of at-risk UPB" % (k, 100 * v / tot))
print("\n  (if the seasoned share is very small, a seasoned-only S-curve fit")
print("   will be thin and the floor estimate correspondingly noisy)")
