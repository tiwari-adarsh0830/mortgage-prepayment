"""
Plot forecast vs realized CPR — v2, starting Jan 2020 only.
Avoids contaminated 2018-2019 realized CPR data (cross-file UPB=0 bug).
The 2020-2025 portion is correct and covers the full rate cycle:
  - Premium market: 2020-2021 (refi boom)
  - Discount market: 2022-2025 (rate rise + burnout)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA = "/scratch/at7095/mortgage_prepayment/outputs/forecast_vs_realized_cpr.csv"
OUT  = "/scratch/at7095/mortgage_prepayment/outputs/forecast_vs_realized_cpr_2020.png"
PMMS_PATH = "/scratch/at7095/mortgage_prepayment/data/pmms_monthly.csv"

df = pd.read_csv(DATA)
df['date'] = pd.to_datetime(df['date'])
# Filter to 2020 onwards (clean data only)
df = df[df['date'] >= '2020-01-01']

pmms_df = pd.read_csv(PMMS_PATH)
def parse_period(p):
    s = str(int(p))
    if len(s)==5: return pd.Timestamp(year=int(s[1:]),month=int(s[0]),day=1)
    elif len(s)==6: return pd.Timestamp(year=int(s[2:]),month=int(s[:2]),day=1)
    return pd.NaT
pmms_df['date'] = pmms_df['reporting_period'].apply(parse_period)
pmms_df = pmms_df.dropna(subset=['date']).sort_values('date')

COUPONS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
WAC = 3.5

fig, axes = plt.subplots(3, 3, figsize=(16, 11))
fig.suptitle('Hazard Model Forecast CPR vs Realized CPR by Coupon\n'
             'FNCL TBA Jan 2020 – Sep 2025 | 21 Fannie Mae Vintages',
             fontsize=14, fontweight='bold', y=0.98)

for ax, coupon in zip(axes.flatten(), COUPONS):
    sub = df[df['coupon'] == coupon].sort_values('date')
    if sub.empty:
        ax.set_visible(False)
        continue

    ax.plot(sub['date'], sub['realized_cpr']*100,
            color='#2166AC', lw=2.0, label='Realized CPR', alpha=0.9)
    ax.plot(sub['date'], sub['forecast_cpr']*100,
            color='#D6604D', lw=2.0, label='Forecast CPR (ML hazard model)',
            linestyle='--', alpha=0.9)

    # Shade PM period
    pm = pmms_df[(pmms_df['date'] >= sub['date'].min()) &
                 (pmms_df['date'] <= sub['date'].max()) &
                 (pmms_df['rate_30yr'] < WAC)]['date']
    if len(pm) > 0:
        ax.axvspan(pm.min(), pm.max(), alpha=0.10, color='green',
                   label='Premium market\n(PMMS < 3.5%)')

    corr = sub['forecast_cpr'].corr(sub['realized_cpr'])
    ax.text(0.97, 0.95, f'r = {corr:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='#444444',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_title(f'FNCL {coupon:.1f}%', fontweight='bold', fontsize=11)
    ax.set_ylabel('CPR (%)', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0, 0].legend(fontsize=7.5, loc='upper right', framealpha=0.9)
fig.text(0.5, 0.01,
         'Green shading = Premium market (PMMS < 3.5% WAC proxy) | '
         'r = Pearson correlation forecast vs realized | '
         'Blue = Fannie Mae pool-level realized | Red dashed = ML hazard model forecast',
         ha='center', fontsize=8.0, color='#444444')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}")

# Summary for email
print("\n=== 2020-2025 Summary ===")
print(f"{'Coupon':>8} {'Forecast':>12} {'Realized':>12} {'Corr':>6}")
print("-"*44)
for coupon in COUPONS:
    sub = df[df['coupon']==coupon]
    if sub.empty: continue
    print(f"{coupon:>8.1f} {sub['forecast_cpr'].mean()*100:>11.2f}% "
          f"{sub['realized_cpr'].mean()*100:>11.2f}% "
          f"{sub['forecast_cpr'].corr(sub['realized_cpr']):>6.2f}")

print("\nPeak 2020-21 (premium regime):")
peak = df[df['date'].between('2020-06-01','2021-12-31')]
for c in [3.5, 4.0, 4.5, 5.0]:
    s = peak[peak['coupon']==c]
    if s.empty: continue
    print(f"  {c:.1f}%: forecast {s['forecast_cpr'].mean()*100:.1f}% "
          f"vs realized {s['realized_cpr'].mean()*100:.1f}%")

print("\nTrough 2022-23 (discount regime):")
trough = df[df['date'].between('2022-06-01','2023-12-31')]
for c in [3.5, 4.5, 6.0, 6.5]:
    s = trough[trough['coupon']==c]
    if s.empty: continue
    print(f"  {c:.1f}%: forecast {s['forecast_cpr'].mean()*100:.1f}% "
          f"vs realized {s['realized_cpr'].mean()*100:.1f}%")
