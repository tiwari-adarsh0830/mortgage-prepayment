import numpy as np, pandas as pd
t = pd.read_excel('data/treasury_yields_clean.xlsx', sheet_name='Treasury_Yields', header=1)
t.columns = [str(c).strip() for c in t.columns]
t['Date'] = pd.to_datetime(t['Date'])
y10c = [c for c in t.columns if '10yr' in c.lower()][0]
d = pd.read_csv('data/treasury_yields.csv', index_col=0, parse_dates=True).sort_index()
pm = pd.read_csv('data/pmms_monthly.csv')
def parse(x):
    s = str(int(x))
    if len(s) == 5: return pd.Timestamp(year=int(s[1:]), month=int(s[0]), day=1)
    if len(s) == 6: return pd.Timestamp(year=int(s[2:]), month=int(s[:2]), day=1)
    return pd.NaT
pm['date'] = pm['reporting_period'].apply(parse)
pms = pm.dropna(subset=['date']).set_index('date')['rate_30yr']

# monthly 10yr from the daily H.15 file, month-end
d10 = d['10yr'].resample('MS').last()
j = pd.DataFrame({'pmms': pms, 'y10': d10}).dropna()
j['sp'] = j['pmms'] - j['y10']
print("PMMS - 10yr spread (H.15 daily, month-end), bp:")
for lo, lab in [('1971-01','full history'), ('2000-01','2000+'), ('2010-01','2010+'),
                ('2013-01','2013+'), ('2018-01','2018+ (panel)'), ('2022-01','2022+')]:
    s = j[j.index >= lo]['sp']
    if len(s): print("  %-18s n=%-4d mean=%3.0fbp  median=%3.0fbp" % (lab, len(s), s.mean()*100, s.median()*100))
print("\nby decade:")
print((j.groupby(j.index.year // 10 * 10)['sp'].mean() * 100).round(0).to_string())
print("\nclosest window to 189bp:")
best = None
for start in pd.date_range('1971-01-01', '2020-01-01', freq='YS'):
    s = j[j.index >= start]['sp']
    if len(s) < 24: continue
    err = abs(s.mean() * 100 - 189)
    if best is None or err < best[0]: best = (err, start.year, s.mean() * 100, len(s))
print("  from %d onward: mean %.0fbp (n=%d), off by %.0fbp" % (best[1], best[2], best[3], best[0]))
