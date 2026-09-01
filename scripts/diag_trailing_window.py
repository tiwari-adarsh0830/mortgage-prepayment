import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import prepare_sequences_trailing_zbc as T

pm = T.load_pmms(); zh = T.load_zhvi()
df = T.load_vintage_filtered("2015Q1", pm, zh, 202012, keep_ids=None, sample_frac=0.01)
print("rows", len(df), "loans", df.loan_id.nunique(), flush=True)

last_true = df.groupby("loan_id")["yyyymm"].max()
d2 = df.copy()
d2["_rev"] = d2.groupby("loan_id").cumcount(ascending=False)
kept = d2[d2["_rev"] < T.MAX_SEQ_LEN]
last_kept  = kept.groupby("loan_id")["yyyymm"].max()
first_kept = kept.groupby("loan_id")["yyyymm"].min()

print("TRAILING CHECK -- last kept row == loan max month for ALL loans:",
      bool((last_kept == last_true).all()), flush=True)
sz = kept.groupby("loan_id").size()
print("kept per loan: min %d max %d" % (sz.min(), sz.max()), flush=True)
print(pd.DataFrame({"first": first_kept, "last": last_kept}).head(5), flush=True)
age0 = kept.groupby("loan_id")["loan_age_months"].first()
print("loan_age at FIRST kept row -- median %.0f min %.0f max %.0f"
      % (age0.median(), age0.min(), age0.max()), flush=True)
