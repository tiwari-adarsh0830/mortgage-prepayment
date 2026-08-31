import numpy as np, pickle, sys
sys.path.insert(0, 'scripts')
from prepare_sequences_rolling_zbc import load_pmms, load_zhvi
from forecast_rolling_cpr import read_coupon_and_realized, mmyyyy_to_yyyymm, yyyymm_to_mmyyyy

d = 'data/sequences_rolling/cutoff_2022_zbc'
seq  = np.load(f'{d}/test_seq.npy', mmap_mode='r')
mask = np.load(f'{d}/test_mask.npy', mmap_mode='r')
ids  = np.load(f'{d}/test_loan_ids.npy', allow_pickle=True)
with open(f'{d}/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
mean_, scale_ = scaler.mean_, scaler.scale_

print("Pulling coupon/zip3/origination for test loans...", flush=True)
coupon_map, active_set, prepaid_set, zip3_map, origdate_map = read_coupon_and_realized(2022, set(ids.tolist()))

target_ids = [lid for lid in ids if lid in coupon_map and 2.75 <= coupon_map[lid] < 4.25][:20000]
print(f"sample size: {len(target_ids):,}", flush=True)

id_to_row = {lid: i for i, lid in enumerate(ids)}
rows = [id_to_row[lid] for lid in target_ids]

last_t = mask[rows].sum(1) - 1
ltv_scaled = seq[rows, last_t, 2]
ltv_raw = ltv_scaled * scale_[2] + mean_[2]
age_scaled_cutoff = seq[rows, last_t, 5]
age_raw_cutoff = age_scaled_cutoff * scale_[5] + mean_[5]

pmms = load_pmms()
zhvi_df = load_zhvi()
zhvi_lookup = dict(zip(zip(zhvi_df['zip3'].astype(int), zhvi_df['reporting_period'].astype(int)), zhvi_df['zhvi'].values))

id_to_target_idx = {lid: i for i, lid in enumerate(target_ids)}
age_2023dec_vals, ltv_now_2023dec_vals = [], []
for lid in target_ids:
    z, od = zip3_map.get(lid), origdate_map.get(lid)
    if z is None or od is None:
        continue
    zo = zhvi_lookup.get((int(z), int(od)))
    zn = zhvi_lookup.get((int(z), yyyymm_to_mmyyyy(202312)))
    orig_yyyymm = mmyyyy_to_yyyymm(int(od))
    age = (202312//100 - orig_yyyymm//100)*12 + (202312%100 - orig_yyyymm%100) - 1
    age_2023dec_vals.append(max(age, 0))
    if zo and zn:
        ltv_raw_i = ltv_raw[id_to_target_idx[lid]]
        ltv_now = ltv_raw_i * zo / zn
        ltv_now_2023dec_vals.append(ltv_now)

print(f"\nFROZEN (cutoff, Dec 2022) loan_age_months: "
      f"min={age_raw_cutoff.min():.1f} max={age_raw_cutoff.max():.1f} "
      f"median={np.median(age_raw_cutoff):.1f}", flush=True)
print(f"TIME-VARYING (forecast month Dec 2023) loan_age_months: "
      f"min={min(age_2023dec_vals):.1f} max={max(age_2023dec_vals):.1f} "
      f"median={np.median(age_2023dec_vals):.1f}", flush=True)
if ltv_now_2023dec_vals:
    lv = np.array(ltv_now_2023dec_vals)
    print(f"TIME-VARYING (forecast month Dec 2023) current_ltv: "
          f"min={lv.min():.1f} max={lv.max():.1f} median={np.median(lv):.1f}", flush=True)

print(f"\nTraining data's loan_age_months distribution (raw, sampled last timestep):", flush=True)
train_seq = np.load(f'{d}/train_seq.npy', mmap_mode='r')
train_mask = np.load(f'{d}/train_mask.npy', mmap_mode='r')
samp = np.random.RandomState(1).choice(len(train_seq), 300000, replace=False)
tlt = train_mask[samp].sum(1) - 1
tage_scaled = train_seq[samp, tlt, 5]
tage_raw = tage_scaled * scale_[5] + mean_[5]
print(f"  min={tage_raw.min():.1f} max={tage_raw.max():.1f} "
      f"p99={np.percentile(tage_raw,99):.1f} median={np.median(tage_raw):.1f}", flush=True)

tltv_scaled = train_seq[samp, tlt, 3]
tltv_raw = tltv_scaled * scale_[3] + mean_[3]
print(f"Training data's current_ltv distribution (raw, sampled last timestep):", flush=True)
print(f"  min={tltv_raw.min():.1f} max={tltv_raw.max():.1f} "
      f"p99={np.percentile(tltv_raw,99):.1f} median={np.median(tltv_raw):.1f}", flush=True)
