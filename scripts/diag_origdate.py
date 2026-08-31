import numpy as np, pickle, sys
sys.path.insert(0, 'scripts')
from forecast_rolling_cpr import read_coupon_and_realized, mmyyyy_to_yyyymm

d = 'data/sequences_rolling/cutoff_2022_zbc'
seq  = np.load(f'{d}/test_seq.npy', mmap_mode='r')
mask = np.load(f'{d}/test_mask.npy', mmap_mode='r')
ids  = np.load(f'{d}/test_loan_ids.npy', allow_pickle=True)
with open(f'{d}/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
mean_, scale_ = scaler.mean_, scaler.scale_

coupon_map, active_set, prepaid_set, zip3_map, origdate_map = read_coupon_and_realized(
    2022, set(ids.tolist()))

target_ids = [lid for lid in ids if lid in coupon_map and 2.75 <= coupon_map[lid] < 4.25][:10]
id_to_row = {lid: i for i, lid in enumerate(ids)}

CUTOFF_YYYYMM = 202212
print(f"{'loan_id':>15} {'trusted_age@cutoff':>19} {'my_origdate_raw':>16} "
      f"{'decoded_yyyymm':>15} {'age@cutoff_from_my_origdate':>28}", flush=True)
for lid in target_ids:
    row = id_to_row[lid]
    lt = mask[row].sum() - 1
    trusted_age_scaled = seq[row, lt, 5]
    trusted_age = trusted_age_scaled * scale_[5] + mean_[5]

    od = origdate_map.get(lid)
    if od is None:
        print(f"{lid:>15} {trusted_age:>19.1f}  NO ORIGDATE FOUND", flush=True)
        continue
    orig_yyyymm = mmyyyy_to_yyyymm(int(od))
    my_age_at_cutoff = ((CUTOFF_YYYYMM//100 - orig_yyyymm//100)*12
                        + (CUTOFF_YYYYMM%100 - orig_yyyymm%100) - 1)
    print(f"{lid:>15} {trusted_age:>19.1f} {int(od):>16} "
          f"{orig_yyyymm:>15} {my_age_at_cutoff:>28.1f}", flush=True)
