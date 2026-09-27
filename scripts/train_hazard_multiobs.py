"""
train_hazard_multiobs.py — Trains on prepare_sequences_multiobs_zbc.py output:
fixed-window (loan_id, ref_month) observations, one forward label each, built
specifically to separate "high refi incentive" from "high incentive but
already declined it repeatedly" (burnout/survivor selection) by observing the
same loan at several ages instead of once at its final age.

THIS IS A NEW FILE, NOT A COPY-EDIT OF train_hazard_rolling.py. The scoring
logic is deliberately different, not an oversight — see "Scoring" below.
Do not merge train_hazard_rolling.py's HazardSampler / evaluate_hazard back
in here; the multiobs builder's own docstring says pointing HazardSampler at
this output destroys the window/label alignment it constructs (random
per-epoch truncation on top of windows that are already fixed at prep time,
plus a second layer of 50/50 oversampling stacked on the mandatory-draw
oversampling already baked into which rows exist).

Model: PrepaymentTransformer, architecture and hyperparameters identical to
train_hazard_rolling.py / train_hazard.py.

Scoring: each multiobs observation is a fixed window ending at ref_month with
ONE forward label (prepay in (ref_month, ref_month+H]) — not an open-ended
sequence that needs per-timestep survival aggregation. So training and eval
both use the masked-mean-pool + classifier forward pass (model(seq, mask),
no return_per_timestep) giving one logit per observation, and evaluate()
scores with sigmoid(logit) directly. No survival-CDF product.

Batching: no HazardSampler, no random truncation — each observation's window
is already fixed at prep time by the builder. ObservationSampler below just
draws observation indices, either uniformly (natural ~8% positive rate) or,
if --pos_ratio is set, at a chosen positive/negative mix.

Do NOT load train_prepay_timestep / test_prepay_timestep — the builder's own
module docstring says these are meaningless for this output (filled with -1
for every observation, kept only for file-shape compatibility with the
trailing builder).

Usage:
    python train_hazard_multiobs.py --cutoff_year 2020
    python train_hazard_multiobs.py --cutoff_year 2020 --pos_ratio 0.5 --use_ipw

Outputs (to outputs/rolling/cutoff_{YEAR}_multiobs_k{K}_h{H}/):
    hazard_best.pt    — best model checkpoint (by test AUC)
    hazard_final.pt   — model checkpoint at the last training epoch, saved
                         unconditionally regardless of whether it beat best_auc
    train_ckpt.pt     — resume checkpoint (model/optimizer/scheduler + both RNG
                         states), written every --ckpt_every epochs and on the
                         final epoch. If present when the script starts, training
                         resumes from it automatically. Deleted automatically on
                         clean completion, so a finished out_dir always re-runs
                         fresh rather than silently replaying its old state —
                         only present while a run is mid-flight or was killed.
    results.json      — best AUC, Platt params (a, b), pos_ratio/use_ipw used, history
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

BASE = '/scratch/at7095/mortgage_prepayment'

# ── Hyperparameters (identical to train_hazard_rolling.py) ────────────────────
BATCH_SIZE      = 2048
N_EPOCHS        = 50      # override with --n_epochs if needed
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
STEPS_PER_EPOCH = 10_000  # steps not full passes — keeps epoch wall-clock predictable
MAX_SEQ         = 33
N_FEATURES      = 9
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model (identical to train_hazard_rolling.py) ───────────────────────────────

class PrepaymentTransformer(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_ff: int = 256, max_seq: int = MAX_SEQ, dropout: float = 0.1):
        super().__init__()
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x, mask=None, return_per_timestep: bool = False):
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out      = self.input_proj(x) + self.pos_embedding(pos)
        pad_mask = ~mask if mask is not None else None       # True=padding for PyTorch
        out      = self.transformer(out, src_key_padding_mask=pad_mask)
        if return_per_timestep:
            return self.classifier(out).squeeze(-1)          # (B, T)
        if mask is not None:
            real = mask.float().unsqueeze(-1)
            out  = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.classifier(out).squeeze(-1)              # (B,)


# ── Batch sampler ───────────────────────────────────────────────────────────
# NOT a HazardSampler. Each observation's window is already fixed at prep
# time by the multiobs builder (right-aligned at ref_month, label already
# resolved) — there is no per-timestep truncation to do here, just draw
# observation indices.

class ObservationSampler:
    """Draws batch_size observation indices.

    pos_ratio=None (default): uniform random draw over ALL observations —
        trains on the natural positive rate as-is (~8.33% for k=5,h=1).
    pos_ratio=f: draw f*batch_size from label==1 indices and the rest from
        label==0 indices, both with replacement.
    """

    def __init__(self, seq, mask, labels, incl_prob, pos_ratio=None):
        self.seq       = seq
        self.mask      = mask
        self.labels    = labels
        self.incl_prob = incl_prob
        self.pos_ratio = pos_ratio
        self.n         = len(labels)

        if pos_ratio is None:
            print(f'  Sampler: uniform over {self.n:,} observations '
                  f'(natural {100 * float(labels.mean()):.2f}% positive rate)', flush=True)
        else:
            self.pos_idx = np.where(labels == 1)[0]
            self.neg_idx = np.where(labels == 0)[0]
            print(f'  Sampler: pos_ratio={pos_ratio} draw | '
                  f'{len(self.pos_idx):,} positive / {len(self.neg_idx):,} negative '
                  f'observations available', flush=True)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        if self.pos_ratio is None:
            idx = rng.integers(0, self.n, size=batch_size)
        else:
            n_pos = int(round(batch_size * self.pos_ratio))
            n_neg = batch_size - n_pos
            idx = np.concatenate([
                rng.choice(self.pos_idx, size=n_pos, replace=True),
                rng.choice(self.neg_idx, size=n_neg, replace=True),
            ])
            rng.shuffle(idx)

        batch_seq    = np.asarray(self.seq[idx])
        batch_mask   = np.asarray(self.mask[idx])
        batch_labels = np.asarray(self.labels[idx])
        batch_weight = np.asarray(self.incl_prob[idx]) if self.incl_prob is not None else None
        return batch_seq, batch_mask, batch_labels, batch_weight


# ── Evaluation ────────────────────────────────────────────────────────────────
# Deliberately NOT a survival-CDF aggregation (contrast with
# train_hazard_rolling.py's evaluate_hazard). Each multiobs observation is a
# single fixed window with one forward label baked in by the builder, so the
# model's single pooled logit for that observation IS the score. Do not
# re-add per-timestep survival aggregation here by copy-pasting
# evaluate_hazard from train_hazard_rolling.py — there is no open-ended
# sequence to aggregate over.

def evaluate(model, seq, mask, labels, batch_size: int = 512):
    """score = sigmoid(logit) directly. roc_auc_score(labels, score)."""
    model.eval()
    n      = len(labels)
    scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = torch.tensor(np.asarray(seq[i:i + batch_size]),  device=DEVICE)
            bm = torch.tensor(np.asarray(mask[i:i + batch_size]), device=DEVICE)
            logits = model(bs, mask=bm)                       # (B,) — no return_per_timestep
            scores[i:i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return roc_auc_score(labels, scores), scores


# ── Platt calibration (identical to train_hazard_rolling.py) ──────────────────

def platt_calibrate(raw_scores: np.ndarray, labels: np.ndarray):
    """Fit P = sigmoid(a * score + b) minimising log-loss.

    raw_scores: sigmoid scores from evaluate() (already in [0,1])
    labels:     binary prepaid-in-horizon labels

    Returns (a, b).
    """
    def nll(params):
        a, b = params
        p = np.clip(1.0 / (1.0 + np.exp(-(a * raw_scores + b))), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 20_000, 'xatol': 1e-7, 'fatol': 1e-7})
    return float(res.x[0]), float(res.x[1])


# ── IPW weight (factored out so the smoke test can assert its DIRECTION,
# not just that the training loop executes) ────────────────────────────────

def ipw_weight(incl_prob: np.ndarray) -> torch.Tensor:
    """Horvitz-Thompson weight: 1/incl_prob, NOT incl_prob itself. Rare pool
    draws (small incl_prob) stand in for more of their stratum's unsampled
    population and must be UPweighted; the mandatory/terminal draw
    (incl_prob=1.0, and by this builder's H=1 window logic the ONLY draw
    that can carry label=1) is already fully sampled and correctly gets the
    minimum weight (1.0). Weighting BY incl_prob directly (the pre-fix bug,
    caught 2026-09-07) inverts this -- it downweights the rare negatives
    that IPW exists to upweight, pushing training even further toward the
    already-oversampled positives than the uncorrected run.
    """
    return 1.0 / torch.tensor(incl_prob, device=DEVICE)


# ── Training loop (factored out so a synthetic smoke test can drive it) ───────

def train_and_evaluate(
    train_seq, train_mask, train_labels, train_incl_prob,
    test_seq, test_mask, test_labels,
    n_epochs: int, pos_ratio, use_ipw: bool,
    out_dir: str | None = None,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
    max_seq: int = MAX_SEQ,
    seed: int = 42,
    ckpt_every: int = 10,
):
    print(f'Active options: pos_ratio={pos_ratio!r}  use_ipw={use_ipw}', flush=True)

    sampler   = ObservationSampler(train_seq, train_mask, train_labels, train_incl_prob,
                                    pos_ratio=pos_ratio)
    print(f'CUBLAS_WORKSPACE_CONFIG={os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET")}', flush=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # manual_seed alone leaves cuDNN algorithm selection and CUDA attention/
    # reduction kernels (flash/mem-efficient SDPA backward, cuDNN autotuned
    # convs) free to pick nondeterministic implementations -- confirmed
    # empirically 2026-09-07: two same-seed runs still differed by up to
    # 0.21 per parameter with only manual_seed set. warn_only=True so any op
    # genuinely lacking a deterministic kernel degrades to a warning instead
    # of a hard crash, rather than failing training outright.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model     = PrepaymentTransformer(max_seq=max_seq).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss(reduction='none' if use_ipw else 'mean')
    rng       = np.random.default_rng(seed)

    best_auc, best_scores = 0.0, None
    history = []

    # Epoch-level resume: a SLURM time-limit kill mid-run previously lost the
    # whole job (only hazard_best.pt/hazard_final.pt existed, written once at
    # the end). train_ckpt.pt captures full state -- model/optimizer/scheduler
    # plus both RNGs (torch and the sampler's numpy Generator) -- so a resumed
    # run continues the same batch-draw/dropout sequence instead of restarting
    # it, matching this project's existing resume-guard discipline elsewhere
    # (prepare_sequences_multiobs_zbc.py's per-vintage/per-pass checkpoints).
    ckpt_path = os.path.join(out_dir, 'train_ckpt.pt') if out_dir is not None else None
    start_epoch = 1
    if ckpt_path is not None and os.path.exists(ckpt_path):
        # weights_only=False: this checkpoint (unlike hazard_best.pt/hazard_final.pt,
        # which hold only tensors/primitives) also carries numpy arrays (best_scores,
        # numpy_rng_state) that PyTorch>=2.6's default weights_only=True rejects.
        # Always our own file in out_dir, never an untrusted source.
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        torch.set_rng_state(ckpt['torch_rng_state'])
        if torch.cuda.is_available() and ckpt['cuda_rng_state_all'] is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state_all'])
        rng.bit_generator.state = ckpt['numpy_rng_state']
        best_auc    = ckpt['best_auc']
        best_scores = ckpt['best_scores']
        history     = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f'RESUME from epoch {start_epoch}/{n_epochs} (checkpoint: {ckpt_path})',
              flush=True)
    epoch = start_epoch - 1  # holds if the loop below never executes (already-done resume)
    auc   = history[-1]['auc'] if history else 0.0  # same: last epoch's auc, for final-save below

    def _save_ckpt():
        # Write-then-rename: a direct write here (same pattern as the builder's
        # pickle.dump checkpoints) left an unreadable 0-byte file when a kill
        # landed between the truncating open() and the write completing --
        # observed directly during this project's own checkpoint/resume testing.
        # That was on small pickle files; this checkpoint carries a full model +
        # optimizer state and is written every few epochs for the entire run, so
        # the same race is more exposed here. os.replace is atomic on the same
        # filesystem, so a kill mid-write leaves the OLD checkpoint intact
        # instead of a truncated new one.
        tmp_path = ckpt_path + '.tmp'
        torch.save({
            'epoch':              epoch,
            'model_state':        model.state_dict(),
            'optimizer_state':    optimizer.state_dict(),
            'scheduler_state':    scheduler.state_dict(),
            'torch_rng_state':    torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state':    rng.bit_generator.state,
            'best_auc':           best_auc,
            'best_scores':        best_scores,
            'history':            history,
        }, tmp_path)
        os.replace(tmp_path, ckpt_path)

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(steps_per_epoch):
            bseq, bmask, lbl, bweight = sampler.sample_batch(batch_size, rng)
            x = torch.tensor(bseq,  device=DEVICE)
            m = torch.tensor(bmask, device=DEVICE)
            y = torch.tensor(lbl,   device=DEVICE)
            logits = model(x, mask=m)

            if use_ipw:
                w = ipw_weight(bweight)
                per_sample = criterion(logits, y)
                loss = (per_sample * w).sum() / w.sum().clamp(min=1e-12)
            else:
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss    = epoch_loss / steps_per_epoch
        auc, scores = evaluate(model, test_seq, test_mask, test_labels)
        elapsed     = time.time() - t0

        history.append({'epoch': epoch, 'loss': avg_loss, 'auc': float(auc)})
        print(f'Epoch {epoch:>2}/{n_epochs}  loss={avg_loss:.5f}  '
              f'auc={auc:.4f}  t={elapsed:.0f}s', flush=True)

        scheduler.step(auc)
        if auc > best_auc:
            best_auc    = auc
            best_scores = scores.copy()
            if out_dir is not None:
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {'input_dim': N_FEATURES, 'n_heads': 4, 'n_layers': 2,
                               'd_model': 64, 'dim_ff': 256, 'dropout': 0.1,
                               'max_seq': max_seq},
                    'epoch': epoch,
                    'auc':   float(auc),
                }, os.path.join(out_dir, 'hazard_best.pt'))
            print(f'  → Best AUC: {best_auc:.4f}' + (' — saved.' if out_dir else ''), flush=True)

        if ckpt_path is not None and (epoch % ckpt_every == 0 or epoch == n_epochs):
            _save_ckpt()
            print(f'  → Checkpoint saved (epoch {epoch}): {ckpt_path}', flush=True)

    if out_dir is not None:
        torch.save({
            'model_state': model.state_dict(),
            'config': {'input_dim': N_FEATURES, 'n_heads': 4, 'n_layers': 2,
                       'd_model': 64, 'dim_ff': 256, 'dropout': 0.1,
                       'max_seq': max_seq},
            'epoch': epoch,
            'auc':   float(auc),
        }, os.path.join(out_dir, 'hazard_final.pt'))
        print(f'  → Final epoch ({epoch}) AUC: {auc:.4f} — saved.', flush=True)

    # Clean completion: remove the resume checkpoint. Otherwise a later re-run
    # pointed at the same out_dir (e.g. same cutoff/k/h/ipw/run_tag re-submitted
    # by mistake, or a future --run_tag omission) would find train_ckpt.pt at
    # epoch==n_epochs, resume into an empty range(n_epochs+1, n_epochs+1), and
    # silently skip straight to Platt/results.json on the OLD state instead of
    # training at all -- a completed dir must re-run fresh, not replay itself.
    if ckpt_path is not None and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print('\nFitting Platt calibration on test set...', flush=True)
    a, b = platt_calibrate(best_scores, test_labels)
    print(f'  Platt: a={a:.4f}, b={b:.4f}', flush=True)

    last10 = history[-10:]
    mean_auc_last10 = float(sum(e['auc'] for e in last10) / len(last10))
    print(f'  Mean AUC, last {len(last10)} epochs: {mean_auc_last10:.4f}', flush=True)

    return {
        'best_auc':        float(best_auc),
        'mean_auc_last10': mean_auc_last10,
        'platt_a':         a,
        'platt_b':         b,
        'history':         history,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff_year',   type=int, default=2020)
    parser.add_argument('--k_draws',       type=int, default=5)
    parser.add_argument('--label_horizon', type=int, default=1)
    parser.add_argument('--max_seq_len',   type=int, default=33,
                         help='Must match the --max_seq_len used by the prep script to build '
                              'this data (default=33). Determines the _L{n} suffix on the '
                              'sequence dir this looks for; the model position-embedding table '
                              'size is derived from the loaded data itself, not hardcoded.')
    parser.add_argument('--sampling_mode', choices=['fixed_k', 'fixed_fraction'], default='fixed_k',
                         help='Must match the --sampling_mode used by the prep script to build '
                              'this data. Determines the k{K}/f{frac} budget tag on the '
                              'sequence dir this looks for.')
    parser.add_argument('--frac_draws',    type=float, default=None,
                         help='Required when --sampling_mode=fixed_fraction; must match the '
                              'prep script\'s --frac_draws used to build this data.')
    parser.add_argument('--n_epochs',      type=int, default=N_EPOCHS)
    parser.add_argument('--pos_ratio',     type=float, default=None,
                         help='Fraction of each batch drawn from label==1 observations. '
                              'Default: unset — uniform draw over all observations at '
                              'the natural positive rate.')
    parser.add_argument('--use_ipw',       action='store_true', default=False,
                         help='Weight the loss per-sample by train_incl_prob '
                              '(weighted mean instead of plain mean). Default: off.')
    parser.add_argument('--run_tag',       type=str, default='',
                         help='Appended to OUT_DIR (e.g. "_seedcheck_a"). Default: empty, '
                              'no-op for every existing invocation. Lets two same-seed runs '
                              'write to distinct directories for a reproducibility check '
                              'instead of one overwriting the other.')
    parser.add_argument('--seed',          type=int, default=42,
                         help='Seed for torch.manual_seed/cuda.manual_seed_all (model init, '
                              'dropout) and the ObservationSampler batch-draw RNG. Default: 42, '
                              'matching every existing invocation\'s previously-hardcoded value '
                              '-- unset is a no-op.')
    parser.add_argument('--ckpt_every',    type=int, default=10,
                         help='Save a full resume checkpoint (train_ckpt.pt) every N epochs, '
                              'plus always on the final epoch. Default: 10, sized so a SLURM '
                              'time-limit kill loses at most ~10 epochs of work (~25 min at the '
                              '~155s/epoch L33/f0.2 rate measured on cutoff_2020), not the whole '
                              'run. A prior run present in OUT_DIR resumes from it automatically.')
    parser.add_argument('--include_pre2013', action='store_true', default=False,
                         help='Must match the --include_pre2013 used by the prep script to build '
                              'this data. Appends the _hist suffix to the sequence dir this looks '
                              'for -- prepare_sequences_multiobs_zbc.py appends _hist to its '
                              'SAVE_DIR when this is set, and this flag was missing from the '
                              'mirrored SEQ_DIR formula below until the historical-era (cutoff_2002) '
                              'rebuild needed it: default False was a silent no-op for every '
                              'modern-era (cutoff>=2013) invocation to date.')
    parser.add_argument('--seq_dir',       type=str, default=None,
                         help='Explicit sequence dir, overriding the --cutoff_year/--sampling_mode/'
                              '--frac_draws/--max_seq_len/--label_horizon/--include_pre2013 formula '
                              'below. Default: None, no-op for every existing invocation -- added '
                              'after a plain SEQ_DIR (data/sequences_rolling/cutoff_2020_zbc_multiobs_'
                              'f0.2_h1) was silently overwritten in place by a later rebuild (job '
                              '18022825, 2026-09-19) between two ensemble seeds trained against the '
                              'same formula-derived path, so seeds trained months apart on a path that '
                              'looks unchanged can silently be trained on different data. Use this to '
                              'pin a run to a specific, verified-by-content directory instead.')
    args = parser.parse_args()

    if args.sampling_mode == 'fixed_fraction':
        if args.frac_draws is None:
            parser.error('--sampling_mode=fixed_fraction requires --frac_draws')
    elif args.frac_draws is not None:
        parser.error('--frac_draws is only used with --sampling_mode=fixed_fraction '
                      '(sampling_mode is fixed_k, --k_draws applies instead)')

    # Must mirror prepare_sequences_multiobs_zbc.py's SAVE_DIR formula exactly
    # (_DEFAULT_SEQ_LEN=33, _budget_tag=k{k}/f{frac}, _hist_suffix) -- do not
    # reimplement this independently a second time, or the two scripts' naming
    # conventions can drift apart silently (as _hist_suffix itself already did
    # once: added to the builder's SAVE_DIR when historical-era support landed,
    # but missed here until the cutoff_2002 training run needed it).
    _cap          = '' if args.max_seq_len == 33 else f'_L{args.max_seq_len}'
    _budget_tag   = f'k{args.k_draws}' if args.sampling_mode == 'fixed_k' else f'f{args.frac_draws}'
    _hist_suffix  = '_hist' if args.include_pre2013 else ''
    SEQ_DIR = args.seq_dir if args.seq_dir is not None else os.path.join(
        BASE, f'data/sequences_rolling/cutoff_{args.cutoff_year}_zbc_multiobs'
              f'_{_budget_tag}_h{args.label_horizon}{_cap}{_hist_suffix}')
    assert os.path.isdir(SEQ_DIR), (
        f'Expected sequence dir not found: {SEQ_DIR} -- check --max_seq_len/'
        f'--sampling_mode/--frac_draws match what prepare_sequences_multiobs_zbc.py built.')
    # _ipw suffix only when --use_ipw is set, so this is a no-op for every
    # existing/default invocation (job 16964657's directory naming is
    # unchanged) -- added because OUT_DIR was otherwise identical for a
    # --use_ipw run at the same cutoff/k_draws/label_horizon, which would
    # silently overwrite that run's hazard_best.pt.
    OUT_DIR = os.path.join(
        BASE, f'outputs/rolling/cutoff_{args.cutoff_year}_multiobs'
              f'_k{args.k_draws}_h{args.label_horizon}'
              f'{"_ipw" if args.use_ipw else ""}'
              f'{args.run_tag}')
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Device: {DEVICE}  |  cutoff: {args.cutoff_year}  |  '
          f'k={args.k_draws}  H={args.label_horizon}', flush=True)
    print(f'Sequences: {SEQ_DIR}', flush=True)
    print(f'Outputs:   {OUT_DIR}', flush=True)

    # ── Load sequences ────────────────────────────────────────────────────────
    # train_seq/train_mask are mmap'd (train_seq alone is 8.15GB). The
    # per-observation metadata arrays (labels, incl_prob) are small (~27MB
    # each for train) and loaded fully. train_prepay_timestep /
    # test_prepay_timestep are NEVER loaded — the builder fills them with -1
    # for every observation and its own docstring calls them meaningless here.
    def _mmap(name):
        return np.load(os.path.join(SEQ_DIR, name), mmap_mode='r')

    def _load(name):
        return np.load(os.path.join(SEQ_DIR, name))

    train_seq       = _mmap('train_seq.npy')
    train_mask      = _mmap('train_mask.npy')
    train_labels    = _load('train_labels.npy')
    train_incl_prob = _load('train_incl_prob.npy')

    test_seq    = _mmap('test_seq.npy')
    test_mask   = _mmap('test_mask.npy')
    test_labels = _load('test_labels.npy')

    print(f'train: {train_seq.shape} | test: {test_seq.shape}', flush=True)

    max_seq = train_seq.shape[1]
    assert max_seq == args.max_seq_len, (
        f'Loaded data has sequence length {max_seq} but --max_seq_len={args.max_seq_len} '
        f'-- SEQ_DIR naming resolved to the wrong directory: {SEQ_DIR}')

    results = train_and_evaluate(
        train_seq, train_mask, train_labels, train_incl_prob,
        test_seq, test_mask, test_labels,
        n_epochs=args.n_epochs, pos_ratio=args.pos_ratio, use_ipw=args.use_ipw,
        out_dir=OUT_DIR, max_seq=max_seq, seed=args.seed, ckpt_every=args.ckpt_every,
    )

    results.update({
        'cutoff_year':    args.cutoff_year,
        'k_draws':        args.k_draws,
        'label_horizon':  args.label_horizon,
        'n_train':        int(len(train_seq)),
        'n_test':         int(len(test_seq)),
        'pos_ratio':      args.pos_ratio,
        'use_ipw':        args.use_ipw,
        'seed':           args.seed,
    })
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. cutoff={args.cutoff_year} | AUC={results['best_auc']:.4f} | "
          f"Platt a={results['platt_a']:.4f} b={results['platt_b']:.4f}", flush=True)


if __name__ == '__main__':
    main()
