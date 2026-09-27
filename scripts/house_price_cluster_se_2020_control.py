"""house_price_cluster_se_2020_control.py -- cluster-robust SE for the
house_price_test_2020_control WLS slopes, clustering zip3-level observations
on the FIRST DIGIT of zip3 (10 possible clusters, 0-9). Mirrors
house_price_cluster_se_2002.py exactly; only OUT_DIR differs. This is a
crude spatial-correlation correction -- 10 clusters is few, and each cluster
still spans a huge, heterogeneous region (e.g. all zip3s starting with '9'
covers the whole West Coast) -- not a substitute for a real spatial-HAC or
fine-grained cluster design, just a cheap check on whether the naive
(non-clustered) SE understates uncertainty from same-region correlated
errors.

Uses the same per-zip3 (x, y, weight) values already saved by
house_price_test_2020_control.py (house_price_zip3_{group}_{weight}.csv) --
does not recompute the underlying error/growth calculation.

Run:
    python scripts/house_price_cluster_se_2020_control.py
"""
import os

import numpy as np
import pandas as pd

BASE = '/scratch/at7095/mortgage_prepayment'
OUT_DIR = os.path.join(BASE, 'outputs/rolling/ensemble_onestep_cutoff_2020_control')


def cluster_robust_wls_se(x, y, w, cluster):
    """WLS y = a + b*x, weight w. Cluster-robust (CR1) sandwich SE for b.
    Returns (slope, naive_se, cluster_se, n_clusters)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ X.T @ W @ y
    slope = beta[1]
    resid = y - X @ beta

    # naive (non-clustered) WLS SE, for comparison
    dof = n - 2
    sigma2 = (w * resid ** 2).sum() / dof
    naive_se = np.sqrt((sigma2 * XtWX_inv)[1, 1])

    # cluster-robust sandwich
    meat = np.zeros((2, 2))
    clusters = pd.Series(cluster).unique()
    for c in clusters:
        mask = np.asarray(cluster) == c
        Xg, Wg, eg = X[mask], w[mask], resid[mask]
        score_g = Xg.T @ (Wg * eg)   # (2,)
        meat += np.outer(score_g, score_g)
    # small-cluster df correction, standard CR1: (G/(G-1)) * ((N-1)/(N-K))
    G = len(clusters)
    correction = (G / (G - 1)) * ((n - 1) / (n - 2))
    var_cluster = correction * (XtWX_inv @ meat @ XtWX_inv)
    cluster_se = np.sqrt(var_cluster[1, 1])

    return float(slope), float(naive_se), float(cluster_se), G


def main():
    rows = []
    for group in ['low_incentive', 'placebo']:
        for wlabel in ['count', 'upb']:
            path = os.path.join(OUT_DIR, f'house_price_zip3_{group}_{wlabel}.csv')
            zdf = pd.read_csv(path)
            zdf['zip3_first_digit'] = (zdf['zip3'] // 100).astype(int)
            slope, naive_se, cluster_se, n_clusters = cluster_robust_wls_se(
                zdf['x'], zdf['y'], zdf['weight'], zdf['zip3_first_digit'])
            rows.append({
                'group': group, 'weight': wlabel, 'n_zip3s': len(zdf), 'n_clusters': n_clusters,
                'slope': slope, 'naive_se': naive_se, 'cluster_se': cluster_se,
                'naive_t': slope / naive_se, 'cluster_t': slope / cluster_se,
                'se_inflation_ratio': cluster_se / naive_se,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'house_price_cluster_se.csv'), index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
