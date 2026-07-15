#!/usr/bin/env python3
"""
Continuous convergence-point tracking
=====================================

The continuous-modulation case of the convergence-point calculus. An inhomogeneous
Poisson voice is driven by a rate that dips at the convergence point and rises away
from it, and we measure how well the achieved event density tracks the intended rate.

The rate profile is the one the paper specifies:

    lambda(t) = 5 + 40 * |t - t_CP| / t_CP,   t_CP = 15 s,   over [0, 30] s

so the instantaneous rate falls to 5 notes/s at the convergence and rises to 45 at the
ends. Events are drawn by thinning, binned into one-second windows, and the achieved
per-window density is correlated against the intended rate at the window centres.

This replaces a hard-coded r = 0.907 that lived in recalculate_statistics.py and was
produced by no measurement. Outputs cp_continuous_tracking.json.

Usage:  python cp_continuous_tracking.py [--seed 42]
"""

import argparse
import json

import numpy as np
from scipy import stats

from config import CODE_DIR

T_CP = 15.0
T_TOTAL = 30.0
LAMBDA_BASE = 5.0
LAMBDA_SLOPE = 40.0


def intended_rate(t):
    return LAMBDA_BASE + LAMBDA_SLOPE * np.abs(t - T_CP) / T_CP


def draw_events(seed):
    """Inhomogeneous Poisson process on [0, T_TOTAL] by thinning at the max rate."""
    rng = np.random.default_rng(seed)
    lam_max = intended_rate(np.array([0.0, T_TOTAL])).max()
    # homogeneous candidates at lam_max, then keep each with prob lambda(t)/lam_max
    n_cand = rng.poisson(lam_max * T_TOTAL)
    cand = np.sort(rng.uniform(0.0, T_TOTAL, size=n_cand))
    keep = rng.uniform(0.0, 1.0, size=n_cand) < intended_rate(cand) / lam_max
    return cand[keep]


def windowed_density(events, w=1.0):
    edges = np.arange(0.0, T_TOTAL + w, w)
    counts, _ = np.histogram(events, bins=edges)
    centres = edges[:-1] + w / 2
    return centres, counts / w


def main(seed):
    events = draw_events(seed)
    centres, achieved = windowed_density(events)
    intended = intended_rate(centres)

    r, p = stats.pearsonr(achieved, intended)
    rmse = float(np.sqrt(np.mean((achieved - intended) ** 2)))

    pre = centres < T_CP
    post = centres >= T_CP
    r_pre, p_pre = stats.pearsonr(achieved[pre], intended[pre])
    r_post, p_post = stats.pearsonr(achieved[post], intended[post])

    near = (centres >= 12) & (centres <= 18)
    ends = (centres <= 5) | (centres >= 25)
    near_density = float(achieved[near].mean())
    end_density = float(achieved[ends].mean())

    result = {
        'seed': seed, 'n_events': int(len(events)),
        'epsilon_ms': 50, 't_cp_s': T_CP, 'total_s': T_TOTAL,
        'rate_profile': 'lambda(t) = 5 + 40*|t - 15|/15',
        'pearson_r': float(r), 'pearson_p': float(p), 'rmse_nps': rmse,
        'r_pre_cp': float(r_pre), 'p_pre_cp': float(p_pre),
        'r_post_cp': float(r_post), 'p_post_cp': float(p_post),
        'near_cp_density_nps': near_density,
        'end_density_nps': end_density,
        'reduction_factor': end_density / near_density if near_density else float('inf'),
    }
    out = CODE_DIR / 'cp_continuous_tracking.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"e:pi canon, CP at {T_CP:.0f}s, epsilon 50 ms, seed {seed}")
    print(f"  n_events = {len(events)}")
    print(f"  continuous tracking: Pearson r = {r:.3f} (p = {p:.2e}), RMSE = {rmse:.2f} notes/s")
    print(f"  pre-CP r = {r_pre:.3f}   post-CP r = {r_post:.3f}")
    print(f"  near-CP density {near_density:.2f} vs ends {end_density:.2f} "
          f"({end_density/near_density:.1f}x)")
    print(f"\nWrote {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args().seed)
