#!/usr/bin/env python3
"""
Density sweep and saturation break point, measured rather than assumed
======================================================================

The saturation break point (the Computational Sensitivity Limit) is the paper's
headline result. It had been fitted to fourteen coherence values that no script
produced: they were typed into density_sweep_results.csv and hard-coded a second
time inside breakpoint_bootstrap.py. This script measures them instead.

For each aggregate density it generates stochastic two-voice textures with the
system's own generator, computes Single-Voice Coherence (the normalized entropy of
the pitch-interval distribution) and Pitch-Class Concentration, and averages over
trials. It then fits a piecewise linear regression to the measured curve and
bootstraps a 95% confidence interval for the break point.

Outputs density_sweep_measured.csv and density_sweep_breakpoint.json.

Usage:  python density_sweep_breakpoint.py [--seed 42] [--trials 5] [--events 100]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from config import CODE_DIR, SUPPLEMENTARY_DIR

sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'experiments'))
sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'core'))

from density_sweep_null_model_comparison import (  # noqa: E402
    generate_amanous_stream_at_density,
    generate_random_midi,
)
from coherence_metrics import single_voice_coherence, tonal_stability  # noqa: E402

# The fourteen levels the paper sweeps, spaced finely through the 20-30 notes/s band
# where the perceptual literature places the tracking-to-texture transition.
DENSITIES = [10, 15, 20, 25, 28, 30, 40, 50, 60, 80, 100, 120, 150, 200]


# ---------------------------------------------------------------- piecewise fit

def fit_segment(x, y):
    if len(x) < 2:
        return np.nan, np.nan, np.inf
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    return intercept, slope, float(np.sum(resid ** 2))


def rss_piecewise(bp, x, y):
    left, right = x <= bp, x > bp
    if left.sum() < 2 or right.sum() < 2:
        return np.inf
    _, _, r1 = fit_segment(x[left], y[left])
    _, _, r2 = fit_segment(x[right], y[right])
    return r1 + r2


def estimate_breakpoint(x, y, bp_min=15, bp_max=50, grid=141):
    """
    Search the break point on a grid over the plausible band. A 0.25 notes/s step is
    coarse enough to bootstrap ten thousand times and far finer than the width of the
    resulting confidence interval, so the quantization is not a limiting factor.
    """
    candidates = np.linspace(bp_min, bp_max, grid)
    rss = [rss_piecewise(bp, x, y) for bp in candidates]
    return float(candidates[int(np.argmin(rss))])


def piecewise_stats(x, y, bp):
    left, right = x <= bp, x > bp
    _, s_pre, r1 = fit_segment(x[left], y[left])
    _, s_post, r2 = fit_segment(x[right], y[right])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_pw = 1 - (r1 + r2) / ss_tot
    _, _, rss_lin = fit_segment(x, y)
    r2_lin = 1 - rss_lin / ss_tot
    return {
        'slope_pre': float(s_pre), 'slope_post': float(s_post),
        'slope_ratio': float(abs(s_pre / s_post)) if s_post else float('inf'),
        'r2_piecewise': float(r2_pw), 'r2_linear': float(r2_lin),
    }


# ---------------------------------------------------------------- sweep

def measure(densities, n_events, n_trials, seed):
    rng = np.random.default_rng(seed)
    rows, per_trial = [], {}
    for d in densities:
        svc_a, pcc_a, svc_null = [], [], []
        for t in range(n_trials):
            np.random.seed(seed + 1000 * t + int(d))
            _, pitches = generate_amanous_stream_at_density(d, n_events, rng)
            _, pitches_null = generate_random_midi(d, n_events, rng)
            for store, p in ((svc_a, pitches), (svc_null, pitches_null)):
                v = single_voice_coherence(np.asarray(p).tolist())
                if not np.isnan(v):
                    store.append(v)
            v = tonal_stability(np.asarray(pitches).tolist())
            if not np.isnan(v):
                pcc_a.append(v)
        per_trial[d] = svc_a
        rows.append({
            'density': d,
            'svc_amanous': float(np.mean(svc_a)),
            'svc_amanous_sd': float(np.std(svc_a, ddof=1)) if len(svc_a) > 1 else 0.0,
            'pcc_amanous': float(np.mean(pcc_a)),
            'svc_null': float(np.mean(svc_null)),
        })
    return rows, per_trial


def bootstrap_ci(rows, per_trial, n_boot, seed):
    """Resample trials within each density and refit the break point."""
    rng = np.random.default_rng(seed)
    x = np.array([r['density'] for r in rows], float)
    bps = []
    for _ in range(n_boot):
        y = np.array([float(np.mean(rng.choice(per_trial[r['density']],
                                               size=len(per_trial[r['density']]),
                                               replace=True)))
                      for r in rows])
        bps.append(estimate_breakpoint(x, y))
    lo, hi = np.percentile(bps, [2.5, 97.5])
    return float(lo), float(hi), len(bps)


def main(seed, trials, events, n_boot):
    rows, per_trial = measure(DENSITIES, events, trials, seed)

    x = np.array([r['density'] for r in rows], float)
    y = np.array([r['svc_amanous'] for r in rows], float)

    bp = estimate_breakpoint(x, y)
    stats_ = piecewise_stats(x, y, bp)
    lo, hi, n_ok = bootstrap_ci(rows, per_trial, n_boot, seed)

    out_csv = CODE_DIR / 'density_sweep_measured.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    result = {
        'seed': seed, 'n_trials': trials, 'n_events': events,
        'n_levels': len(DENSITIES), 'densities': DENSITIES,
        'breakpoint_nps': bp, 'ci95': [lo, hi], 'n_bootstrap': n_ok,
        **stats_,
        'null_svc_range': [min(r['svc_null'] for r in rows),
                           max(r['svc_null'] for r in rows)],
        'amanous_pcc_range': [min(r['pcc_amanous'] for r in rows),
                              max(r['pcc_amanous'] for r in rows)],
    }
    out_json = CODE_DIR / 'density_sweep_breakpoint.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"measured sweep: {len(DENSITIES)} density levels, "
          f"{events} events x {trials} trials, seed {seed}")
    for r in rows:
        print(f"  {r['density']:5.0f} notes/s   SVC {r['svc_amanous']:.3f} "
              f"+-{r['svc_amanous_sd']:.3f}   PCC {r['pcc_amanous']:.3f}   "
              f"null SVC {r['svc_null']:.3f}")
    print(f"\nbreak point   {bp:.1f} notes/s   95% CI [{lo:.1f}, {hi:.1f}] "
          f"({n_ok} bootstrap resamples)")
    print(f"slopes        pre {stats_['slope_pre']:.4f}  post {stats_['slope_post']:.4f}  "
          f"ratio {stats_['slope_ratio']:.1f}x")
    print(f"fit           R2 piecewise {stats_['r2_piecewise']:.3f}  "
          f"vs linear {stats_['r2_linear']:.3f}")
    print(f"\nWrote {out_csv}\nWrote {out_json}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--events', type=int, default=100)
    ap.add_argument('--bootstrap', type=int, default=10000)
    a = ap.parse_args()
    main(a.seed, a.trials, a.events, a.bootstrap)
