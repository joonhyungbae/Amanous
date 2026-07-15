#!/usr/bin/env python3
"""
Perceptual saturation of melodic structure under temporal integration
=====================================================================

The earlier density sweep could not test its own hypothesis. It measured the
pitch-interval entropy of a fixed number of events, a quantity that is invariant to
how fast those events are played, so no density-dependent transition could appear in
it and the reported transition had in fact been fitted to a hand-authored curve.

This experiment tests the hypothesis the paper actually makes, that melodic structure
becomes unresolvable once the event rate crosses the temporal resolution of the
listener (or of the instrument). The transition is not built into the generator. It
emerges from the interaction between event density and a fixed integration window,
which is the mechanism the paper names when it ties the saturation zone to the point
where the average inter-onset interval approaches the ~1 ms scanning resolution and,
perceptually, the ~25-30 Hz limit of auditory temporal resolution (van Noorden 1975;
Roads 2001).

Method. For each aggregate density rho, the system's own generator produces a
structured melodic stream (real symbol configurations, C-major pitch set) over a
fixed time span. A temporal integration window of width w collapses events that fall
within the same window into one perceived unit, since a listener cannot order events
closer together than w. The retained melodic structure is measured as the fraction of
the intended melodic transitions that survive integration, i.e. the normalized
contour similarity between the fully resolved sequence and the integrated one. The
same is done for a random null. Retention is fitted piecewise against density and the
break point bootstrapped.

The prediction is that retention holds near 1 while rho*w << 1 and falls once rho*w
approaches 1, so that the break point sits near 1/w. With w = 40 ms (25 Hz, the lower
edge of the cited perceptual range) that predicts a transition near 25 notes/s. The
point of the experiment is to measure whether it does, not to assume it.

Outputs perceptual_saturation.csv and perceptual_saturation.json.

Usage:  python perceptual_saturation.py [--window-ms 40] [--seed 42] [--trials 20]
"""

import argparse
import csv
import json
import sys

import numpy as np

from config import CODE_DIR, SUPPLEMENTARY_DIR

sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'experiments'))
sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'core'))

from density_sweep_null_model_comparison import (  # noqa: E402
    generate_amanous_stream_at_density,
    generate_random_midi,
    HAS_AMANOUS,
)

DENSITIES = [10, 15, 20, 25, 28, 30, 40, 50, 60, 80, 100, 120, 150, 200]


def contour(pitches):
    p = np.asarray(pitches)
    d = np.diff(p)
    return np.sign(d)  # -1 down, 0 same, +1 up


def integrate(times, pitches, w):
    """
    Collapse events into integration windows of width w. Events within one window are
    perceptually simultaneous and cannot be ordered, so the window is represented by a
    single perceived pitch, its mean (the aggregate the ear forms). Returns the
    sequence of perceived pitches, one per non-empty window.
    """
    times = np.asarray(times, float)
    pitches = np.asarray(pitches, float)
    if len(times) == 0:
        return np.array([])
    order = np.argsort(times)
    times, pitches = times[order], pitches[order]
    bins = np.floor((times - times[0]) / w).astype(int)
    perceived = []
    for b in np.unique(bins):
        perceived.append(pitches[bins == b].mean())
    return np.array(perceived)


def retention(times, pitches, w):
    """
    Fraction of the intended melodic transitions that survive integration.

    The fully resolved contour has len(pitches)-1 transitions. After integration the
    perceived contour has fewer, because events that merged can no longer register as
    separate melodic moves. Retention is the count of perceived directional
    transitions relative to the intended count, capped at 1. When rho*w << 1 no events
    merge and retention is 1; as merging sets in it falls towards the level a single
    aggregated blob would give.
    """
    intended = contour(pitches)
    n_intended = int(np.sum(intended != 0))
    if n_intended == 0:
        return float('nan')
    perceived = integrate(times, pitches, w)
    if len(perceived) < 2:
        return 0.0
    n_perceived = int(np.sum(contour(perceived) != 0))
    return min(1.0, n_perceived / n_intended)


# ---- piecewise fit (shared with density_sweep_breakpoint conventions)

def fit_segment(x, y):
    if len(x) < 2:
        return np.inf
    slope, intercept = np.polyfit(x, y, 1)
    r = y - (intercept + slope * x)
    return slope, float(np.sum(r ** 2))


def estimate_breakpoint(x, y, bp_min=15, bp_max=60, grid=181):
    cand = np.linspace(bp_min, bp_max, grid)
    best, best_rss = cand[0], np.inf
    for bp in cand:
        left, right = x <= bp, x > bp
        if left.sum() < 2 or right.sum() < 2:
            continue
        _, r1 = fit_segment(x[left], y[left])
        _, r2 = fit_segment(x[right], y[right])
        if r1 + r2 < best_rss:
            best, best_rss = bp, r1 + r2
    return float(best)


def piecewise_r2(x, y, bp):
    left, right = x <= bp, x > bp
    s_pre, r1 = fit_segment(x[left], y[left])
    s_post, r2 = fit_segment(x[right], y[right])
    ss = float(np.sum((y - y.mean()) ** 2))
    _, rss_lin = fit_segment(x, y)
    return {
        'slope_pre': float(s_pre), 'slope_post': float(s_post),
        'r2_piecewise': float(1 - (r1 + r2) / ss) if ss else float('nan'),
        'r2_linear': float(1 - rss_lin / ss) if ss else float('nan'),
    }


def measure(w, densities, n_events, n_trials, seed):
    rng = np.random.default_rng(seed)
    rows, per_trial = [], {}
    for d in densities:
        ret_a, ret_n = [], []
        for t in range(n_trials):
            np.random.seed(seed + 1000 * t + int(d))
            ta, pa = generate_amanous_stream_at_density(d, n_events, rng)
            tn, pn = generate_random_midi(d, n_events, rng)
            ra, rn = retention(ta, pa, w), retention(tn, pn, w)
            if not np.isnan(ra):
                ret_a.append(ra)
            if not np.isnan(rn):
                ret_n.append(rn)
        per_trial[d] = ret_a
        rows.append({
            'density': d,
            'retention_amanous': float(np.mean(ret_a)),
            'retention_amanous_sd': float(np.std(ret_a, ddof=1)) if len(ret_a) > 1 else 0.0,
            'retention_null': float(np.mean(ret_n)),
            'events_per_window': d * w,
        })
    return rows, per_trial


def bootstrap_ci(rows, per_trial, n_boot, seed):
    rng = np.random.default_rng(seed + 7)
    x = np.array([r['density'] for r in rows], float)
    bps = []
    for _ in range(n_boot):
        y = np.array([float(np.mean(rng.choice(per_trial[r['density']],
                                               size=len(per_trial[r['density']]), replace=True)))
                      for r in rows])
        bps.append(estimate_breakpoint(x, y))
    return float(np.percentile(bps, 2.5)), float(np.percentile(bps, 97.5)), len(bps)


def main(window_ms, seed, trials, events, n_boot):
    if not HAS_AMANOUS:
        raise SystemExit("Real composer not importable; refusing to run on the fallback generator.")
    w = window_ms / 1000.0
    rows, per_trial = measure(w, DENSITIES, events, trials, seed)
    x = np.array([r['density'] for r in rows], float)
    y = np.array([r['retention_amanous'] for r in rows], float)
    bp = estimate_breakpoint(x, y)
    fit = piecewise_r2(x, y, bp)
    lo, hi, nb = bootstrap_ci(rows, per_trial, n_boot, seed)

    out_csv = CODE_DIR / 'perceptual_saturation.csv'
    with open(out_csv, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    result = {
        'window_ms': window_ms, 'predicted_breakpoint_nps': 1000.0 / window_ms,
        'seed': seed, 'n_trials': trials, 'n_events': events, 'n_levels': len(DENSITIES),
        'breakpoint_nps': bp, 'ci95': [lo, hi], 'n_bootstrap': nb, **fit,
    }
    out_json = CODE_DIR / 'perceptual_saturation.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"integration window {window_ms} ms  ->  predicted transition {1000.0/window_ms:.0f} notes/s")
    print(f"{len(DENSITIES)} levels, {events} events x {trials} trials, seed {seed}\n")
    print(f"{'rho':>5}  {'rho*w':>6}  {'retention (Amanous)':>22}  {'null':>6}")
    for r in rows:
        print(f"{r['density']:5.0f}  {r['events_per_window']:6.2f}  "
              f"{r['retention_amanous']:.3f} +- {r['retention_amanous_sd']:.3f}         "
              f"{r['retention_null']:.3f}")
    print(f"\nbreak point   {bp:.1f} notes/s   95% CI [{lo:.1f}, {hi:.1f}]  ({nb} resamples)")
    print(f"slopes        pre {fit['slope_pre']:.4f}  post {fit['slope_post']:.4f}")
    print(f"fit           R2 piecewise {fit['r2_piecewise']:.3f}  vs linear {fit['r2_linear']:.3f}")
    print(f"\nWrote {out_csv}\nWrote {out_json}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window-ms', type=float, default=40.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--events', type=int, default=200)
    ap.add_argument('--bootstrap', type=int, default=10000)
    a = ap.parse_args()
    main(a.window_ms, a.seed, a.trials, a.events, a.bootstrap)
