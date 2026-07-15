#!/usr/bin/env python3
"""
Discriminability of structured from random streams across density
=================================================================

The reframed contribution. The earlier claim of a sharp saturation threshold is not
supported (perceptual_saturation_wsweep.py shows the retention break point does not
track the integration window and is a floor artifact). What the data do support is a
density of maximal separability between structured and random streams, after which
that separability declines, which is the paper's stated claim that single-domain
metrics lose discriminative power at high density.

For each aggregate density this generates structured Amanous streams and random null
streams with the system's own generator, applies the temporal integration window, and
measures melodic-transition retention for each. It then tests, at every density,
whether the structured streams are distinguishable from the null (Welch t-test,
Cohen's d), locates the density of peak discriminability, and bootstraps a confidence
interval for that location.

Outputs discriminability_analysis.csv and discriminability_analysis.json.

Usage:  python discriminability_analysis.py [--window-ms 40] [--seed 42] [--trials 40]
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy import stats

from config import CODE_DIR, SUPPLEMENTARY_DIR

sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'experiments'))
sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'core'))

from density_sweep_null_model_comparison import (  # noqa: E402
    generate_amanous_stream_at_density, generate_random_midi, HAS_AMANOUS,
)
from perceptual_saturation import DENSITIES, retention  # noqa: E402


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else float('nan')


def measure(w, densities, n_events, n_trials, seed):
    rng = np.random.default_rng(seed)
    per_density = {}
    for d in densities:
        ra, rn = [], []
        for t in range(n_trials):
            np.random.seed(seed + 1000 * t + int(d))
            ta, pa = generate_amanous_stream_at_density(d, n_events, rng)
            tn, pn = generate_random_midi(d, n_events, rng)
            a, n = retention(ta, pa, w), retention(tn, pn, w)
            if not np.isnan(a):
                ra.append(a)
            if not np.isnan(n):
                rn.append(n)
        per_density[d] = (np.array(ra), np.array(rn))
    return per_density


def analyze(per_density, densities):
    rows = []
    for d in densities:
        ra, rn = per_density[d]
        t, p = stats.ttest_ind(ra, rn, equal_var=False)
        rows.append({
            'density': d,
            'retention_amanous': float(ra.mean()),
            'retention_amanous_sd': float(ra.std(ddof=1)),
            'retention_null': float(rn.mean()),
            'retention_null_sd': float(rn.std(ddof=1)),
            'gap': float(ra.mean() - rn.mean()),
            't': float(t), 'p': float(p), 'cohens_d': float(cohens_d(ra, rn)),
        })
    return rows


def peak_ci(per_density, densities, n_boot, seed):
    """Bootstrap the density at which the retention gap is largest."""
    rng = np.random.default_rng(seed + 11)
    x = np.array(densities, float)
    peaks = []
    for _ in range(n_boot):
        gaps = []
        for d in densities:
            ra, rn = per_density[d]
            ba = rng.choice(ra, size=len(ra), replace=True).mean()
            bn = rng.choice(rn, size=len(rn), replace=True).mean()
            gaps.append(ba - bn)
        peaks.append(x[int(np.argmax(gaps))])
    lo, hi = np.percentile(peaks, [2.5, 97.5])
    return float(lo), float(hi), float(np.median(peaks))


def main(window_ms, seed, trials, events, n_boot):
    if not HAS_AMANOUS:
        raise SystemExit("Real composer not importable; refusing to run on the fallback generator.")
    w = window_ms / 1000.0
    per_density = measure(w, DENSITIES, events, trials, seed)
    rows = analyze(per_density, DENSITIES)

    peak_density = max(rows, key=lambda r: r['gap'])['density']
    lo, hi, med = peak_ci(per_density, DENSITIES, n_boot, seed)

    # Half-max decline: density at which the gap has fallen to half its peak, on the
    # high-density side of the peak.
    peak_gap = max(r['gap'] for r in rows)
    half = peak_gap / 2
    half_density = None
    for r in rows:
        if r['density'] > peak_density and r['gap'] <= half:
            half_density = r['density']
            break

    out_csv = CODE_DIR / 'discriminability_analysis.csv'
    with open(out_csv, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    result = {
        'window_ms': window_ms, 'seed': seed, 'n_trials': trials, 'n_events': events,
        'n_levels': len(DENSITIES),
        'peak_discriminability_density': peak_density,
        'peak_gap': peak_gap,
        'peak_ci95': [lo, hi], 'peak_median_bootstrap': med,
        'half_max_density_high_side': half_density,
        'gap_at_lowest': rows[0]['gap'], 'gap_at_highest': rows[-1]['gap'],
    }
    out_json = CODE_DIR / 'discriminability_analysis.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"integration window {window_ms} ms   {len(DENSITIES)} levels, "
          f"{events} events x {trials} trials, seed {seed}\n")
    print(f"{'rho':>5} {'amanous':>16} {'null':>16} {'gap':>7} {'t':>7} {'p':>9} {'d':>6}")
    for r in rows:
        print(f"{r['density']:5.0f} {r['retention_amanous']:.3f}+-{r['retention_amanous_sd']:.3f}   "
              f"{r['retention_null']:.3f}+-{r['retention_null_sd']:.3f}   "
              f"{r['gap']:.3f} {r['t']:7.2f} {r['p']:9.2e} {r['cohens_d']:6.2f}")
    print(f"\npeak discriminability at {peak_density:.0f} notes/s (gap {peak_gap:.3f}), "
          f"95% CI [{lo:.0f}, {hi:.0f}]")
    print(f"gap falls to half-peak by {half_density} notes/s on the high-density side")
    print(f"\nWrote {out_csv}\nWrote {out_json}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window-ms', type=float, default=40.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--events', type=int, default=200)
    ap.add_argument('--bootstrap', type=int, default=10000)
    a = ap.parse_args()
    main(a.window_ms, a.seed, a.trials, a.events, a.bootstrap)
