#!/usr/bin/env python3
"""
Null Model (Random Baseline) for Fig 3: Melodic Coherence & Tonal Stability.

Generates fully random MIDI streams (White Noise) at 10--200 notes/s and
computes the same metrics as Amanous to show that the 30 notes/s breakpoint
reflects structural intent, not metric saturation.

- Pitch: uniform 0--127
- Velocity: uniform 0--1023
- IOI: exponential (scale = 1/density)

Output CSV: density, random_single_voice_coherence, random_tonal_stability
(for comparison with Amanous: coherence_density_results.csv).
"""

import sys
import os
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from coherence_metrics import single_voice_coherence, tonal_stability


def generate_random_stream(density_notes_per_sec: float, n_events: int, rng: np.random.Generator):
    """
    One random MIDI stream: pitch U[0,127], velocity U[0,1023], IOI ~ Exp(1/density).
    Returns (times, pitches, velocities) and derived lists for metrics.
    """
    # Mean IOI = 1/density (seconds)
    scale_ioi = 1.0 / density_notes_per_sec
    iois = rng.exponential(scale=scale_ioi, size=n_events)
    iois = np.maximum(iois, 0.001)  # avoid zero
    times = np.concatenate([[0], np.cumsum(iois)])
    pitches = rng.integers(0, 128, size=n_events)
    velocities = rng.integers(0, 1024, size=n_events)
    return times, pitches, velocities


def run_density_level(density: float, n_events: int, rng: np.random.Generator, n_trials: int = 5):
    """Average single-voice coherence and tonal stability over n_trials random streams."""
    svc_list, ts_list = [], []
    for _ in range(n_trials):
        times, pitches, velocities = generate_random_stream(density, n_events, rng)
        svc = single_voice_coherence(pitches.tolist())
        ts = tonal_stability(pitches.tolist())
        if not np.isnan(svc):
            svc_list.append(svc)
        if not np.isnan(ts):
            ts_list.append(ts)
    return (
        np.mean(svc_list) if svc_list else np.nan,
        np.std(svc_list) if len(svc_list) > 1 else 0,
        np.mean(ts_list) if ts_list else np.nan,
        np.std(ts_list) if len(ts_list) > 1 else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Random baseline for Fig 3 (null model)")
    parser.add_argument("--densities", type=str, default="10,15,20,25,28,30,40,50,60,80,100,120,150,200",
                        help="Comma-separated density levels (notes/s)")
    parser.add_argument("--n-events", type=int, default=100, help="Events per stream (per trial)")
    parser.add_argument("--n-trials", type=int, default=5, help="Trials per density level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    densities = [float(x.strip()) for x in args.densities.split(",")]
    rng = np.random.default_rng(args.seed)

    rows = []
    for d in densities:
        svc_mean, svc_std, ts_mean, ts_std = run_density_level(d, args.n_events, rng, args.n_trials)
        rows.append({
            "density": d,
            "random_single_voice_coherence": svc_mean,
            "random_svc_std": svc_std,
            "random_tonal_stability": ts_mean,
            "random_ts_std": ts_std,
        })

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    w = csv.DictWriter(out, fieldnames=["density", "random_single_voice_coherence", "random_svc_std",
                                        "random_tonal_stability", "random_ts_std"])
    w.writeheader()
    w.writerows(rows)
    if args.output:
        out.close()
        print("Wrote", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
