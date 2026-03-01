#!/usr/bin/env python3
"""
For Figure 3 update: generate fully random MIDI at 10--200 notes/s and compute MC/TS.

- Density: same 14 bins as paper Figure 3 (10, 15, 20, 25, 28, 30, 40, 50, 60, 80, 100, 120, 150, 200).
- Null model: Pitch U[0,127], Velocity U[0,1023], IOI U(0, 2/ρ) — all Uniform (mean IOI = 1/ρ).
- Metrics: Melodic Coherence (MC), Tonal Stability (TS).
- Output: CSV (density, random_mc_mean, random_mc_std, random_ts_mean, random_ts_std) and TikZ coordinates.
"""

import sys
import os
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
from coherence_metrics import single_voice_coherence, tonal_stability

# Same density range as Figure 3 (paper §4.3.1)
FIG3_DENSITIES = [10, 15, 20, 25, 28, 30, 40, 50, 60, 80, 100, 120, 150, 200]
# Paper Figure 3 Amanous pre-saturation curve (Single-Voice Coherence, normalized 0–1)
AMANOUS_MC_FIGURE3 = {10: 1.00, 15: 0.92, 20: 0.78, 25: 0.55, 28: 0.38, 30: 0.25}
# Pre-saturation densities (Gap quantification; paper saturation ~28–30 notes/s)
PRE_SATURATION_DENSITIES = [10, 15, 20, 25, 28]


def generate_random_midi_stream(
    density_notes_per_sec: float,
    n_events: int,
    rng: np.random.Generator,
):
    """
    Generate one fully random MIDI stream.
    Pitch U[0,127], Velocity U[0,1023], IOI U(0, 2/ρ) — all Uniform (mean IOI = 1/ρ).
    """
    mean_ioi = 1.0 / density_notes_per_sec
    # U(0, 2/ρ) -> mean = 1/ρ, uniform
    iois = rng.uniform(0.0, 2.0 * mean_ioi, size=n_events)
    iois = np.maximum(iois, 1e-6)
    times = np.concatenate([[0], np.cumsum(iois)])
    pitches = rng.integers(0, 128, size=n_events)
    velocities = rng.integers(0, 1024, size=n_events)
    return times, pitches, velocities


def run_null_model_sweep(
    densities=None,
    n_events: int = 100,
    n_trials: int = 10,
    seed: int = 42,
):
    """Generate n_trials random streams per density; return MC/TS mean and std."""
    if densities is None:
        densities = FIG3_DENSITIES
    rng = np.random.default_rng(seed)
    rows = []
    for d in densities:
        mc_list, ts_list = [], []
        for _ in range(n_trials):
            _t, pitches, _v = generate_random_midi_stream(d, n_events, rng)
            mc = single_voice_coherence(pitches.tolist())
            ts = tonal_stability(pitches.tolist())
            if not np.isnan(mc):
                mc_list.append(mc)
            if not np.isnan(ts):
                ts_list.append(ts)
        mean_mc = np.mean(mc_list) if mc_list else np.nan
        std_mc = np.std(mc_list) if len(mc_list) > 1 else 0.0
        mean_ts = np.mean(ts_list) if ts_list else np.nan
        std_ts = np.std(ts_list) if len(ts_list) > 1 else 0.0
        rows.append({
            "density": d,
            "random_mc_mean": mean_mc,
            "random_mc_std": std_mc,
            "random_ts_mean": mean_ts,
            "random_ts_std": std_ts,
        })
    return rows


def compute_gap_vs_figure3(rows: list) -> dict:
    """
    Compare paper Figure 3 Amanous curve with Random baseline.
    Quantify Gap = Amanous_MC − Random_MC in pre-saturation range (10–28 notes/s).
    """
    gaps = []
    gap_at_10 = None
    for r in rows:
        d = int(r["density"]) if r["density"] == int(r["density"]) else r["density"]
        if d not in AMANOUS_MC_FIGURE3 or d not in PRE_SATURATION_DENSITIES:
            continue
        a_mc = AMANOUS_MC_FIGURE3[d]
        r_mc = r["random_mc_mean"]
        if np.isnan(r_mc):
            continue
        g = a_mc - r_mc
        gaps.append(g)
        if d == 10:
            gap_at_10 = g
    return {
        "mean_gap_mc": float(np.mean(gaps)) if gaps else np.nan,
        "gap_mc_at_10": gap_at_10,
        "pre_saturation_densities": PRE_SATURATION_DENSITIES,
        "n_points": len(gaps),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fig 3: random baseline Melodic Coherence (10--200 notes/s)"
    )
    parser.add_argument(
        "--densities",
        type=str,
        default=None,
        help="Comma-separated densities (default: Fig 3 levels)",
    )
    parser.add_argument("--n-events", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--tikz", action="store_true", help="Print TikZ coordinates for random curve")
    parser.add_argument("--gap", action="store_true", help="Gap vs Figure 3 Amanous (pre-saturation 10–28 n/s)")
    args = parser.parse_args()

    densities = FIG3_DENSITIES
    if args.densities:
        densities = [float(x.strip()) for x in args.densities.split(",")]

    rows = run_null_model_sweep(
        densities=densities,
        n_events=args.n_events,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    if args.output:
        fieldnames = ["density", "random_mc_mean", "random_mc_std", "random_ts_mean", "random_ts_std", "random_single_voice_coherence", "random_tonal_stability"]
        # merge compatibility: column names expected by fig3_null_model_merge.py
        out_rows = []
        for r in rows:
            out_rows.append({
                **r,
                "random_single_voice_coherence": r["random_mc_mean"],
                "random_tonal_stability": r["random_ts_mean"],
            })
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote: {args.output}")

    print("\nDensity (notes/s)  Random MC (mean ± std)   Random TS (mean ± std)")
    print("-" * 65)
    for r in rows:
        print(f"  {r['density']:6.1f}           {r['random_mc_mean']:.3f} ± {r['random_mc_std']:.3f}        {r['random_ts_mean']:.3f} ± {r['random_ts_std']:.3f}")

    if args.tikz:
        print("\n% TikZ coordinates for random baseline MC (Fig 3):")
        print("\\addplot[orange, thick, mark=triangle*, mark size=1.2] coordinates {")
        for r in rows:
            print(f"    ({r['density']:.0f},{r['random_mc_mean']:.3f})")
        print("};")

    # Gap vs paper Figure 3 Amanous curve (pre-saturation)
    if args.gap:
        gap_summary = compute_gap_vs_figure3(rows)
        if gap_summary["n_points"] > 0:
            print("\n--- Gap vs Figure 3 Amanous (pre-saturation 10–28 notes/s) ---")
            print(f"  Mean Gap MC (Amanous − Random): {gap_summary['mean_gap_mc']:.4f}")
            print(f"  Gap MC @ 10 n/s:               {gap_summary['gap_mc_at_10']:.4f}")
            print(f"  Pre-saturation densities:      {gap_summary['pre_saturation_densities']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
