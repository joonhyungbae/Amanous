#!/usr/bin/env python3
"""
Merge Amanous coherence_density_results.csv with Random Baseline CSV for Fig 3.

Usage:
  python fig3_null_model_merge.py \\
    supplementary_code/data/csv/coherence_density_results.csv \\
    random_baseline_results.csv \\
    -o fig3_combined.csv

Output columns: density, amanous_coherence, amanous_ts (if present), random_svc, random_ts
"""

import sys
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("amanous_csv", help="Amanous coherence_density_results.csv")
    parser.add_argument("random_csv", help="Random baseline output CSV")
    parser.add_argument("-o", "--output", required=True, help="Output merged CSV")
    args = parser.parse_args()

    # Load Amanous: density, melodic_coherence or harmonic_coherence_v3, (optional tonal stability)
    amanous = {}
    with open(args.amanous_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            d = float(row["density"])
            amanous[d] = {
                "melodic_coherence": row.get("melodic_coherence"),
                "harmonic_coherence_v3": row.get("harmonic_coherence_v3"),
            }

    # Load Random baseline
    random = {}
    with open(args.random_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            d = float(row["density"])
            random[d] = {
                "random_single_voice_coherence": row.get("random_single_voice_coherence"),
                "random_tonal_stability": row.get("random_tonal_stability"),
            }

    densities = sorted(set(amanous) | set(random))
    fieldnames = [
        "density",
        "amanous_melodic_coherence",
        "amanous_harmonic_coherence_v3",
        "random_single_voice_coherence",
        "random_tonal_stability",
    ]
    with open(args.output, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for d in densities:
            a = amanous.get(d, {})
            r = random.get(d, {})
            row = {
                "density": d,
                "amanous_melodic_coherence": a.get("melodic_coherence", ""),
                "amanous_harmonic_coherence_v3": a.get("harmonic_coherence_v3", ""),
                "random_single_voice_coherence": r.get("random_single_voice_coherence", ""),
                "random_tonal_stability": r.get("random_tonal_stability", ""),
            }
            w.writerow(row)
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
