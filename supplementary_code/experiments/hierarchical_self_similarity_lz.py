#!/usr/bin/env python3
"""
Hierarchical self-similarity analysis: MIDI with recursion-depth weighting vs symbol-only sequence.

- Compare the Layer 1+2 pipeline (IOI/Pitch distribution by expand_lsystem generation depth) with
  MIDI generated from the same symbol sequence but without depth weighting.
- Discretize both event sequences (IOI bins, pitch class) and compare via Lempel-Ziv complexity (LZ phrase count).
- With depth weighting, hierarchical structure is stronger so LZ can be lower (more compressible = higher self-similarity).

Run: python hierarchical_self_similarity_lz.py [--seed 42] [--out-dir .]
"""

import argparse
import os
import sys

import numpy as np

# Path so amanous_composer is importable (code dir under repo root)
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from amanous_composer import (
    compose,
    get_canonical_config,
    generate_lsystem,
)


def lempel_ziv_phrase_count(s: str) -> int:
    """LZ78-style phrase count; lower = more compressible / more self-similar."""
    if not s:
        return 0
    n = len(s)
    phrases = 0
    i = 0
    while i < n:
        phrases += 1
        j = i + 1
        while j <= n:
            sub = s[i:j]
            if s[:i].find(sub) >= 0 or len(sub) == 1:
                j += 1
            else:
                break
        i = j - 1 if j > i + 1 else i + 1
    return phrases


def events_to_discrete_string(events, time_key="onset_time", n_ioi_bins=4):
    """
    Convert event list (sorted by time) to a discrete string for LZ.
    Each event → (IOI bin, pitch_class); encode as single char 0-9A-Za-z for up to 62 symbols.
    Uses onset_time for reproducibility (no hardware comp).
    """
    if not events:
        return ""
    sorted_events = sorted(events, key=lambda e: e.get(time_key, e.get("onset_time", 0)))
    onsets = [e.get(time_key, e.get("onset_time", 0)) for e in sorted_events]
    pitches = [e.get("pitch", 60) for e in sorted_events]
    # IOIs in seconds; quantize to n_ioi_bins (log-spaced or linear)
    iois = np.diff(onsets)
    if len(iois) == 0:
        iois = np.array([0.2])
    ioi_min, ioi_max = np.percentile(iois[iois > 0] if np.any(iois > 0) else [0.01, 1.0], [2, 98])
    if ioi_max <= ioi_min:
        ioi_max = ioi_min + 0.01
    ioi_bins = np.clip(
        np.floor(n_ioi_bins * (np.log1p(iois) - np.log1p(ioi_min)) / (np.log1p(ioi_max) - np.log1p(ioi_min))),
        0,
        n_ioi_bins - 1,
    ).astype(int)
    pitch_classes = (np.array(pitches[1:]) % 12).astype(int)  # first event has no IOI
    # Encode (ioi_bin, pc) -> char: 0..n_ioi_bins*12-1, map to 0-9 A-Z a-z
    n_syms = n_ioi_bins * 12
    indices = ioi_bins * 12 + pitch_classes
    indices = np.clip(indices, 0, n_syms - 1)
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[: n_syms]
    return "".join(charset[i] for i in indices)


def run_comparison(seed: int = 42, out_dir: str = "."):
    config = get_canonical_config()
    config.seed = seed
    sequence = generate_lsystem(config.axiom, config.production_rules, config.iterations)

    # 1) Depth-weighted: normal compose (uses expand_lsystem + apply_depth_weight_to_config)
    events_depth, seq1, _ = compose(config, lsystem_sequence_override=None, apply_hw_compensation=False)
    # 2) Symbol-only (no depth weight): override with same sequence → all generation=0
    events_flat, seq2, _ = compose(config, lsystem_sequence_override=sequence, apply_hw_compensation=False)

    assert seq1 == seq2 == sequence, "Sequence mismatch"
    # Use onset_time for discrete string (no trigger_time)
    time_key = "onset_time"

    s_depth = events_to_discrete_string(events_depth, time_key=time_key)
    s_flat = events_to_discrete_string(events_flat, time_key=time_key)

    lz_depth = lempel_ziv_phrase_count(s_depth)
    lz_flat = lempel_ziv_phrase_count(s_flat)
    n_depth = len(s_depth)
    n_flat = len(s_flat)

    results = {
        "seed": seed,
        "sequence": sequence,
        "n_events_depth": len(events_depth),
        "n_events_flat": len(events_flat),
        "n_chars_depth": n_depth,
        "n_chars_flat": n_flat,
        "LZ_depth_weighted": lz_depth,
        "LZ_symbol_only": lz_flat,
        "LZ_norm_depth": lz_depth / max(1, n_depth),
        "LZ_norm_flat": lz_flat / max(1, n_flat),
    }
    return results, events_depth, events_flat


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical self-similarity: LZ complexity of depth-weighted vs symbol-only MIDI"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".", help="Directory for CSV/optional MIDI")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results, ev_depth, ev_flat = run_comparison(seed=args.seed, out_dir=args.out_dir)

    print("=" * 60)
    print("Hierarchical self-similarity: Depth-weighted vs Symbol-only")
    print("=" * 60)
    print(f"Seed: {results['seed']}")
    print(f"L-system sequence: {results['sequence']}")
    print(f"Events (depth-weighted / symbol-only): {results['n_events_depth']} / {results['n_events_flat']}")
    print(f"Discrete string length: {results['n_chars_depth']} / {results['n_chars_flat']}")
    print()
    print("Lempel-Ziv phrase count (lower = more compressible / more self-similar):")
    print(f"  Depth-weighted (recursion depth → IOI λ, pitch range): LZ = {results['LZ_depth_weighted']}")
    print(f"  Symbol-only (same symbol order, no depth weight):     LZ = {results['LZ_symbol_only']}")
    print(f"  Normalized (phrases/length): depth {results['LZ_norm_depth']:.4f}  vs  flat {results['LZ_norm_flat']:.4f}")
    # Compare normalized LZ (same-length scale); depth-weighted has more events (denser IOI in deep sections)
    if results["LZ_norm_depth"] < results["LZ_norm_flat"]:
        print("  → Normalized LZ: depth-weighted is lower (higher hierarchical self-similarity per event).")
    else:
        print("  → Normalized LZ: run multiple seeds for statistical comparison.")
    print("=" * 60)

    csv_path = os.path.join(args.out_dir, "hierarchical_self_similarity_lz.csv")
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            if k != "sequence" and isinstance(v, (int, float)):
                f.write(f"{k},{v}\n")
        f.write(f"sequence,{results['sequence']}\n")
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
