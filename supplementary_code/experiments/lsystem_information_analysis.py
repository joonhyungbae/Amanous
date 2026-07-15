#!/usr/bin/env python3
"""
Layer 1 L-system output sequence (S) information-theoretic and complexity analysis

1. Prepare actual generated symbol sequence and a randomly shuffled sequence with same composition.
2. Compute Information Rate I(X_t; X_{t-1}) for both.
3. Compute Lempel-Ziv complexity for both -> compare compressibility and self-similarity.
4. Output results as tables per depth 4, 6, 7.

Default run: python lsystem_information_analysis.py
Other depths: python lsystem_information_analysis.py --depths 4,5,6,7
"""

import sys
import argparse
import numpy as np
from collections import Counter


def lsystem_expand(axiom: str, rules: dict, depth: int) -> str:
    """L-system expansion: axiom, rules (e.g. A->AB, B->A), depth."""
    current = axiom
    for _ in range(depth):
        current = "".join(rules.get(s, s) for s in current)
    return current


def shuffled_preserve_composition(sequence: str, rng: np.random.Generator) -> str:
    """Preserve same composition ratio, shuffle order only."""
    chars = list(sequence)
    rng.shuffle(chars)
    return "".join(chars)


def mutual_information_1step(symbols: list) -> float:
    """I(X_t; X_{t-1}) for discrete symbol sequence (e.g. 'A','B')."""
    if len(symbols) < 2:
        return 0.0
    joint = Counter(zip(symbols[:-1], symbols[1:]))
    margin_prev = Counter(symbols[:-1])
    margin_curr = Counter(symbols[1:])
    n = len(symbols) - 1
    mi = 0.0
    for (a, b), count in joint.items():
        p_ab = count / n
        p_a = margin_prev[a] / n
        p_b = margin_curr[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * np.log2(p_ab / (p_a * p_b))
    return mi


def entropy(symbols: list) -> float:
    """Shannon entropy H(X)."""
    if not symbols:
        return 0.0
    c = Counter(symbols)
    n = len(symbols)
    return -sum((k / n) * np.log2(k / n) for k in c.values() if k > 0)


def information_rate(symbols: list, order: int = 1) -> float:
    """IR = I(X_t; X_{t-1}) when order=1."""
    if len(symbols) <= order:
        return 0.0
    if order == 1:
        return mutual_information_1step(symbols)
    past = [tuple(symbols[i : i + order]) for i in range(len(symbols) - order)]
    curr = symbols[order:]
    joint = Counter(zip(past, curr))
    margin_past = Counter(past)
    margin_curr = Counter(curr)
    n = len(curr)
    mi = 0.0
    for (p, c), count in joint.items():
        p_j = count / n
        p_p = margin_past[p] / n
        p_c = margin_curr[c] / n
        if p_j > 0 and p_p > 0 and p_c > 0:
            mi += p_j * np.log2(p_j / (p_p * p_c))
    return mi


def lempel_ziv_complexity(sequence: str) -> int:
    """
    Lempel-Ziv (LZ78-style) complexity: number of distinct phrases.
    Lower = more compressible / self-similar.
    """
    if not sequence:
        return 0
    n = len(sequence)
    phrases = 0
    i = 0
    while i < n:
        phrases += 1
        # Find longest match of prefix seen so far
        j = i + 1
        while j <= n:
            sub = sequence[i:j]
            if sequence[:i].find(sub) >= 0 or len(sub) == 1:
                j += 1
            else:
                break
        i = j - 1 if j > i + 1 else i + 1
    return phrases


def normalized_lz_complexity(sequence: str) -> float:
    """LZ complexity / length. Normalized toward 0–1 (higher = less compressible)."""
    n = len(sequence)
    if n == 0:
        return 0.0
    c = lempel_ziv_complexity(sequence)
    # Upper bound: theoretical by alphabet size
    return c / n


def run_analysis(depth: int, seed: int = 42) -> dict:
    """Run L-system vs Shuffled analysis."""
    axiom, rules = "A", {"A": "AB", "B": "A"}
    seq_a = lsystem_expand(axiom, rules, depth)
    rng = np.random.default_rng(seed)
    seq_b = shuffled_preserve_composition(seq_a, rng)

    symbols_a = list(seq_a)
    symbols_b = list(seq_b)

    ir_a = information_rate(symbols_a, order=1)
    ir_b = information_rate(symbols_b, order=1)
    h_a = entropy(symbols_a)
    h_b = entropy(symbols_b)
    lz_a = lempel_ziv_complexity(seq_a)
    lz_b = lempel_ziv_complexity(seq_b)
    n = len(seq_a)
    n_lz_a = lz_a / n if n else 0
    n_lz_b = lz_b / n if n else 0

    return {
        "depth": depth,
        "length": n,
        "n_A": seq_a.count("A"),
        "n_B": seq_a.count("B"),
        "IR_Lsystem": ir_a,
        "IR_Shuffled": ir_b,
        "H_Lsystem": h_a,
        "H_Shuffled": h_b,
        "LZ_phrases_Lsystem": lz_a,
        "LZ_phrases_Shuffled": lz_b,
        "LZ_normalized_Lsystem": n_lz_a,
        "LZ_normalized_Shuffled": n_lz_b,
    }


def print_table(results: list):
    """Print results as table."""
    if not results:
        return
    r0 = results[0]
    print("\n" + "=" * 80)
    print("L-system information-theoretic analysis (Sequence A: L-system, Sequence B: Shuffled)")
    print("=" * 80)
    print(f"{'Metric':<35} {'L-system (A)':>18} {'Shuffled (B)':>18} {'Difference (A-B)':>18}")
    print("-" * 80)

    for r in results:
        print(f"  [Depth {r['depth']}, length={r['length']}]")
        print(f"  {'Information Rate I(X_t; X_{t-1})':<33} {r['IR_Lsystem']:>18.4f} {r['IR_Shuffled']:>18.4f} {r['IR_Lsystem']-r['IR_Shuffled']:>18.4f}")
        print(f"  {'Shannon Entropy H(X)':<33} {r['H_Lsystem']:>18.4f} {r['H_Shuffled']:>18.4f} {r['H_Lsystem']-r['H_Shuffled']:>18.4f}")
        print(f"  {'LZ complexity (phrases)':<33} {r['LZ_phrases_Lsystem']:>18d} {r['LZ_phrases_Shuffled']:>18d} {r['LZ_phrases_Lsystem']-r['LZ_phrases_Shuffled']:>18d}")
        print(f"  {'LZ normalized (phrases/length)':<33} {r['LZ_normalized_Lsystem']:>18.4f} {r['LZ_normalized_Shuffled']:>18.4f} {r['LZ_normalized_Lsystem']-r['LZ_normalized_Shuffled']:>18.4f}")
        print("-" * 80)
    print()


def print_depth_table(results: list):
    """Print summary table per depth 4, 6, 7 (requested format)."""
    if not results:
        return
    # Header
    col_w = 14
    sep = " | "
    header = (
        "Depth".ljust(6) + sep +
        "|S|".rjust(col_w) + sep +
        "IR (L-system)".rjust(col_w) + sep +
        "IR (Shuffled)".rjust(col_w) + sep +
        "LZ (L-system)".rjust(col_w) + sep +
        "LZ (Shuffled)".rjust(col_w)
    )
    print("\n" + "=" * len(header))
    print("Layer 1 L-system vs random shuffle: Information Rate I(X_t; X_{t-1}) and LZ complexity")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            str(r["depth"]).ljust(6) + sep +
            str(r["length"]).rjust(col_w) + sep +
            f"{r['IR_Lsystem']:.4f}".rjust(col_w) + sep +
            f"{r['IR_Shuffled']:.4f}".rjust(col_w) + sep +
            str(r["LZ_phrases_Lsystem"]).rjust(col_w) + sep +
            str(r["LZ_phrases_Shuffled"]).rjust(col_w)
        )
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="L-system IR & LZ complexity vs Shuffled")
    parser.add_argument("--depths", type=str, default="4,6,7", help="Comma-separated L-system depths (default: 4,6,7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Print per-depth metric details")
    args = parser.parse_args()

    depths = [int(x.strip()) for x in args.depths.split(",")]
    results = [run_analysis(d, seed=args.seed) for d in depths]

    # Per-depth table (default)
    print_depth_table(results)

    if args.verbose:
        print_table(results)

    # LaTeX table snippet
    print("\n--- LaTeX table snippet ---\n")
    print(r"\begin{table}[htbp]")
    print(r"\caption{Information-theoretic comparison: L-system vs.\ Shuffled (same composition).}")
    print(r"\label{tab:lsystem_ir_lz}")
    print(r"\begin{tabular}{@{}lrrrrr@{}} \toprule")
    print(r"Depth & $|\Sigma|$ & IR (L-system) & IR (Shuffled) & LZ (L-system) & LZ (Shuffled) \\ \midrule")
    for r in results:
        print(rf"{r['depth']} & {r['length']} & {r['IR_Lsystem']:.4f} & {r['IR_Shuffled']:.4f} & {r['LZ_phrases_Lsystem']} & {r['LZ_phrases_Shuffled']} \\")
    print(r"\\ \bottomrule \end{tabular}")
    print(r"\end{table}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
