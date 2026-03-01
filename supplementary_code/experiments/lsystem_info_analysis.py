#!/usr/bin/env python3
"""
L-system information-theoretic and complexity analysis (Section 4.1.6 Ablation)

Uses project L-system generator (amanous_composer.generate_lsystem):
1. Generate L-system symbol sequence up to given depth
2. Generate randomly shuffled sequence with same A:B composition
3. For both: Information Rate I(s_t; s_{t+1}), Lempel-Ziv complexity (lower = more compressible = higher self-similarity)
4. Output statistical table showing L-system has (i) higher compressibility and (ii) significantly higher IR (multi-shuffle p-value).

Run: python lsystem_info_analysis.py [--depths 4,5,6,7] [--n-shuffles 1000]
"""

import sys
import os
import argparse
import numpy as np
from collections import Counter

# Project L-system generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
try:
    from amanous_composer import generate_lsystem
except ImportError:
    # Fallback: local implementation of same logic
    def generate_lsystem(axiom: str, rules: dict, iterations: int) -> str:
        current = axiom
        for _ in range(iterations):
            next_string = "".join(rules.get(s, s) for s in current)
            current = next_string
        return current


def shuffled_preserve_composition(sequence: str, rng: np.random.Generator) -> str:
    """Shuffle order only, preserving same A:B composition."""
    chars = list(sequence)
    rng.shuffle(chars)
    return "".join(chars)


def mutual_information_1step(symbols: list) -> float:
    """I(s_t; s_{t+1}) for discrete symbol sequence (Information Rate)."""
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


def lempel_ziv_complexity(sequence: str) -> int:
    """
    Lempel-Ziv (LZ78-style) complexity: number of distinct phrases.
    Lower = higher compressibility (higher self-similarity).
    """
    if not sequence:
        return 0
    n = len(sequence)
    phrases = 0
    i = 0
    while i < n:
        phrases += 1
        j = i + 1
        while j <= n:
            sub = sequence[i:j]
            if sequence[:i].find(sub) >= 0 or len(sub) == 1:
                j += 1
            else:
                break
        i = j - 1 if j > i + 1 else i + 1
    return phrases


def run_analysis(
    depth: int,
    seed: int = 42,
    n_shuffles: int = 1000,
) -> dict:
    """L-system vs Shuffled: single depth, null distribution from n_shuffles shuffles."""
    axiom, rules = "A", {"A": "AB", "B": "A"}
    seq_lsystem = generate_lsystem(axiom, rules, depth)
    symbols_lsystem = list(seq_lsystem)

    ir_lsystem = mutual_information_1step(symbols_lsystem)
    lz_lsystem = lempel_ziv_complexity(seq_lsystem)
    n = len(seq_lsystem)
    n_a, n_b = seq_lsystem.count("A"), seq_lsystem.count("B")

    rng = np.random.default_rng(seed)
    ir_shuffled = []
    lz_shuffled = []
    for _ in range(n_shuffles):
        seq_s = shuffled_preserve_composition(seq_lsystem, rng)
        ir_shuffled.append(mutual_information_1step(list(seq_s)))
        lz_shuffled.append(lempel_ziv_complexity(seq_s))

    ir_shuffled = np.array(ir_shuffled)
    lz_shuffled = np.array(lz_shuffled)

    # One-sided p-value: L-system IR > shuffled → p = P(shuffled >= L-system)
    p_ir = np.mean(ir_shuffled >= ir_lsystem)
    # L-system LZ < shuffled (more compressible) -> p = P(shuffled <= L-system)
    p_lz = np.mean(lz_shuffled <= lz_lsystem)

    return {
        "depth": depth,
        "length": n,
        "n_A": n_a,
        "n_B": n_b,
        "n_shuffles": n_shuffles,
        "IR_Lsystem": ir_lsystem,
        "IR_Shuffled_mean": float(np.mean(ir_shuffled)),
        "IR_Shuffled_std": float(np.std(ir_shuffled)),
        "p_IR": p_ir,
        "LZ_Lsystem": lz_lsystem,
        "LZ_Shuffled_mean": float(np.mean(lz_shuffled)),
        "LZ_Shuffled_std": float(np.std(lz_shuffled)),
        "p_LZ": p_lz,
        "LZ_norm_Lsystem": lz_lsystem / n if n else 0,
        "LZ_norm_Shuffled_mean": float(np.mean(lz_shuffled) / n) if n else 0,
    }


def print_statistical_table(results: list):
    """Statistical table for Section 4.1.6: L-system significantly better in compressibility/dependency."""
    if not results:
        return
    w = 14
    sep = " | "
    print("\n" + "=" * 100)
    print("L-system vs Randomly Shuffled (same A:B composition) — Section 4.1.6 Ablation")
    print("Information Rate I(s_t; s_{{t+1}}), Lempel-Ziv Complexity; p-values from {} shuffles (one-sided).".format(
        results[0].get("n_shuffles", "N")
    ))
    print("=" * 100)
    header = (
        "Depth".ljust(6) + sep +
        "|S|".rjust(8) + sep +
        "A:B".rjust(8) + sep +
        "IR (L-sys)".rjust(w) + sep +
        "IR (Shuffled)".rjust(w) + sep +
        "p (IR)".rjust(10) + sep +
        "LZ (L-sys)".rjust(w) + sep +
        "LZ (Shuffled)".rjust(w) + sep +
        "p (LZ)".rjust(10)
    )
    print(header)
    print("-" * 100)
    for r in results:
        ab = f"{r['n_A']}:{r['n_B']}"
        ir_s = f"{r['IR_Shuffled_mean']:.4f}±{r['IR_Shuffled_std']:.4f}"
        lz_s = f"{r['LZ_Shuffled_mean']:.1f}±{r['LZ_Shuffled_std']:.1f}"
        row = (
            str(r["depth"]).ljust(6) + sep +
            str(r["length"]).rjust(8) + sep +
            ab.rjust(8) + sep +
            f"{r['IR_Lsystem']:.4f}".rjust(w) + sep +
            ir_s.rjust(w) + sep +
            f"{r['p_IR']:.4f}".rjust(10) + sep +
            str(r["LZ_Lsystem"]).rjust(w) + sep +
            lz_s.rjust(w) + sep +
            f"{r['p_LZ']:.4f}".rjust(10)
        )
        print(row)
    print("=" * 100)
    print("Interpretation: Lower p_IR ⇒ L-system has significantly higher sequence dependency (IR).")
    print("                Lower p_LZ ⇒ L-system is significantly more compressible (self-similar).")
    print()


def print_latex_table(results: list):
    """LaTeX table snippet for paper."""
    print("\n--- LaTeX table (Section 4.1.6) ---\n")
    print(r"\begin{table}[htbp]")
    print(r"\caption{L-system vs.\ randomly shuffled sequence (same symbol composition): ")
    print(r"Information Rate $I(s_t; s_{t+1})$ and Lempel-Ziv complexity. ")
    print(r"$p$-values from one-sided permutation test (shuffled null).}")
    print(r"\label{tab:lsystem_ir_lz_ablation}")
    print(r"\begin{tabular}{@{}lrrrrrrr@{}} \toprule")
    print(r"Depth & $|S|$ & A:B & IR (L-sys) & IR (Shuffled) & $p_{\mathrm{IR}}$ & LZ (L-sys) & LZ (Shuffled) & $p_{\mathrm{LZ}}$ \\ \midrule")
    for r in results:
        ir_s = "{:.4f} $\\pm$ {:.4f}".format(r["IR_Shuffled_mean"], r["IR_Shuffled_std"])
        lz_s = "{:.1f} $\\pm$ {:.1f}".format(r["LZ_Shuffled_mean"], r["LZ_Shuffled_std"])
        print(rf"{r['depth']} & {r['length']} & {r['n_A']}:{r['n_B']} & {r['IR_Lsystem']:.4f} & {ir_s} & {r['p_IR']:.4f} & {r['LZ_Lsystem']} & {lz_s} & {r['p_LZ']:.4f} \\")
    print(r"\\ \bottomrule \end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="L-system Information Rate & LZ complexity vs Shuffled (Section 4.1.6)"
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="4,5,6,7",
        help="Comma-separated L-system depths (default: 4,5,6,7)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=1000,
        help="Number of shuffled null samples for p-value (default: 1000)",
    )
    args = parser.parse_args()

    depths = [int(x.strip()) for x in args.depths.split(",")]
    results = [
        run_analysis(d, seed=args.seed, n_shuffles=args.n_shuffles)
        for d in depths
    ]

    print_statistical_table(results)
    print_latex_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
