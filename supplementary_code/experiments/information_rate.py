#!/usr/bin/env python3
"""
Information Rate (IR) analysis following Lattner et al. / JCMS-style predictability.

IR quantifies how much the past reduces uncertainty about the present (mutual information
between consecutive or short-horizon symbolic states). We compute IR on event sequences
(pitch class and/or IOI bin) and compare Section A (deterministic) vs Section B (textural):
expect higher IR in A (more predictable structure).

Reference: Information dynamics in music; Lattner et al. (JCMS/ISMIR context).
"""

import sys
import os
import argparse
import numpy as np
from collections import Counter, defaultdict


def events_to_symbol_sequence(events, use_ioi: bool = False, ioi_bins: int = 10):
    """
    Convert event list to discrete symbol sequence for IR.
    Symbol = pitch class (0--11) or (pitch_class, ioi_bin) if use_ioi.
    """
    events = sorted(events, key=lambda e: e["onset_time"])
    if not events:
        return [], []
    times = [e["onset_time"] for e in events]
    pcs = [e["pitch"] % 12 for e in events]
    if not use_ioi or len(times) < 2:
        return pcs, None
    iois = np.diff(times)
    ioi_min, ioi_max = max(1e-6, iois.min()), max(iois.max(), 1)
    bins = np.linspace(ioi_min, ioi_max, ioi_bins + 1)
    ioi_bin_idx = np.digitize(iois, bins) - 1
    ioi_bin_idx = np.clip(ioi_bin_idx, 0, ioi_bins - 1)
    symbols = [(pcs[i], int(ioi_bin_idx[i])) for i in range(len(iois))]
    return symbols, (pcs, ioi_bin_idx)


def mutual_information_1step(symbols):
    """
    I(X_t; X_{t-1}) for discrete symbol sequence.
    symbols: list of hashable (e.g. int 0--11 or tuple (pc, ioi_bin))
    """
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


def entropy(symbols):
    """Shannon entropy of symbol distribution."""
    if not symbols:
        return 0.0
    c = Counter(symbols)
    n = len(symbols)
    return -sum((k / n) * np.log2(k / n) for k in c.values() if k > 0)


def information_rate(symbols, order: int = 1):
    """
    Information Rate as mean mutual information between past (order symbols) and present.
    order=1: IR = I(X_t; X_{t-1}).
    """
    if len(symbols) <= order:
        return 0.0
    if order == 1:
        return mutual_information_1step(symbols)
    # Higher order: past = (X_{t-order}, ..., X_{t-1})
    past_symbols = [tuple(symbols[i : i + order]) for i in range(len(symbols) - order)]
    curr_symbols = symbols[order:]
    joint = Counter(zip(past_symbols, curr_symbols))
    margin_past = Counter(past_symbols)
    margin_curr = Counter(curr_symbols)
    n = len(curr_symbols)
    mi = 0.0
    for (past, curr), count in joint.items():
        p_j = count / n
        p_p = margin_past[past] / n
        p_c = margin_curr[curr] / n
        if p_j > 0 and p_p > 0 and p_c > 0:
            mi += p_j * np.log2(p_j / (p_p * p_c))
    return mi


def load_events_from_csv(path: str):
    import csv
    events = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row.get("onset_time", row.get("time", 0)))
            p = int(row.get("pitch", 60))
            events.append({"onset_time": t, "pitch": p})
    return events


def main():
    parser = argparse.ArgumentParser(description="Information Rate: A (deterministic) vs B (textural)")
    parser.add_argument("section_a_csv", nargs="?", help="CSV events for Section A (deterministic)")
    parser.add_argument("section_b_csv", nargs="?", help="CSV events for Section B (textural)")
    parser.add_argument("--order", type=int, default=1, help="IR order (past context length)")
    parser.add_argument("--use-ioi", action="store_true", help="Include IOI bin in symbol")
    parser.add_argument("--ioi-bins", type=int, default=10)
    args = parser.parse_args()

    if args.section_a_csv and args.section_b_csv:
        events_a = load_events_from_csv(args.section_a_csv)
        events_b = load_events_from_csv(args.section_b_csv)
    else:
        # Synthetic demo: A = repetitive, B = random
        rng = np.random.default_rng(42)
        n_a, n_b = 500, 500
        # A: scale-like, low entropy
        events_a = [{"onset_time": 0.2 * i, "pitch": 60 + (i % 7)} for i in range(n_a)]
        # B: random pitch
        events_b = [{"onset_time": np.cumsum(rng.exponential(0.03, n_b))[i], "pitch": int(rng.integers(40, 80))} for i in range(n_b)]
        print("No CSV provided; using synthetic A (scale) vs B (random).")

    sym_a, _ = events_to_symbol_sequence(events_a, use_ioi=args.use_ioi, ioi_bins=args.ioi_bins)
    sym_b, _ = events_to_symbol_sequence(events_b, use_ioi=args.use_ioi, ioi_bins=args.ioi_bins)

    if isinstance(sym_a[0], tuple):
        pass
    else:
        sym_a = list(sym_a)
        sym_b = list(sym_b)

    ir_a = information_rate(sym_a, order=args.order)
    ir_b = information_rate(sym_b, order=args.order)
    h_a = entropy(sym_a)
    h_b = entropy(sym_b)

    print("Information Rate (order={})".format(args.order))
    print("  Section A (deterministic): IR = {:.4f}, H = {:.4f}".format(ir_a, h_a))
    print("  Section B (textural):     IR = {:.4f}, H = {:.4f}".format(ir_b, h_b))
    print("  Difference (A - B):       IR = {:.4f} (expected > 0 for more predictable A)".format(ir_a - ir_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
