#!/usr/bin/env python3
"""
Piecewise linear regression with breakpoint estimation and 95% bootstrap CI.
Uses only numpy and scipy.
"""

import numpy as np
from scipy import optimize

# Data: 14 (density, coherence) pairs
density = np.array([
    10, 15, 20, 25, 28, 30,  # pre-breakpoint
    40, 50, 60, 80, 100, 120, 150, 200
])
coherence = np.array([
    1.00, 0.92, 0.78, 0.55, 0.38, 0.25,  # pre-breakpoint
    0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12
])
n = len(density)


def fit_segment(x, y):
    """OLS slope and intercept for y ~ x. Returns (intercept, slope), rss."""
    if len(x) < 2:
        return np.nan, np.nan, np.inf
    X = np.column_stack([np.ones_like(x), x])
    beta, rss, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta[0], beta[1], (rss[0] if np.size(rss) else np.sum((y - X @ beta) ** 2))


def rss_piecewise(bp, x, y):
    """Total RSS for piecewise linear with breakpoint bp. Left: x < bp, Right: x >= bp."""
    left = x < bp
    right = x >= bp
    if np.sum(left) < 2 or np.sum(right) < 2:
        return 1e30  # finite penalty for scipy
    _, _, rss_left = fit_segment(x[left], y[left])
    _, _, rss_right = fit_segment(x[right], y[right])
    return rss_left + rss_right


def fit_piecewise(x, y, bp):
    """Given breakpoint bp, return (a1, b1, a2, b2, rss_total)."""
    left = x < bp
    right = x >= bp
    a1, b1, rss_left = fit_segment(x[left], y[left])
    a2, b2, rss_right = fit_segment(x[right], y[right])
    return a1, b1, a2, b2, rss_left + rss_right


def estimate_breakpoint(x, y, bp_min=15, bp_max=50):
    """Find breakpoint minimising RSS via scipy (fast)."""
    res = optimize.minimize_scalar(
        rss_piecewise, bounds=(bp_min, bp_max), method="bounded", args=(x, y)
    )
    return res.x


def r2_simple_linear(x, y):
    """R² for simple linear regression y ~ x."""
    _, _, rss = fit_segment(x, y)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - rss / ss_tot if ss_tot > 0 else 0


def main():
    # 1) Best-fit breakpoint (grid search 15--50, step 0.1)
    bp_best = estimate_breakpoint(density, coherence)
    a1, b1, a2, b2, rss_pw = fit_piecewise(density, coherence, bp_best)

    # R² piecewise
    ss_tot = np.sum((coherence - np.mean(coherence)) ** 2)
    r2_piecewise = 1 - rss_pw / ss_tot
    r2_linear = r2_simple_linear(density, coherence)

    # 2) Bootstrap 95% CI for breakpoint
    B = 10000
    np.random.seed(42)
    bp_boot = np.zeros(B)
    for i in range(B):
        idx = np.random.choice(n, size=n, replace=True)
        x_b, y_b = density[idx], coherence[idx]
        bp_boot[i] = estimate_breakpoint(x_b, y_b)
    ci_low = np.percentile(bp_boot, 2.5)
    ci_high = np.percentile(bp_boot, 97.5)

    # 3) Slope ratio (pre / post; both negative, ratio as stated "XX×")
    slope_pre = b1
    slope_post = b2
    ratio = slope_pre / slope_post if slope_post != 0 else np.nan

    # 4) Summary
    print("--- Piecewise linear breakpoint analysis ---")
    print(f"Best-fit breakpoint:     {bp_best:.1f} notes/s")
    print(f"95% bootstrap CI:       [{ci_low:.1f}, {ci_high:.1f}] notes/s")
    print(f"Pre-breakpoint slope:   {slope_pre:.4f}")
    print(f"Post-breakpoint slope:  {slope_post:.4f}")
    print(f"Slope ratio (pre/post): {ratio:.2f}×")
    print(f"R² piecewise:           {r2_piecewise:.3f}")
    print(f"R² simple linear:       {r2_linear:.3f}")
    print()
    print("--- LaTeX-ready summary ---")
    print(f"Breakpoint: {bp_best:.1f} notes/s [95\\% CI: {ci_low:.1f}--{ci_high:.1f}]; slope ratio: {ratio:.2f}×; R²_piecewise = {r2_piecewise:.3f}")


if __name__ == "__main__":
    main()
