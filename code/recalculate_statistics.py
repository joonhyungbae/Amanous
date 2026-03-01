#!/usr/bin/env python3
"""
Recalculation script: compute missing statistics for cmj.tex paper.

Item 1: Uncalibrated pre-comp. mean absolute error
Item 2: Uncalibrated + filter mean absolute error verification
Item 3: CP-related 5 test p-values (N=60/30 windows)
Item 4: Melodic breakpoint U and r (N=14)
"""

import numpy as np
import pandas as pd
from scipy import stats

# Base path for extracted code outputs. If your folder names differ, adjust these.
CODE_EXTRACTED = "/home/jhbae/Amanous/code_extracted"
DIR_CALIBRATED_PRECOMP = "r02_n006_calibrated_precomp"
DIR_ROBUSTNESS_FILTER = "r02_n005_robustness_filter"
DIR_DISCRETE_SWITCH = "r04_n002_discrete_parameter_switch"
DIR_MELODIC_BREAKPOINT = "r03_n001_melodic_texture_breakpoint"

print("=" * 70)
print("CMJ.TEX recalculated results")
print("=" * 70)

# =============================================================================
# Items 1 & 2: Hardware Compensation (Table 7)
# =============================================================================
print("\n" + "=" * 70)
print("Items 1 & 2: Hardware Compensation (Table 7)")
print("=" * 70)

# Load note_by_note_error_analysis.csv
df_errors = pd.read_csv(f"{CODE_EXTRACTED}/{DIR_CALIBRATED_PRECOMP}/note_by_note_error_analysis.csv")

# Absolute error by condition:
# distorted_error: uncorrected
# theoretical_error: uncalibrated pre-comp (linear model)
# calibrated_only_error: calibrated pre-comp
# corrected_error: calibrated + filter

n_events = len(df_errors)
print(f"\nTotal events N = {n_events}")

# Uncorrected stats
uncorrected_abs_errors = np.abs(df_errors['distorted_error'])
uncorrected_mean = uncorrected_abs_errors.mean()
uncorrected_sd = uncorrected_abs_errors.std()
print(f"\n[Uncorrected]")
print(f"  Mean absolute error: {uncorrected_mean:.2f} ms")
print(f"  SD: {uncorrected_sd:.2f} ms")

# Item 1: Uncalibrated pre-comp (linear model)
uncal_precomp_abs_errors = np.abs(df_errors['theoretical_error'])
uncal_precomp_mean = uncal_precomp_abs_errors.mean()
uncal_precomp_sd = uncal_precomp_abs_errors.std()
reduction_uncal = (1 - uncal_precomp_mean / uncorrected_mean) * 100

print(f"\n[Uncalibrated pre-comp. (Item 1)]")
print(f"  Mean absolute error: {uncal_precomp_mean:.2f} ms")
print(f"  SD: {uncal_precomp_sd:.2f} ms")
print(f"  vs. Uncorrected: -{reduction_uncal:.1f}%")
print(f"  >>> For Table 7: ${uncal_precomp_mean:.2f} \\pm {uncal_precomp_sd:.2f}$ | $-{reduction_uncal:.1f}\\%$")

# Calibrated pre-comp
cal_precomp_abs_errors = np.abs(df_errors['calibrated_only_error'])
cal_precomp_mean = cal_precomp_abs_errors.mean()
cal_precomp_sd = cal_precomp_abs_errors.std()
reduction_cal = (1 - cal_precomp_mean / uncorrected_mean) * 100

print(f"\n[Calibrated pre-comp.]")
print(f"  Mean absolute error: {cal_precomp_mean:.2f} ms")
print(f"  SD: {cal_precomp_sd:.2f} ms")
print(f"  vs. Uncorrected: -{reduction_cal:.1f}%")

# Calibrated + filter
cal_filter_abs_errors = np.abs(df_errors['corrected_error'])
cal_filter_mean = cal_filter_abs_errors.mean()
cal_filter_sd = cal_filter_abs_errors.std()
reduction_cal_filter = (1 - cal_filter_mean / uncorrected_mean) * 100

print(f"\n[Calibrated + filter]")
print(f"  Mean absolute error: {cal_filter_mean:.2f} ms")
print(f"  SD: {cal_filter_sd:.2f} ms")
print(f"  vs. Uncorrected: -{reduction_cal_filter:.1f}%")

# Item 2: Uncalibrated + filter
# Linear model then robustness filter; compare with latency_filter_effectiveness_comparison.csv
print("\n[Item 2: Table 6 vs Table 7 comparison]")
print(f"  Table 6 'Before' SD (error SD): 4.244 ms")
print(f"  Table 6 'After' SD (filtered): 2.870 ms")

df_filter = pd.read_csv(f"{CODE_EXTRACTED}/{DIR_ROBUSTNESS_FILTER}/latency_filter_effectiveness_comparison.csv")
print(f"\n  Filter effectiveness data:")
print(df_filter.to_string(index=False))

# =============================================================================
# Item 3: CP-related test p-value recalculation
# =============================================================================
print("\n" + "=" * 70)
print("Item 3: CP-related test p-value recalculation (N=60/30 windows)")
print("=" * 70)

df_cp = pd.read_csv(f"{CODE_EXTRACTED}/{DIR_DISCRETE_SWITCH}/statistical_summary.csv")
print("\nCP Statistical Summary:")
print(df_cp.to_string(index=False))

# Mann-Whitney U p-value (N=30 vs N=30 windows), U=0 case (complete separation)
n1, n2 = 30, 30

from scipy.stats import mannwhitneyu

# Simulate complete separation
group1 = np.arange(30)
group2 = np.arange(30, 60)

stat, pval = mannwhitneyu(group1, group2, alternative='two-sided')
print(f"\n[CP density/TS switch: complete separation (U=0)]")
print(f"  N per group: {n1}")
print(f"  Mann-Whitney U: {stat}")
print(f"  p-value: {pval:.2e}")

# Pearson correlation p-value (N=60 windows), r = 0.907
n_continuous = 60
r_continuous = 0.907

t_stat = r_continuous * np.sqrt(n_continuous - 2) / np.sqrt(1 - r_continuous**2)
p_continuous = 2 * (1 - stats.t.cdf(abs(t_stat), n_continuous - 2))

print(f"\n[Continuous tracking: Pearson r = {r_continuous}]")
print(f"  N (windows): {n_continuous}")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_continuous:.2e}")

# Pre-CP symmetry (N=30)
n_half = 30
r_pre = 0.888
t_pre = r_pre * np.sqrt(n_half - 2) / np.sqrt(1 - r_pre**2)
p_pre = 2 * (1 - stats.t.cdf(abs(t_pre), n_half - 2))

print(f"\n[Pre-CP symmetry: Pearson r = {r_pre}]")
print(f"  N (windows): {n_half}")
print(f"  t-statistic: {t_pre:.2f}")
print(f"  p-value: {p_pre:.2e}")

# Post-CP symmetry (N=30)
r_post = 0.933
t_post = r_post * np.sqrt(n_half - 2) / np.sqrt(1 - r_post**2)
p_post = 2 * (1 - stats.t.cdf(abs(t_post), n_half - 2))

print(f"\n[Post-CP symmetry: Pearson r = {r_post}]")
print(f"  N (windows): {n_half}")
print(f"  t-statistic: {t_post:.2f}")
print(f"  p-value: {p_post:.2e}")

# =============================================================================
# Item 4: Melodic Breakpoint U and r
# =============================================================================
print("\n" + "=" * 70)
print("Item 4: Melodic Breakpoint U and r (N=14 density levels)")
print("=" * 70)

df_coherence = pd.read_csv(f"{CODE_EXTRACTED}/{DIR_MELODIC_BREAKPOINT}/coherence_density_results.csv")
print("\nCoherence by density level:")
print(df_coherence[['density', 'melodic_coherence', 'harmonic_coherence']].to_string(index=False))

# Breakpoint at 30 notes/s
breakpoint = 30
pre_breakpoint = df_coherence[df_coherence['density'] <= breakpoint]
post_breakpoint = df_coherence[df_coherence['density'] > breakpoint]

print(f"\nBreakpoint: {breakpoint} notes/s")
print(f"Pre-breakpoint densities: {list(pre_breakpoint['density'].values)}")
print(f"Post-breakpoint densities: {list(post_breakpoint['density'].values)}")

# Use harmonic_coherence_v3 (0-1) for Mann-Whitney U
pre_values = pre_breakpoint['harmonic_coherence_v3'].values
post_values = post_breakpoint['harmonic_coherence_v3'].values

print(f"\nPre-breakpoint coherence values: {pre_values}")
print(f"Post-breakpoint coherence values: {post_values}")

n1_bp = len(pre_values)
n2_bp = len(post_values)

stat_bp, pval_bp = mannwhitneyu(pre_values, post_values, alternative='two-sided')
r_bp = 1 - 2 * stat_bp / (n1_bp * n2_bp)

print(f"\n[Melodic Breakpoint Mann-Whitney U]")
print(f"  N pre-breakpoint: {n1_bp}")
print(f"  N post-breakpoint: {n2_bp}")
print(f"  Mann-Whitney U: {stat_bp}")
print(f"  p-value: {pval_bp:.6f}")
print(f"  Rank-biserial r: {r_bp:.3f}")

if stat_bp <= 3:
    u_check = 1.0
    r_check = 1 - 2 * u_check / (n1_bp * n2_bp)
    print(f"\n[Check] For U={u_check}, r = {r_check:.3f}")

# =============================================================================
# Summary for table updates
# =============================================================================
print("\n" + "=" * 70)
print("Summary: values for cmj.tex update")
print("=" * 70)

print("""
### Table 7 (tab:calibration) update:
""")
print(f"Uncalibrated pre-comp.: ${uncal_precomp_mean:.2f} \\pm {uncal_precomp_sd:.2f}$ | $-{reduction_uncal:.1f}\\%$")
print(f"  (current placeholder: $X.XX \\pm 4.24$ | $-YY.Y\\%$)")

print("""
### Appendix A (tab:stats_full) update:
""")
print(f"CP density switch: N=60, U=0, p={pval:.2e}, r=1.00")
print(f"CP TS switch: N=60, U=900, p={pval:.2e}, r=-1.00")
print(f"Continuous tracking: N=60, r=0.907, p={p_continuous:.2e}")
print(f"Pre-CP symmetry: N=30, r=0.888, p={p_pre:.2e}")
print(f"Post-CP symmetry: N=30, r=0.933, p={p_post:.2e}")
print(f"\nMelodic breakpoint: N=14, U={stat_bp:.1f}, p={pval_bp:.6f}, r={r_bp:.3f}")
