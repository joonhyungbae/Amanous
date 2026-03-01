"""
nwVSS (range-normalised wVSS) analysis for paper revision.

Computes:
1. nwVSS weights: same effect-size procedure as wVSS but on range-normalised
   components (W_pitch/127, W_vel/1023, W_temporal/R_temporal).
2. R_temporal: theoretical log-IOI range (ln(IOI_max) - ln(IOI_min)); default
   from IOI in [0.001, 10] s => ln(10/0.001) ≈ 9.21.
3. Split-half cross-validation for nwVSS weights (split pairwise rows by half).

Inputs: vss_component_analysis_results.csv, vss_wvss_pairwise_data.csv
Output: printed table and numbers for paper.tex.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# Theoretical ranges (paper Eq. nwVSS)
R_PITCH = 127
R_VEL = 1023
# log-IOI range: ln(IOI_max) - ln(IOI_min). IOI in seconds, e.g. 0.001--10 => 9.21
R_TEMPORAL_THEORETICAL = np.log(10.0) - np.log(0.001)  # ~9.21


def load_component_data(data_dir):
    path = os.path.join(data_dir, 'vss_component_analysis_results.csv')
    df = pd.read_csv(path)
    return df


def load_pairwise_data(data_dir):
    path = os.path.join(data_dir, 'vss_wvss_pairwise_data.csv')
    df = pd.read_csv(path)
    return df


def compute_weights_from_means(ctrl_means, multi_means):
    """Effect = |multi - ctrl| per component; weights = effect / sum(effects) * 100."""
    effects = {
        'pitch': abs(multi_means['pitch'] - ctrl_means['pitch']),
        'velocity': abs(multi_means['velocity'] - ctrl_means['velocity']),
        'temporal': abs(multi_means['temporal'] - ctrl_means['temporal']),
    }
    total = sum(effects.values())
    if total == 0:
        return {'pitch': 0, 'velocity': 0, 'temporal': 0}
    return {k: v / total * 100 for k, v in effects.items()}


def component_means_by_condition(df_component):
    """Extract Mean per (Condition, Component) from component CSV."""
    out = {}
    for cond in ['Control', 'Multi-Constraint']:
        out[cond] = {}
        for comp_name, key in [
            ('Pitch Distance', 'pitch'),
            ('Velocity Distance', 'velocity'),
            ('Temporal Distance', 'temporal'),
        ]:
            row = df_component[(df_component['Condition'] == cond) & (df_component['Component'] == comp_name)]
            out[cond][key] = row['Mean'].values[0] if len(row) else 0
    return out


def pairwise_means_by_condition(df_pair):
    """From pairwise CSV: mean of Pitch_Distance, Velocity_Distance, Temporal_Distance per condition."""
    out = {}
    for cond in ['Control', 'Multi-Constraint']:
        sub = df_pair[df_pair['Condition'] == cond]
        out[cond] = {
            'pitch': sub['Pitch_Distance'].mean(),
            'velocity': sub['Velocity_Distance'].mean(),
            'temporal': sub['Temporal_Distance'].mean(),
        }
    return out


def observed_log_ioi_range_from_pairwise(df_pair):
    """
    We don't have raw log-IOI; use observed range of Temporal_Distance as proxy
    for scale. Alternatively use theoretical. Here we return theoretical and
    optionally the max observed W_temporal for reporting.
    """
    max_w_temporal = df_pair['Temporal_Distance'].max()
    return R_TEMPORAL_THEORETICAL, max_w_temporal


def run_nwvss_weights(df_component, R_temporal):
    """Compute wVSS (raw) and nwVSS (normalised) weights from component means."""
    means = component_means_by_condition(df_component)
    ctrl = means['Control']
    multi = means['Multi-Constraint']

    # Raw wVSS weights (effect = |multi - ctrl|)
    wvss_weights = compute_weights_from_means(ctrl, multi)

    # Normalised components: divide by range
    ctrl_norm = {
        'pitch': ctrl['pitch'] / R_PITCH,
        'velocity': ctrl['velocity'] / R_VEL,
        'temporal': ctrl['temporal'] / R_temporal,
    }
    multi_norm = {
        'pitch': multi['pitch'] / R_PITCH,
        'velocity': multi['velocity'] / R_VEL,
        'temporal': multi['temporal'] / R_temporal,
    }
    nwvss_weights = compute_weights_from_means(ctrl_norm, multi_norm)

    return wvss_weights, nwvss_weights


def run_split_half(df_pair, R_temporal):
    """
    Split pairwise data into two halves (by row index: even vs odd).
    Compute component means per condition per half, then effects and weights.
    """
    control = df_pair[df_pair['Condition'] == 'Control'].sort_index().reset_index(drop=True)
    multi = df_pair[df_pair['Condition'] == 'Multi-Constraint'].sort_index().reset_index(drop=True)
    n = len(control)
    if n != len(multi) or n < 2:
        return None

    half1_idx = list(range(0, n, 2))  # 0, 2, 4
    half2_idx = list(range(1, n, 2))  # 1, 3, 5

    def means_for_indices(cond_df, indices):
        sub = cond_df.iloc[indices]
        return {
            'pitch': sub['Pitch_Distance'].mean(),
            'velocity': sub['Velocity_Distance'].mean(),
            'temporal': sub['Temporal_Distance'].mean(),
        }

    ctrl_h1 = means_for_indices(control, half1_idx)
    ctrl_h2 = means_for_indices(control, half2_idx)
    multi_h1 = means_for_indices(multi, half1_idx)
    multi_h2 = means_for_indices(multi, half2_idx)

    # Raw wVSS weights per half
    wvss_h1 = compute_weights_from_means(ctrl_h1, multi_h1)
    wvss_h2 = compute_weights_from_means(ctrl_h2, multi_h2)

    # nwVSS: normalise by ranges then effect
    def norm_means(m, Rt):
        return {
            'pitch': m['pitch'] / R_PITCH,
            'velocity': m['velocity'] / R_VEL,
            'temporal': m['temporal'] / Rt,
        }
    nwvss_h1 = compute_weights_from_means(norm_means(ctrl_h1, R_temporal), norm_means(multi_h1, R_temporal))
    nwvss_h2 = compute_weights_from_means(norm_means(ctrl_h2, R_temporal), norm_means(multi_h2, R_temporal))

    return {
        'wvss_h1': wvss_h1,
        'wvss_h2': wvss_h2,
        'nwvss_h1': nwvss_h1,
        'nwvss_h2': nwvss_h2,
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')
    df_component = load_component_data(data_dir)
    df_pair = load_pairwise_data(data_dir)

    R_temporal, max_w_temporal = observed_log_ioi_range_from_pairwise(df_pair)

    wvss_weights, nwvss_weights = run_nwvss_weights(df_component, R_temporal)
    split = run_split_half(df_pair, R_temporal)

    # Report
    print("=" * 60)
    print("nwVSS analysis (range-normalised wVSS)")
    print("=" * 60)
    print(f"R_pitch = {R_PITCH}, R_vel = {R_VEL}, R_temporal = {R_temporal:.4f} (theoretical ln(10/0.001))")
    print()
    print("Weights from component means (Control vs Multi-Constraint):")
    print("  wVSS (raw):   pitch {:.3f}%, velocity {:.3f}%, temporal {:.3f}%".format(
        wvss_weights['pitch'], wvss_weights['velocity'], wvss_weights['temporal']))
    print("  nwVSS (norm): pitch {:.3f}%, velocity {:.3f}%, temporal {:.3f}%".format(
        nwvss_weights['pitch'], nwvss_weights['velocity'], nwvss_weights['temporal']))
    print()

    if split:
        print("Split-half cross-validation (pairwise rows split into two halves):")
        print("  wVSS  half1: pitch {:.2f}%, vel {:.2f}%, temp {:.2f}%".format(
            split['wvss_h1']['pitch'], split['wvss_h1']['velocity'], split['wvss_h1']['temporal']))
        print("  wVSS  half2: pitch {:.2f}%, vel {:.2f}%, temp {:.2f}%".format(
            split['wvss_h2']['pitch'], split['wvss_h2']['velocity'], split['wvss_h2']['temporal']))
        print("  nwVSS half1: pitch {:.2f}%, vel {:.2f}%, temp {:.2f}%".format(
            split['nwvss_h1']['pitch'], split['nwvss_h1']['velocity'], split['nwvss_h1']['temporal']))
        print("  nwVSS half2: pitch {:.2f}%, vel {:.2f}%, temp {:.2f}%".format(
            split['nwvss_h2']['pitch'], split['nwvss_h2']['velocity'], split['nwvss_h2']['temporal']))
        dev_wvss = max(
            abs(split['wvss_h1']['velocity'] - split['wvss_h2']['velocity']),
            abs(split['wvss_h1']['pitch'] - split['wvss_h2']['pitch']),
            abs(split['wvss_h1']['temporal'] - split['wvss_h2']['temporal']),
        )
        dev_nwvss = max(
            abs(split['nwvss_h1']['velocity'] - split['nwvss_h2']['velocity']),
            abs(split['nwvss_h1']['pitch'] - split['nwvss_h2']['pitch']),
            abs(split['nwvss_h1']['temporal'] - split['nwvss_h2']['temporal']),
        )
        print("  Max weight deviation: wVSS {:.2f} pp, nwVSS {:.2f} pp".format(dev_wvss, dev_nwvss))
    print()
    print("-- For paper.tex --")
    print("nwVSS weights: pitch {:.2f}%, velocity {:.2f}%, temporal {:.2f}%".format(
        nwvss_weights['pitch'], nwvss_weights['velocity'], nwvss_weights['temporal']))
    print("R_temporal (theoretical): {:.2f}".format(R_temporal))
    if split:
        print("nwVSS split-half: half1 vel {:.2f}%, half2 vel {:.2f}%; max deviation {:.2f} pp".format(
            split['nwvss_h1']['velocity'], split['nwvss_h2']['velocity'], dev_nwvss))


if __name__ == "__main__":
    main()
