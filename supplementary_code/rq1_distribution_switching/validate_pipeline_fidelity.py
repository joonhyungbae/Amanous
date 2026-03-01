"""
RQ1 Validation: Distribution-Switching Pipeline Fidelity

Validates that distributional separations specified at Layer 2 survive
the full four-layer processing chain (Section 5).

Expected results (Paper Table 3):
- Melodic Coherence: d = 3.70 [95% CI: 2.65, 4.73]
- Rhythmic Coherence: d = 5.34 [95% CI: 3.96, 6.70]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import pandas as pd
import numpy as np
from scipy import stats
from coherence_metrics import cohens_d


def validate_coherence_matrices(mc_path, rc_path):
    """
    Validate melodic and rhythmic coherence from pre-computed matrices.
    
    Parameters
    ----------
    mc_path : str
        Path to melodic coherence matrix CSV
    rc_path : str
        Path to rhythmic coherence matrix CSV
        
    Returns
    -------
    dict
        Validation results
    """
    mc = pd.read_csv(mc_path, index_col=0)
    rc = pd.read_csv(rc_path, index_col=0)
    
    sections = list(mc.columns)
    symbols = {s: 'A' if 'A' in s else 'B' for s in sections}
    
    same_mc, cross_mc = [], []
    same_rc, cross_rc = [], []
    
    for i, s1 in enumerate(sections):
        for j, s2 in enumerate(sections):
            if i >= j:
                continue
            if symbols[s1] == symbols[s2]:
                same_mc.append(mc.loc[s1, s2])
                same_rc.append(rc.loc[s1, s2])
            else:
                cross_mc.append(mc.loc[s1, s2])
                cross_rc.append(rc.loc[s1, s2])
    
    # Calculate statistics
    t_mc, p_mc = stats.ttest_ind(same_mc, cross_mc)
    t_rc, p_rc = stats.ttest_ind(same_rc, cross_rc)
    
    d_mc = cohens_d(same_mc, cross_mc)
    d_rc = cohens_d(same_rc, cross_rc)
    
    results = {
        'melodic': {
            'same_mean': np.mean(same_mc),
            'same_std': np.std(same_mc, ddof=1),
            'cross_mean': np.mean(cross_mc),
            'cross_std': np.std(cross_mc, ddof=1),
            't_statistic': t_mc,
            'p_value': p_mc,
            'cohens_d': d_mc,
            'expected_d': 3.70,
            'pass': abs(d_mc - 3.70) < 0.1
        },
        'rhythmic': {
            'same_mean': np.mean(same_rc),
            'same_std': np.std(same_rc, ddof=1),
            'cross_mean': np.mean(cross_rc),
            'cross_std': np.std(cross_rc, ddof=1),
            't_statistic': t_rc,
            'p_value': p_rc,
            'cohens_d': d_rc,
            'expected_d': 5.34,
            'pass': abs(d_rc - 5.34) < 0.1
        }
    }
    
    return results


def print_validation_report(results):
    """Print formatted validation report."""
    print("=" * 60)
    print("RQ1 PIPELINE FIDELITY VALIDATION")
    print("=" * 60)
    
    for metric, data in results.items():
        status = "✅ PASS" if data['pass'] else "⚠️ CHECK"
        print(f"\n{metric.upper()} COHERENCE:")
        print(f"  Same-symbol:  {data['same_mean']:.4f} ± {data['same_std']:.4f}")
        print(f"  Cross-symbol: {data['cross_mean']:.4f} ± {data['cross_std']:.4f}")
        print(f"  t = {data['t_statistic']:.2f}, p = {data['p_value']:.2e}")
        print(f"  Cohen's d: {data['cohens_d']:.2f} (expected: {data['expected_d']:.2f})")
        print(f"  Status: {status}")


if __name__ == "__main__":
    # Default paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')
    mc_path = os.path.join(data_dir, 'melodic_coherence_matrix.csv')
    rc_path = os.path.join(data_dir, 'rhythmic_coherence_matrix.csv')
    
    if os.path.exists(mc_path) and os.path.exists(rc_path):
        results = validate_coherence_matrices(mc_path, rc_path)
        print_validation_report(results)
    else:
        print("Data files not found. Please ensure CSV files are in data/csv/")
