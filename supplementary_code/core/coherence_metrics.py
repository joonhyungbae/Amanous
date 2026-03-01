"""
Coherence Metrics Module for Algorithmic Composition Analysis

Implements information-theoretic metrics for measuring musical coherence:
- Melodic Coherence (MC): Contour edit distance
- Rhythmic Coherence (RC): IOI distribution similarity  
- Tonal Stability (TS): Pitch-class entropy
- Voice Separation Score (VSS/wVSS): Wasserstein distance
- nwVSS (range-normalized wVSS): W/R per domain, weights from normalized effects

Reference: Paper Section 4.1 (Evaluation Metrics), Eq. nwVSS
"""

import numpy as np

# --- nwVSS domain ranges (paper Eq. nwVSS) ---
R_PITCH = 127       # MIDI note range
R_VEL = 1023        # XPMIDI range
# log-IOI range: ln(IOI_max) - ln(IOI_min), e.g. IOI in [0.001, 10] s
R_TEMPORAL = float(np.log(10.0) - np.log(0.001))  # ~9.21
NWVSS_RANGES = {"pitch": R_PITCH, "velocity": R_VEL, "temporal": R_TEMPORAL}
from scipy import stats
from scipy.spatial.distance import cdist
from collections import Counter


def melodic_coherence(seq1, seq2):
    """
    Normalized Levenshtein edit distance on pitch contour.
    
    MC(X, Y) = 1 - d_Lev(contour(X), contour(Y)) / max(|X|, |Y|)
    
    Parameters
    ----------
    seq1, seq2 : array-like
        Pitch sequences
        
    Returns
    -------
    float
        Melodic coherence in [0, 1]
    """
    def to_contour(pitches):
        """Convert pitch sequence to Up/Down/Same contour."""
        contour = []
        for i in range(1, len(pitches)):
            diff = pitches[i] - pitches[i-1]
            if diff > 0:
                contour.append('U')
            elif diff < 0:
                contour.append('D')
            else:
                contour.append('S')
        return contour
    
    def levenshtein(s1, s2):
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
    
    c1 = to_contour(seq1)
    c2 = to_contour(seq2)
    
    if len(c1) == 0 and len(c2) == 0:
        return 1.0
    
    max_len = max(len(c1), len(c2))
    edit_dist = levenshtein(c1, c2)
    
    return 1 - edit_dist / max_len


def rhythmic_coherence(ioi1, ioi2):
    """
    One minus Kolmogorov-Smirnov distance between IOI distributions.
    
    RC(X, Y) = 1 - D_KS(F_IOI^X, F_IOI^Y)
    
    Parameters
    ----------
    ioi1, ioi2 : array-like
        Inter-onset interval sequences
        
    Returns
    -------
    float
        Rhythmic coherence in [0, 1]
    """
    statistic, _ = stats.ks_2samp(ioi1, ioi2)
    return 1 - statistic


def single_voice_coherence(pitches):
    """
    Single-voice coherence from pitch-interval entropy (paper Fig 3).
    High when intervals are predictable (low entropy), low when random.
    SVC = 1 - H(interval_class) / log2(N_classes).

    Parameters
    ----------
    pitches : array-like
        MIDI pitch sequence (at least 2 notes)

    Returns
    -------
    float
        Coherence in [0, 1]
    """
    if pitches is None or len(pitches) < 2:
        return np.nan
    intervals = [int(round((pitches[i] - pitches[i - 1]))) for i in range(1, len(pitches))]
    # Clip to ±12 semitones (interval class range for comparison)
    interval_classes = [max(-12, min(12, i)) for i in intervals]
    # Map to 0..24 for histogram
    bins = [i + 12 for i in interval_classes]
    counts = Counter(bins)
    total = sum(counts.values())
    entropy = 0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * np.log2(p)
    max_entropy = np.log2(25)  # 25 bins (-12..+12)
    return 1 - (entropy / max_entropy)


def tonal_stability(pitches):
    """
    Entropy-based tonal stability index.
    
    TS = 1 - H(pitch-class) / log2(12)
    
    Parameters
    ----------
    pitches : array-like
        MIDI pitch values
        
    Returns
    -------
    float
        Tonal stability in [0, 1]
        0 = maximally entropic (uniform)
        1 = single pitch class
    """
    # Convert to pitch classes (0-11)
    pitch_classes = [p % 12 for p in pitches]
    
    # Count occurrences
    counts = Counter(pitch_classes)
    total = sum(counts.values())
    
    # Calculate entropy
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Normalize by maximum entropy (log2(12))
    max_entropy = np.log2(12)
    normalized_entropy = entropy / max_entropy
    
    return 1 - normalized_entropy


def voice_separation_score(voice1_events, voice2_events, weighted=False, weights=None):
    """
    Mean Wasserstein distance across pitch, velocity, and log-IOI.
    
    VSS = (1/3) * (W_pitch + W_vel + W_temporal)
    wVSS uses empirically derived weights.
    
    Parameters
    ----------
    voice1_events, voice2_events : list of dict
        Event lists with 'pitch', 'velocity', 'onset_time' keys
    weighted : bool
        If True, use weighted VSS
    weights : dict, optional
        Custom weights {pitch, velocity, temporal}
        Default: {pitch: 0.00394, velocity: 0.97216, temporal: 0.02390}
        
    Returns
    -------
    float
        Voice separation score
    """
    if weights is None:
        weights = {
            'pitch': 0.00394,
            'velocity': 0.97216,
            'temporal': 0.02390
        }
    
    def wasserstein_1d(x, y):
        """1D Wasserstein distance."""
        return stats.wasserstein_distance(x, y)
    
    # Extract features
    pitch1 = [e['pitch'] for e in voice1_events]
    pitch2 = [e['pitch'] for e in voice2_events]
    
    vel1 = [e['velocity'] for e in voice1_events]
    vel2 = [e['velocity'] for e in voice2_events]
    
    # Calculate IOIs
    times1 = sorted([e['onset_time'] for e in voice1_events])
    times2 = sorted([e['onset_time'] for e in voice2_events])
    
    ioi1 = np.diff(times1) if len(times1) > 1 else [0]
    ioi2 = np.diff(times2) if len(times2) > 1 else [0]
    
    # Log-IOI (add small epsilon to avoid log(0))
    log_ioi1 = np.log(np.array(ioi1) + 1e-6)
    log_ioi2 = np.log(np.array(ioi2) + 1e-6)
    
    # Calculate Wasserstein distances
    w_pitch = wasserstein_1d(pitch1, pitch2)
    w_vel = wasserstein_1d(vel1, vel2)
    w_temporal = wasserstein_1d(log_ioi1, log_ioi2)
    
    if weighted:
        return (weights['pitch'] * w_pitch + 
                weights['velocity'] * w_vel + 
                weights['temporal'] * w_temporal)
    else:
        return (w_pitch + w_vel + w_temporal) / 3


def vss_component_distances(voice1_events, voice2_events):
    """
    Raw Wasserstein distances for pitch, velocity, and log-IOI (temporal).
    Used as input for nwVSS (normalize by range then apply weights).

    Parameters
    ----------
    voice1_events, voice2_events : list of dict
        Event lists with 'pitch', 'velocity', 'onset_time' keys

    Returns
    -------
    dict
        {'pitch': W_pitch, 'velocity': W_vel, 'temporal': W_temporal}
    """
    def wasserstein_1d(x, y):
        return stats.wasserstein_distance(x, y)

    pitch1 = [e['pitch'] for e in voice1_events]
    pitch2 = [e['pitch'] for e in voice2_events]
    vel1 = [e['velocity'] for e in voice1_events]
    vel2 = [e['velocity'] for e in voice2_events]
    times1 = sorted([e['onset_time'] for e in voice1_events])
    times2 = sorted([e['onset_time'] for e in voice2_events])
    ioi1 = np.diff(times1) if len(times1) > 1 else np.array([0.0])
    ioi2 = np.diff(times2) if len(times2) > 1 else np.array([0.0])
    log_ioi1 = np.log(np.array(ioi1) + 1e-6)
    log_ioi2 = np.log(np.array(ioi2) + 1e-6)

    return {
        'pitch': float(wasserstein_1d(pitch1, pitch2)),
        'velocity': float(wasserstein_1d(vel1, vel2)),
        'temporal': float(wasserstein_1d(log_ioi1, log_ioi2)),
    }


def nwvss_score(voice1_events, voice2_events, weights=None, ranges=None):
    """
    Range-normalized wVSS (nwVSS).

    nwVSS = w_pitch * (W_pitch / R_pitch) + w_vel * (W_vel / R_vel) + w_temporal * (W_temporal / R_temporal).

    If weights is None, returns the mean of the three normalized components
    (equal-weight nwVSS). Weights should be derived from the same
    weight-extraction procedure on normalized components (e.g. compute_nwvss_weights).

    Parameters
    ----------
    voice1_events, voice2_events : list of dict
        Event lists with 'pitch', 'velocity', 'onset_time'
    weights : dict, optional
        {'pitch': w_p, 'velocity': w_v, 'temporal': w_t} in [0, 1], sum = 1.
        If None, uses equal weights (1/3 each).
    ranges : dict, optional
        {'pitch': R_p, 'velocity': R_v, 'temporal': R_t}. Default: R_PITCH, R_VEL, R_TEMPORAL.

    Returns
    -------
    float
        nwVSS score
    """
    if ranges is None:
        ranges = NWVSS_RANGES
    raw = vss_component_distances(voice1_events, voice2_events)
    norm = {
        'pitch': raw['pitch'] / ranges['pitch'],
        'velocity': raw['velocity'] / ranges['velocity'],
        'temporal': raw['temporal'] / ranges['temporal'],
    }
    if weights is None:
        return float(np.mean([norm['pitch'], norm['velocity'], norm['temporal']]))
    return float(
        weights['pitch'] * norm['pitch'] +
        weights['velocity'] * norm['velocity'] +
        weights['temporal'] * norm['temporal']
    )


def compute_wvss_weights(ctrl_component_means, multi_component_means):
    """
    Compute raw wVSS weights from condition-wise component means (Control vs Multi-Constraint).
    Effect per component = |multi - ctrl|; weights = effect / sum(effects) * 100.
    Used for split-half validation and paper wVSS (Section 4).

    Parameters
    ----------
    ctrl_component_means : dict
        {'pitch': mean_W_pitch, 'velocity': mean_W_vel, 'temporal': mean_W_temporal} for Control.
    multi_component_means : dict
        Same keys for Multi-Constraint condition.

    Returns
    -------
    dict
        {'pitch': pct, 'velocity': pct, 'temporal': pct} summing to 100.
    """
    effects = {
        'pitch': abs(multi_component_means['pitch'] - ctrl_component_means['pitch']),
        'velocity': abs(multi_component_means['velocity'] - ctrl_component_means['velocity']),
        'temporal': abs(multi_component_means['temporal'] - ctrl_component_means['temporal']),
    }
    total = sum(effects.values())
    if total == 0:
        return {'pitch': 0.0, 'velocity': 0.0, 'temporal': 0.0}
    return {k: v / total * 100 for k, v in effects.items()}


def compute_nwvss_weights(ctrl_component_means, multi_component_means, ranges=None):
    """
    Compute nwVSS weights from condition-wise component means (Control vs Multi-Constraint).
    Effect per component = |multi_norm - ctrl_norm|; weights = effect / sum(effects) * 100.

    So weights are derived in the range-normalized space, making contributions
    comparable across pitch, velocity, and temporal domains.

    Parameters
    ----------
    ctrl_component_means : dict
        {'pitch': mean_W_pitch, 'velocity': mean_W_vel, 'temporal': mean_W_temporal} for Control.
    multi_component_means : dict
        Same keys for Multi-Constraint condition.
    ranges : dict, optional
        Domain ranges; default R_PITCH, R_VEL, R_TEMPORAL.

    Returns
    -------
    dict
        {'pitch': pct, 'velocity': pct, 'temporal': pct} summing to 100.
    """
    if ranges is None:
        ranges = NWVSS_RANGES
    ctrl_n = {
        'pitch': ctrl_component_means['pitch'] / ranges['pitch'],
        'velocity': ctrl_component_means['velocity'] / ranges['velocity'],
        'temporal': ctrl_component_means['temporal'] / ranges['temporal'],
    }
    multi_n = {
        'pitch': multi_component_means['pitch'] / ranges['pitch'],
        'velocity': multi_component_means['velocity'] / ranges['velocity'],
        'temporal': multi_component_means['temporal'] / ranges['temporal'],
    }
    effects = {
        'pitch': abs(multi_n['pitch'] - ctrl_n['pitch']),
        'velocity': abs(multi_n['velocity'] - ctrl_n['velocity']),
        'temporal': abs(multi_n['temporal'] - ctrl_n['temporal']),
    }
    total = sum(effects.values())
    if total == 0:
        return {'pitch': 0.0, 'velocity': 0.0, 'temporal': 0.0}
    return {k: v / total * 100 for k, v in effects.items()}


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size with pooled standard deviation.
    
    Parameters
    ----------
    group1, group2 : array-like
        Two groups to compare
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    
    if pooled_std == 0:
        return np.inf if np.mean(group1) != np.mean(group2) else 0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


if __name__ == "__main__":
    # Test the module
    print("Coherence Metrics Module - Test")
    print("=" * 50)
    
    # Test melodic coherence
    seq1 = [60, 62, 64, 65, 67]
    seq2 = [60, 62, 64, 65, 67]
    print(f"MC (identical): {melodic_coherence(seq1, seq2):.3f}")
    
    seq3 = [60, 58, 56, 54, 52]  # Descending
    print(f"MC (opposite): {melodic_coherence(seq1, seq3):.3f}")
    
    # Test tonal stability
    c_major = [60, 62, 64, 65, 67, 69, 71]  # C major scale
    chromatic = list(range(60, 72))  # All 12 pitch classes
    print(f"TS (C major): {tonal_stability(c_major):.3f}")
    print(f"TS (chromatic): {tonal_stability(chromatic):.3f}")
