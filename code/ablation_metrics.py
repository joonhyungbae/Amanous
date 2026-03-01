"""
Shared metrics for ablation experiments (Section 4.1).
Uses paper metrics: MC, RC (KS-based), VSS from supplementary_code/core/coherence_metrics.
Section boundaries and same-symbol / sequential metrics.
"""

import os
import sys
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Any

# Use paper coherence metrics (KS-based RC, contour MC, Wasserstein VSS)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'supplementary_code', 'core'))
from coherence_metrics import (
    melodic_coherence,
    rhythmic_coherence,
    voice_separation_score,
    tonal_stability,
)

# Latency models (linear matches amanous_composer; power-law for ablation c, Eq. 5, c=0.5)
def _latency_ms(velocity_midi: int) -> float:
    velocity_dk = velocity_midi * 8
    return 30 - 20 * (velocity_dk / 1023)


def _latency_powerlaw_ms(velocity_midi: int, exponent: float = 0.5) -> float:
    """Power-law L(v): 10--30 ms, (v/v_max)^c. exponent c=0.5 per paper Eq. 5."""
    v_dk = velocity_midi * 8
    v_norm = v_dk / 1023.0
    return 30.0 - 20.0 * (v_norm ** exponent)


def latency_powerlaw_c05_fn():
    """Return a callable (velocity_midi -> ms) for power-law with c=0.5 (paper Eq. 5)."""
    return lambda v: _latency_powerlaw_ms(v, 0.5)


def section_bounds(sequence: str, config: Any) -> List[Tuple[float, float, str]]:
    """Return list of (start_time, end_time, symbol) for each section."""
    bounds = []
    t = 0.0
    for symbol in sequence:
        sym_config = config.symbol_configs[symbol]
        duration = sym_config.duration
        bounds.append((t, t + duration, symbol))
        t += duration
    return bounds


def events_to_sections(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> List[List[Dict]]:
    """Assign events to sections by time_key (onset_time or trigger_time). Returns list of event lists per section."""
    bounds = section_bounds(sequence, config)
    sections = [[] for _ in bounds]
    for e in events:
        t = e.get(time_key, e["onset_time"])
        for i, (start, end, _) in enumerate(bounds):
            if start <= t < end:
                sections[i].append(e)
                break
    return sections


def _section_pitches(sec_events: List[Dict], time_key: str = "onset_time") -> np.ndarray:
    """Pitch sequence for one section, ordered by time."""
    if not sec_events:
        return np.array([])
    ordered = sorted(sec_events, key=lambda x: x.get(time_key, x["onset_time"]))
    return np.array([e["pitch"] for e in ordered])


def _section_iois(sec_events: List[Dict], time_key: str = "onset_time") -> np.ndarray:
    """IOI sequence for one section (all voices merged, sorted by time)."""
    if len(sec_events) < 2:
        return np.array([])
    times = sorted(e.get(time_key, e["onset_time"]) for e in sec_events)
    return np.diff(times)


def same_symbol_mc(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> float:
    """Average melodic coherence over all same-symbol section pairs (paper MC)."""
    sections = events_to_sections(events, sequence, config, time_key)
    bounds = section_bounds(sequence, config)
    symbols = [s for (_, _, s) in bounds]
    values = []
    for i in range(len(sections)):
        for j in range(i + 1, len(sections)):
            if symbols[i] != symbols[j]:
                continue
            p1 = _section_pitches(sections[i], time_key)
            p2 = _section_pitches(sections[j], time_key)
            if len(p1) >= 2 and len(p2) >= 2:  # contour needs at least 2
                mc = melodic_coherence(p1, p2)
                values.append(mc)
    return float(np.mean(values)) if values else np.nan


def same_symbol_rc(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> float:
    """Average rhythmic coherence over all same-symbol section pairs (paper RC = 1 - KS)."""
    sections = events_to_sections(events, sequence, config, time_key)
    bounds = section_bounds(sequence, config)
    symbols = [s for (_, _, s) in bounds]
    values = []
    for i in range(len(sections)):
        for j in range(i + 1, len(sections)):
            if symbols[i] != symbols[j]:
                continue
            ioi1 = _section_iois(sections[i], time_key)
            ioi2 = _section_iois(sections[j], time_key)
            if len(ioi1) >= 1 and len(ioi2) >= 1:
                rc = rhythmic_coherence(ioi1, ioi2)
                values.append(rc)
    return float(np.mean(values)) if values else np.nan


def sequential_self_similarity_mc(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> float:
    """Average MC over adjacent same-symbol section pairs (L-system structure)."""
    sections = events_to_sections(events, sequence, config, time_key)
    bounds = section_bounds(sequence, config)
    symbols = [s for (_, _, s) in bounds]
    values = []
    for i in range(len(sections) - 1):
        if symbols[i] != symbols[i + 1]:
            continue
        p1 = _section_pitches(sections[i], time_key)
        p2 = _section_pitches(sections[i + 1], time_key)
        if len(p1) >= 2 and len(p2) >= 2:
            values.append(melodic_coherence(p1, p2))
    return float(np.mean(values)) if values else np.nan


# ---------------------------------------------------------------------------
# Section-order-sensitive metrics (event-derived; capture L-system macro-form)
# ---------------------------------------------------------------------------

def _section_feature_sequence(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> np.ndarray:
    """
    Per-section feature from events only (no symbol labels): tonal stability.
    A-sections tend to have higher TS (scale), B-sections lower (textural).
    Returns one value per section; NaN for sections with < 2 notes.
    """
    sections = events_to_sections(events, sequence, config, time_key)
    out = []
    for sec_events in sections:
        pitches = _section_pitches(sec_events, time_key)
        if len(pitches) >= 2:
            out.append(tonal_stability(pitches))
        else:
            out.append(np.nan)
    return np.array(out, dtype=float)


def section_sequence_autocorrelation(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> float:
    """
    Lag-1 autocorrelation of section-level feature (tonal stability from events).
    L-system order tends to place same-symbol sections adjacently (A-A, B-B),
    so adjacent sections often have similar feature values → higher autocorrelation.
    Random section order → lower autocorrelation.
    """
    x = _section_feature_sequence(events, sequence, config, time_key)
    valid = np.isfinite(x)
    if np.sum(valid) < 3:
        return np.nan
    x = x[valid]
    if np.std(x) < 1e-10:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def section_sequence_information_rate(sequence: str) -> float:
    """
    Information rate I(s_t; s_{t+1}) of the section (symbol) sequence used in this run.
    Derived from the pipeline output (the sequence that generated the events).
    L-system order (e.g. ABAABABA) has high IR; random shuffle has low IR.
    """
    if len(sequence) < 2:
        return np.nan
    from collections import Counter
    symbols = list(sequence)
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
    return float(mi)


def _lempel_ziv_phrase_count(s: str) -> int:
    """LZ78-style phrase count; lower = more compressible / more structured."""
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


def section_sequence_lz_complexity(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> float:
    """
    LZ complexity of discretized section-feature sequence (from events).
    Section feature = tonal stability; discretize by run median (0/1).
    L-system order yields more compressible sequence (lower phrase count).
    """
    x = _section_feature_sequence(events, sequence, config, time_key)
    valid = np.isfinite(x)
    if np.sum(valid) < 2:
        return np.nan
    median = np.nanmedian(x)
    # Discretize: >= median -> '1', else '0'; NaN sections -> '0' to preserve length
    s = "".join("1" if valid[i] and x[i] >= median else "0" for i in range(len(x)))
    if len(s) < 2:
        return np.nan
    return float(_lempel_ziv_phrase_count(s))


def vss_components(
    events: List[Dict],
    time_key: str = "trigger_time",
) -> Dict[str, float]:
    """VSS and per-component Wasserstein (pitch, velocity, temporal). Uses time_key for IOI (e.g. trigger_time for output)."""
    voices = sorted(set(e.get("voice_id", 0) for e in events))
    if len(voices) < 2:
        return {"pitch": 0.0, "velocity": 0.0, "temporal": 0.0, "vss": 0.0, "wvss": 0.0}
    v1_events = [e for e in events if e.get("voice_id") == voices[0]]
    v2_events = [e for e in events if e.get("voice_id") == voices[1]]
    # Pass events with desired time for IOI
    use_events = []
    for ev_list in (v1_events, v2_events):
        use_events.append([
            {**e, "onset_time": e.get(time_key, e["onset_time"])}
            for e in ev_list
        ])
    vss = voice_separation_score(use_events[0], use_events[1], weighted=False)
    wvss = voice_separation_score(use_events[0], use_events[1], weighted=True)
    # Raw components (replicate coherence_metrics logic)
    def w1d(x, y):
        return stats.wasserstein_distance(x, y)
    p1 = [e["pitch"] for e in use_events[0]]
    p2 = [e["pitch"] for e in use_events[1]]
    t1 = sorted([e["onset_time"] for e in use_events[0]])
    t2 = sorted([e["onset_time"] for e in use_events[1]])
    ioi1 = np.diff(t1) if len(t1) > 1 else np.array([0.0])
    ioi2 = np.diff(t2) if len(t2) > 1 else np.array([0.0])
    log_ioi1 = np.log(ioi1 + 1e-6)
    log_ioi2 = np.log(ioi2 + 1e-6)
    return {
        "pitch": float(w1d(p1, p2)),
        "velocity": float(w1d([e["velocity"] for e in use_events[0]], [e["velocity"] for e in use_events[1]])),
        "temporal": float(w1d(log_ioi1, log_ioi2)),
        "vss": float(vss),
        "wvss": float(wvss),
    }


def _get_latency_fn(latency_fn=None):
    """Return latency function (velocity_midi -> ms). Default: linear."""
    if latency_fn is None:
        return _latency_ms
    return latency_fn


def onset_alignment_sd_ms(
    events: List[Dict],
    tol_sec: float = 0.002,
    compensated: bool = False,
    latency_fn=None,
) -> float:
    """
    Mean SD of timing deviation (ms) within groups of events intended to be simultaneous.
    compensated=True: actual_onset = trigger_time + L(v)/1000 (so deviation = 0 per group).
    compensated=False: actual_onset = onset_time + L(v)/1000, deviation = L(v) in ms.
    latency_fn: optional (velocity_midi -> ms); used only when compensated=False.
    """
    from collections import defaultdict
    get_lat = _get_latency_fn(latency_fn)
    groups = defaultdict(list)
    for e in events:
        t = e["onset_time"]
        bucket = round(t / tol_sec) * tol_sec
        groups[bucket].append(e)
    sds = []
    for evs in groups.values():
        if len(evs) < 2:
            continue
        if compensated:
            # actual_onset = trigger_time + L/1000; intended = onset_time. So deviation_s = trigger + L/1000 - onset.
            # For compensated events trigger = onset - L/1000, so deviation = 0.
            trigger = np.array([e.get("trigger_time", e["onset_time"]) for e in evs])
            latency_s = np.array([get_lat(e["velocity"]) / 1000.0 for e in evs])
            intended = np.array([e["onset_time"] for e in evs])
            actual_s = trigger + latency_s
            deviation_ms = (actual_s - intended) * 1000.0
        else:
            deviation_ms = np.array([get_lat(e["velocity"]) for e in evs])
        sds.append(float(np.std(deviation_ms)))
    return float(np.mean(sds)) if sds else 0.0


def velocity_timing_correlation(
    events: List[Dict],
    compensated: bool = False,
    latency_fn=None,
) -> Tuple[float, float]:
    """
    Pearson r(velocity, onset_error).
    compensated=True: onset_error = 0 for all → r = 0 (no velocity-timing bias).
    compensated=False: onset_error = L(v) in ms → r(velocity, L(v)); use latency_fn for L.
    Returns (r, p-value).
    """
    if compensated:
        return (0.0, np.nan)
    get_lat = _get_latency_fn(latency_fn)
    vels = np.array([e["velocity"] for e in events])
    lat = np.array([get_lat(e["velocity"]) for e in events])
    if len(vels) < 2 or np.std(lat) == 0:
        return (np.nan, np.nan)
    r, p = stats.pearsonr(vels, lat)
    return (float(r), float(p))


def vss_temporal_per_section(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "trigger_time",
) -> List[float]:
    """VSS temporal component (Wasserstein on log-IOI) per section. One value per section."""
    sections = events_to_sections(events, sequence, config, time_key)
    bounds = section_bounds(sequence, config)
    out = []
    for sec_events in sections:
        voices = sorted(set(e.get("voice_id", 0) for e in sec_events))
        if len(voices) < 2:
            out.append(np.nan)
            continue
        v1 = [e for e in sec_events if e.get("voice_id") == voices[0]]
        v2 = [e for e in sec_events if e.get("voice_id") == voices[1]]
        t1 = sorted(e.get(time_key, e["onset_time"]) for e in v1)
        t2 = sorted(e.get(time_key, e["onset_time"]) for e in v2)
        ioi1 = np.diff(t1) if len(t1) > 1 else np.array([0.0])
        ioi2 = np.diff(t2) if len(t2) > 1 else np.array([0.0])
        log1 = np.log(ioi1 + 1e-6)
        log2 = np.log(ioi2 + 1e-6)
        out.append(float(stats.wasserstein_distance(log1, log2)))
    return out


def rc_per_section(
    events: List[Dict],
    sequence: str,
    config: Any,
    time_key: str = "onset_time",
) -> List[float]:
    """Per-section RC: for section i, mean of RC(IOI_i, IOI_j) over same-symbol sections j != i."""
    sections = events_to_sections(events, sequence, config, time_key)
    bounds = section_bounds(sequence, config)
    symbols = [s for (_, _, s) in bounds]
    section_iois = [_section_iois(sec, time_key) for sec in sections]
    out = []
    for i in range(len(sections)):
        ioi_i = section_iois[i]
        if len(ioi_i) < 1:
            out.append(np.nan)
            continue
        vals = []
        for j in range(len(sections)):
            if i == j or symbols[i] != symbols[j]:
                continue
            ioi_j = section_iois[j]
            if len(ioi_j) >= 1:
                vals.append(rhythmic_coherence(ioi_i, ioi_j))
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def ioi_ks_l3_vs_l4(events: List[Dict]) -> float:
    """KS distance between IOI distribution (sorted by onset_time) vs (sorted by trigger_time)."""
    by_onset = sorted(events, key=lambda e: e["onset_time"])
    by_trigger = sorted(events, key=lambda e: e.get("trigger_time", e["onset_time"]))
    t_onset = np.array([e["onset_time"] for e in by_onset])
    t_trig = np.array([e.get("trigger_time", e["onset_time"]) for e in by_trigger])
    ioi_onset = np.diff(t_onset) if len(t_onset) > 1 else np.array([0.0])
    ioi_trig = np.diff(t_trig) if len(t_trig) > 1 else np.array([0.0])
    if len(ioi_onset) == 0 or len(ioi_trig) == 0:
        return 0.0
    stat, _ = stats.ks_2samp(ioi_onset, ioi_trig)
    return float(stat)
