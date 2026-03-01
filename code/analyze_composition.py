#!/usr/bin/env python3
"""
Amanous Composition Analyzer
============================

Analyzes generated compositions using paper metrics:
- Melodic Coherence
- Rhythmic Coherence
- Tonal Stability
- Voice Separation Score
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import os

def load_composition(csv_path: str) -> pd.DataFrame:
    """Load composition events from CSV."""
    return pd.read_csv(csv_path)

def calculate_tonal_stability(pitches: np.ndarray) -> float:
    """
    Tonal stability: 1 - H(pitch-class) / log2(12).
    High = focused on a key; low = uniform (atonal).
    """
    pitch_classes = pitches % 12
    counts = Counter(pitch_classes)

    total = sum(counts.values())
    probs = np.array([counts.get(i, 0) / total for i in range(12)])
    probs = probs[probs > 0]

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(12)
    
    return 1 - (entropy / max_entropy)

def calculate_rhythmic_coherence(iois: np.ndarray) -> float:
    """
    Rhythmic coherence: inverse of IOI CV. High = regular rhythm, low = irregular.
    """
    if len(iois) < 2:
        return 1.0
    
    cv = np.std(iois) / np.mean(iois) if np.mean(iois) > 0 else 0
    return 1 / (1 + cv)  # normalize to 0-1

def calculate_density(events: pd.DataFrame, window_size: float = 1.0) -> np.ndarray:
    """Compute note density (notes/s) per time window."""
    max_time = events['onset_time'].max()
    windows = np.arange(0, max_time, window_size)
    densities = []
    
    for t in windows:
        count = len(events[(events['onset_time'] >= t) & 
                          (events['onset_time'] < t + window_size)])
        densities.append(count / window_size)
    
    return np.array(densities)

def calculate_voice_separation(events: pd.DataFrame) -> dict:
    """Voice separation score (Wasserstein-distance style)."""
    voices = events['voice_id'].unique()

    if len(voices) < 2:
        return {'pitch': 0, 'velocity': 0, 'temporal': 0, 'total': 0}

    voice_data = {v: events[events['voice_id'] == v] for v in voices}

    # Pitch separation
    pitch_dists = []
    for v1 in voices:
        for v2 in voices:
            if v1 < v2:
                dist = abs(voice_data[v1]['pitch'].mean() - voice_data[v2]['pitch'].mean())
                pitch_dists.append(dist)
    
    # Velocity separation
    vel_dists = []
    for v1 in voices:
        for v2 in voices:
            if v1 < v2:
                dist = abs(voice_data[v1]['velocity'].mean() - voice_data[v2]['velocity'].mean())
                vel_dists.append(dist)
    
    # Temporal separation (IOI difference)
    temporal_dists = []
    for v1 in voices:
        for v2 in voices:
            if v1 < v2:
                ioi1 = np.diff(voice_data[v1]['onset_time'].sort_values())
                ioi2 = np.diff(voice_data[v2]['onset_time'].sort_values())
                if len(ioi1) > 0 and len(ioi2) > 0:
                    dist = abs(np.mean(ioi1) - np.mean(ioi2))
                    temporal_dists.append(dist)
    
    return {
        'pitch': np.mean(pitch_dists) if pitch_dists else 0,
        'velocity': np.mean(vel_dists) if vel_dists else 0,
        'temporal': np.mean(temporal_dists) if temporal_dists else 0,
        'total': np.mean(pitch_dists) + np.mean(vel_dists) + np.mean(temporal_dists) * 100
    }

def analyze_composition(csv_path: str, title: str = None) -> dict:
    """Full composition analysis."""
    events = load_composition(csv_path)

    if title is None:
        title = os.path.basename(csv_path).replace('_events.csv', '')

    # Basic stats
    duration = events['onset_time'].max()
    n_events = len(events)
    n_voices = events['voice_id'].nunique()
    avg_density = n_events / duration if duration > 0 else 0
    
    # Metrics
    pitches = events['pitch'].values
    onset_times = events['onset_time'].sort_values().values
    iois = np.diff(onset_times)
    
    ts = calculate_tonal_stability(pitches)
    rc = calculate_rhythmic_coherence(iois)
    vss = calculate_voice_separation(events)
    densities = calculate_density(events)
    
    # Pitch stats
    pitch_range = pitches.max() - pitches.min()
    pitch_mean = pitches.mean()
    pitch_std = pitches.std()
    
    # Velocity stats
    velocities = events['velocity'].values
    vel_mean = velocities.mean()
    vel_std = velocities.std()
    
    return {
        'title': title,
        'duration_s': duration,
        'n_events': n_events,
        'n_voices': n_voices,
        'avg_density_nps': avg_density,
        'max_density_nps': densities.max() if len(densities) > 0 else 0,
        'tonal_stability': ts,
        'rhythmic_coherence': rc,
        'voice_separation_pitch': vss['pitch'],
        'voice_separation_velocity': vss['velocity'],
        'voice_separation_temporal': vss['temporal'],
        'pitch_range': pitch_range,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'velocity_mean': vel_mean,
        'velocity_std': vel_std,
    }

def print_analysis(analysis: dict):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"Composition: {analysis['title']}")
    print(f"{'='*60}")

    print(f"\nBasic info:")
    print(f"  - Duration: {analysis['duration_s']:.2f}s")
    print(f"  - Total events: {analysis['n_events']:,}")
    print(f"  - Voices: {analysis['n_voices']}")
    print(f"  - Avg density: {analysis['avg_density_nps']:.1f} notes/s")
    print(f"  - Max density: {analysis['max_density_nps']:.1f} notes/s")

    print(f"\nCoherence metrics:")
    print(f"  - Tonal stability (TS): {analysis['tonal_stability']:.4f}")
    if analysis['tonal_stability'] > 0.3:
        print(f"    -> High tonality (key-focused)")
    elif analysis['tonal_stability'] > 0.15:
        print(f"    -> Medium tonality")
    else:
        print(f"    -> Low tonality (atonal/chromatic)")

    print(f"  - Rhythmic coherence (RC): {analysis['rhythmic_coherence']:.4f}")
    if analysis['rhythmic_coherence'] > 0.7:
        print(f"    -> Regular rhythm")
    elif analysis['rhythmic_coherence'] > 0.4:
        print(f"    -> Medium regularity")
    else:
        print(f"    -> Free/probabilistic rhythm")

    print(f"\nVoice separation (VSS):")
    print(f"  - Pitch: {analysis['voice_separation_pitch']:.2f} semitones")
    print(f"  - Velocity: {analysis['voice_separation_velocity']:.2f}")
    print(f"  - Temporal: {analysis['voice_separation_temporal']:.4f}s")

    print(f"\nPitch:")
    print(f"  - Range: {analysis['pitch_range']} semitones")
    print(f"  - Mean: MIDI {analysis['pitch_mean']:.1f}")
    print(f"  - Std: {analysis['pitch_std']:.2f}")

    print(f"\nDynamics:")
    print(f"  - Mean velocity: {analysis['velocity_mean']:.1f}")
    print(f"  - Velocity std: {analysis['velocity_std']:.2f}")

    if analysis['avg_density_nps'] > 30:
        print(f"\n⚡ Above density threshold!")
        print(f"   Paper 24-30 notes/s coherence threshold exceeded.")
        print(f"   Likely perceived as texture rather than melodic tracking.")

if __name__ == "__main__":
    # Analyze all generated compositions
    composition_dirs = [
        "/home/jhbae/Amanous/code",
        "/home/jhbae/Amanous/compositions"
    ]
    
    all_analyses = []
    
    print("=" * 70)
    print("AMANOUS COMPOSITION ANALYSIS")
    print("Paper-metrics composition analyzer")
    print("=" * 70)
    
    for comp_dir in composition_dirs:
        if os.path.exists(comp_dir):
            csv_files = [f for f in os.listdir(comp_dir) if f.endswith('_events.csv')]
            
            for csv_file in csv_files:
                csv_path = os.path.join(comp_dir, csv_file)
                analysis = analyze_composition(csv_path)
                all_analyses.append(analysis)
                print_analysis(analysis)
    
    if len(all_analyses) > 1:
        print("\n" + "=" * 70)
        print("Comparison summary")
        print("=" * 70)

        df = pd.DataFrame(all_analyses)
        print("\n{:<25} {:>10} {:>10} {:>8} {:>8}".format(
            "Composition", "Events", "Density", "TS", "RC"))
        print("-" * 65)
        
        for _, row in df.iterrows():
            print("{:<25} {:>10,} {:>10.1f} {:>8.4f} {:>8.4f}".format(
                row['title'][:24],
                int(row['n_events']),
                row['avg_density_nps'],
                row['tonal_stability'],
                row['rhythmic_coherence']
            ))
