#!/usr/bin/env python3
"""
Multi-Layer Algorithmic Composition System for Disklavier
==========================================================

This script implements a hierarchical generative music system integrating:
  1. L-System macro-form generation
  2. Nancarrow-style tempo canons for meso-structure
  3. Xenakis-style stochastic generation with psychoacoustic constraints
  4. Hardware latency compensation for Yamaha Disklavier Pro

Author: Generated for algorithmic composition research
Date: 2024
"""

import numpy as np
from disklavier_latency_compensation import precompensate_onset_linear


# =============================================================================
# LAYER 1: L-SYSTEM MACRO-FORM GENERATION
# =============================================================================

def generate_lsystem(axiom, rules, iterations):
    """
    Generate L-System string by applying production rules iteratively.

    Parameters
    ----------
    axiom : str
        Initial string
    rules : dict
        Production rules {symbol: replacement}
    iterations : int
        Number of iterations to apply

    Returns
    -------
    str
        Generated string after all iterations
    """
    current = axiom
    for i in range(iterations):
        next_string = ""
        for symbol in current:
            next_string += rules.get(symbol, symbol)
        current = next_string
    return current


# =============================================================================
# LAYER 2: NANCARROW TEMPO CANON MAPPING
# =============================================================================

def build_section_timeline(lsystem_sequence, canon_parameters):
    """
    Build timeline of sections based on L-System sequence.

    Parameters
    ----------
    lsystem_sequence : str
        Generated L-System string
    canon_parameters : dict
        Mapping from symbols to canon parameters

    Returns
    -------
    list of dict
        Section information with timing and parameters
    """
    sections = []
    cumulative_time = 0.0

    for i, symbol in enumerate(lsystem_sequence):
        params = canon_parameters[symbol]
        section = {
            'index': i,
            'symbol': symbol,
            'start_time': cumulative_time,
            'end_time': cumulative_time + params['duration'],
            'duration': params['duration'],
            'tempo_ratio': params['ratio'],
            'mode': params['psychoacoustic_mode']
        }
        sections.append(section)
        cumulative_time += params['duration']

    return sections


# =============================================================================
# LAYER 3: XENAKIS-STYLE STOCHASTIC GENERATION
# =============================================================================

def generate_stochastic_notes_melodic(start_time, end_time, tempo_ratio, voice_id):
    """
    Generate notes for melodic mode (clear voices, separated pitches).

    Parameters
    ----------
    start_time : float
        Section start time in seconds
    end_time : float
        Section end time in seconds
    tempo_ratio : float
        Tempo scaling factor for this voice
    voice_id : int
        Voice identifier (0 or 1)

    Returns
    -------
    list of dict
        Note events with keys: onset_time, pitch, velocity, voice_id
    """
    duration = end_time - start_time
    base_density = 5.0  # notes per second
    density = base_density * tempo_ratio
    n_notes = int(density * duration)

    # Pitch ranges for two voices (separated by >5 semitones)
    if voice_id == 0:
        pitch_mean = 60  # C4
        pitch_std = 5
    else:
        pitch_mean = 72  # C5
        pitch_std = 5

    notes = []
    for _ in range(n_notes):
        onset = start_time + np.random.uniform(0, duration)
        pitch = int(np.clip(np.random.normal(pitch_mean, pitch_std), 21, 108))
        velocity = int(np.random.uniform(400, 700))

        notes.append({
            'onset_time': onset,
            'pitch': pitch,
            'velocity': velocity,
            'voice_id': voice_id
        })

    return notes


def generate_stochastic_notes_textural(start_time, end_time, tempo_ratio, voice_id):
    """
    Generate notes for textural mode (high density, fusion).

    Parameters
    ----------
    start_time : float
        Section start time in seconds
    end_time : float
        Section end time in seconds
    tempo_ratio : float
        Tempo scaling factor for this voice
    voice_id : int
        Voice identifier (0 or 1)

    Returns
    -------
    list of dict
        Note events with keys: onset_time, pitch, velocity, voice_id
    """
    duration = end_time - start_time
    base_density = 50.0  # notes per second
    density = base_density * tempo_ratio
    n_notes = int(density * duration)

    # Overlapping pitch ranges for both voices
    pitch_mean = 60
    pitch_std = 12

    notes = []
    for _ in range(n_notes):
        onset = start_time + np.random.uniform(0, duration)
        pitch = int(np.clip(np.random.normal(pitch_mean, pitch_std), 21, 108))
        velocity = int(np.random.uniform(500, 900))

        notes.append({
            'onset_time': onset,
            'pitch': pitch,
            'velocity': velocity,
            'voice_id': voice_id
        })

    return notes


# =============================================================================
# LAYER 4: HARDWARE LATENCY COMPENSATION
# =============================================================================

def apply_hardware_compensation(notes):
    """
    Apply velocity-dependent latency compensation to all notes.

    Parameters
    ----------
    notes : list of dict
        Note events with onset_time, pitch, velocity, voice_id

    Returns
    -------
    list of dict
        Compensated notes with adjusted onset times
    """
    compensated_notes = []

    for note in notes:
        intended_onset_ms = note['onset_time'] * 1000.0
        trigger_time_ms = precompensate_onset_linear(intended_onset_ms, note['velocity'])
        trigger_time_s = trigger_time_ms / 1000.0

        compensated_note = {
            'onset_time': trigger_time_s,
            'pitch': note['pitch'],
            'velocity': note['velocity'],
            'voice_id': note['voice_id'],
            'intended_onset': note['onset_time'],
            'compensation_ms': intended_onset_ms - trigger_time_ms
        }

        compensated_notes.append(compensated_note)

    return compensated_notes


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_composition(axiom='A', iterations=4, seed=42):
    """
    Generate complete multi-layer composition.

    Parameters
    ----------
    axiom : str
        L-System starting symbol
    iterations : int
        Number of L-System iterations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list of dict
        Final event list with compensated onset times
    """
    np.random.seed(seed)

    # Layer 1: Generate L-System sequence
    production_rules = {'A': 'AB', 'B': 'A'}
    lsystem_sequence = generate_lsystem(axiom, production_rules, iterations)

    # Layer 2: Map to tempo canon parameters
    canon_parameters = {
        'A': {
            'ratio': (3, 4),
            'duration': 10.0,
            'psychoacoustic_mode': 'melodic'
        },
        'B': {
            'ratio': (1, np.sqrt(2)),
            'duration': 8.0,
            'psychoacoustic_mode': 'textural'
        }
    }

    sections = build_section_timeline(lsystem_sequence, canon_parameters)

    # Layer 3: Generate stochastic notes
    all_notes = []
    for section in sections:
        for voice_id in [0, 1]:
            voice_tempo_ratio = section['tempo_ratio'][voice_id]

            if section['mode'] == 'melodic':
                notes = generate_stochastic_notes_melodic(
                    section['start_time'], section['end_time'], 
                    voice_tempo_ratio, voice_id
                )
            else:
                notes = generate_stochastic_notes_textural(
                    section['start_time'], section['end_time'], 
                    voice_tempo_ratio, voice_id
                )

            all_notes.extend(notes)

    all_notes.sort(key=lambda x: x['onset_time'])

    # Layer 4: Apply hardware compensation
    final_events = apply_hardware_compensation(all_notes)
    final_events.sort(key=lambda x: x['onset_time'])

    return final_events, lsystem_sequence, sections


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-LAYER ALGORITHMIC COMPOSITION SYSTEM")
    print("=" * 80)

    # Generate composition
    events, sequence, sections = generate_composition(axiom='A', iterations=4, seed=42)

    print(f"\nL-System sequence: '{sequence}'")
    print(f"Total sections: {len(sections)}")
    print(f"Total events: {len(events)}")
    print(f"Duration: {events[-1]['onset_time']:.2f} seconds")

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(events)
    output_file = 'multilayer_composition_events.csv'
    df.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")

    print("\nFirst 10 events:")
    for i, event in enumerate(events[:10]):
        print(f"  {i:4d}: t={event['onset_time']:8.4f}s, "
              f"pitch={event['pitch']:3d}, vel={event['velocity']:3d}, "
              f"voice={event['voice_id']}")
