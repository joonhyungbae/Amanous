"""
Latency Robustness Filter for Disklavier Composition System

This module implements a post-processing filter that detects and transforms
latency-sensitive musical events to minimize temporal distortion caused by
velocity-dependent latency in the Yamaha Disklavier Pro.

Based on research findings that latency varies from 10-30ms across the velocity
range (Goebl & Bresin, 2003), creating temporal distortion in chords, hockets,
and rapid sequences with large velocity ranges.

Author: Research Project on Disklavier Algorithmic Composition
Date: 2024
"""

import numpy as np
from typing import List, Dict, Tuple
import copy


def calculate_latency(velocity):
    """
    Calculate Disklavier latency based on velocity.
    Linear model from literature: 10-30ms range.

    Parameters:
    velocity: int (1-1024) - MIDI velocity value

    Returns:
    latency in seconds
    """
    v_norm = velocity / 1024.0
    latency_ms = 30 - (v_norm * 20)
    return latency_ms / 1000.0


class LatencySensitivityDetector:
    """
    Detects musical events that are highly sensitive to latency model uncertainty.
    """

    def __init__(self, 
                 chord_threshold_notes=4,
                 chord_threshold_velocity_range=500,
                 hocket_threshold_ioi=0.075,
                 hocket_threshold_velocity_diff=400,
                 rapid_sequence_threshold_ioi=0.050,
                 rapid_sequence_threshold_velocity_range=600):
        """Initialize detection thresholds."""
        self.chord_threshold_notes = chord_threshold_notes
        self.chord_threshold_velocity_range = chord_threshold_velocity_range
        self.hocket_threshold_ioi = hocket_threshold_ioi
        self.hocket_threshold_velocity_diff = hocket_threshold_velocity_diff
        self.rapid_sequence_threshold_ioi = rapid_sequence_threshold_ioi
        self.rapid_sequence_threshold_velocity_range = rapid_sequence_threshold_velocity_range

    def detect_sensitive_chords(self, events):
        """
        Detect chords with large velocity range that will experience temporal spread.

        Rule: IF (notes_in_chord > threshold) AND (velocity_range > threshold)
              THEN flag as 'sensitive_chord'
        """
        sensitive_chords = []

        time_groups = {}
        for idx, event in enumerate(events):
            time_key = round(event['time'] * 1000)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(idx)

        for time_key, event_indices in time_groups.items():
            if len(event_indices) >= self.chord_threshold_notes:
                velocities = [events[idx]['velocity'] for idx in event_indices]
                velocity_range = max(velocities) - min(velocities)

                if velocity_range >= self.chord_threshold_velocity_range:
                    latencies = [calculate_latency(v) for v in velocities]
                    latency_spread = max(latencies) - min(latencies)

                    sensitive_chords.append({
                        'type': 'sensitive_chord',
                        'time': time_key / 1000.0,
                        'event_indices': event_indices,
                        'num_notes': len(event_indices),
                        'velocity_range': velocity_range,
                        'latency_spread_ms': latency_spread * 1000,
                        'velocities': velocities
                    })

        return sensitive_chords

    def detect_sensitive_hockets(self, events):
        """
        Detect rapid alternations between voices with large velocity differences.

        Rule: IF (inter_onset_interval < threshold) AND 
              (abs(velocity_voice1 - velocity_voice2) > threshold)
              THEN flag as 'sensitive_hocket'
        """
        sensitive_hockets = []

        sorted_events = sorted(enumerate(events), key=lambda x: x[1]['time'])

        for i in range(len(sorted_events) - 1):
            idx1, event1 = sorted_events[i]
            idx2, event2 = sorted_events[i + 1]

            ioi = event2['time'] - event1['time']
            velocity_diff = abs(event1['velocity'] - event2['velocity'])

            different_voices = (event1['pitch'] != event2['pitch']) or \
                             (event1.get('voice_id') != event2.get('voice_id'))

            if (ioi < self.hocket_threshold_ioi and 
                velocity_diff > self.hocket_threshold_velocity_diff and
                different_voices):

                latency_diff = abs(calculate_latency(event1['velocity']) - 
                                  calculate_latency(event2['velocity']))

                sensitive_hockets.append({
                    'type': 'sensitive_hocket',
                    'time': event1['time'],
                    'event_indices': [idx1, idx2],
                    'ioi_ms': ioi * 1000,
                    'velocity_diff': velocity_diff,
                    'latency_diff_ms': latency_diff * 1000,
                    'velocities': [event1['velocity'], event2['velocity']]
                })

        return sensitive_hockets

    def detect_rapid_sequences(self, events):
        """
        Detect rapid sequences within a single voice with large velocity variation.

        Rule: IF (IOI < threshold) AND (velocity_range_in_sequence > threshold)
              THEN flag as 'sensitive_sequence'
        """
        sensitive_sequences = []

        voice_groups = {}
        for idx, event in enumerate(events):
            voice_id = event.get('voice_id', 0)
            if voice_id not in voice_groups:
                voice_groups[voice_id] = []
            voice_groups[voice_id].append((idx, event))

        for voice_id, voice_events in voice_groups.items():
            voice_events = sorted(voice_events, key=lambda x: x[1]['time'])

            for i in range(len(voice_events) - 2):
                sequence_indices = []
                sequence_velocities = []

                current_idx = i
                while current_idx < len(voice_events) - 1:
                    idx1, event1 = voice_events[current_idx]
                    idx2, event2 = voice_events[current_idx + 1]

                    ioi = event2['time'] - event1['time']

                    if ioi < self.rapid_sequence_threshold_ioi:
                        if len(sequence_indices) == 0:
                            sequence_indices.append(idx1)
                            sequence_velocities.append(event1['velocity'])
                        sequence_indices.append(idx2)
                        sequence_velocities.append(event2['velocity'])
                        current_idx += 1
                    else:
                        break

                if len(sequence_indices) >= 3:
                    velocity_range = max(sequence_velocities) - min(sequence_velocities)

                    if velocity_range >= self.rapid_sequence_threshold_velocity_range:
                        latencies = [calculate_latency(v) for v in sequence_velocities]
                        latency_spread = max(latencies) - min(latencies)

                        sensitive_sequences.append({
                            'type': 'sensitive_sequence',
                            'time': voice_events[i][1]['time'],
                            'event_indices': sequence_indices,
                            'voice_id': voice_id,
                            'num_notes': len(sequence_indices),
                            'velocity_range': velocity_range,
                            'latency_spread_ms': latency_spread * 1000,
                            'velocities': sequence_velocities
                        })

        return sensitive_sequences

    def detect_all(self, events):
        """Run all detection rules and return comprehensive report."""
        chords = self.detect_sensitive_chords(events)
        hockets = self.detect_sensitive_hockets(events)
        sequences = self.detect_rapid_sequences(events)

        return {
            'sensitive_chords': chords,
            'sensitive_hockets': hockets,
            'sensitive_sequences': sequences,
            'total_flagged': len(chords) + len(hockets) + len(sequences)
        }


class LatencyRobustnessTransformer:
    """
    Transforms latency-sensitive events to minimize temporal distortion.
    """

    def __init__(self, 
                 chord_compression_factor=0.5,
                 hocket_alignment_strategy='mean',
                 sequence_smoothing_factor=0.7):
        """Initialize transformation parameters."""
        self.chord_compression_factor = chord_compression_factor
        self.hocket_alignment_strategy = hocket_alignment_strategy
        self.sequence_smoothing_factor = sequence_smoothing_factor

    def transform_sensitive_chord(self, events, chord_info):
        """
        Compress velocity range of chord to reduce latency spread.
        """
        event_indices = chord_info['event_indices']
        velocities = chord_info['velocities']

        mean_velocity = np.mean(velocities)

        new_velocities = []
        for v in velocities:
            compressed = mean_velocity + (v - mean_velocity) * (1 - self.chord_compression_factor)
            compressed = max(1, min(1024, int(compressed)))
            new_velocities.append(compressed)

        transformations = []
        for idx, new_vel in zip(event_indices, new_velocities):
            old_vel = events[idx]['velocity']
            events[idx]['velocity'] = new_vel
            transformations.append({
                'event_idx': idx,
                'old_velocity': old_vel,
                'new_velocity': new_vel,
                'transformation': 'chord_compression'
            })

        old_latencies = [calculate_latency(v) for v in velocities]
        new_latencies = [calculate_latency(v) for v in new_velocities]
        old_spread = max(old_latencies) - min(old_latencies)
        new_spread = max(new_latencies) - min(new_latencies)

        return transformations, {
            'old_latency_spread_ms': old_spread * 1000,
            'new_latency_spread_ms': new_spread * 1000,
            'improvement_ms': (old_spread - new_spread) * 1000,
            'improvement_pct': ((old_spread - new_spread) / old_spread * 100) if old_spread > 0 else 0
        }

    def transform_sensitive_hocket(self, events, hocket_info):
        """
        Align velocities of hocket events to reduce latency difference.
        """
        event_indices = hocket_info['event_indices']
        velocities = hocket_info['velocities']

        if self.hocket_alignment_strategy == 'mean':
            target_velocity = int(np.mean(velocities))
        elif self.hocket_alignment_strategy == 'median':
            target_velocity = int(np.median(velocities))
        elif self.hocket_alignment_strategy == 'min':
            target_velocity = min(velocities)
        elif self.hocket_alignment_strategy == 'max':
            target_velocity = max(velocities)
        else:
            target_velocity = int(np.mean(velocities))

        target_velocity = max(1, min(1024, target_velocity))

        transformations = []
        for idx in event_indices:
            old_vel = events[idx]['velocity']
            events[idx]['velocity'] = target_velocity
            transformations.append({
                'event_idx': idx,
                'old_velocity': old_vel,
                'new_velocity': target_velocity,
                'transformation': 'hocket_alignment'
            })

        old_latencies = [calculate_latency(v) for v in velocities]
        new_latency = calculate_latency(target_velocity)
        old_diff = abs(old_latencies[0] - old_latencies[1])
        new_diff = 0

        return transformations, {
            'old_latency_diff_ms': old_diff * 1000,
            'new_latency_diff_ms': new_diff * 1000,
            'improvement_ms': old_diff * 1000,
            'improvement_pct': 100.0
        }

    def transform_sensitive_sequence(self, events, sequence_info):
        """
        Smooth velocity variation in rapid sequences.
        """
        event_indices = sequence_info['event_indices']
        velocities = sequence_info['velocities']

        smoothed_velocities = []
        for i, v in enumerate(velocities):
            if i == 0:
                smoothed = v
            elif i == len(velocities) - 1:
                smoothed = v
            else:
                smoothed = (velocities[i-1] * 0.25 + v * 0.5 + velocities[i+1] * 0.25)

            blended = v + (smoothed - v) * self.sequence_smoothing_factor
            blended = max(1, min(1024, int(blended)))
            smoothed_velocities.append(blended)

        transformations = []
        for idx, new_vel in zip(event_indices, smoothed_velocities):
            old_vel = events[idx]['velocity']
            events[idx]['velocity'] = new_vel
            transformations.append({
                'event_idx': idx,
                'old_velocity': old_vel,
                'new_velocity': new_vel,
                'transformation': 'sequence_smoothing'
            })

        old_latencies = [calculate_latency(v) for v in velocities]
        new_latencies = [calculate_latency(v) for v in smoothed_velocities]
        old_spread = max(old_latencies) - min(old_latencies)
        new_spread = max(new_latencies) - min(new_latencies)

        return transformations, {
            'old_latency_spread_ms': old_spread * 1000,
            'new_latency_spread_ms': new_spread * 1000,
            'improvement_ms': (old_spread - new_spread) * 1000,
            'improvement_pct': ((old_spread - new_spread) / old_spread * 100) if old_spread > 0 else 0
        }

    def transform_all(self, events, detection_results):
        """
        Apply all transformations based on detection results.
        """
        all_transformations = []
        improvement_stats = {
            'chords': [],
            'hockets': [],
            'sequences': []
        }

        for chord_info in detection_results['sensitive_chords']:
            trans, improvement = self.transform_sensitive_chord(events, chord_info)
            all_transformations.extend(trans)
            improvement_stats['chords'].append(improvement)

        for hocket_info in detection_results['sensitive_hockets']:
            trans, improvement = self.transform_sensitive_hocket(events, hocket_info)
            all_transformations.extend(trans)
            improvement_stats['hockets'].append(improvement)

        for sequence_info in detection_results['sensitive_sequences']:
            trans, improvement = self.transform_sensitive_sequence(events, sequence_info)
            all_transformations.extend(trans)
            improvement_stats['sequences'].append(improvement)

        return all_transformations, improvement_stats


def apply_latency_robustness_filter(event_list, 
                                     chord_compression=0.5,
                                     hocket_strategy='mean',
                                     sequence_smoothing=0.7,
                                     verbose=False):
    """
    Complete latency-robustness post-processing pipeline.

    This function implements a post-processing filter that:
    1. Detects latency-sensitive musical events
    2. Transforms them to minimize temporal distortion
    3. Returns modified event list and detailed report

    Parameters:
    -----------
    event_list : list of dict
        Event list in format: [{'time': float, 'pitch': int, 'velocity': int, 'voice_id': int}, ...]
    chord_compression : float (0-1)
        Velocity range compression factor for chords
    hocket_strategy : str
        Velocity alignment strategy for hockets ('mean', 'median', 'min', 'max')
    sequence_smoothing : float (0-1)
        Smoothing factor for rapid sequences
    verbose : bool
        Print detailed report

    Returns:
    --------
    dict with keys:
        - 'transformed_events': modified event list
        - 'detection_results': sensitivity detection report
        - 'transformations': list of applied transformations
        - 'improvement_stats': improvement statistics
        - 'summary': summary statistics
    """
    transformed_events = copy.deepcopy(event_list)

    detector = LatencySensitivityDetector()
    transformer = LatencyRobustnessTransformer(
        chord_compression_factor=chord_compression,
        hocket_alignment_strategy=hocket_strategy,
        sequence_smoothing_factor=sequence_smoothing
    )

    detection_results = detector.detect_all(transformed_events)
    transformations, improvement_stats = transformer.transform_all(
        transformed_events, 
        detection_results
    )

    unique_affected = set(t['event_idx'] for t in transformations)

    total_improvement = (
        sum([s['improvement_ms'] for s in improvement_stats['chords']]) +
        sum([s['improvement_ms'] for s in improvement_stats['hockets']]) +
        sum([s['improvement_ms'] for s in improvement_stats['sequences']])
    )

    summary = {
        'total_events': len(event_list),
        'flagged_events': detection_results['total_flagged'],
        'modified_events': len(unique_affected),
        'transformations_applied': len(transformations),
        'total_latency_improvement_ms': total_improvement,
        'num_chords_processed': len(improvement_stats['chords']),
        'num_hockets_processed': len(improvement_stats['hockets']),
        'num_sequences_processed': len(improvement_stats['sequences'])
    }

    if verbose:
        print("Latency Robustness Filter Applied")
        print(f"  Input events: {summary['total_events']}")
        print(f"  Flagged as sensitive: {summary['flagged_events']}")
        print(f"  Events modified: {summary['modified_events']}")
        print(f"  Total latency improvement: {summary['total_latency_improvement_ms']:.2f} ms")

    return {
        'transformed_events': transformed_events,
        'detection_results': detection_results,
        'transformations': transformations,
        'improvement_stats': improvement_stats,
        'summary': summary
    }
