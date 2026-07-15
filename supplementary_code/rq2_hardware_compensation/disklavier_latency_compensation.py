"""
Disklavier Latency Compensation Module

This module implements velocity-dependent latency models and pre-compensation
algorithms for the Yamaha Disklavier Pro, based on empirical data from 
Goebl & Bresin (2003) showing latency range of 10-30ms with the physical
principle "Louder = Earlier" (higher velocity → lower latency).

Author: Generated for algorithmic composition research
Date: 2024
Reference: Goebl, W., & Bresin, R. (2003). Measurement and reproduction of 
          piano pedaling parameters using a computer-controlled grand piano.
"""

import numpy as np

# Hardware constants from Goebl & Bresin (2003)
VELOCITY_MIN = 0
VELOCITY_MAX = 1023  # 10-bit resolution
LATENCY_MIN = 10     # ms (at maximum velocity - loudest)
LATENCY_MAX = 30     # ms (at minimum velocity - softest)


def latency_linear(velocity, v_max=VELOCITY_MAX, 
                   latency_min=LATENCY_MIN, latency_max=LATENCY_MAX):
    """
    Linear latency model: delay_ms = a - b * (velocity / v_max)

    This is the simplest model with constant rate of change.
    Suitable as a baseline or when true relationship is unknown.

    Parameters
    ----------
    velocity : float or array-like
        MIDI velocity value [0, 1023]
    v_max : float, optional
        Maximum velocity (default: 1023)
    latency_min : float, optional
        Minimum latency at max velocity in ms (default: 10)
    latency_max : float, optional
        Maximum latency at min velocity in ms (default: 30)

    Returns
    -------
    float or array-like
        Latency in milliseconds

    Examples
    --------
    >>> latency_linear(0)
    30.0
    >>> latency_linear(1023)
    10.0
    >>> latency_linear(512)
    19.99
    """
    v_norm = velocity / v_max
    delay_ms = latency_max - (latency_max - latency_min) * v_norm
    return delay_ms


def latency_powerlaw(velocity, exponent=2.0, v_max=VELOCITY_MAX, 
                     latency_min=LATENCY_MIN, latency_max=LATENCY_MAX):
    """
    Power-law latency model: delay_ms = a - b * (velocity/v_max)^c

    Provides non-linear relationship with tunable exponent.
    - c > 1: Latency decreases slowly at low velocities, rapidly at high
    - c < 1: Latency decreases rapidly at low velocities, slowly at high
    - c = 1: Reduces to linear model

    Parameters
    ----------
    velocity : float or array-like
        MIDI velocity value [0, 1023]
    exponent : float, optional
        Power-law exponent c (default: 2.0)
    v_max : float, optional
        Maximum velocity (default: 1023)
    latency_min : float, optional
        Minimum latency at max velocity in ms (default: 10)
    latency_max : float, optional
        Maximum latency at min velocity in ms (default: 30)

    Returns
    -------
    float or array-like
        Latency in milliseconds

    Examples
    --------
    >>> latency_powerlaw(512, exponent=2.0)
    24.99
    >>> latency_powerlaw(512, exponent=0.5)
    15.85
    """
    v_norm = velocity / v_max
    delay_ms = latency_max - (latency_max - latency_min) * (v_norm ** exponent)
    return delay_ms


def latency_logarithmic(velocity, v_max=VELOCITY_MAX, 
                        latency_min=LATENCY_MIN, latency_max=LATENCY_MAX):
    """
    Logarithmic latency model: delay_ms = a - b * log(velocity+1) / log(v_max+1)

    Models perceptual phenomena following Weber-Fechner law.
    Latency decreases rapidly at low velocities, slowly at high velocities.

    Parameters
    ----------
    velocity : float or array-like
        MIDI velocity value [0, 1023]
    v_max : float, optional
        Maximum velocity (default: 1023)
    latency_min : float, optional
        Minimum latency at max velocity in ms (default: 10)
    latency_max : float, optional
        Maximum latency at min velocity in ms (default: 30)

    Returns
    -------
    float or array-like
        Latency in milliseconds

    Examples
    --------
    >>> latency_logarithmic(0)
    30.0
    >>> latency_logarithmic(512)
    11.99
    """
    v_norm = np.log(velocity + 1) / np.log(v_max + 1)
    delay_ms = latency_max - (latency_max - latency_min) * v_norm
    return delay_ms


def precompensate_onset_linear(intended_onset_ms, velocity):
    """
    Pre-compensation algorithm using linear latency model.

    Adjusts note trigger time to ensure sound arrives at intended time.

    Parameters
    ----------
    intended_onset_ms : float
        Desired time when sound should reach the listener's ear (ms)
    velocity : float or array-like
        MIDI velocity value [0, 1023]

    Returns
    -------
    float or array-like
        Adjusted trigger time in milliseconds

    Examples
    --------
    >>> precompensate_onset_linear(1000, 0)     # soft note
    970.0
    >>> precompensate_onset_linear(1000, 1023)  # loud note
    990.0
    """
    delay = latency_linear(velocity)
    adjusted_onset_ms = intended_onset_ms - delay
    return adjusted_onset_ms


def precompensate_onset_powerlaw(intended_onset_ms, velocity, exponent=2.0):
    """
    Pre-compensation algorithm using power-law latency model.

    Parameters
    ----------
    intended_onset_ms : float
        Desired time when sound should reach the listener's ear (ms)
    velocity : float or array-like
        MIDI velocity value [0, 1023]
    exponent : float, optional
        Power-law exponent (default: 2.0)

    Returns
    -------
    float or array-like
        Adjusted trigger time in milliseconds
    """
    delay = latency_powerlaw(velocity, exponent=exponent)
    adjusted_onset_ms = intended_onset_ms - delay
    return adjusted_onset_ms


def precompensate_onset_logarithmic(intended_onset_ms, velocity):
    """
    Pre-compensation algorithm using logarithmic latency model.

    Parameters
    ----------
    intended_onset_ms : float
        Desired time when sound should reach the listener's ear (ms)
    velocity : float or array-like
        MIDI velocity value [0, 1023]

    Returns
    -------
    float or array-like
        Adjusted trigger time in milliseconds
    """
    delay = latency_logarithmic(velocity)
    adjusted_onset_ms = intended_onset_ms - delay
    return adjusted_onset_ms


def compensate_chord(note_data, intended_onset_ms, model='linear', **kwargs):
    """
    Apply pre-compensation to a chord (multiple simultaneous notes).

    Parameters
    ----------
    note_data : list of dict
        List of note dictionaries with 'velocity' key
    intended_onset_ms : float
        Desired simultaneous arrival time
    model : str, optional
        Latency model to use: 'linear', 'powerlaw', or 'logarithmic'
    **kwargs : dict
        Additional parameters (e.g., exponent for power-law)

    Returns
    -------
    list of float
        Adjusted trigger times for each note

    Examples
    --------
    >>> notes = [{'velocity': 128}, {'velocity': 640}, {'velocity': 896}]
    >>> compensate_chord(notes, 1000.0, model='linear')
    [972.5, 982.51, 987.52]
    """
    trigger_times = []

    for note in note_data:
        velocity = note['velocity']

        if model == 'linear':
            trigger_time = precompensate_onset_linear(intended_onset_ms, velocity)
        elif model == 'powerlaw':
            exponent = kwargs.get('exponent', 2.0)
            trigger_time = precompensate_onset_powerlaw(intended_onset_ms, velocity, exponent)
        elif model == 'logarithmic':
            trigger_time = precompensate_onset_logarithmic(intended_onset_ms, velocity)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'linear', 'powerlaw', or 'logarithmic'.")

        trigger_times.append(trigger_time)

    return trigger_times


if __name__ == "__main__":
    # Test the module
    print("Disklavier Latency Compensation Module - Test")
    print("=" * 60)

    # Test individual functions
    print("\nTest velocities: 0 (softest), 512 (medium), 1023 (loudest)")
    test_velocities = [0, 512, 1023]

    print("\nLinear model:")
    for v in test_velocities:
        print(f"  v={v:4d} → latency={latency_linear(v):6.2f}ms")

    print("\nPower-law model (c=2.0):")
    for v in test_velocities:
        print(f"  v={v:4d} → latency={latency_powerlaw(v, 2.0):6.2f}ms")

    print("\nLogarithmic model:")
    for v in test_velocities:
        print(f"  v={v:4d} → latency={latency_logarithmic(v):6.2f}ms")

    # Test chord compensation
    print("\nChord compensation test:")
    chord = [{'velocity': 128}, {'velocity': 512}, {'velocity': 896}]
    triggers = compensate_chord(chord, 1000.0, model='linear')
    print(f"  Intended onset: 1000.0ms")
    print(f"  Trigger times: {triggers}")
