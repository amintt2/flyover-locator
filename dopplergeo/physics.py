"""Doppler physics for a moving source and a fixed listener."""

import numpy as np


def speed_of_sound(temp_c=15.0):
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


def straight_flyover(t, f0, v, d, tc, c=343.0):
    """Frequency heard at time t from a source in straight, constant-speed flight.

    v: source speed (m/s), d: slant distance at closest approach (m),
    tc: time at which the sound *emitted* at closest approach reaches the listener.
    Uses the exact retarded-time solution, not the naive instantaneous one.
    """
    t = np.asarray(t, float)
    tp = t - tc + d / c  # time measured from the CPA emission instant
    tau = (c * c * tp - c * np.sqrt(v * v * tp * tp + (1 - (v / c) ** 2) * d * d)) / (c * c - v * v)
    vr = v * v * tau / np.sqrt(d * d + v * v * tau * tau)
    return f0 * c / (c + vr)


def heard_curves(tau, src, obs, f0, c):
    """Emission-parametrised Doppler model along a real trajectory.

    tau: (N,) emission times, src: (N, 3) source positions, obs: (M, 3) listeners.
    Each emission instant tau_i produces a sound that arrives at tau_i + r/c with
    frequency f0 * c / (c + dr/dtau), so the curve is built from where the plane was
    when it emitted, which is what the listener actually hears.
    Returns (t_arrival, freq, range), each (M, N).
    """
    r = np.linalg.norm(src[None, :, :] - obs[:, None, :], axis=2)
    vr = np.gradient(r, tau, axis=1)
    t_arr = tau[None, :] + r / c
    f = f0 * c / (c + vr)
    return t_arr, f, r
