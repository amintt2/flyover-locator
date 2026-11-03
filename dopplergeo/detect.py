"""Automatic detection of propeller flyovers in a spectrogram."""

import numpy as np

from .audio import event_from_fit, fit_flyover, track_ridge


def detect_flyovers(spec, c=343.0, fmin=50.0, fmax=300.0, win_s=110.0, step_s=20.0,
                    min_drop=0.06, max_rms=1.5, min_strength=1.5, log=print):
    """Scan the recording with overlapping windows, track the strongest line in
    each, and keep windows whose ridge fits a flyover Doppler curve.

    Returns a list of dicts (fit + 'strength' + 'window'), strongest first,
    de-duplicated so each flyover appears once.
    """
    found = []
    t_end = spec.duration
    starts = np.arange(0.0, max(t_end - win_s, 0.0) + step_s, step_s)
    for t0 in starts:
        t1 = min(t0 + win_s, t_end)
        if t1 - t0 < 40:
            continue
        t, f, w = track_ridge(spec, t0, t1, fmin, fmax, min_z=1.0)
        if len(t) < 0.35 * (t1 - t0) / spec.hop:
            continue
        try:
            fit = fit_flyover(t, f, w, c)
        except (ValueError, np.linalg.LinAlgError):
            continue
        drop = 2 * fit["v"] / c
        tc = fit["tc"]
        if not (t0 + 0.2 * (t1 - t0) <= tc <= t1 - 0.2 * (t1 - t0)):
            continue
        if fit["rms_hz"] > max_rms or drop < min_drop or not (15 <= fit["v"] <= 160):
            continue
        inl = fit.pop("inliers")
        near = (np.abs(t - tc) < 30) & inl
        # the curve must be followed continuously around the CPA, not just touched
        if near.sum() < 0.6 * 60 / spec.hop:
            continue
        strength = float(np.median(w[near]))
        if strength < min_strength:
            continue
        span = t[inl & (w > 1.5)]
        win = (max(tc - 60, float(span.min()) if span.size else tc - 40, 0.0),
               min(tc + 60, float(span.max()) if span.size else tc + 40, t_end))
        fit.update(strength=strength, window=win)
        found.append(fit)
    found.sort(key=lambda x: -x["strength"])
    events = []
    for fit in found:
        if all(abs(fit["tc"] - e["tc"]) > 40 for e in events):
            events.append(fit)
    events.sort(key=lambda x: x["tc"])
    for e in events:
        log(f"  flyover at {e['tc']:.1f}s: f0={e['f0']:.1f}Hz v={e['v']:.0f}m/s d={e['d']:.0f}m "
            f"rms={e['rms_hz']:.2f}Hz strength={e['strength']:.1f}")
    return events


def events_for_search(fits, max_events=3, min_ratio=0.6):
    """Pick the strongest flyovers and turn them into search events."""
    top = max(f["strength"] for f in fits)
    best = sorted((f for f in fits if f["strength"] >= min_ratio * top), key=lambda x: -x["strength"])[:max_events]
    best.sort(key=lambda x: x["tc"])
    return [event_from_fit(f, f["window"]) for f in best]
