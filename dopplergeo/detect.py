"""Automatic detection of propeller flyovers in a spectrogram."""

import numpy as np

from .audio import event_from_fit, fit_flyover, harmonic_spectrogram, track_ridge, track_ridges
from .physics import straight_flyover


GROUND = {"car": "voiture / moto", "truck": "camion / bus", "boat": "bateau", "train": "train ?"}


def detect_flyovers(spec, c=343.0, fmin=50.0, fmax=300.0, win_s=110.0, step_s=20.0,
                    min_drop=0.06, max_rms=1.5, min_strength=1.5, log=print, clips=None, others=None):
    """Scan the recording with overlapping windows, track the strongest line in
    each, and keep windows whose ridge fits a flyover Doppler curve.

    clips: continuous spans [(a, b)] of an edited video; windows never cross a cut
    and each fit records its clip index.

    others: optional list that receives the Doppler passes no aircraft could make
    (too close or too slow), each with a guessed 'kind' and French 'label'.

    Returns a list of dicts (fit + 'strength' + 'window' + 'clip'), strongest first,
    de-duplicated so each flyover appears once.
    """
    found = []
    harm = harmonic_spectrogram(spec, fmin * 0.8, fmax * 2 / 3)
    for ci, (ca, cb) in enumerate(clips or [(0.0, spec.duration)]):
        found += _scan(spec, c, ca, cb, ci, fmin, fmax, win_s, step_s, min_drop, max_rms, min_strength)
        # distant passes: each harmonic alone is too faint, all of them together are not
        for fit in _scan(harm, c, ca, cb, ci, harm.freqs[0], harm.freqs[-1], win_s, step_s, min_drop, max_rms,
                         min_strength):
            if fit["kind"] == "aircraft":
                found.append(_loudest_harmonic(spec, fit, c))
    events = _dedupe([f for f in found if f["kind"] == "aircraft"], 40)
    if others is not None:
        ground = [f for f in found if f["kind"] != "aircraft"
                  and all(abs(f["tc"] - e["tc"]) > 20 for e in events)]
        for g in _dedupe(ground, 20):
            g["label"] = GROUND[g["kind"]]
            others.append(g)
    for e in events:
        log(f"  flyover at {e['tc']:.1f}s: f0={e['f0']:.1f}Hz v={e['v']:.0f}m/s d={e['d']:.0f}m "
            f"rms={e['rms_hz']:.2f}Hz strength={e['strength']:.1f}" + (f" clip {e['clip'] + 1}" if clips else ""))
    return events


def _bands(fmin, fmax, width=2.5, step=1.6):
    """Overlapping bands, each wide enough for a full Doppler drop, so the loudest
    line elsewhere (a motor, a road, music) cannot pull the tracker away."""
    out, a = [], fmin
    while True:
        b = min(a * width, fmax)
        out.append((a, b))
        if b >= fmax:
            return out
        a *= step


def _ridges(spec, windows, fa, fb, **kw):
    """Ridges for [(t0, t1)]: full-length windows in one batch, shorter ones singly."""
    full = [t0 for t0, t1 in windows if t1 - t0 >= windows[0][1] - windows[0][0] - 1e-6]
    out = track_ridges(spec, full, windows[0][1] - windows[0][0], fa, fb, **kw) if full else {}
    kw.pop("pool", None)
    for t0, t1 in windows:
        if float(t0) not in out:
            out[float(t0)] = track_ridge(spec, t0, t1, fa, fb, **kw)
    return out


def _scan(spec, c, ca, cb, clip, fmin, fmax, win_s, step_s, min_drop, max_rms, min_strength):
    windows = [(t0, min(t0 + win_s, cb)) for t0 in np.arange(ca, max(cb - win_s, ca) + step_s, step_s)]
    windows = [(t0, t1) for t0, t1 in windows if t1 - t0 >= 40]
    found = []
    if not windows:
        return found
    for fa, fb in _bands(fmin, fmax):
        ridges = _ridges(spec, windows, fa, fb, min_z=1.0)
        for t0, t1 in windows:
            ridge = ridges[float(t0)]
            fit = _fit_window(spec, c, t0, t1, ridge, ca, cb, min_drop, max_rms, min_strength)
            # the tracker may run along a steady engine line before or after the pass:
            # fit each continuous stretch of the ridge on its own
            for piece in ([] if fit else _pieces(ridge, 0.35 * (t1 - t0))):
                fit = _fit_window(spec, c, t0, t1, piece, ca, cb, min_drop, max_rms, min_strength)
                if fit:
                    break
            if fit:
                fit["clip"] = clip
                found.append(fit)
    return found


def _loudest_harmonic(spec, fit, c, kmax=6):
    """A fit on the harmonic sum is in terms of the fundamental, which may be silent:
    move it onto the harmonic that is actually loudest (v, d and tc are unchanged)."""
    t = np.arange(*fit["window"], spec.hop)
    curve = straight_flyover(t, fit["f0"], fit["v"], fit["d"], fit["tc"], c)
    ks = [k for k in range(1, kmax + 1) if k * curve.max() <= spec.freqs[-1]] or [1]
    k = max(ks, key=lambda k: float(np.median(spec.lookup(t, k * curve))))
    fit.update(f0=fit["f0"] * k, rms_hz=fit["rms_hz"] * k, harmonic=k)
    return fit


def _pieces(ridge, min_s, max_gap_s=2.0, max_jump_hz=3.0):
    """Stretches of a ridge with no time gap or frequency jump, longest first."""
    t, f, w = ridge
    if len(t) < 2:
        return []
    cut = np.where((np.diff(t) > max_gap_s) | (np.abs(np.diff(f)) > max_jump_hz))[0] + 1
    out = [(t[a:b], f[a:b], w[a:b]) for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)])]
    if len(out) == 1:
        return []  # one stretch: the whole ridge, already tried
    return sorted((p for p in out if p[0][-1] - p[0][0] >= min_s), key=lambda p: p[0][0] - p[0][-1])


def _dedupe(found, gap):
    """Strongest first, one per pass, then in time order."""
    kept = []
    for fit in sorted(found, key=lambda x: -x["strength"]):
        if all(abs(fit["tc"] - e["tc"]) > gap for e in kept):
            kept.append(fit)
    return sorted(kept, key=lambda x: x["tc"])


def _ground_kind(fit):
    """A Doppler pass no aircraft could make, guessed from speed and engine tone:
    boats are slow, heavy diesels (trucks, buses) have a low tone. A guess, not a fact."""
    v, d = fit["v"], fit["d"]
    if v < 1.5 or d > 600 or 2 * v / 343.0 < 0.015:
        return None
    if v < 9:
        return "boat"
    if v <= 45:
        return "truck" if fit["f0"] < 80 else "car"
    return "train"


def _fit_window(spec, c, t0, t1, ridge, ca, cb, min_drop, max_rms, min_strength, min_d=150.0):
    t, f, w = ridge
    if len(t) < 0.35 * (t1 - t0) / spec.hop:
        return None
    try:
        fit = fit_flyover(t, f, w, c)
    except (ValueError, np.linalg.LinAlgError):
        return None
    drop = 2 * fit["v"] / c
    tc = fit["tc"]
    if not (t0 + 0.2 * (t1 - t0) <= tc <= t1 - 0.2 * (t1 - t0)):
        return None
    if fit["rms_hz"] > max_rms:
        return None
    # an aircraft does not pass within a few tens of metres or crawl: those are road
    # traffic, boats or trains, kept apart as "other sounds"
    kind = "aircraft"
    if drop < min_drop or not (25 <= fit["v"] <= 160) or fit["d"] < min_d:
        kind = _ground_kind(fit)
        if kind is None:
            return None
    inl = fit.pop("inliers")
    near = (np.abs(t - tc) < 30) & inl
    # the curve must be followed continuously around the CPA, not just touched
    if near.sum() < 0.6 * 60 / spec.hop:
        return None
    if _drop_missing(t, f, inl, fit, c):
        return None
    strength = float(np.median(w[near]))
    if strength < min_strength:
        return None
    span = t[inl & (w > 1.5)]
    win = (max(tc - 60, float(span.min()) if span.size else tc - 40, ca),
           min(tc + 60, float(span.max()) if span.size else tc + 40, cb))
    fit.update(strength=strength, window=win, kind=kind)
    return fit


def _drop_missing(t, f, inl, fit, c, share=0.5):
    """True when less than `share` of the middle of the drop was tracked (a jump, not a curve)."""
    f0, v = fit["f0"], fit["v"]
    hi, lo = f0 / (1 - v / c), f0 / (1 + v / c)
    mid = inl & (f > lo + 0.3 * (hi - lo)) & (f < hi - 0.3 * (hi - lo))
    tv = fit["d"] / max(v, 1e-6)
    return mid.sum() < share * min(tv, 10.0) / max(np.median(np.diff(t)) if len(t) > 1 else 1.0, 1e-3)


def detect_short(spec, c=343.0, win_s=14.0, step_s=5.0, max_tv=4.0, min_strength=1.5, log=print, clips=None,
                 fmin=None, fmax=None, label="fast pass", cover=0.5, min_d=80.0):
    """Close, fast passes (jets on approach, low flybys): the frequency drop lasts only
    d/v = 1-4 s and is often high pitched, with music or speech on top. Short windows are
    tracked in overlapping frequency bands so a louder sound elsewhere cannot hide the
    line, and a fit is kept only if it follows the Doppler S curve through its drop.
    `spec` is a wide-band spectrogram (short frames). With longer windows and a larger
    max_tv the same search finds slower passes above the main detector's band;
    `cover` is the share of the window the line must be seen in (lower under speech).
    """
    lo_f, hi_f = max(float(spec.freqs[0]), fmin or 0), min(float(spec.freqs[-1]), fmax or np.inf)
    bands, a = [], lo_f
    while a < hi_f:
        bands.append((a, min(a * 2.5, hi_f)))
        a *= 1.6
    found = []
    for ci, (ca, cb) in enumerate(clips or [(0.0, spec.duration)]):
        windows = [(t0, min(t0 + win_s, cb)) for t0 in np.arange(ca, max(cb - win_s, ca) + step_s, step_s)]
        windows = [(t0, t1) for t0, t1 in windows if t1 - t0 >= 6]
        if not windows:
            continue
        for fa, fb in bands:
            # a 300 m/s pass at 100 m on a tone at the top of the band, per frame
            # bins finer than ~0.4% of the frequency add cost, not information
            pool = max(1, int(fa / 250 / spec.df))
            ridges = _ridges(spec, windows, fa, fb, max_jump_hz=0.15 * fb * spec.hop, jump_cost=0.5, min_z=1.0,
                             pool=pool)
            for t0, t1 in windows:
                fit = _short_fit(spec, c, t0, t1, ridges[float(t0)], max_tv, min_strength, cover, min_d)
                if fit:
                    fit["clip"] = ci
                    found.append(fit)
    found.sort(key=lambda x: -x["strength"])
    events = []
    for fit in found:
        if all(abs(fit["tc"] - e["tc"]) > 10 for e in events):
            events.append(fit)
    events.sort(key=lambda x: x["tc"])
    for e in events:
        log(f"  {label} at {e['tc']:.1f}s: f0={e['f0']:.0f}Hz v={e['v']:.0f}m/s d={e['d']:.0f}m "
            f"drop {e['d'] / e['v']:.1f}s strength={e['strength']:.1f}")
    return events


def _short_fit(spec, c, t0, t1, ridge, max_tv, min_strength, cover=0.5, min_d=80.0):
    t, f, w = ridge
    if len(t) < cover * (t1 - t0) / spec.hop:
        return None
    tol = max(3.0, 0.006 * float(np.median(f)))
    try:
        fit = fit_flyover(t, f, w, c, tol_hz=tol)
    except (ValueError, np.linalg.LinAlgError):
        return None
    tc, v, d = fit["tc"], fit["v"], fit["d"]
    tv = d / max(v, 1e-6)
    if not (t0 + 0.3 * (t1 - t0) <= tc <= t1 - 0.3 * (t1 - t0)) or not (25 <= v <= 300) or tv > max_tv or d < min_d:
        return None
    if d < 150 and v < 60:  # slow and very close: a boat or a vehicle, not an aircraft
        return None
    if fit["rms_hz"] > tol / 2:
        return None
    inl = fit.pop("inliers")
    if _drop_missing(t, f, inl, fit, c, share=min(0.5, cover)):
        return None
    # the drop itself must be followed, and most of the modelled shift seen in the data
    half = min(max(1.0, 2 * tv), 0.3 * (t1 - t0))
    near = (np.abs(t - tc) < half) & inl
    if near.sum() < 1.2 * cover * 2 * half / spec.hop:
        return None
    seen = np.percentile(f[inl], 95) - np.percentile(f[inl], 5)
    if seen < 0.5 * fit["f0"] * 2 * v / c:
        return None
    strength = float(np.median(w[near]))
    if strength < min_strength:
        return None
    span = max(6.0, min(20.0, 4 * tv))
    fit.update(strength=strength, window=(max(tc - span, t0 - 6), min(tc + span, t1 + 6)), band="wide")
    return fit


def events_for_search(fits, max_events=3, min_ratio=0.6):
    """Pick the strongest flyovers and turn them into search events."""
    top = max(f["strength"] for f in fits)
    best = sorted((f for f in fits if f["strength"] >= min_ratio * top), key=lambda x: -x["strength"])[:max_events]
    best.sort(key=lambda x: x["tc"])
    return [event_from_fit(f, f["window"]) for f in best]
