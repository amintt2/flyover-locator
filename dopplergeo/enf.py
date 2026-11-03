"""Electrical network frequency (mains hum) analysis.

Microphones near powered devices pick up the 50/60 Hz hum and its harmonics.
The nominal frequency narrows the region; the tiny fluctuations of the actual
frequency are identical across one synchronous grid at a given second, so a
trace correlated with a reference series identifies the grid (and the time).
"""

import numpy as np
from scipy.signal import welch

REGIONS = {
    50: "Europe, Afrique, Asie (sauf Japon de l'Est partiel), Océanie, Argentine/Chili",
    60: "Amérique du Nord, Brésil, Colombie, Venezuela, Corée du Sud, Taïwan, Philippines, Arabie saoudite, ouest du Japon",
}

# Where a search can run without a user-picked zone (good ADS-B coverage).
DEFAULT_AREAS = {
    50: {"name": "Europe", "lat_min": 36.0, "lat_max": 60.0, "lon_min": -10.0, "lon_max": 25.0},
    60: {"name": "USA (48 états)", "lat_min": 24.5, "lat_max": 49.5, "lon_min": -125.0, "lon_max": -66.5},
}


def _prominence_db(f, pxx, target, half=0.6, ctx=6.0):
    peak = (f >= target - half) & (f <= target + half)
    ctxm = (f >= target - ctx) & (f <= target + ctx) & ~peak
    if not peak.any() or not ctxm.any():
        return 0.0
    return float(10 * np.log10(pxx[peak].max() / np.median(pxx[ctxm])))


def analyze_enf(x, sr, win_s=8.0, hop_s=1.0, min_db=8.0):
    """Detect the grid nominal frequency and extract the ENF trace.

    Returns dict(nominal, strength (dB), harmonic, regions, t (s), f (Hz), quality).
    """
    out = {"nominal": None, "strength": 0.0, "harmonic": None, "regions": None, "t": [], "f": [], "quality": 0.0}
    if len(x) < sr * 10:
        return out
    nper = int(min(len(x), 16 * sr))
    f, pxx = welch(x, sr, nperseg=nper, noverlap=nper // 2)
    best = (0.0, None, None)
    for nominal in (50, 60):
        for h in (1, 2, 3):
            if h * nominal + 7 >= sr / 2:
                continue
            db = _prominence_db(f, pxx, h * nominal)
            if db > best[0]:
                best = (db, nominal, h)
    if best[0] < min_db:
        out["strength"] = best[0]
        return out
    db, nominal, h = best
    out.update(nominal=nominal, strength=db, harmonic=h, regions=REGIONS[nominal])

    # ENF trace on the strongest harmonic, sub-bin peak interpolation
    n = int(win_s * sr)
    hop = int(hop_s * sr)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = np.where(np.abs(freqs - h * nominal) <= 0.5 * h)[0]
    win = np.hanning(n)
    ts, fs, snr = [], [], []
    for s in range(0, len(x) - n, hop):
        spec = np.abs(np.fft.rfft(x[s : s + n] * win, nfft))[band]
        k = int(spec.argmax())
        if 0 < k < len(spec) - 1:
            a, b, c = np.log(spec[k - 1 : k + 2] + 1e-12)
            d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
        else:
            d = 0.0
        fs.append((freqs[band][k] + d * (freqs[1] - freqs[0])) / h)
        ts.append((s + n / 2) / sr)
        snr.append(spec[k] / (np.median(spec) + 1e-12))
    fs = np.array(fs)
    ok = np.abs(fs - nominal) < 0.25  # grids stay within ±0.2 Hz
    out["t"] = np.round(np.array(ts)[ok], 2).tolist()
    out["f"] = np.round(fs[ok], 5).tolist()
    out["quality"] = float(np.median(snr)) if snr else 0.0
    return out


def correlate_reference(t, f, ref_t, ref_f, t0):
    """Pearson correlation between a recording's ENF trace (times relative to
    recording start t0, unix) and a reference grid series (unix times)."""
    t, f, ref_t, ref_f = map(lambda a: np.asarray(a, float), (t, f, ref_t, ref_f))
    if len(t) < 30 or len(ref_t) < 30:
        return None
    order = np.argsort(ref_t)
    ref = np.interp(t + t0, ref_t[order], ref_f[order], left=np.nan, right=np.nan)
    m = np.isfinite(ref)
    if m.sum() < 30 or np.std(f[m]) == 0 or np.std(ref[m]) == 0:
        return None
    r = float(np.corrcoef(f[m], ref[m])[0, 1])
    return r if np.isfinite(r) else None
