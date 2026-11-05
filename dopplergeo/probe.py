"""Quick look at a video before downloading all of it: is its soundtrack worth a scan?

A few short excerpts (30 s, spread over the video) and its metadata give an indicative
score. The thresholds come from a handful of checked videos and webcams, not from a
calibrated model: the score ranks videos, it is not a probability of finding a plane.
"""

import re

import numpy as np

from .audio import load_audio, spectrogram
from .survey import audio_stats

SR = 1000
MONTAGE = re.compile(r"compilation|moments|best of|funniest|highlights?|#shorts|tiktok|clips?\b|reacts?", re.I)
RAW = re.compile(r"full stream|\bvod\b|\birl\b|walk(ing)?\b|live ?stream|\bstream\b|webcam|live cam|\bcam\b|"
                 r"plane ?spotting|airport|beach", re.I)


def steady_lines(x, sr=SR):
    """Narrow lines lasting through the excerpt (mains hum, engines, air conditioning) in
    40-400 Hz. An untouched soundtrack has some; denoising and noise gates remove them,
    together with the tone of a passing propeller."""
    s = spectrogram(x, sr, 40.0, 400.0, 1.0, 0.25)
    med = np.median(s.db, axis=1)
    k = int(8 / s.df) | 1
    pad = np.pad(med, k // 2, mode="edge")
    local = np.array([np.median(pad[i:i + k]) for i in range(len(med))])
    peak = (med - local > 6) & (med >= np.maximum(np.r_[med[1:], -np.inf], np.r_[-np.inf, med[:-1]]))
    # a line must hold in most of the excerpt, not be one loud note
    on = (s.db[peak] - local[peak, None]) > 3
    return int(np.sum(on.mean(axis=1) > 0.6))


def excerpt_stats(path):
    x, sr = load_audio(path, sr=SR)
    if len(x) < 5 * sr:
        raise ValueError("excerpt too short")
    return audio_stats(x, sr) | {"lines": steady_lines(x, sr), "seconds": round(len(x) / sr)}


def score(excerpts, info):
    """excerpts: excerpt_stats of each sample; info: duration, title, live_status.
    Returns 0-100 and the reasons, in French, for the extension."""
    s, why = 50, []
    dur = info.get("duration") or 0
    title = info.get("title") or ""
    if dur and dur < 300:
        s -= 30; why.append(("-", "vidéo de moins de 5 min : un passage dure 1 à 2 min, peu de chances"))
    elif dur and dur < 600:
        s -= 10; why.append(("-", "moins de 10 min"))
    elif dur >= 3600:
        s += 15; why.append(("+", f"longue ({dur / 3600:.1f} h) : plus de temps pour qu'un avion passe"))
    elif dur >= 1800:
        s += 10; why.append(("+", "plus de 30 min"))
    if info.get("live_status") == "was_live":
        s += 15; why.append(("+", "rediffusion d'un live : son brut, sans montage"))
    if MONTAGE.search(title):
        s -= 20; why.append(("-", "titre de montage / compilation : extraits courts, coupés"))
    elif RAW.search(title):
        s += 5; why.append(("+", "titre de stream brut, balade ou caméra"))

    if excerpts:
        silent = float(np.median([e["silent_share"] for e in excerpts]))
        floor = float(np.median([e["floor_50_300_db"] for e in excerpts]))
        lines = float(np.median([e["lines"] for e in excerpts]))
        if silent > 0.15:
            s -= 40; why.append(("-", f"{silent:.0%} de silence numérique : son coupé par un noise gate"))
        elif silent > 0.05:
            s -= 20; why.append(("-", f"{silent:.0%} de silence numérique : son probablement traité"))
        if floor < -95:
            s -= 30; why.append(("-", f"fond à {floor:.0f} dB : bien trop bas pour un micro réel, son débruité"))
        elif floor > -45:
            s -= 15; why.append(("-", f"fond à {floor:.0f} dB : très bruyant (rue, vent, musique)"))
        else:
            s += 5; why.append(("+", f"fond à {floor:.0f} dB : bruit ambiant naturel"))
        if lines >= 3:
            s += 10; why.append(("+", f"{lines:.0f} sons continus (moteurs, clim, secteur) : rien n'a été filtré"))
        elif lines == 0:
            s -= 5; why.append(("-", "aucun son continu : voix ou musique par-dessus, ou spectre lissé"))
    return max(0, min(100, s)), [{"sign": a, "text": b} for a, b in why]
