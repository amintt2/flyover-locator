"""Close jet passes: a 2-4 s, high-pitched frequency drop with music on top,
as in most videos. Detected in the wide band and located from two passes."""

import numpy as np
import pandas as pd

from dopplergeo.geo import M_PER_DEG_LAT, haversine_m, m_per_deg_lon
from dopplergeo.pipeline import analyze_audio, locate
from dopplergeo.physics import straight_flyover
from dopplergeo.search import SearchParams

SR, DUR, C = 8000, 210.0, 343.0
LAT, LON = 40.30, -74.60
REC0 = 1_757_300_000.0
# (heard CPA s, heading deg, speed m/s, lateral m, height m, fan tone Hz)
PASSES = [(60.0, 45.0, 78.0, 140.0, 160.0, 1150.0), (150.0, 300.0, 85.0, -180.0, 120.0, 820.0)]


def _music(rng, t):
    """Chords that change every 0.4 s, louder than the plane most of the time."""
    out = np.zeros_like(t)
    for k in range(int(DUR / 0.4)):
        sel = (t >= k * 0.4) & (t < (k + 1) * 0.4)
        for note in rng.choice(np.arange(48, 90), 3, replace=False):
            f = 440 * 2 ** ((note - 69) / 12)
            out[sel] += 0.5 * np.sin(2 * np.pi * f * t[sel])
    return out


def _scene(rng, music=True):
    t = np.arange(int(DUR * SR)) / SR
    x = rng.normal(0, 0.3, len(t)) + (_music(rng, t) if music else 0)
    rows = []
    for i, (tc, hd, v, lat_m, h, f0) in enumerate(PASSES):
        d = float(np.hypot(lat_m, h))
        f = straight_flyover(t, f0, v, d, tc, C)
        r = np.sqrt(d * d + (v * (t - tc)) ** 2)
        x += 120 / r * (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * rng.normal(0, 1, len(t)))
        # ADS-B: straight track, CPA emitted at tc - d/c
        tt = np.arange(-400, 400, 1.0)
        a = np.radians(hd)
        e = lat_m * np.cos(a) + v * tt * np.sin(a)
        n = -lat_m * np.sin(a) + v * tt * np.cos(a)
        rows.append(pd.DataFrame({"flight_id": f"JET{i}", "time": REC0 + tc - d / C + tt,
                                  "lat": LAT + n / M_PER_DEG_LAT, "lon": LON + e / m_per_deg_lon(LAT),
                                  "alt_m": h + 1.5, "typecode": "A320"}))
    for i in range(40):  # unrelated high traffic
        tt = np.arange(-900, 900, 1.0)
        a, v = rng.uniform(0, 2 * np.pi), rng.uniform(180, 240)
        la, lo = LAT + rng.uniform(-0.1, 0.1), LON + rng.uniform(-0.1, 0.1)
        rows.append(pd.DataFrame({"flight_id": f"R{i}", "time": REC0 + rng.uniform(0, DUR) + tt,
                                  "lat": la + v * tt * np.cos(a) / M_PER_DEG_LAT,
                                  "lon": lo + v * tt * np.sin(a) / m_per_deg_lon(LAT),
                                  "alt_m": rng.uniform(4000, 11000), "typecode": "B738"}))
    return (x / np.abs(x).max() * 0.9).astype(np.float32), pd.concat(rows, ignore_index=True)


def test_fast_passes_under_music_are_detected():
    x, _ = _scene(np.random.default_rng(3))
    a = analyze_audio((x, SR), plots=False)
    fast = [f for f in a.fits if f.get("band") == "wide"]
    assert len(fast) == 2
    for f, (tc, hd, v, lat_m, h, f0) in zip(fast, PASSES):
        assert abs(f["tc"] - tc) < 1.0
        assert abs(f["v"] - v) / v < 0.25
        assert abs(f["f0"] - f0) / f0 < 0.03


def test_music_alone_is_not_a_flyover():
    rng = np.random.default_rng(4)
    t = np.arange(int(DUR * SR)) / SR
    x = (_music(rng, t) + rng.normal(0, 0.3, len(t))).astype(np.float32)
    a = analyze_audio((x / np.abs(x).max(), SR), plots=False)
    assert not a.fits


def test_fast_passes_locate(tmp_path):
    x, adsb = _scene(np.random.default_rng(3))
    a = analyze_audio((x, SR), plots=False)
    area = {"lat_min": LAT - 0.1, "lat_max": LAT + 0.1, "lon_min": LON - 0.1, "lon_max": LON + 0.1}
    res = locate(a.spec, a.events, adsb, area, lambda la, lo: np.zeros(np.shape(la)), tmp_path / "out",
                 params=SearchParams(), plots=True)
    best = res["top"][0]
    err = haversine_m(LAT, LON, best["lat"], best["lon"])
    print(res["top"][:2], "error m:", err)
    assert err < 150
    assert abs(best["rec_start_unix"] - REC0) < 5
