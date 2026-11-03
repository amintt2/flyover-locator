"""Synthetic scenario generator: ADS-B table + WAV recording at a hidden listener.

Used to validate the pipeline end to end without real data.
"""

import json

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.io import wavfile

from .geo import M_PER_DEG_LAT, ecef, m_per_deg_lon

PROPS = ["C172", "PA28", "BE36", "SR22", "C208", "DHC6", "PC12", "BE20"]
JETS = ["B738", "A320", "CRJ9", "E75L", "A321", "B739"]


def _simulate(rng, lat0, lon0, t0, heading, speed, alt, duration, dt=0.5, turbulence=1.0):
    """Integrate a wandering flight forward from (lat0, lon0) at t0."""
    n = int(duration / dt)
    hd = heading + np.cumsum(rng.normal(0, 0.15 * turbulence, n)) * dt
    # occasional gentle turns
    for _ in range(rng.poisson(duration / 600)):
        a = rng.integers(0, n)
        hd[a:] += rng.normal(0, 25)
    sp = speed + np.cumsum(rng.normal(0, 0.05 * turbulence, n)) * dt
    sp = np.clip(sp, speed * 0.85, speed * 1.15)
    vs = np.convolve(rng.normal(0, 1.5 * turbulence, n), np.ones(120) / 120, "same")  # smooth vertical speed
    al = alt + np.cumsum(vs) * dt
    e = np.cumsum(sp * np.sin(np.radians(hd))) * dt
    nn = np.cumsum(sp * np.cos(np.radians(hd))) * dt
    t = t0 + np.arange(n) * dt
    return t, lat0 + nn / M_PER_DEG_LAT, lon0 + e / m_per_deg_lon(lat0), al


def _through(rng, lat, lon, t_pass, heading, speed, alt, half=1200):
    """Flight passing through (lat, lon) at t_pass, with wander both before and after."""
    tf, laf, lof, alf = _simulate(rng, lat, lon, t_pass, heading, speed, alt, half)
    tb, lab, lob, alb = _simulate(rng, lat, lon, t_pass, heading + 180, speed, alt, half)
    tb = 2 * t_pass - tb
    return (np.r_[tb[:0:-1], tf], np.r_[lab[:0:-1], laf], np.r_[lob[:0:-1], lof], np.r_[alb[:0:-1], alf])


def make_scenario(out_dir, seed=1, n_random=220, region=(40.0, 41.0, -75.0, -74.0), offset_s=920.0):
    from pathlib import Path

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    la0, la1, lo0, lo1 = region
    base = 1_757_300_000.0  # 2025-09-08 ~03:33 UTC
    span = 3 * 3600

    # terrain: rolling hills 50-400 m
    glat = np.linspace(la0 - 0.3, la1 + 0.3, 241)
    glon = np.linspace(lo0 - 0.3, lo1 + 0.3, 241)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    elev = 200 + 120 * np.sin(LA * 9.0) * np.cos(LO * 7.0) + 60 * np.sin(LA * 23 + LO * 17)
    np.savez(out / "dem.npz", lat=glat, lon=glon, elev=elev)

    def dem(la, lo):
        return 200 + 120 * np.sin(la * 9.0) * np.cos(lo * 7.0) + 60 * np.sin(la * 23 + lo * 17)

    flights = []  # (id, type, prop, t, lat, lon, alt, f0)
    for i in range(n_random):
        prop = rng.random() < 0.55
        typ = rng.choice(PROPS if prop else JETS)
        lat = rng.uniform(la0, la1)
        lon = rng.uniform(lo0, lo1)
        if prop:
            sp, alt = rng.uniform(50, 120), rng.uniform(500, 3000)
        else:
            sp, alt = rng.uniform(160, 240), rng.uniform(4000, 11000)
        t, la, lo, al = _through(rng, lat, lon, base + rng.uniform(0, span), rng.uniform(0, 360), sp, alt, 900)
        flights.append([f"r{i:04d}", typ, prop, t, la, lo, al, rng.uniform(70, 150)])

    # hidden listener + the two flyovers it hears
    obs_lat, obs_lon = la0 + 0.43 * (la1 - la0), lo0 + 0.61 * (lo1 - lo0)
    obs_h = dem(obs_lat, obs_lon) + 1.5
    t_a = base + span / 2

    def inject(fid, typ, t_pass, lateral, alt_agl, heading, speed, f0, lat_c=obs_lat, lon_c=obs_lon):
        hd = np.radians(heading + 90)
        la = lat_c + lateral * np.cos(hd) / M_PER_DEG_LAT
        lo = lon_c + lateral * np.sin(hd) / m_per_deg_lon(lat_c)
        t, a, b, c = _through(rng, la, lo, t_pass, heading, speed, dem(lat_c, lon_c) + alt_agl, 1500)
        flights.append([fid, typ, True, t, a, b, c, f0])

    inject("TRUE_A", "C172", t_a, 900, 700, 35, 95.0, 118.0)
    inject("TRUE_B", "PA28", t_a + offset_s, -1300, 900, 250, 72.0, 92.0)
    # decoy pair elsewhere with similar numbers but different geometry
    q_lat, q_lon = la0 + 0.7 * (la1 - la0), lo0 + 0.25 * (lo1 - lo0)
    inject("DECOY_A", "C182", t_a + 1800, 1000, 600, 120, 90.0, 110.0, q_lat, q_lon)
    inject("DECOY_B", "PA32", t_a + 1800 + offset_s, -1100, 1000, 300, 75.0, 95.0, q_lat, q_lon)

    obs_xyz = ecef(obs_lat, obs_lon, obs_h)
    c = 343.0
    rec_start = t_a - 300.0
    duration = offset_s + 600.0
    sr = 2000
    t_rec = rec_start + np.arange(int(duration * sr)) / sr

    # enforce silence: drop random props loudly heard during the recording
    kept = []
    for f in flights:
        xyz = ecef(f[4], f[5], f[6])
        r = np.linalg.norm(xyz - obs_xyz, axis=1)
        heard = f[3] + r / c
        inrec = (heard >= rec_start - 30) & (heard <= rec_start + duration + 30)
        if f[0].startswith("r") and inrec.any() and r[inrec].min() < 3500:
            continue
        kept.append(f)
    flights = kept

    audio = np.zeros(len(t_rec))
    for fid, typ, prop, t, la, lo, al, f0 in flights:
        xyz = ecef(la, lo, al)
        r = np.linalg.norm(xyz - obs_xyz, axis=1)
        heard = t + r / c
        sel = (t_rec >= heard[0]) & (t_rec <= heard[-1])
        if not sel.any() or r.min() > 12000:
            continue
        tau = np.interp(t_rec[sel], heard, t)
        rr = np.interp(t_rec[sel], heard, r)
        amp = 1.0 / rr
        if prop:
            sig = sum(a * np.sin(2 * np.pi * f0 * h * tau + h) for h, a in ((1, 1.0), (2, 0.6), (3, 0.3)))
        else:
            sig = rng.normal(0, 1.0, sel.sum())
        audio[sel] += 500 * amp * sig
    noise = rng.normal(0, 1.0, len(audio))
    brown = np.cumsum(rng.normal(0, 0.02, len(audio)))
    brown -= np.convolve(brown, np.ones(4001) / 4001, "same")
    hum = 0.3 * np.sin(2 * np.pi * 60 * t_rec) + 0.15 * np.sin(2 * np.pi * 120 * t_rec)
    audio = audio + noise + brown + hum
    audio = audio / np.abs(audio).max() * 0.9
    wavfile.write(out / "recording.wav", sr, (audio * 32767).astype(np.int16))

    rows = []
    for fid, typ, prop, t, la, lo, al, f0 in flights:
        idx = np.sort(rng.choice(len(t), size=len(t) // 3, replace=False))  # ~1.5 s ADS-B cadence
        alt_q = np.round(al[idx] / (25 * 0.3048)) * (25 * 0.3048)  # 25 ft resolution
        rows.append(pd.DataFrame({"flight_id": fid, "time": t[idx], "lat": la[idx], "lon": lo[idx],
                                  "alt_m": alt_q, "typecode": typ}))
    pd.concat(rows, ignore_index=True).to_parquet(out / "adsb.parquet", index=False)

    def heard_cpa(fid):
        f = next(x for x in flights if x[0] == fid)
        xyz = ecef(f[4], f[5], f[6])
        r = np.linalg.norm(xyz - obs_xyz, axis=1)
        i = r.argmin()
        return float(f[3][i] + r[i] / c - rec_start), float(r[i]), f[7]

    truth = {"lat": obs_lat, "lon": obs_lon, "rec_start": rec_start, "duration": duration,
             "events": [dict(zip(("t_cpa", "r", "f0"), heard_cpa(x))) for x in ("TRUE_A", "TRUE_B")]}
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
