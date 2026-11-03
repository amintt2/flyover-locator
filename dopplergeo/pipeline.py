"""End-to-end run: audio -> events -> coarse search -> fine scoring -> results.

Split in two so callers can check the audio (cheap) before fetching ADS-B data
(expensive): `analyze_audio` then `locate`.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adsb import build_segments
from .audio import Spectrogram, load_audio, plot_spectrogram, spectrogram
from .detect import detect_flyovers, events_for_search
from .enf import analyze_enf
from .physics import straight_flyover
from .search import Event, SearchParams, best_per_combination, coarse_search, fine_search, model_curve

SPEC = {"fmin": 40.0, "fmax": 400.0, "win_s": 1.0, "hop_s": 0.1}
SR = 1000


@dataclass
class AudioAnalysis:
    spec: Spectrogram
    fits: list
    events: list
    enf: dict


def analyze_audio(source, out_dir=None, p=None, log=print, plots=True, sr=SR):
    """source: a media path or a (samples, sample_rate) tuple."""
    p = p or SearchParams()
    if isinstance(source, tuple):
        x, sr = source
    else:
        log("decoding audio")
        x, sr = load_audio(source, sr=sr)
    spec = spectrogram(x, sr, SPEC["fmin"], SPEC["fmax"], SPEC["win_s"], SPEC["hop_s"])
    log(f"  {spec.duration:.0f}s of audio")
    enf = analyze_enf(x, sr)
    if enf["nominal"]:
        log(f"  mains hum: {enf['nominal']} Hz grid (strength {enf['strength']:.1f})")
    log("detecting flyovers")
    fits = detect_flyovers(spec, p.c, log=log)
    events = events_for_search(fits) if fits else []
    if plots and out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        curves = []
        for f in fits:
            ft = np.linspace(f["window"][0], f["window"][1], 400)
            curves.append((ft, straight_flyover(ft, f["f0"], f["v"], f["d"], f["tc"], p.c),
                           f"{f['tc']:.0f}s  ~{f['v']:.0f} m/s  ~{f['d']:.0f} m"))
        plot_spectrogram(spec, out / "flyovers.png", fmin=40, fmax=300, curves=curves,
                         title=f"{len(fits)} flyover(s) detected")
    return AudioAnalysis(spec, fits, events, enf)


def locate(spec, events, adsb_df, area, dem, out_dir, params=None, log=print, plots=True):
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = params or SearchParams()
    events = [e if isinstance(e, Event) else Event.from_dict(e) for e in events]
    (out / "events.json").write_text(json.dumps([e.__dict__ for e in events], indent=2, default=list))

    log(f"preparing trajectories ({adsb_df['flight_id'].nunique()} flights)")
    segs = build_segments(adsb_df)
    log(f"coarse search with {len(events)} event(s)")
    cands = coarse_search(segs, events, area, dem, spec.duration, p, log=log)
    if cands.empty:
        raise RuntimeError("no location matches the aircraft heard in the recording")
    log(f"{len(cands)} coarse candidates")
    scored = fine_search(cands, segs, events, spec, dem, p, log=log)
    if scored.empty:
        raise RuntimeError("fine scoring produced no result")
    best = best_per_combination(scored, len(events))
    scored.to_csv(out / "scored_points.csv", index=False)
    best.to_csv(out / "best_per_combination.csv", index=False)

    by_sid = {s.sid: s for s in segs}
    top = []
    for i, row in best.head(10).iterrows():
        top.append({
            "rank": i + 1, "lat": row["lat"], "lon": row["lon"], "score": row["score"],
            "scores": [row[f"score{k+1}"] for k in range(len(events))],
            "flights": [row[f"sid{k+1}"].split("#")[0] for k in range(len(events))],
            "rec_start_unix": row["rec_start"],
        })
    if plots and top:
        for i, row in best.head(3).iterrows():
            curves = []
            for k, e in enumerate(events):
                seg = by_sid[row[f"sid{k+1}"]]
                ct, cf = model_curve(row["lat"], row["lon"], row["elev"], seg, e, row["rec_start"], p)
                for h in e.harmonics:
                    curves.append((ct, cf * h, f"{seg.flight} h{h}"))
            fmax = max(e.f0 * max(e.harmonics) for e in events) * 1.4
            fmin = min(e.f0 for e in events) * 0.6
            plot_spectrogram(spec, out / f"match_{i + 1}.png", fmin=fmin, fmax=fmax, curves=curves,
                             title=f"#{i + 1}  {row['lat']:.5f}, {row['lon']:.5f}  score {row['score']:.2f}")
    # heat layer: best score per fine point, normalised
    heat = scored.groupby(["lat", "lon"], as_index=False)["score"].max()
    heat = heat.nlargest(5000, "score")
    if len(heat):
        lo, hi = heat["score"].min(), heat["score"].max()
        heat["w"] = (heat["score"] - lo) / max(hi - lo, 1e-9)
    result = {
        "top": top,
        "events": [{"t_cpa": e.t_cpa, "v": list(e.v), "d": list(e.d), "f0": e.f0} for e in events],
        "heat": heat[["lat", "lon", "w"]].round(6).values.tolist() if len(heat) else [],
        "n_coarse": int(len(cands)), "n_scored": int(len(scored)),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    if len(top) > 1 and top[1]["score"] > 0:
        result["confidence"] = round(top[0]["score"] / top[1]["score"], 2)
    (out / "result.json").write_text(json.dumps(result, default=float))
    log(f"done in {result['elapsed_s']}s")
    return result


def run(audio_path, adsb_df, area, dem, out_dir, events=None, params=None, log=print, plots=True):
    p = params or SearchParams()
    a = analyze_audio(audio_path, out_dir, p, log=log, plots=plots)
    events = events if events is not None else a.events
    if not events:
        raise RuntimeError("no aircraft Doppler curve found in the recording")
    res = locate(a.spec, events, adsb_df, area, dem, out_dir, p, log=log, plots=plots)
    res["enf"] = a.enf
    return res
