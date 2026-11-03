"""End-to-end run: audio -> events -> coarse search -> fine scoring -> results.

Split in two so callers can check the audio (cheap) before fetching ADS-B data
(expensive): `analyze_audio` then `locate`.
"""

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .adsb import build_segments
from .audio import Bands, Spectrogram, clips_from_cuts, load_audio, plot_spectrogram, spectrogram
from .detect import detect_flyovers, detect_short, events_for_search
from .enf import analyze_enf
from .physics import straight_flyover
from .search import Event, SearchParams, best_per_combination, coarse_search, fine_search, model_curve

SPEC = {"fmin": 40.0, "fmax": 400.0, "win_s": 1.0, "hop_s": 0.1}
SR = 1000
WIDE = {"sr": 8000, "fmin": 100.0, "fmax": 3000.0, "win_s": 0.25, "hop_s": 0.05}  # fast, high-pitched passes


@dataclass
class AudioAnalysis:
    spec: Spectrogram
    fits: list
    events: list
    enf: dict
    clips: list | None = None  # edited video: continuous spans in recording time


def analyze_audio(source, out_dir=None, p=None, log=print, plots=True, sr=SR, cuts=None):
    """source: a media path or a (samples, sample_rate) tuple.

    cuts: times (s) where an edited video jumps; flyovers are then detected per
    clip and the search links clips only by place, not by time."""
    p = p or SearchParams()
    if isinstance(source, tuple):
        x, sr = source
        xw, srw = (x, sr) if sr >= 4000 else (None, None)
    else:
        log("decoding audio")
        x, sr = load_audio(source, sr=sr)
        xw, srw = load_audio(source, sr=WIDE["sr"])
    spec = spectrogram(x, sr, SPEC["fmin"], SPEC["fmax"], SPEC["win_s"], SPEC["hop_s"])
    log(f"  {spec.duration:.0f}s of audio")
    enf = analyze_enf(x, sr)
    if enf["nominal"]:
        log(f"  mains hum: {enf['nominal']} Hz grid (strength {enf['strength']:.1f})")
    log("detecting flyovers")
    clips = clips_from_cuts(cuts, spec.duration) if cuts else None
    if clips:
        log(f"  {len(clips)} clips separated by cuts at " + ", ".join(f"{c:.0f}s" for c in cuts))
    fits = detect_flyovers(spec, p.c, log=log, clips=clips)
    wide = None
    if xw is not None:
        long_ = len(xw) / srw > 1800  # keep memory bounded on long recordings
        wide = spectrogram(xw, srw, WIDE["fmin"], min(WIDE["fmax"] * (0.5 if long_ else 1), 0.45 * srw),
                           WIDE["win_s"], WIDE["hop_s"] * (2 if long_ else 1), pad=2, smooth_hz=2.0, smooth_s=0.1)
        del xw
        for sf in detect_short(wide, p.c, log=log, clips=clips):
            if all(abs(sf["tc"] - f["tc"]) > 20 for f in fits):
                fits.append(sf)
        fits.sort(key=lambda f: f["tc"])
        if any(f.get("band") == "wide" for f in fits):
            spec = Bands(spec, wide)
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
        if wide is not None and any(f.get("band") == "wide" for f in fits):
            wc = [c_ for c_, f in zip(curves, fits) if f.get("band") == "wide"]
            plot_spectrogram(wide, out / "flyovers_fast.png", curves=wc,
                             title=f"{len(wc)} fast pass(es) detected")
    return AudioAnalysis(spec, fits, events, enf, clips)


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
    if len({e.clip for e in events}) > 1:
        cands, scored = _search_clips(segs, events, spec, area, dem, p, log)
    else:
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
                ct, cf = model_curve(row["lat"], row["lon"], row["elev"], seg, e,
                                     row.get(f"origin{k+1}", row["rec_start"]), p)
                for h in e.harmonics:
                    curves.append((ct, cf * h, f"{seg.flight} h{h}"))
            fmax = max(e.f0 * max(e.harmonics) for e in events) * 1.4
            fmin = min(e.f0 for e in events) * 0.6
            ps = spec.wide if isinstance(spec, Bands) and fmax > spec.low.freqs[-1] else spec
            plot_spectrogram(ps, out / f"match_{i + 1}.png", fmin=max(fmin, ps.freqs[0]), fmax=min(fmax, ps.freqs[-1]),
                             curves=curves,
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


def _key(df):
    return np.rint(df["lat"].to_numpy() * 1e6).astype(np.int64) * 1_000_000_000 + np.rint(df["lon"].to_numpy() * 1e6).astype(np.int64)


def _search_clips(segs, events, spec, area, dem, p, log):
    """Edited video: each continuous clip is searched on its own (times only compare
    inside a clip), then only places that fit every clip are kept and their scores
    added. Matches add up per clip instead of multiplying across clips."""
    order = sorted({e.clip for e in events}, key=lambda c: next(k for k, e in enumerate(events) if e.clip == c))
    idx = [[k for k, e in enumerate(events) if e.clip == c] for c in order]
    spans = p.clips or [(0.0, spec.duration)]
    runs = []
    for n, (c, ks) in enumerate(zip(order, idx)):
        sub = [replace(events[k], clip=0) for k in ks]
        t = sub[0].t_cpa
        span = next((s for s in spans if s[0] <= t <= s[1]), (0.0, spec.duration))
        sw = p.start_window
        if sw and n > 0:
            sw = (sw[0] - p.clip_gap_s, sw[1] + p.clip_gap_s)
        runs.append((ks, sub, replace(p, clips=[tuple(span)], start_window=sw)))
    cands = []
    for n, (ks, sub, pc) in enumerate(runs):
        log(f"  clip {n + 1}: {len(sub)} flyover(s)")
        c = coarse_search(segs, sub, area, dem, spec.duration, pc, log=log)
        if c.empty:
            raise RuntimeError(f"no location matches the aircraft heard in clip {n + 1}")
        cands.append(c)
    common = set(_key(cands[0]).tolist())
    for c in cands[1:]:
        common &= set(_key(c).tolist())
    if not common:
        raise RuntimeError("no location matches the aircraft heard in every clip")
    common = np.fromiter(common, np.int64)
    cands = [c[np.isin(_key(c), common)].reset_index(drop=True) for c in cands]
    log(f"{len(common)} coarse places fit every clip")

    merged = None
    for n, ((ks, sub, pc), c) in enumerate(zip(runs, cands)):
        sc = fine_search(c, segs, sub, spec, dem, pc, log=log)
        if sc.empty:
            raise RuntimeError("fine scoring produced no result")
        sc = sc.loc[sc.assign(key=_key(sc)).groupby("key")["score"].idxmax()]
        cols = {"score": f"s{n}", "dt": f"dt{n}", "rec_start": f"rs{n}"}
        for j, k in enumerate(ks):
            cols.update({f"score{j+1}": f"score{k+1}", f"origin{j+1}": f"origin{k+1}", f"sid{j+1}": f"sid{k+1}"})
        sc = sc.rename(columns=cols).assign(key=_key(sc))
        if merged is None:
            merged = sc
        else:
            merged = merged.merge(sc.drop(columns=["lat", "lon", "elev"]), on="key")
    if merged.empty:
        raise RuntimeError("fine scoring produced no result")
    merged["score"] = merged[[f"s{n}" for n in range(len(runs))]].sum(axis=1)
    merged["dt"], merged["rec_start"] = merged["dt0"], merged["rs0"]
    keep = ["lat", "lon", "elev", "score", "dt", "rec_start"]
    for k in range(len(events)):
        keep += [f"score{k+1}", f"origin{k+1}", f"sid{k+1}"]
    scored = merged[keep].sort_values("score", ascending=False, ignore_index=True)
    return pd.concat(cands, ignore_index=True), scored


def run(audio_path, adsb_df, area, dem, out_dir, events=None, params=None, log=print, plots=True):
    p = params or SearchParams()
    a = analyze_audio(audio_path, out_dir, p, log=log, plots=plots)
    events = events if events is not None else a.events
    if not events:
        raise RuntimeError("no aircraft Doppler curve found in the recording")
    res = locate(a.spec, events, adsb_df, area, dem, out_dir, p, log=log, plots=plots)
    res["enf"] = a.enf
    return res
