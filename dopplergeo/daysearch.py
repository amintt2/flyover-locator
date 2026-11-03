"""Find the day: run the search on every indexed day of a range and rank days.

Days are independent, so each one runs in its own process. The coarse stage
already fixes the recording start inside the day (time offset between heard
flyovers); a day without a match costs only the screening pass.
"""

import datetime as dt
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .dem import load_dem
from .detect import events_for_search
from .geo import haversine_m
from .index import load_day
from .pipeline import locate
from .search import Event, SearchParams

DAY = 86400
CHUNK_DEG = 5.0
BUF = 0.12  # deg, > hearing radius


def day_events(fits):
    """Every clear flyover: each extra one must line up too, which is what
    separates the right day from look-alike traffic on other days."""
    return events_for_search(fits, max_events=4, min_ratio=0.3)


def prefilter(df, events, margin=12.0):
    """Drop flights that never fly at a speed one of the heard aircraft had.

    Cheap (vectorised on the compact points), so a worldwide day fits in memory
    before trajectories are resampled to 1 s.
    """
    if df.empty:
        return df
    v_lo = min(e.v[0] for e in events) - margin
    v_hi = max(e.v[1] for e in events) + margin
    df = df.sort_values(["flight_id", "time"], kind="stable")
    same = df["flight_id"].to_numpy()[1:] == df["flight_id"].to_numpy()[:-1]
    t, la, lo = (df[c].to_numpy(float) for c in ("time", "lat", "lon"))
    dt_ = np.diff(t)
    v = haversine_m(la[:-1], lo[:-1], la[1:], lo[1:]) / np.maximum(dt_, 1e-3)
    ok = same & (dt_ > 0) & (dt_ <= 30) & (v >= v_lo) & (v <= v_hi)
    keep = set(df["flight_id"].to_numpy()[1:][ok])
    return df[df["flight_id"].isin(keep)]


def _chunks(lat, lon, area, deg=CHUNK_DEG):
    """Occupied deg x deg cells of the area (a worldwide day never fits in memory at once)."""
    i0, j0 = np.floor(area["lat_min"] / deg), np.floor(area["lon_min"] / deg)
    i1, j1 = np.floor(area["lat_max"] / deg), np.floor(area["lon_max"] / deg)
    cells = set(zip(np.floor(lat / deg).astype(int).tolist(), np.floor(lon / deg).astype(int).tolist()))
    for i, j in sorted(cells):
        if i0 <= i <= i1 and j0 <= j <= j1:
            yield {"lat_min": max(i * deg, area["lat_min"]), "lat_max": min((i + 1) * deg, area["lat_max"]),
                   "lon_min": max(j * deg, area["lon_min"]), "lon_max": min((j + 1) * deg, area["lon_max"])}


def _one_day(day, root, spec, events, area, dem_cfg, out_dir, bbox):
    t_start = time.time()
    events = [e if isinstance(e, Event) else Event.from_dict(e) for e in events]
    d0 = dt.datetime.combine(day, dt.time(), dt.timezone.utc).timestamp()
    out = Path(out_dir) / f"{day}"
    lines, best, n_flights = [], None, 0
    p = SearchParams(start_window=(d0, d0 + DAY), tile_deg=0.5)
    dem = load_dem(dem_cfg)
    day_df = load_day(root, day, bbox, (d0 - 600, d0 + DAY + spec.duration + 600), compact=True)
    lat, lon = day_df["lat"].to_numpy(), day_df["lon"].to_numpy()
    for k, cell in enumerate(_chunks(lat, lon, area)):
        # aircraft heard from inside the cell can be up to ~12 km outside it
        m = ((lat >= cell["lat_min"] - BUF) & (lat <= cell["lat_max"] + BUF)
             & (lon >= cell["lon_min"] - BUF * 1.5) & (lon <= cell["lon_max"] + BUF * 1.5))
        df = prefilter(_canonical(day_df[m]), events)
        if df.empty:
            continue
        n_flights += df["flight_id"].nunique()
        try:
            res = locate(spec, events, df, cell, dem, out / f"c{k}", params=p, log=lambda m: None)
        except RuntimeError:
            continue
        if res["top"] and (best is None or res["top"][0]["score"] > best[0]["top"][0]["score"]):
            best = (res, out / f"c{k}")
    lines.append(f"{day}: {n_flights} flights at a matching speed searched")
    if best is None:
        lines.append(f"{day}: no match")
        return {"day": str(day), "score": 0.0, "result": None, "elapsed_s": round(time.time() - t_start, 1),
                "log": lines}
    res, src = best
    for png in src.glob("match_*.png"):
        shutil.copy(png, out / png.name)
    res.pop("heat", None)
    return {"day": str(day), "score": float(res["top"][0]["score"]), "result": res,
            "elapsed_s": round(time.time() - t_start, 1), "log": lines}


def _canonical(df):
    df = df.copy()
    for c in ("flight_id", "typecode"):
        df[c] = df[c].astype(str)
    for c in ("lat", "lon", "alt_m"):
        df[c] = df[c].astype(float)
    return df


def search_days(spec, events, days, root, area, dem_cfg, out_dir, log=print, workers=None, on_day=None):
    """Search each day; returns (best day's result or None, [{day, score, ...}] ranked)."""
    events = [e if isinstance(e, Event) else Event.from_dict(e) for e in events]
    bbox = (area["lat_min"] - BUF, area["lat_max"] + BUF, area["lon_min"] - BUF * 1.5, area["lon_max"] + BUF * 1.5)
    workers = workers or int(os.environ.get("DAY_WORKERS", max(1, min(4, (os.cpu_count() or 2) - 1))))
    days = sorted(days)
    log(f"searching {len(days)} day(s), {workers} at a time")
    ranked, done = [], 0
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = [ex.submit(_one_day, d, root, spec, events, area, dem_cfg, out_dir, bbox) for d in days]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            for line in r.pop("log"):
                log(line)
            ranked.append(r)
            if on_day:
                on_day(done, len(days), r)
    ranked.sort(key=lambda r: -r["score"])
    summary = [{"day": r["day"], "score": round(r["score"], 3), "elapsed_s": r["elapsed_s"],
                "top": (r["result"] or {}).get("top", [])[:1]} for r in ranked]
    (Path(out_dir) / "days.json").write_text(json.dumps(summary, default=float))
    best = ranked[0] if ranked and ranked[0]["result"] else None
    if best is None:
        return None, summary
    res = best["result"]
    for png in (Path(out_dir) / best["day"]).glob("match_*.png"):
        shutil.copy(png, Path(out_dir) / png.name)
    res["day"] = best["day"]
    if len(ranked) > 1 and ranked[1]["score"] > 0:
        res["day_confidence"] = round(best["score"] / ranked[1]["score"], 2)
    return res, summary
