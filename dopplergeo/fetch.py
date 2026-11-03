"""Automatic ADS-B download from adsb.lol daily `globe_history` archives.

Each day is ~3 GB worldwide. It is streamed and filtered on the fly to the
search box and time window, and only the small filtered result is cached.
"""

import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from .adsb import stream_readsb_archive

API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
DL = "https://github.com/adsblol/globe_history_{year}/releases/download/{tag}/{name}"


def _asset_urls(day):
    tag = f"v{day:%Y.%m.%d}-planes-readsb-prod-0"
    try:
        req = urllib.request.Request(API.format(year=day.year, tag=tag),
                                     headers={"User-Agent": "dopplergeo", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            names = sorted(a["name"] for a in json.load(r)["assets"] if ".tar" in a["name"])
    except Exception:
        names = [f"{tag}.tar.aa", f"{tag}.tar.ab"]  # usual split
    if not names:
        raise RuntimeError(f"no adsb.lol archive for {day}")
    return [DL.format(year=day.year, tag=tag, name=n) for n in names]


def fetch_adsb(bbox, t_range, cache_dir, log=print):
    """Points within bbox=(lat_min, lat_max, lon_min, lon_max) and t_range (unix)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    d0 = dt.datetime.fromtimestamp(t_range[0], dt.timezone.utc).date()
    d1 = dt.datetime.fromtimestamp(t_range[1], dt.timezone.utc).date()
    frames = []
    day = d0
    while day <= d1:
        key = hashlib.sha1(json.dumps([str(day), [round(x, 3) for x in bbox]]).encode()).hexdigest()[:12]
        path = cache / f"adsb_{day}_{key}.parquet"
        if path.exists():
            log(f"ADS-B {day}: using cache")
            df = pd.read_parquet(path)
        else:
            log(f"ADS-B {day}: streaming the adsb.lol archive (~3 GB, filtered on the fly; takes a while)")
            day_start = dt.datetime.combine(day, dt.time(), dt.timezone.utc).timestamp()
            df = stream_readsb_archive(_asset_urls(day), path, bbox, (day_start, day_start + 86400 + 600),
                                       progress=False)
            log(f"ADS-B {day}: {df['flight_id'].nunique()} flights kept in the area")
        frames.append(df)
        day += dt.timedelta(days=1)
    df = pd.concat(frames, ignore_index=True)
    return df[(df["time"] >= t_range[0]) & (df["time"] <= t_range[1])]
