"""Automatic ADS-B download from adsb.lol daily `globe_history` archives.

Each day is ~3 GB worldwide. It is streamed once and reduced to a compact
daily index (see index.py) that every later search reads.
"""

import datetime as dt
import json
import urllib.error
import urllib.request
import pandas as pd


API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
DL = "https://github.com/adsblol/globe_history_{year}/releases/download/{tag}/{name}"


def _asset_urls(day):
    tag = f"v{day:%Y.%m.%d}-planes-readsb-prod-0"
    try:
        req = urllib.request.Request(API.format(year=day.year, tag=tag),
                                     headers={"User-Agent": "dopplergeo", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            names = sorted(a["name"] for a in json.load(r)["assets"] if ".tar" in a["name"])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Les données d'avions du {day} ne sont pas encore publiées (elles sortent le "
                               "lendemain). Réessaie demain, ou utilise le mode Enregistrer pour le direct.")
        names = [f"{tag}.tar.aa", f"{tag}.tar.ab"]  # API rate-limited: use the usual split
    except Exception:
        names = [f"{tag}.tar.aa", f"{tag}.tar.ab"]
    if not names:
        raise RuntimeError(f"no adsb.lol archive for {day}")
    return [DL.format(year=day.year, tag=tag, name=n) for n in names]


def fetch_adsb(bbox, t_range, index_root, log=print, max_alt_m=None):
    """Points within bbox=(lat_min, lat_max, lon_min, lon_max) and t_range (unix).

    Days missing from the compact index are streamed once (worldwide) and kept
    for every later search; max_alt_m is fixed by the index (index.MAX_ALT_M).
    """
    from .index import build_day, day_path, load_day

    d0 = dt.datetime.fromtimestamp(t_range[0], dt.timezone.utc).date()
    d1 = dt.datetime.fromtimestamp(t_range[1], dt.timezone.utc).date()
    frames, day = [], d0
    while day <= d1:
        if day_path(index_root, day).exists():
            log(f"ADS-B {day}: from the index")
        else:
            log(f"ADS-B {day}: first search on this day, streaming the adsb.lol archive "
                "(~3 GB, keeps only low propeller traffic; ~10 min, then instant)")
            build_day(day, index_root, log=log)
        frames.append(load_day(index_root, day, bbox, t_range))
        day += dt.timedelta(days=1)
    return pd.concat(frames, ignore_index=True)
