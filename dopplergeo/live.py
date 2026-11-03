"""Field validation against a source with a known position.

Records audio from a live stream (any URL ffmpeg can read, e.g. an HLS .m3u8
webcam) while polling the adsb.lol live API around the known position, then
runs the locator and reports the error.

    python -m dopplergeo.live --url <stream> --lat 38.95 --lon -121.08 --minutes 20 --out data/live1
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .dem import load_dem
from .geo import M_PER_DEG_LAT, haversine_m, m_per_deg_lon
from .pipeline import run
from .search import SearchParams

API = "https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/{nm}"
FT = 0.3048


def poll_adsb(lat, lon, nm, stop, rows, period=3.0, log=print):
    """Poll the live API until `stop`; rows grow with (hex, t, lat, lon, alt_m, type).
    When the API says to slow down (HTTP 429) the period doubles up to two minutes,
    honouring Retry-After, and comes back down once requests succeed again."""
    url = API.format(lat=lat, lon=lon, nm=nm)
    wait, last_log, failures = period, 0.0, 0
    while not stop.is_set():
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dopplergeo-validation"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
            now = d.get("now", t0 * 1000) / 1000
            for a in d.get("ac", []):
                alt = a.get("alt_geom")
                if alt is None:
                    alt = a.get("alt_baro")
                if a.get("lat") is None or alt in (None, "ground"):
                    continue
                rows.append((a["hex"], now - float(a.get("seen_pos", 0)), a["lat"], a["lon"], alt * FT,
                             a.get("t") or ""))
            if failures:
                log(f"ADS-B back after {failures} failed request(s)")
            failures, wait = 0, max(period, wait / 2)
        except Exception as e:
            failures += 1
            retry = 0.0
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                try:
                    retry = float(e.headers.get("Retry-After") or 0)
                except ValueError:
                    pass
                wait = min(120.0, max(wait * 2, retry))
            if time.time() - last_log > 60:
                log(f"ADS-B poll failed ({e}); next try in {wait:.0f}s")
                last_log = time.time()
        stop.wait(max(0.0, wait - (time.time() - t0)))


class SharedPoller:
    """One poller per zone, however many recordings use it: several sessions on the
    same area (a restart, two phones) must not multiply requests to the API."""

    _lock = threading.Lock()
    _all = {}

    def __init__(self, key, lat, lon, nm, period, log):
        self.key, self.rows, self.users, self.stop = key, [], 0, threading.Event()
        self.logs = []
        threading.Thread(target=poll_adsb, args=(lat, lon, nm, self.stop, self.rows, period, self._log),
                         daemon=True).start()

    def _log(self, msg):
        for fn in list(self.logs):
            fn(msg)

    @classmethod
    def acquire(cls, lat, lon, nm, log=print, period=5.0):
        key = (round(lat, 2), round(lon, 2), int(nm))
        with cls._lock:
            p = cls._all.get(key)
            if p is None:
                p = cls._all[key] = SharedPoller(key, lat, lon, nm, period, log)
            p.users += 1
            p.logs.append(log)
            return p

    def release(self, log=None):
        with SharedPoller._lock:
            self.users -= 1
            if log in self.logs:
                self.logs.remove(log)
            if self.users <= 0:
                self.stop.set()
                SharedPoller._all.pop(self.key, None)


def record(url, seconds, path):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", url, "-t", str(seconds), "-vn", "-ac", "1", "-ar", "8000", str(path)]
    start = time.time()
    subprocess.run(cmd, check=True)
    return start


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="stream URL (m3u8, rtsp, http) or a local file")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--radius-km", type=float, default=15, help="search area radius around the true point")
    ap.add_argument("--out", default="data/live")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, stop = [], threading.Event()
    th = threading.Thread(target=poll_adsb, args=(a.lat, a.lon, 30, stop, rows), daemon=True)
    th.start()
    print(f"recording {a.minutes} min from {a.url}")
    start = record(a.url, a.minutes * 60, out / "audio.wav")
    stop.set()
    th.join()
    adsb = pd.DataFrame(rows, columns=["flight_id", "time", "lat", "lon", "alt_m", "typecode"])
    adsb = adsb.drop_duplicates(["flight_id", "time"])
    adsb.to_parquet(out / "adsb.parquet", index=False)
    print(f"{adsb['flight_id'].nunique()} aircraft seen, {len(adsb)} positions")

    r = a.radius_km * 1000
    dlat, dlon = r / M_PER_DEG_LAT, r / m_per_deg_lon(a.lat)
    area = {"lat_min": a.lat - dlat, "lat_max": a.lat + dlat, "lon_min": a.lon - dlon, "lon_max": a.lon + dlon}
    dem = load_dem({"type": "hgt", "dir": "data/dem", "download": True})
    # stream latency is unknown: allow up to a minute
    p = SearchParams(start_window=(start - 60, start + 60), tile_deg=0.5)
    res = run(out / "audio.wav", adsb, area, dem, out, params=p)
    for t in res["top"][:5]:
        err = haversine_m(a.lat, a.lon, t["lat"], t["lon"])
        print(f"#{t['rank']} score {t['score']:.2f} error {err:.0f} m flights {t['flights']}")
    json.dump({"truth": [a.lat, a.lon], "result": res["top"][:5]}, open(out / "validation.json", "w"), default=float)


if __name__ == "__main__":
    main()
