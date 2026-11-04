"""Field validation against a source with a known position.

Records audio from a live stream (any URL ffmpeg can read, e.g. an HLS .m3u8
webcam) while polling the adsb.lol live API around the known position, then
runs the locator and reports the error.

    python -m dopplergeo.live --url <stream> --lat 38.95 --lon -121.08 --minutes 20 --out data/live1
"""

import argparse
import json
import math
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .aircraft import is_propeller
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


class Radar:
    """What is flying in a map view right now.

    The world is cut into fixed CELL-degree cells. A view only marks the cells it
    covers as wanted and reads a shared store of latest positions; one background
    thread refreshes the wanted cells in turn, at most one API request per
    `min_gap` seconds and each cell at most every `refresh` seconds, backing off
    after HTTP 429. However many people look and zoom, the request rate to the API
    stays bounded. Each aircraft carries the time of its fix ("t"), track and speed:
    the maps move it along its track between refreshes, so these can be slow."""

    CELL = 4.0  # degrees; one request (radius 160 nm) covers a cell

    def __init__(self, refresh=30.0, keep=180.0, min_gap=1.0, want_for=60.0, max_cells=12):
        self.refresh, self.keep, self.min_gap, self.want_for, self.max_cells = refresh, keep, min_gap, want_for, max_cells
        self.lock = threading.Lock()
        self.planes = {}  # hex -> latest aircraft dict, with its fix time "t"
        self.wanted = {}  # cell -> last time a view covered it
        self.fetched = {}  # cell -> last fetch attempt
        self.blocked_until = 0.0
        self.thread = None

    def _cells(self, south, west, north, east):
        c = self.CELL
        cells = [(i, j) for i in range(math.floor(south / c), math.floor(north / c) + 1)
                 for j in range(math.floor(west / c), math.floor(east / c) + 1)]
        mid = ((south + north) / 2, (west + east) / 2)
        # too wide a view: the cells nearest its centre only
        cells.sort(key=lambda ij: ((ij[0] + 0.5) * c - mid[0]) ** 2 + ((ij[1] + 0.5) * c - mid[1]) ** 2)
        return cells[: self.max_cells]

    def view(self, south, west, north, east, props_only=False, limit=800):
        now = time.time()
        with self.lock:
            for cell in self._cells(south, west, north, east):
                self.wanted[cell] = now
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            out = [a for a in self.planes.values()
                   if now - a["t"] < self.keep and south <= a["lat"] <= north and west <= a["lon"] <= east
                   and (a["prop"] or not props_only)]
            blocked = now < self.blocked_until
        out.sort(key=lambda a: (not a["prop"], a["alt_m"]))  # propellers, then low aircraft, first
        return out[:limit], blocked

    def _run(self):
        while True:
            now = time.time()
            with self.lock:
                for k in [k for k, t in self.wanted.items() if now - t > self.want_for]:
                    del self.wanted[k]
                for k in [k for k, a in self.planes.items() if now - a["t"] > self.keep]:
                    del self.planes[k]
                due = sorted((self.fetched.get(c, 0.0), c) for c in self.wanted
                             if now - self.fetched.get(c, 0.0) >= self.refresh)
                wait = self.blocked_until - now
            if wait > 0 or not due:
                time.sleep(max(wait, 0.5))
                continue
            cell = due[0][1]
            with self.lock:
                self.fetched[cell] = now
            self._fetch(cell)
            time.sleep(self.min_gap)

    def _fetch(self, cell):
        c = self.CELL
        lat, lon = (cell[0] + 0.5) * c, (cell[1] + 0.5) * c
        try:
            req = urllib.request.Request(API.format(lat=lat, lon=lon, nm=160), headers={"User-Agent": "dopplergeo-map"})
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                try:
                    retry = float(e.headers.get("Retry-After") or 0)
                except ValueError:
                    retry = 0.0
                with self.lock:
                    self.blocked_until = time.time() + max(20.0, retry)
            return
        except Exception:
            return
        now = d.get("now", time.time() * 1000) / 1000
        got = {}
        for a in d.get("ac", []):
            alt = a.get("alt_geom")
            if alt is None:
                alt = a.get("alt_baro")
            if a.get("lat") is None or alt is None:
                continue
            typ = a.get("t") or ""
            got[a["hex"]] = {"hex": a["hex"], "lat": a["lat"], "lon": a["lon"],
                             "alt_m": 0 if alt == "ground" else round(alt * FT), "type": typ,
                             "prop": bool(typ) and is_propeller(typ), "track": a.get("track"),
                             "speed_ms": round(a["gs"] * 0.5144) if a.get("gs") is not None else None,
                             "flight": (a.get("flight") or "").strip(),
                             "t": round(now - float(a.get("seen_pos", 0)), 1)}
        with self.lock:
            self.planes.update(got)


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
