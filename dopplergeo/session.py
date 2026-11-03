"""Live recording sessions: the browser streams microphone PCM while the server
polls live ADS-B and the grid frequency, re-analyses the audio as it grows and
re-runs the search whenever a new flyover is heard."""

import json
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .aircraft import is_propeller
from .enf import correlate_reference
from .geo import M_PER_DEG_LAT, m_per_deg_lon
from .live import SharedPoller
from .pipeline import SR, analyze_audio, locate
from .search import SearchParams

GRID_URL = "https://www.netzfrequenz.info/json/act.json"  # continental Europe, live
MAX_SECONDS = 45 * 60


def poll_grid(stop, rows, period=1.0):
    while not stop.is_set():
        t0 = time.time()
        try:
            req = urllib.request.Request(GRID_URL, headers={"User-Agent": "dopplergeo"})
            with urllib.request.urlopen(req, timeout=5) as r:
                v = float(r.read().decode().strip())
            if 49.5 < v < 50.5:
                rows.append((t0, v))
        except Exception:
            pass
        stop.wait(max(0.0, period - (time.time() - t0)))


class LiveSession:
    def __init__(self, sid, directory, lat, lon, radius_km, t0, dem_factory, log_cap=200, sr=SR):
        self.sid, self.dir = sid, Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lat, self.lon, self.radius_km = lat, lon, radius_km
        self.t0 = t0  # unix time of the first audio sample (server clock)
        self.sr = sr  # 1000 Hz (propeller tones only) or 8000 Hz (jet whine too)
        self.pcm = self.dir / "audio.pcm"
        self.pcm.touch()
        self.samples = 0
        self.grid_rows = []
        self.stop_evt = threading.Event()
        self.state = "recording"
        self.fits, self.enf, self.result, self.error = [], {}, None, None
        self.others = []
        self.searched_events = None
        self.log_lines, self.log_cap = [], log_cap
        self.dem_factory = dem_factory
        self.lock = threading.Lock()
        self.updated = time.time()
        nm = min(250, max(5, radius_km / 1.852))
        self.poller = SharedPoller.acquire(lat, lon, nm, log=self.log)
        threading.Thread(target=poll_grid, args=(self.stop_evt, self.grid_rows), daemon=True).start()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def log(self, msg):
        with self.lock:
            self.log_lines = (self.log_lines + [str(msg)])[-self.log_cap:]

    def add_audio(self, data: bytes):
        if self.state != "recording":
            return
        n = len(data) // 2
        if self.samples + n > MAX_SECONDS * self.sr:
            self.finish()
            return
        with open(self.pcm, "ab") as fh:
            fh.write(data[: n * 2])
        self.samples += n
        self.updated = time.time()

    @property
    def adsb_rows(self):
        return self.poller.rows

    def finish(self):
        if self.state == "recording":
            self.state = "finishing"

    @property
    def duration(self):
        return self.samples / self.sr

    def _area(self):
        r = self.radius_km * 1000
        dlat, dlon = r / M_PER_DEG_LAT, r / m_per_deg_lon(self.lat)
        return {"lat_min": self.lat - dlat, "lat_max": self.lat + dlat,
                "lon_min": self.lon - dlon, "lon_max": self.lon + dlon}

    def _analyze(self):
        x = np.fromfile(self.pcm, "<i2").astype(np.float32) / 32768.0
        if len(x) < self.sr * 40:
            return
        a = analyze_audio((x, self.sr), self.dir, log=lambda m: None)
        self.fits = [{k: v for k, v in f.items() if k not in ("sigma",)} for f in a.fits]
        self.others = [{k: f[k] for k in ("tc", "f0", "v", "d", "kind", "label", "strength")} for f in a.others or []]
        self.enf = {k: a.enf[k] for k in ("nominal", "strength", "regions", "quality")}
        if a.enf["nominal"] == 50 and self.grid_rows:
            g = np.array(self.grid_rows)
            r = correlate_reference(a.enf["t"], a.enf["f"], g[:, 0], g[:, 1], self.t0)
            self.enf["europe_corr"] = r
        if not a.events or not self.adsb_rows:
            return
        key = json.dumps([(round(e["t_cpa"]), round(e["f0"])) for e in a.events])
        if key == self.searched_events:
            return
        self.searched_events = key
        self.log(f"{len(a.events)} flyover(s) heard: searching")
        adsb = pd.DataFrame(self.adsb_rows, columns=["flight_id", "time", "lat", "lon", "alt_m", "typecode"])
        adsb = adsb.drop_duplicates(["flight_id", "time"])
        p = SearchParams(start_window=(self.t0 - 3, self.t0 + 3), tile_deg=0.5)
        try:
            self.result = locate(a.spec, a.events, adsb, self._area(), self.dem_factory(), self.dir, p,
                                 log=self.log)
            self.result.pop("heat_points", None)
        except RuntimeError as e:
            self.log(str(e))

    def _loop(self):
        last = 0
        while True:
            try:
                if self.state == "finishing":
                    self.log("final analysis")
                    self._analyze()
                    self.state = "done"
                    self.stop_evt.set()
                    self.poller.release(self.log)
                    return
                if self.samples - last >= 15 * self.sr:
                    last = self.samples
                    self._analyze()
                if time.time() - self.updated > 900:  # browser gone for good (iOS pauses the mic when locked)
                    self.finish()
            except Exception as e:
                traceback.print_exc()
                self.error = str(e)
                self.log("ERROR: " + str(e))
            time.sleep(2)

    def status(self):
        with self.lock:
            log = list(self.log_lines)
        props = {r[0] for r in self.adsb_rows if r[5] and is_propeller(r[5])}
        latest = {}
        now = time.time()
        for hexid, t, la, lo, alt, typ in self.adsb_rows[-3000:]:
            if now - t < 30:
                latest[hexid] = {"hex": hexid, "lat": la, "lon": lo, "alt_m": round(alt), "type": typ,
                                 "prop": bool(typ) and is_propeller(typ)}
        return {
            "aircraft": list(latest.values()),
            "state": self.state, "duration": round(self.duration, 1), "t0": self.t0,
            "area": {"lat": self.lat, "lon": self.lon, "radius_km": self.radius_km},
            "aircraft_seen": len({r[0] for r in self.adsb_rows}), "typed_aircraft": len(props),
            "grid_samples": len(self.grid_rows),
            "fits": self.fits, "others": self.others, "enf": self.enf, "result": self.result, "error": self.error, "log": log,
        }
