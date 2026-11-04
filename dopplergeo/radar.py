"""Live radar for the maps: what is flying in view, where it has been and where it is going.

Positions come from the adsb.lol live API, fetched by fixed cells and shared by every
viewer (see `Radar`). Each aircraft carries a short trail of past fixes and a predicted
path for the next five minutes, built from what the transponder says: turn rate,
vertical rate, the altitude selected on the autopilot, and, for airline flights whose
callsign has a known route (adsbdb.com), the bearing to the destination airport.
"""

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .aircraft import is_propeller

API = "https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/{nm}"
ROUTE_API = "https://api.adsbdb.com/v0/callsign/{cs}"
FT = 0.3048
KT = 0.5144
FPM = 0.00508  # ft/min -> m/s
R_EARTH = 6371000.0

AHEAD_S, STEP_S = 300, 10
MAX_TURN_DEG_S = 1.5  # a standard-rate turn is 3 deg/s; airliners steer gentler
TRAIL_S, TRAIL_EVERY_S, TRAIL_MAX = 600, 15, 40


def _move(lat, lon, hdg, m):
    r = math.radians(hdg)
    return (lat + m * math.cos(r) / 111320,
            lon + m * math.sin(r) / (111320 * math.cos(math.radians(lat))))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def _dist(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def predict(a, dest=None):
    """Points [lon, lat, seconds after the fix, alt_m] every STEP_S for AHEAD_S.

    The current turn goes on for half a minute; with a destination the aircraft then
    steers towards it at MAX_TURN_DEG_S. Climbs and descents go on at the current rate
    until the altitude selected on the autopilot (when sent), never below the ground
    or the destination; the path stops at the destination."""
    v, hdg = a.get("speed_ms"), a.get("track")
    if not v or hdg is None or a["alt_m"] <= 0:
        return []
    lat, lon, alt = a["lat"], a["lon"], float(a["alt_m"])
    vs, target, turn = a.get("vs_ms") or 0.0, a.get("sel_alt_m"), a.get("turn_deg_s") or 0.0
    if target is not None and (vs < 0 and target >= alt or vs > 0 and target <= alt):
        target = None  # already past it (on final, the go-around altitude is set): not where it is heading
    floor = dest["elev_m"] if dest else 0.0
    pts = [[round(lon, 5), round(lat, 5), 0, round(alt)]]
    for k in range(1, AHEAD_S // STEP_S + 1):
        t = k * STEP_S
        if dest and t > 20:
            diff = (_bearing(lat, lon, dest["lat"], dest["lon"]) - hdg + 540) % 360 - 180
            hdg += max(-MAX_TURN_DEG_S * STEP_S, min(MAX_TURN_DEG_S * STEP_S, diff))
        elif t <= 30:
            hdg += turn * STEP_S
        if dest and _dist(lat, lon, dest["lat"], dest["lon"]) < v * STEP_S:
            pts.append([round(dest["lon"], 5), round(dest["lat"], 5), t, round(floor)])
            break
        lat, lon = _move(lat, lon, hdg, v * STEP_S)
        if vs:
            alt += vs * STEP_S
            if target is not None and (vs < 0 and alt <= target or vs > 0 and alt >= target):
                alt, vs = float(target), 0.0
        alt = max(alt, floor)
        pts.append([round(lon, 5), round(lat, 5), t, round(alt)])
    return pts


class Routes:
    """Origin and destination of airline flights by callsign, from adsbdb.com: looked up
    in the background, one request every `gap` s, kept a day (unknown ones too)."""

    def __init__(self, gap=1.5, keep=86400.0):
        self.gap, self.keep = gap, keep
        self.lock = threading.Lock()
        self.known = {}  # callsign -> (time, dest dict or None)
        self.queue = []
        self.thread = None

    def get(self, cs):
        """Destination if known; otherwise queue the lookup and return None for now."""
        cs = cs.strip().upper()
        if not cs:
            return None
        with self.lock:
            hit = self.known.get(cs)
            if hit and time.time() - hit[0] < self.keep:
                return hit[1]
            if cs not in self.queue and len(self.queue) < 500:
                self.queue.append(cs)
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
        return None

    def _run(self):
        while True:
            with self.lock:
                cs = self.queue.pop(0) if self.queue else None
            if cs is None:
                time.sleep(1.0)
                continue
            dest = None
            try:
                req = urllib.request.Request(ROUTE_API.format(cs=urllib.parse.quote(cs)),
                                             headers={"User-Agent": "dopplergeo-map"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    route = json.load(r).get("response", {}).get("flightroute") or {}
                d = route.get("destination") or {}
                if d.get("latitude") is not None:
                    dest = {"lat": d["latitude"], "lon": d["longitude"], "code": d.get("iata_code") or d.get("icao_code"),
                            "name": d.get("name") or "", "elev_m": round((d.get("elevation") or 0) * FT)}
            except urllib.error.HTTPError as e:
                if e.code == 429:  # come back to it later
                    with self.lock:
                        self.queue.append(cs)
                    time.sleep(30)
                    continue
            except Exception:
                pass
            with self.lock:
                self.known[cs] = (time.time(), dest)
            time.sleep(self.gap)


class Radar:
    """What is flying in a map view right now.

    The world is cut into fixed CELL-degree cells. A view only marks the cells it
    covers as wanted and reads a shared store of latest positions; one background
    thread refreshes the wanted cells in turn, at most one API request per
    `min_gap` seconds and each cell at most every `refresh` seconds, backing off
    after HTTP 429. However many people look and zoom, the request rate to the API
    stays bounded. Each aircraft carries the time of its fix ("t") and its predicted
    path: the maps move it along that path between refreshes, so these can be slow."""

    CELL = 4.0  # degrees; one request (radius 160 nm) covers a cell

    def __init__(self, refresh=30.0, keep=180.0, min_gap=1.0, want_for=60.0, max_cells=12):
        self.refresh, self.keep, self.min_gap, self.want_for, self.max_cells = refresh, keep, min_gap, want_for, max_cells
        self.lock = threading.Lock()
        self.planes = {}  # hex -> latest aircraft dict, with its fix time "t"
        self.trails = {}  # hex -> [[lon, lat, alt_m, t], ...] oldest first
        self.wanted = {}  # cell -> last time a view covered it
        self.fetched = {}  # cell -> last fetch attempt
        self.blocked_until = 0.0
        self.thread = None
        self.routes = Routes()

    def _cells(self, south, west, north, east):
        c = self.CELL
        cells = [(i, j) for i in range(math.floor(south / c), math.floor(north / c) + 1)
                 for j in range(math.floor(west / c), math.floor(east / c) + 1)]
        mid = ((south + north) / 2, (west + east) / 2)
        # too wide a view: the cells nearest its centre only
        cells.sort(key=lambda ij: ((ij[0] + 0.5) * c - mid[0]) ** 2 + ((ij[1] + 0.5) * c - mid[1]) ** 2)
        return cells[: self.max_cells]

    def view(self, south, west, north, east, props_only=False, detail=False, limit=800):
        """Aircraft in the box, propellers then low aircraft first. `detail` (a zoomed-in
        view): also trails and destinations, and routes are looked up for these flights."""
        now = time.time()
        with self.lock:
            for cell in self._cells(south, west, north, east):
                self.wanted[cell] = now
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            sel = [a for a in self.planes.values()
                   if now - a["t"] < self.keep and south <= a["lat"] <= north and west <= a["lon"] <= east
                   and (a["prop"] or not props_only)]
            trails = {a["hex"]: list(self.trails.get(a["hex"], [])) for a in sel} if detail else {}
            blocked = now < self.blocked_until
        sel.sort(key=lambda a: (not a["prop"], a["alt_m"]))
        out = []
        for a in sel[:limit]:
            dest = self.routes.get(a["flight"]) if detail and a["flight"] else None
            if dest and not self._heading_to(a, dest):
                dest = None
            b = {k: v for k, v in a.items() if k not in ("vs_ms", "sel_alt_m", "turn_deg_s")}
            b["vs_ms"] = a.get("vs_ms")
            b["path"] = predict(a, dest)
            if detail:
                b["trail"] = [p[:3] for p in trails.get(a["hex"], [])]
                if dest:
                    b["dest"] = {k: dest[k] for k in ("lat", "lon", "code", "name")}
            out.append(b)
        return out, blocked

    @staticmethod
    def _heading_to(a, dest):
        """A route table can be stale: trust it only if the aircraft roughly points there."""
        if a.get("track") is None:
            return False
        d = _dist(a["lat"], a["lon"], dest["lat"], dest["lon"])
        diff = abs((_bearing(a["lat"], a["lon"], dest["lat"], dest["lon"]) - a["track"] + 540) % 360 - 180)
        return d < 2_500_000 and (diff < 70 or d < 60_000)

    def _run(self):
        while True:
            now = time.time()
            with self.lock:
                for k in [k for k, t in self.wanted.items() if now - t > self.want_for]:
                    del self.wanted[k]
                for k in [k for k, a in self.planes.items() if now - a["t"] > self.keep]:
                    del self.planes[k]
                    self.trails.pop(k, None)
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
            rate = a.get("baro_rate", a.get("geom_rate"))
            sel = a.get("nav_altitude_mcp", a.get("nav_altitude_fms"))
            got[a["hex"]] = {"hex": a["hex"], "lat": a["lat"], "lon": a["lon"],
                             "alt_m": 0 if alt == "ground" else round(alt * FT), "type": typ,
                             "prop": bool(typ) and is_propeller(typ), "track": a.get("track"),
                             "speed_ms": round(a["gs"] * KT) if a.get("gs") is not None else None,
                             "flight": (a.get("flight") or "").strip(),
                             "vs_ms": round(rate * FPM, 1) if rate is not None and abs(rate) >= 200 else 0.0,
                             "sel_alt_m": round(sel * FT) if sel is not None else None,
                             "turn_deg_s": a.get("track_rate") or 0.0,
                             "t": round(now - float(a.get("seen_pos", 0)), 1)}
        with self.lock:
            self.planes.update(got)
            for h, a in got.items():
                tr = self.trails.setdefault(h, [])
                if not tr or a["t"] - tr[-1][3] >= TRAIL_EVERY_S:
                    tr.append([round(a["lon"], 5), round(a["lat"], 5), a["alt_m"], a["t"]])
                while tr and (a["t"] - tr[0][3] > TRAIL_S or len(tr) > TRAIL_MAX):
                    tr.pop(0)
