"""ADS-B ingestion and trajectory preparation.

Canonical point table columns: flight_id, time (unix s), lat, lon, alt_m
(geometric height, metres), typecode.

Storage-friendly ingestion: `stream_readsb_archive` reads a daily readsb /
adsb.lol `globe_history` tar (local file or URLs, optionally split into parts)
as a stream and keeps only points inside a lat/lon box and time window, so
the multi-GB archive never touches the disk.
"""

import glob
import gzip
import io
import json
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter1d

from .aircraft import is_propeller
from .geo import ecef

FT = 0.3048


# ---------------------------------------------------------------- readers


def read_table(paths, aircraft_db=None):
    """Parquet/CSV in canonical or OpenSky state-vector format (glob allowed)."""
    files = sorted(glob.glob(paths)) if isinstance(paths, str) else list(paths)
    if not files:
        raise FileNotFoundError(paths)
    dfs = [pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if "flight_id" not in df.columns and "icao24" in df.columns:  # OpenSky state vectors
        alt = df["geoaltitude"] if "geoaltitude" in df.columns else df["baroaltitude"]
        df = pd.DataFrame({
            "flight_id": df["icao24"].astype(str),
            "time": df["time"].astype(float),
            "lat": df["lat"], "lon": df["lon"], "alt_m": alt,
            "typecode": df["typecode"] if "typecode" in df.columns else "",
            **({"onground": df["onground"]} if "onground" in df.columns else {}),
        })
        if "onground" in df.columns:
            df = df[~df.pop("onground").astype(bool)]
    if aircraft_db:
        db = pd.read_csv(aircraft_db, usecols=lambda c: c.lower() in ("icao24", "typecode"))
        db.columns = [c.lower() for c in db.columns]
        m = dict(zip(db["icao24"].astype(str).str.lower(), db["typecode"].fillna("")))
        icao = df["flight_id"].str.split("_").str[0].str.lower()
        df["typecode"] = icao.map(m).fillna(df.get("typecode", ""))
    df["typecode"] = df.get("typecode", "").fillna("").astype(str)
    return df.dropna(subset=["lat", "lon", "alt_m"])[["flight_id", "time", "lat", "lon", "alt_m", "typecode"]]


def _parse_trace(obj, bbox=None, t_range=None, max_alt_m=None):
    """readsb trace JSON -> list of canonical rows."""
    base = float(obj.get("timestamp", 0))
    icao = str(obj.get("icao", "")).lower()
    tcode = obj.get("t") or ""
    rows, leg = [], 0
    for p in obj.get("trace", []):
        flags = p[6] if len(p) > 6 and isinstance(p[6], int) else 0
        if flags & 2:
            leg += 1
        t, lat, lon, alt = base + p[0], p[1], p[2], p[3]
        if lat is None or alt is None or alt == "ground" or flags & 1:
            continue
        if bbox and not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
            continue
        if t_range and not (t_range[0] <= t <= t_range[1]):
            continue
        geom = p[10] if len(p) > 10 else None
        alt_ft = geom if isinstance(geom, (int, float)) else alt
        if len(p) > 8 and isinstance(p[8], dict) and not tcode:
            tcode = p[8].get("t", "") or ""
        if max_alt_m is not None and alt_ft * FT > max_alt_m:
            continue  # cruising traffic is never heard as a Doppler curve
        rows.append((f"{icao}_{leg}", t, lat, lon, alt_ft * FT, tcode))
    return rows


def _load_json_bytes(b):
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return json.loads(b)


def read_readsb_traces(paths, bbox=None, t_range=None):
    """Directory/glob of extracted readsb trace files (*.json, possibly gzip)."""
    files = sorted(glob.glob(paths, recursive=True)) if isinstance(paths, str) else list(paths)
    rows = []
    for f in files:
        with open(f, "rb") as fh:
            rows += _parse_trace(_load_json_bytes(fh.read()), bbox, t_range)
    return pd.DataFrame(rows, columns=["flight_id", "time", "lat", "lon", "alt_m", "typecode"])


class _ChainedStream(io.RawIOBase):
    """Concatenate several URLs/files (e.g. archive.tar.aa, .ab, ...) into one stream."""

    def __init__(self, sources):
        self.sources = list(sources)
        self.cur = None
        self.bytes = 0

    def readable(self):
        return True

    def _open(self, src):
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "dopplergeo"})
            return urllib.request.urlopen(req)
        return open(src, "rb")

    def readinto(self, buf):
        while True:
            if self.cur is None:
                if not self.sources:
                    return 0
                self.cur = self._open(self.sources.pop(0))
            n = self.cur.readinto(buf)
            if n:
                self.bytes += n
                return n
            self.cur.close()
            self.cur = None


def stream_readsb_archive(sources, out_parquet, bbox, t_range=None, log=print, every_mb=250, max_alt_m=None):
    """Stream a globe_history tar (parts in order) and keep only matching points.

    bbox = (lat_min, lat_max, lon_min, lon_max). Writes a small parquet file.
    """
    raw = _ChainedStream(sources)
    stream = io.BufferedReader(raw, buffer_size=1 << 20)
    rows, n, next_mb = [], 0, every_mb
    with tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile() or "trace" not in member.name:
                continue
            data = tar.extractfile(member).read()
            n += 1
            try:
                rows += _parse_trace(_load_json_bytes(data), bbox, t_range, max_alt_m)
            except (ValueError, OSError):
                continue
            if log and raw.bytes / 1e6 >= next_mb:
                next_mb += every_mb
                log(f"  {raw.bytes / 1e9:.2f} GB read, {n} aircraft scanned, {len(rows)} points kept")
    df = pd.DataFrame(rows, columns=["flight_id", "time", "lat", "lon", "alt_m", "typecode"])
    df.to_parquet(out_parquet, index=False)
    return df


def load_adsb(cfg):
    fmt = cfg.get("format", "table")
    if fmt == "table":
        df = read_table(cfg["path"], cfg.get("aircraft_db"))
    elif fmt == "readsb":
        df = read_readsb_traces(cfg["path"])
    else:
        raise ValueError(f"unknown ADS-B format {fmt!r}")
    df = df.copy()
    df["alt_m"] = df["alt_m"].astype(float) + float(cfg.get("alt_offset_m", 0.0))
    return df


# ---------------------------------------------------------------- trajectories


@dataclass
class Segment:
    """A continuous piece of one flight resampled to 1 s and lightly smoothed."""

    sid: str
    flight: str
    typecode: str
    prop: bool
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    xyz: np.ndarray  # ECEF (N, 3)
    speed: np.ndarray  # 3D speed m/s

    def bbox(self):
        return self.lat.min(), self.lat.max(), self.lon.min(), self.lon.max()

    def sub(self, mask, min_len=5):
        """Contiguous runs of `mask` as views (same sid)."""
        out = []
        if not mask.any():
            return out
        edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= min_len:
                out.append(Segment(self.sid, self.flight, self.typecode, self.prop, self.t[a:b],
                                   self.lat[a:b], self.lon[a:b], self.alt[a:b], self.xyz[a:b], self.speed[a:b]))
        return out


def build_segments(df, type_db=None, max_gap=30.0, min_duration=20.0, smooth_s=2.0, smooth_alt_s=8.0,
                   geoid_offset_m=0.0):
    """Split flights at data gaps, resample at 1 s, convert to ECEF.

    Altitude is smoothed much more than position: it is quantised to 25 ft, and
    differentiating the steps would invent tens of m/s of vertical speed.

    geoid_offset_m converts the DEM datum to the altitude datum (if ADS-B heights
    are ellipsoidal and the DEM is MSL, pass the local geoid undulation).
    """
    segs = []
    for fid, g in df.sort_values("time").groupby("flight_id", sort=False):
        g = g.drop_duplicates("time")
        t = g["time"].to_numpy(float)
        if len(t) < 3:
            continue
        tcode = next((x for x in g["typecode"] if x), "")
        prop = is_propeller(tcode, type_db)
        cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
        for k, idx in enumerate(np.split(np.arange(len(t)), cuts)):
            ts = t[idx]
            if ts[-1] - ts[0] < min_duration:
                continue
            t1 = np.arange(np.ceil(ts[0]), ts[-1], 1.0)
            lat = np.interp(t1, ts, g["lat"].to_numpy(float)[idx])
            lon = np.interp(t1, ts, g["lon"].to_numpy(float)[idx])
            alt = np.interp(t1, ts, g["alt_m"].to_numpy(float)[idx])
            if smooth_s > 0:
                lat = gaussian_filter1d(lat, smooth_s, mode="nearest")
                lon = gaussian_filter1d(lon, smooth_s, mode="nearest")
            if smooth_alt_s > 0:
                alt = gaussian_filter1d(alt, smooth_alt_s, mode="nearest")
            xyz = ecef(lat, lon, alt)
            speed = np.linalg.norm(np.gradient(xyz, axis=0), axis=1)
            segs.append(Segment(f"{fid}#{k}", str(fid), tcode, prop, t1, lat, lon, alt, xyz, speed))
    return segs


def speed_runs(seg, v_lo, v_hi, margin=5.0, pad_s=60):
    """Sub-segments where speed is near [v_lo, v_hi] (padded so CPAs are complete)."""
    m = (seg.speed >= v_lo - margin) & (seg.speed <= v_hi + margin)
    if pad_s and m.any():
        m = binary_dilation(m, iterations=int(pad_s))
    return seg.sub(m)
