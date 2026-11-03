"""Ground elevation providers. All return metres above mean sea level for
vectorised (lat, lon) inputs."""

import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ConstantDEM:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self, lat, lon):
        return np.full(np.shape(lat), self.value)


class GridDEM:
    """In-memory grid: npz with `lat` (ascending), `lon` (ascending), `elev[lat, lon]`."""

    def __init__(self, lat, lon, elev):
        self._f = RegularGridInterpolator((lat, lon), elev, bounds_error=False, fill_value=None)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        return cls(d["lat"], d["lon"], d["elev"])

    def __call__(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        return self._f(np.stack([lat.ravel(), lon.ravel()], axis=1)).reshape(lat.shape)


class HGTDEM:
    """Directory of SRTM .hgt tiles (1 or 3 arc-second). Missing tiles read as 0 m."""

    SKADI = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{folder}/{name}.gz"

    def __init__(self, directory, download=False):
        self.dir = Path(directory)
        self.download = download
        self._cache = {}

    def _fetch(self, name):
        import gzip
        import urllib.request

        self.dir.mkdir(parents=True, exist_ok=True)
        url = self.SKADI.format(folder=name[:3], name=name)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                (self.dir / name).write_bytes(gzip.decompress(r.read()))
        except Exception:
            pass  # ocean / missing tile -> 0 m

    def _tile(self, la, lo):
        key = (la, lo)
        if key not in self._cache:
            name = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}.hgt"
            if self.download and not (self.dir / name).exists():
                self._fetch(name)
            path = next((p for p in (self.dir / name, self.dir / name.lower()) if p.exists()), None)
            if path is None:
                self._cache[key] = None
            else:
                n = int(round(np.sqrt(os.path.getsize(path) / 2)))
                a = np.fromfile(path, ">i2").reshape(n, n).astype(np.float32)
                a[a < -1000] = 0.0  # voids
                self._cache[key] = a
        return self._cache[key]

    def __call__(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        out = np.zeros(lat.shape)
        la_f, lo_f = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for la, lo in set(zip(la_f.ravel().tolist(), lo_f.ravel().tolist())):
            a = self._tile(la, lo)
            if a is None:
                continue
            m = (la_f == la) & (lo_f == lo)
            n = a.shape[0] - 1
            r = (la + 1 - lat[m]) * n
            c = (lon[m] - lo) * n
            r0, c0 = np.clip(np.floor(r).astype(int), 0, n - 1), np.clip(np.floor(c).astype(int), 0, n - 1)
            fr, fc = r - r0, c - c0
            out[m] = (a[r0, c0] * (1 - fr) * (1 - fc) + a[r0 + 1, c0] * fr * (1 - fc)
                      + a[r0, c0 + 1] * (1 - fr) * fc + a[r0 + 1, c0 + 1] * fr * fc)
        return out


class GeoTiffDEM:
    """Any single-band GeoTIFF in EPSG:4326 (needs rasterio)."""

    def __init__(self, path):
        import rasterio

        with rasterio.open(path) as ds:
            self.a = ds.read(1).astype(np.float32)
            self.inv = ~ds.transform
            nod = ds.nodata
        if nod is not None:
            self.a[self.a == nod] = 0.0

    def __call__(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        c, r = self.inv * (lon, lat)
        c, r = np.asarray(c) - 0.5, np.asarray(r) - 0.5
        h, w = self.a.shape
        r0 = np.clip(np.floor(r).astype(int), 0, h - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, w - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        a = self.a
        return (a[r0, c0] * (1 - fr) * (1 - fc) + a[r0 + 1, c0] * fr * (1 - fc)
                + a[r0, c0 + 1] * (1 - fr) * fc + a[r0 + 1, c0 + 1] * fr * fc)


def load_dem(cfg):
    cfg = cfg or {"type": "constant", "value": 0}
    kind = cfg.get("type", "constant")
    if kind == "constant":
        return ConstantDEM(cfg.get("value", 0))
    if kind == "npz":
        return GridDEM.from_npz(cfg["path"])
    if kind == "hgt":
        return HGTDEM(cfg["dir"], cfg.get("download", False))
    if kind == "geotiff":
        return GeoTiffDEM(cfg["path"])
    raise ValueError(f"unknown DEM type {kind!r}")
