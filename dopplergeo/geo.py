"""Geodesy helpers. All range computations are done in ECEF, which is exact 3D
geometry (no projection or earth-curvature error)."""

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
M_PER_DEG_LAT = 111_132.0


def ecef(lat, lon, h):
    """Geodetic (deg, deg, metres above ellipsoid) -> ECEF metres, shape (..., 3)."""
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + h) * s
    return np.stack([x, y, z], axis=-1)


def m_per_deg_lon(lat):
    return M_PER_DEG_LAT * np.cos(np.radians(lat))


def latlon_grid(lat_min, lat_max, lon_min, lon_max, step_m):
    """Regular lat/lon grid with roughly `step_m` spacing (lon spacing set at mid-latitude)."""
    dlat = step_m / M_PER_DEG_LAT
    dlon = step_m / m_per_deg_lon((lat_min + lat_max) / 2)
    lats = np.arange(lat_min + dlat / 2, lat_max, dlat)
    lons = np.arange(lon_min + dlon / 2, lon_max, dlon)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return la.ravel(), lo.ravel()


def offset_latlon(lat, lon, east_m, north_m):
    return lat + north_m / M_PER_DEG_LAT, lon + east_m / m_per_deg_lon(lat)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))
