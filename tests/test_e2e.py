"""End-to-end: synthetic scenario -> automatic pipeline -> location within 150 m."""

import json

import pandas as pd

from dopplergeo.dem import GridDEM
from dopplergeo.geo import haversine_m
from dopplergeo.pipeline import run
from dopplergeo.synth import make_scenario


def test_synthetic_scenario(tmp_path):
    truth = make_scenario(tmp_path / "scn")
    adsb = pd.read_parquet(tmp_path / "scn" / "adsb.parquet")
    area = {"lat_min": 40.0, "lat_max": 41.0, "lon_min": -75.0, "lon_max": -74.0}
    res = run(tmp_path / "scn" / "recording.wav", adsb, area, GridDEM.from_npz(tmp_path / "scn" / "dem.npz"),
              tmp_path / "out", plots=True)
    best = res["top"][0]
    err = haversine_m(truth["lat"], truth["lon"], best["lat"], best["lon"])
    print(json.dumps(res["top"][:3], indent=1), "error m:", err)
    assert err < 150
    assert set(best["flights"]) == {"TRUE_A_0", "TRUE_B_0"} or all(f.startswith("TRUE") for f in best["flights"])
    assert abs(best["rec_start_unix"] - truth["rec_start"]) < 5
