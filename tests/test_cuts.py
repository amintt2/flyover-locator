"""Edited video: 400 s are cut out between two flyovers, so their time offset is wrong.
With the cut known, clips are linked by place only and the location is still found."""

import time

import numpy as np
import pandas as pd

from dopplergeo.audio import load_audio
from dopplergeo.dem import GridDEM
from dopplergeo.geo import haversine_m
from dopplergeo.pipeline import SR, analyze_audio, locate
from dopplergeo.search import SearchParams
from dopplergeo.synth import make_scenario

CUT, REMOVED = 600.0, 400.0


def _edited(tmp_path):
    truth = make_scenario(tmp_path / "scn")
    x, sr = load_audio(tmp_path / "scn" / "recording.wav", sr=SR)
    x = np.concatenate([x[: int(CUT * sr)], x[int((CUT + REMOVED) * sr):]])
    adsb = pd.read_parquet(tmp_path / "scn" / "adsb.parquet")
    dem = GridDEM.from_npz(tmp_path / "scn" / "dem.npz")
    area = {"lat_min": 40.0, "lat_max": 41.0, "lon_min": -75.0, "lon_max": -74.0}
    return truth, (x, sr), adsb, dem, area


def test_edited_video_with_cut(tmp_path):
    truth, audio, adsb, dem, area = _edited(tmp_path)
    a = analyze_audio(audio, plots=False, cuts=[CUT])
    assert a.clips and len(a.clips) == 2
    assert {e["clip"] for e in a.events} == {0, 1}  # flyovers on both sides of the cut
    p = SearchParams(clips=a.clips)
    res = locate(a.spec, a.events, adsb, area, dem, tmp_path / "out", params=p, plots=True,
                 log=lambda m: print(time.strftime("%X"), m, flush=True))
    best = res["top"][0]
    err = haversine_m(truth["lat"], truth["lon"], best["lat"], best["lon"])
    print(res["top"][:2], "error m:", err)
    assert err < 150
    assert abs(best["rec_start_unix"] - truth["rec_start"]) < 5
