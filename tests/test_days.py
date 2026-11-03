"""Find the day: the true flyovers are on one of three indexed days."""

import datetime as dt

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dopplergeo.daysearch import day_events, search_days
from dopplergeo.geo import haversine_m
from dopplergeo.index import SCHEMA, day_path
from dopplergeo.pipeline import analyze_audio
from dopplergeo.synth import make_scenario


def _write(df, root, day):
    p = day_path(root, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df[SCHEMA.names].astype({"lat": "float32", "lon": "float32",
                                                                  "alt_m": "float32"}), schema=SCHEMA,
                                        preserve_index=False), p)


def test_find_the_day(tmp_path):
    truth = make_scenario(tmp_path / "scn")
    adsb = pd.read_parquet(tmp_path / "scn" / "adsb.parquet")
    adsb["typecode"] = adsb["typecode"].fillna("").astype(str)
    day = dt.datetime.fromtimestamp(truth["rec_start"], dt.timezone.utc).date()
    decoys = adsb[~adsb["flight_id"].str.startswith("TRUE")]
    root = tmp_path / "idx"
    for k in (-1, 1):  # same traffic minus the aircraft really heard, one day before/after
        _write(decoys.assign(time=decoys["time"] + k * 86400), root, day + dt.timedelta(k))
    _write(adsb, root, day)

    a = analyze_audio(tmp_path / "scn" / "recording.wav", plots=False)
    area = {"lat_min": 40.0, "lat_max": 41.0, "lon_min": -75.0, "lon_max": -74.0}
    res, ranked = search_days(a.spec, day_events(a.fits), [day + dt.timedelta(k) for k in (-1, 0, 1)], root, area,
                              {"type": "npz", "path": str(tmp_path / "scn" / "dem.npz")}, tmp_path / "out",
                              workers=3)
    print(ranked)
    assert res is not None and res["day"] == str(day)
    assert res["day_confidence"] > 1.4  # every heard flyover must line up: decoy days fall well behind
    best = res["top"][0]
    assert haversine_m(truth["lat"], truth["lon"], best["lat"], best["lon"]) < 150
    assert abs(best["rec_start_unix"] - truth["rec_start"]) < 5
