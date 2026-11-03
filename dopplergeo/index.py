"""Compact daily ADS-B index.

A day of adsb.lol `globe_history` is ~3 GB, almost all of it useless here:
jets, cruising traffic, ground points and per-point fields we never read. Each
day is streamed once and only what can be heard as a propeller Doppler curve
is kept (low altitude, not a known jet, one point every few seconds), which is
~15x smaller. Searches then read these files instead of the archive.

    python -m dopplergeo.index 2026-10-01 2026-10-31 --root data/adsb_index --workers 3
"""

import argparse
import datetime as dt
import io
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .adsb import _ChainedStream, _load_json_bytes, _parse_trace
from .aircraft import is_propeller

MAX_ALT_M = 6000.0  # cruising traffic is never heard as a Doppler curve
STEP_S = 4.0  # trajectories are splined back to 1 s; straight legs lose nothing
SCHEMA = pa.schema([("flight_id", pa.string()), ("time", pa.float64()), ("lat", pa.float32()),
                    ("lon", pa.float32()), ("alt_m", pa.float32()), ("typecode", pa.string())])


def day_path(root, day):
    return Path(root) / f"{day:%Y}" / f"{day:%Y-%m-%d}.parquet"


def available_days(root):
    return sorted(dt.date.fromisoformat(p.stem) for p in Path(root).glob("*/*.parquet"))


def _decimate(rows, step):
    out, last = [], -np.inf
    for r in rows:
        if r[1] - last >= step:
            out.append(r)
            last = r[1]
    return out


def build_day(day, root, log=print, step_s=STEP_S, max_alt_m=MAX_ALT_M, every_mb=500):
    """Stream one day of adsb.lol worldwide and write its compact index file."""
    from .fetch import _asset_urls

    path = day_path(root, day)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    raw = _ChainedStream(_asset_urls(day))
    t0, n, kept, next_mb, batch = time.time(), 0, 0, every_mb, []
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd", compression_level=9)

    def flush():
        nonlocal batch
        if batch:
            cols = list(zip(*batch))
            writer.write_table(pa.table(dict(zip(SCHEMA.names, cols)), schema=SCHEMA))
            batch = []

    try:
        with tarfile.open(fileobj=io.BufferedReader(raw, 1 << 20), mode="r|*") as tar:
            for member in tar:
                if not member.isfile() or "trace" not in member.name:
                    continue
                n += 1
                try:
                    rows = _parse_trace(_load_json_bytes(tar.extractfile(member).read()), max_alt_m=max_alt_m)
                except (ValueError, OSError):
                    continue
                if not rows or not is_propeller(rows[-1][5]):
                    continue
                rows = _decimate(rows, step_s)
                kept += len(rows)
                batch += rows
                if len(batch) > 500_000:
                    flush()
                if log and raw.bytes / 1e6 >= next_mb:
                    next_mb += every_mb
                    log(f"  {day}: {raw.bytes / 1e9:.1f} GB read, {n} aircraft, {kept} points kept")
        flush()
        writer.close()
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(path)
    log(f"{day}: indexed {kept} points in {time.time() - t0:.0f}s "
        f"({raw.bytes / 1e9:.2f} GB -> {path.stat().st_size / 1e6:.0f} MB)")
    return path


def load_day(root, day, bbox=None, t_range=None, compact=False):
    """Points of one indexed day inside bbox=(lat0, lat1, lon0, lon1) and t_range (unix).

    compact: categorical ids and float32 positions (a worldwide day in ~1 GB)."""
    f = None
    if bbox is not None:
        f = ((ds.field("lat") >= bbox[0]) & (ds.field("lat") <= bbox[1])
             & (ds.field("lon") >= bbox[2]) & (ds.field("lon") <= bbox[3]))
    if t_range is not None:
        ft = (ds.field("time") >= t_range[0]) & (ds.field("time") <= t_range[1])
        f = ft if f is None else f & ft
    df = ds.dataset(day_path(root, day)).to_table(filter=f).to_pandas(strings_to_categorical=compact)
    if compact:
        return df
    for c in ("lat", "lon", "alt_m"):
        df[c] = df[c].astype(float)
    return df


def _build_quiet(day, root):
    return build_day(day, root, log=lambda m: print(m, flush=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("start", type=dt.date.fromisoformat)
    ap.add_argument("end", type=dt.date.fromisoformat)
    ap.add_argument("--root", default=os.path.join(os.environ.get("DATA_DIR", "data"), "adsb_index"))
    ap.add_argument("--workers", type=int, default=2, help="days streamed at once (network-bound)")
    a = ap.parse_args(argv)
    days = [a.start + dt.timedelta(d) for d in range((a.end - a.start).days + 1)]
    days = [d for d in days if not day_path(a.root, d).exists()]
    print(f"{len(days)} day(s) to index into {a.root}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_build_quiet, d, a.root): d for d in days}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"{futs[f]}: FAILED {e}", flush=True)


if __name__ == "__main__":
    main()
