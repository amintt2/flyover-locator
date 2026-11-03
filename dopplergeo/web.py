"""Web app: drop a video, get a map.

Run: uvicorn dopplergeo.web:app --host 0.0.0.0 --port 8000
Env: DATA_DIR (default ./data), APP_PASSWORD (optional shared password),
     MAX_UPLOAD_MB (default 500), MAX_RADIUS_KM (default 100).
"""

import datetime as dt
import json
import os
import secrets
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .dem import load_dem
from .fetch import fetch_adsb
from .geo import M_PER_DEG_LAT, m_per_deg_lon
from .audio import media_start_time
from .pipeline import run
from .search import SearchParams

DATA = Path(os.environ.get("DATA_DIR", "data")).resolve()
JOBS = DATA / "jobs"
CACHE = DATA / "adsb_cache"
PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MAX_RADIUS_KM = float(os.environ.get("MAX_RADIUS_KM", "100"))
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Flyover Locator")
pool = ThreadPoolExecutor(max_workers=1)  # one heavy search at a time
lock = threading.Lock()


def _check(pw):
    if PASSWORD and not secrets.compare_digest(pw or "", PASSWORD):
        raise HTTPException(401, "wrong password")


def _status(job, **kw):
    path = JOBS / job / "status.json"
    with lock:
        st = json.loads(path.read_text()) if path.exists() else {"log": []}
        st.update(kw)
        path.write_text(json.dumps(st, default=float))


def _log(job):
    def f(msg):
        with lock:
            path = JOBS / job / "status.json"
            st = json.loads(path.read_text())
            st["log"] = (st["log"] + [str(msg)])[-200:]
            path.write_text(json.dumps(st, default=float))
    return f


def _work(job, media, lat, lon, radius_km, start_unix, start_tol_s, use_meta):
    log = _log(job)
    try:
        _status(job, state="running")
        meta = media_start_time(media)
        if meta and use_meta:
            log(f"time from video metadata: {dt.datetime.fromtimestamp(meta, dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
            start_unix, start_tol_s = meta, max(start_tol_s if start_unix else 0, 180.0)
        if start_unix is None:
            raise RuntimeError("no recording date: enter it, or upload the original video file with its metadata")
        r_m = radius_km * 1000
        dlat, dlon = r_m / M_PER_DEG_LAT, r_m / m_per_deg_lon(lat)
        area = {"lat_min": lat - dlat, "lat_max": lat + dlat, "lon_min": lon - dlon, "lon_max": lon + dlon}
        buf = 12000 / M_PER_DEG_LAT
        bbox = (area["lat_min"] - buf, area["lat_max"] + buf, area["lon_min"] - buf * 1.4, area["lon_max"] + buf * 1.4)
        t_range = (start_unix - start_tol_s - 1800, start_unix + start_tol_s + 7200)
        if os.environ.get("ADSB_FILE"):  # local/offline data (testing, or your own export)
            from .adsb import read_table

            adsb = read_table(os.environ["ADSB_FILE"])
        else:
            adsb = fetch_adsb(bbox, t_range, CACHE, log=log)
        if adsb.empty:
            raise RuntimeError("no aircraft data in this area/time")
        p = SearchParams(start_window=(start_unix - start_tol_s, start_unix + start_tol_s),
                         tile_deg=0.5)
        dem = (load_dem({"type": "npz", "path": os.environ["DEM_NPZ"]}) if os.environ.get("DEM_NPZ")
               else load_dem({"type": "hgt", "dir": str(DATA / "dem"), "download": True}))
        res = run(media, adsb, area, dem, JOBS / job, params=p, log=log)
        res["area"] = {"lat": lat, "lon": lon, "radius_km": radius_km}
        _status(job, state="done", result=res)
    except Exception as e:  # report every failure to the UI
        log("ERROR: " + str(e))
        traceback.print_exc()
        _status(job, state="error", error=str(e))
    finally:
        try:
            media.unlink()
        except OSError:
            pass


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/api/config")
def config():
    return {"password": bool(PASSWORD), "max_radius_km": MAX_RADIUS_KM}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), lat: float = Form(...), lon: float = Form(...),
                     radius_km: float = Form(30.0), start: str = Form(""), start_tol_min: float = Form(30.0),
                     use_metadata: bool = Form(True), x_password: str | None = Header(None)):
    _check(x_password)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "bad coordinates")
    radius_km = min(max(radius_km, 1.0), MAX_RADIUS_KM)
    job = uuid.uuid4().hex[:12]
    d = JOBS / job
    d.mkdir(parents=True)
    media = d / ("upload" + Path(file.filename or "x").suffix[:8])
    size = 0
    with open(media, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "file too large")
            fh.write(chunk)
    start_unix = None
    if start:
        t = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        start_unix = t.timestamp()
    _status(job, state="queued", log=[f"uploaded {size / 1e6:.1f} MB"])
    pool.submit(_work, job, media, lat, lon, radius_km, start_unix, start_tol_min * 60, use_metadata)
    return {"job": job}


@app.get("/api/jobs/{job}")
def job_status(job: str, x_password: str | None = Header(None)):
    _check(x_password)
    path = JOBS / job / "status.json"
    if not job.isalnum() or not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())


@app.get("/api/jobs/{job}/img/{name}")
def job_image(job: str, name: str, pw: str = ""):
    _check(pw)
    if not job.isalnum() or name not in {"flyovers.png", "match_1.png", "match_2.png", "match_3.png"}:
        raise HTTPException(404)
    path = JOBS / job / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")
