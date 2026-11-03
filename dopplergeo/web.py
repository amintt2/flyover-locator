"""Web app: drop a video (or record live), get a map.

Run: uvicorn dopplergeo.web:app --host 0.0.0.0 --port 8000
Env: DATA_DIR (default ./data), APP_PASSWORD (optional shared password),
     MAX_UPLOAD_MB (default 500), MAX_RADIUS_KM (default 100),
     ADSB_FILE / DEM_NPZ (offline testing).
"""

import datetime as dt
import json
import math
import os
import secrets
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .audio import media_start_time
from .dem import load_dem
from .enf import DEFAULT_AREAS
from .fetch import fetch_adsb
from .geo import M_PER_DEG_LAT, m_per_deg_lon
from .pipeline import analyze_audio, locate
from .search import SearchParams
from .session import LiveSession

DATA = Path(os.environ.get("DATA_DIR", "data")).resolve()
JOBS = DATA / "jobs"
LIVE = DATA / "live"
CACHE = DATA / "adsb_cache"
PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MAX_RADIUS_KM = float(os.environ.get("MAX_RADIUS_KM", "100"))
MAX_LIVE_RADIUS_KM = 450.0  # adsb.lol live point queries go up to 250 nm
JOBS.mkdir(parents=True, exist_ok=True)
LIVE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Flyover Locator")
pool = ThreadPoolExecutor(max_workers=1)  # one heavy search at a time
lock = threading.Lock()
sessions: dict[str, LiveSession] = {}
IMAGES = {"flyovers.png", "match_1.png", "match_2.png", "match_3.png"}


def _check(pw):
    if PASSWORD and not secrets.compare_digest(pw or "", PASSWORD):
        raise HTTPException(401, "wrong password")


def _dem():
    if os.environ.get("DEM_NPZ"):
        return load_dem({"type": "npz", "path": os.environ["DEM_NPZ"]})
    return load_dem({"type": "hgt", "dir": str(DATA / "dem"), "download": True})


def _clean(o):
    """JSON-safe copy: NaN/inf become null (browsers and FastAPI reject them)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) or hasattr(o, "dtype"):
        try:
            v = float(o)
        except (TypeError, ValueError):
            return o
        return v if math.isfinite(v) else None
    return o


def _status(job, **kw):
    path = JOBS / job / "status.json"
    with lock:
        st = json.loads(path.read_text()) if path.exists() else {"log": []}
        st.update(_clean(kw))
        path.write_text(json.dumps(st))


def _log(job):
    def f(msg):
        with lock:
            path = JOBS / job / "status.json"
            st = json.loads(path.read_text())
            st["log"] = (st["log"] + [str(msg)])[-300:]
            path.write_text(json.dumps(st, default=float))
    return f


def _circle_area(lat, lon, radius_km):
    r = radius_km * 1000
    dlat, dlon = r / M_PER_DEG_LAT, r / m_per_deg_lon(lat)
    return {"lat_min": lat - dlat, "lat_max": lat + dlat, "lon_min": lon - dlon, "lon_max": lon + dlon}


WORLD = {"lat_min": -60.0, "lat_max": 75.0, "lon_min": -180.0, "lon_max": 180.0}


def _work(job, media, zone, start_unix, start_tol_s, use_meta, region, mode="locate"):
    log = _log(job)
    try:
        _status(job, state="running", stage="audio")
        meta = media_start_time(media)
        if meta and use_meta:
            log(f"time from video metadata: {dt.datetime.fromtimestamp(meta, dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
            start_unix, start_tol_s = meta, 180.0
        a = analyze_audio(media, JOBS / job, log=log)
        info = {"fits": [{k: v for k, v in f.items() if k != "sigma"} for f in a.fits],
                "enf": {k: a.enf[k] for k in ("nominal", "strength", "regions", "quality")},
                "duration": a.spec.duration}
        _status(job, audio=info, stage="audio_done")
        n = len(a.fits)
        if mode == "detect":
            msg = f"{n} passage(s) d'avion à hélice détecté(s)." if n else "Aucun passage d'avion à hélice détecté."
            _status(job, state="done", stage="detect_done", result={"top": [], "heat": [], "events": [], "message": msg})
            return
        if not a.events:
            _status(job, state="done", stage="no_flyover",
                    result={"top": [], "heat": [], "events": [], "message": "Aucun passage d'avion à hélice détecté."})
            return
        if start_unix is None:
            raise RuntimeError("Il faut la date de l'enregistrement : remplis-la, ou envoie le fichier original "
                               "(les applis de messagerie effacent l'heure).")
        if zone:
            area = _circle_area(*zone)
            area_info = {"lat": zone[0], "lon": zone[1], "radius_km": zone[2]}
        else:
            # no zone: everywhere aircraft data exists; only tiles crossed by matching aircraft are visited
            area = dict(WORLD)
            log("no zone given: searching everywhere matching aircraft flew")
            area_info = None
            if start_tol_s > 3600 and len(a.events) < 2:
                raise RuntimeError("Avec seulement la date et sans zone, il faut au moins 2 passages d'avion "
                                   "(un seul passage colle à des milliers d'endroits). Donne l'heure ou une zone.")
        buf = 12000 / M_PER_DEG_LAT
        bbox = (area["lat_min"] - buf, area["lat_max"] + buf, area["lon_min"] - buf * 1.5, area["lon_max"] + buf * 1.5)
        t_range = (start_unix - start_tol_s - 600, start_unix + start_tol_s + a.spec.duration + 600)
        _status(job, stage="adsb")
        if os.environ.get("ADSB_FILE"):  # local/offline data (testing, or your own export)
            from .adsb import read_table

            adsb = read_table(os.environ["ADSB_FILE"])
        else:
            adsb = fetch_adsb(bbox, t_range, CACHE, log=log, max_alt_m=6000.0)
        if adsb.empty:
            raise RuntimeError("Aucune donnée d'avion dans cette zone à cette heure.")
        _status(job, stage="search")
        p = SearchParams(start_window=(start_unix - start_tol_s, start_unix + start_tol_s), tile_deg=0.5)
        res = locate(a.spec, a.events, adsb, area, _dem(), JOBS / job, params=p, log=log)
        if area_info is None and res["top"]:
            lats = [t["lat"] for t in res["top"]]
            lons = [t["lon"] for t in res["top"]]
            area_info = {"bbox": [min(lats), min(lons), max(lats), max(lons)]}
        res["area"] = area_info
        res["enf"] = info["enf"]
        _status(job, state="done", stage="done", result=res)
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
    return {"password": bool(PASSWORD), "max_radius_km": MAX_RADIUS_KM, "max_live_radius_km": MAX_LIVE_RADIUS_KM,
            "regions": {k: v["name"] for k, v in {"europe": DEFAULT_AREAS[50], "usa": DEFAULT_AREAS[60]}.items()}}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), lat: float | None = Form(None), lon: float | None = Form(None),
                     radius_km: float = Form(30.0), start: str = Form(""), start_tol_min: float = Form(30.0),
                     use_metadata: bool = Form(True), region: str = Form("auto"), mode: str = Form("locate"),
                     x_password: str | None = Header(None)):
    _check(x_password)
    zone = None
    if lat is not None and lon is not None:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(400, "bad coordinates")
        zone = (lat, lon, min(max(radius_km, 1.0), MAX_RADIUS_KM))
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
    _status(job, state="queued", stage="queued", mode="detect" if mode == "detect" else "locate",
            log=[f"uploaded {size / 1e6:.1f} MB"])
    pool.submit(_work, job, media, zone, start_unix, start_tol_min * 60, use_metadata, region,
                "detect" if mode == "detect" else "locate")
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
    path = JOBS / job / name
    if not job.isalnum() or name not in IMAGES or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ live recording


def _session(sid):
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s


@app.post("/api/live")
async def live_start(request: Request, x_password: str | None = Header(None)):
    """Body JSON: {lat, lon, radius_km, client_now_ms, audio_start_ms}."""
    _check(x_password)
    body = await request.json()
    lat, lon = float(body["lat"]), float(body["lon"])
    radius = min(max(float(body.get("radius_km", 200)), 5.0), MAX_LIVE_RADIUS_KM)
    offset = time.time() - float(body["client_now_ms"]) / 1000  # server minus client clock
    t0 = float(body.get("audio_start_ms", body["client_now_ms"])) / 1000 + offset
    for k, s in list(sessions.items()):  # drop stale sessions
        if s.state == "done" and time.time() - s.updated > 3600:
            sessions.pop(k, None)
    if sum(s.state == "recording" for s in sessions.values()) >= 3:
        raise HTTPException(429, "trop d'enregistrements en cours")
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = LiveSession(sid, LIVE / sid, lat, lon, radius, t0, _dem)
    return {"session": sid, "sample_rate": 1000, "clock_offset_s": offset}


@app.post("/api/live/{sid}/audio")
async def live_audio(sid: str, request: Request, x_password: str | None = Header(None)):
    """Body: raw little-endian int16 mono PCM at 1000 Hz."""
    _check(x_password)
    s = _session(sid)
    s.add_audio(await request.body())
    return {"duration": s.duration, "state": s.state}


@app.post("/api/live/{sid}/stop")
def live_stop(sid: str, x_password: str | None = Header(None)):
    _check(x_password)
    s = _session(sid)
    s.finish()
    return {"state": s.state}


@app.get("/api/live/{sid}")
def live_status(sid: str, x_password: str | None = Header(None)):
    _check(x_password)
    return _clean(_session(sid).status())


@app.get("/api/live/{sid}/img/{name}")
def live_image(sid: str, name: str, pw: str = ""):
    _check(pw)
    path = LIVE / sid / name
    if not sid.isalnum() or name not in IMAGES or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})
