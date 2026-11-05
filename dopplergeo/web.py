"""Web app: drop a video (or record live), get a map.

Run: uvicorn dopplergeo.web:app --host 0.0.0.0 --port 8000
Env: DATA_DIR (default ./data), APP_PASSWORD (optional shared password),
     MAX_UPLOAD_MB (default 4000), MAX_RADIUS_KM (default 100),
     MAX_DAYS (default 31, find-the-day range), DAY_WORKERS (days searched at once),
     ADSB_FILE / DEM_NPZ (offline testing).
"""

import asyncio
import datetime as dt
import json
import math
import os
import secrets
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .audio import detect_cuts, media_start_time
from .dem import load_dem
from .daysearch import day_events, search_days
from .enf import DEFAULT_AREAS
from .fetch import fetch_adsb
from .index import build_day, day_path
from .geo import M_PER_DEG_LAT, m_per_deg_lon
from .pipeline import analyze_audio, locate
from .radar import Radar
from .search import SearchParams
from .session import LiveSession
from . import probe, survey

DATA = Path(os.environ.get("DATA_DIR", "data")).resolve()
JOBS = DATA / "jobs"
LIVE = DATA / "live"
INDEX = DATA / "adsb_index"
PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "4000")) * 1024 * 1024
MAX_RADIUS_KM = float(os.environ.get("MAX_RADIUS_KM", "100"))
MAX_DAYS = int(os.environ.get("MAX_DAYS", "31"))
MAX_LIVE_RADIUS_KM = 450.0  # adsb.lol live point queries go up to 250 nm
JOBS.mkdir(parents=True, exist_ok=True)
LIVE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Flyover Locator")
# the YouTube extension sends excerpts and soundtracks straight from the video page
app.add_middleware(CORSMiddleware, allow_origins=["https://www.youtube.com"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])
EXTENSION = Path(os.environ.get("EXTENSION_DIR", Path(__file__).resolve().parent.parent / "extension"))


@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError):
    fields = ", ".join(".".join(str(x) for x in e.get("loc", [])[1:]) + f" ({e.get('type')})" for e in exc.errors())
    print(f"422 on {request.url.path}: {fields}", flush=True)
    return JSONResponse({"detail": f"Requête refusée par le serveur : {fields}"}, status_code=422)


def _num(v, default=None):
    """Lenient form number: browsers send '', 'undefined', 'null' or '48,85'."""
    try:
        x = float(str(v).strip().replace(",", "."))
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default
pool = ThreadPoolExecutor(max_workers=1)  # one heavy search at a time
lock = threading.Lock()
sessions: dict[str, LiveSession] = {}
IMAGES = {"flyovers.png", "match_1.png", "match_2.png", "match_3.png"}


def _check(pw):
    if PASSWORD and not secrets.compare_digest(pw or "", PASSWORD):
        raise HTTPException(401, "wrong password")


def _dem_cfg():
    if os.environ.get("DEM_NPZ"):
        return {"type": "npz", "path": os.environ["DEM_NPZ"]}
    return {"type": "hgt", "dir": str(DATA / "dem"), "download": True}


def _dem():
    return load_dem(_dem_cfg())


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


def _find_day(job, a, info, zone, days, log):
    """No time at all, only a range of days: search every day, rank them."""
    events = day_events(a.fits)
    if len(events) < 2:
        raise RuntimeError("Pour retrouver le jour il faut au moins 2 passages d'avion bien nets dans le son "
                           "(un seul passage colle à trop d'endroits et de jours).")
    area = _circle_area(*zone) if zone else dict(WORLD)
    _status(job, stage="adsb", days={"done": 0, "total": len(days)})
    usable = []
    for i, d in enumerate(days):
        if day_path(INDEX, d).exists():
            usable.append(d)
            continue
        log(f"{d}: not indexed yet, streaming it once (~3 GB, ~10 min)")
        try:
            build_day(d, INDEX, log=log)
            usable.append(d)
        except RuntimeError as e:  # not published yet
            log(f"{d}: skipped ({e})")
        _status(job, days={"done": 0, "total": len(days), "indexed": i + 1})
    if not usable:
        raise RuntimeError("Aucun jour de données d'avions disponible sur cette période.")
    _status(job, stage="search")

    def on_day(done, total, r):
        _status(job, days={"done": done, "total": total, "last": r["day"]})

    res, ranked = search_days(a.spec, events, usable, INDEX, area, _dem_cfg(), JOBS / job, log=log, on_day=on_day,
                              clips=a.clips)
    if res is None and len(a.events) >= 2 and len(a.events) < len(events):
        log("no day matches every flyover; retrying with the strongest ones only")
        res, ranked = search_days(a.spec, a.events, usable, INDEX, area, _dem_cfg(), JOBS / job, log=log,
                                  on_day=on_day, clips=a.clips)
    if res is None:
        _status(job, state="done", stage="done", result={
            "top": [], "heat": [], "events": [], "days": ranked[:10],
            "message": f"Aucun jour sur {len(usable)} ne colle aux avions entendus."})
        return
    res["days"] = ranked[:10]
    res["area"] = {"lat": zone[0], "lon": zone[1], "radius_km": zone[2]} if zone else {
        "bbox": [min(t["lat"] for t in res["top"]), min(t["lon"] for t in res["top"]),
                 max(t["lat"] for t in res["top"]), max(t["lon"] for t in res["top"])]}
    res["enf"] = info["enf"]
    res.setdefault("heat", [])
    _status(job, state="done", stage="done", result=res)


def _work(job, media, zone, start_unix, start_tol_s, use_meta, region, mode="locate", days=None, cuts=False):
    log = _log(job)
    try:
        _status(job, state="running", stage="audio")
        meta = media_start_time(media)
        if meta and use_meta:
            log(f"time from video metadata: {dt.datetime.fromtimestamp(meta, dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
            start_unix, start_tol_s = meta, 180.0
        cut_times = None
        if cuts:
            log("looking for cuts in the video")
            cut_times = detect_cuts(media)
            log(f"  {len(cut_times)} cut(s) found" if cut_times else "  no cut found: treated as one continuous recording")
        a = analyze_audio(media, JOBS / job, log=log, cuts=cut_times)
        info = {"fits": [{k: v for k, v in f.items() if k != "sigma"} for f in a.fits],
                "others": [{k: f[k] for k in ("tc", "f0", "v", "d", "kind", "label", "strength")} for f in a.others or []],
                "enf": {k: a.enf[k] for k in ("nominal", "strength", "regions", "quality")},
                "duration": a.spec.duration, "clips": len(a.clips) if a.clips else 1}
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
        if days and not (meta and use_meta):  # the video's own timestamp beats a range of days
            return _find_day(job, a, info, zone, days, log)
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
            adsb = fetch_adsb(bbox, t_range, INDEX, log=log, max_alt_m=6000.0)
        if adsb.empty:
            raise RuntimeError("Aucune donnée d'avion dans cette zone à cette heure.")
        _status(job, stage="search")
        p = SearchParams(start_window=(start_unix - start_tol_s, start_unix + start_tol_s), tile_deg=0.5,
                         clips=a.clips)
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


@app.get("/flyover-extension.zip")
def extension_zip():
    """The Chrome extension, to unzip and load in chrome://extensions (developer mode)."""
    import io
    import zipfile

    if not EXTENSION.is_dir():
        raise HTTPException(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(EXTENSION.iterdir()):
            if f.is_file():
                z.write(f, f"flyover-extension/{f.name}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="flyover-extension.zip"'})


@app.get("/api/config")
def config():
    return {"password": bool(PASSWORD), "max_radius_km": MAX_RADIUS_KM, "max_days": MAX_DAYS, "max_live_radius_km": MAX_LIVE_RADIUS_KM,
            "max_upload_mb": MAX_UPLOAD // (1024 * 1024),
            "regions": {k: v["name"] for k, v in {"europe": DEFAULT_AREAS[50], "usa": DEFAULT_AREAS[60]}.items()}}


RADAR = Radar()


@app.get("/api/aircraft")
def aircraft(south: float, west: float, north: float, east: float, zoom: float = 8.0,
             x_password: str | None = Header(None)):
    """Aircraft flying in a map view now. Zoomed out (a country), only propellers:
    those are the ones a recording can hear."""
    _check(x_password)
    ac, stale = RADAR.view(south, west, north, east, props_only=zoom < 7.5, detail=zoom >= 7.5)
    return {"aircraft": ac, "props_only": zoom < 7.5, "stale": stale, "t": time.time()}


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(None), lat: str = Form(""), lon: str = Form(""),
                     radius_km: str = Form("30"), start: str = Form(""), start_tol_min: str = Form("30"),
                     use_metadata: str = Form("true"), region: str = Form("auto"), mode: str = Form("locate"),
                     day_from: str = Form(""), day_to: str = Form(""), cuts: str = Form(""), x_password: str | None = Header(None)):
    _check(x_password)
    if file is None or not hasattr(file, "read"):
        raise HTTPException(400, "Le fichier n'est pas arrivé : réessaie (ou choisis-le depuis Fichiers plutôt que Photos).")
    lat, lon = _num(lat), _num(lon)
    radius_km, start_tol_min = _num(radius_km, 30.0), _num(start_tol_min, 30.0)
    use_metadata = str(use_metadata).strip().lower() not in ("false", "0", "no", "off")
    zone = None
    if lat is not None and lon is not None:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(400, "bad coordinates")
        zone = (lat, lon, min(max(radius_km, 1.0), MAX_RADIUS_KM))
    days = None
    if day_from and day_to:
        d0, d1 = dt.date.fromisoformat(day_from), dt.date.fromisoformat(day_to)
        if d1 < d0:
            d0, d1 = d1, d0
        d1 = min(d1, dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1))  # published the next day
        if (d1 - d0).days + 1 > MAX_DAYS:
            raise HTTPException(400, f"période trop longue (max {MAX_DAYS} jours)")
        if d1 < d0:
            raise HTTPException(400, "les données d'avions d'aujourd'hui sortent demain")
        days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]
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
                raise HTTPException(413, f"Fichier trop gros (max {MAX_UPLOAD // (1024 * 1024)} Mo).")
            fh.write(chunk)
    start_unix = None
    if start:
        try:
            t = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, f"date/heure illisible : {start!r}")
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        start_unix = t.timestamp()
    _status(job, state="queued", stage="queued", mode="detect" if mode == "detect" else "locate",
            log=[f"uploaded {size / 1e6:.1f} MB"])
    pool.submit(_work, job, media, zone, start_unix, start_tol_min * 60, use_metadata, region,
                "detect" if mode == "detect" else "locate", days,
                str(cuts).strip().lower() in ("1", "true", "on", "auto"))
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


# ------------------------------------------------------------------ video survey

SURVEY = DATA / "survey"
_survey_lock = threading.Lock()
_survey_queue = set()  # uploads waiting for the scan in progress


@app.post("/api/survey")
def survey_start(body: dict, x_password: str | None = Header(None)):
    """Download and scan a few video URLs on the server (one run at a time)."""
    _check(x_password)
    urls = [u for u in body.get("urls", []) if isinstance(u, str) and u.startswith("https://")][:10]
    if not urls:
        raise HTTPException(400, "urls")
    if not _survey_lock.acquire(blocking=False):
        raise HTTPException(409, "survey already running")

    def run():
        try:
            survey.main([*urls, "--out", str(SURVEY), "--max-hours", str(body.get("max_hours", 3))])
        except Exception:
            traceback.print_exc()
        finally:
            _survey_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": urls}


@app.post("/api/survey/file")
async def survey_file(file: UploadFile = File(...), info: str = Form("{}"), x_password: str | None = Header(None)):
    """Scan a soundtrack downloaded elsewhere (YouTube refuses datacenter addresses)."""
    _check(x_password)
    meta = json.loads(info)
    vid = str(meta.get("id") or uuid.uuid4().hex[:11])
    if not vid.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "id")
    d = SURVEY / "uploads" / vid
    d.mkdir(parents=True, exist_ok=True)
    src = d / ("a" + Path(file.filename or "a.webm").suffix)
    with src.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)

    def run():
        _survey_lock.acquire()
        _survey_queue.discard(vid)
        row = {"url": meta.get("webpage_url"), "query": "upload", "ts": int(time.time())}
        try:
            row |= survey.analyse(str(src), meta | {"id": vid}, SURVEY / "review" / vid)
        except Exception as e:
            traceback.print_exc()
            row |= meta | {"error": str(e)}
        finally:
            shutil.rmtree(d, ignore_errors=True)
            with (SURVEY / "results.jsonl").open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            _survey_lock.release()

    # one scan at a time: later uploads wait their turn instead of being refused
    _survey_queue.add(vid)
    threading.Thread(target=run, daemon=True).start()
    return {"id": vid, "waiting": len(_survey_queue) - 1}


@app.post("/api/probe")
async def probe_video(files: list[UploadFile] | None = File(None), info: str = Form("{}"), x_password: str | None = Header(None)):
    """Score a video from a few short excerpts before it is downloaded in full."""
    _check(x_password)
    meta = json.loads(info)
    d = Path(tempfile.mkdtemp(dir=DATA))
    try:
        paths = []
        for i, f in enumerate((files or [])[:8]):
            p = d / f"{i}{Path(f.filename or 'a.webm').suffix[:8]}"
            p.write_bytes(await f.read(20 << 20))
            paths.append(p)

        def run():
            ex = []
            for p in paths:
                try:
                    ex.append(probe.excerpt_stats(p))
                except Exception:  # noqa: BLE001 - a bad excerpt should not sink the others
                    traceback.print_exc()
            return ex

        ex = await asyncio.to_thread(run)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    s, why = probe.score(ex, meta)
    return {"score": s, "reasons": why, "excerpts": ex}


@app.get("/api/survey")
def survey_results(x_password: str | None = Header(None)):
    _check(x_password)
    path = SURVEY / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return {"running": _survey_lock.locked() or bool(_survey_queue), "waiting": sorted(_survey_queue), "results": rows}


@app.get("/api/survey/img/{vid}/{name}")
def survey_image(vid: str, name: str, pw: str = ""):
    _check(pw)
    path = SURVEY / "review" / vid / name
    if not vid.replace("-", "").replace("_", "").isalnum() or not name.endswith(".png") or "/" in name or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/traces")
def traces_start(body: dict, x_password: str | None = Header(None)):
    """Every aircraft (jets too, unlike the index) in a small box for one day, streamed
    from the adsb.lol archive on the server; fetch the parquet when it is ready."""
    _check(x_password)
    from .adsb import stream_readsb_archive
    from .fetch import _asset_urls

    day = dt.date.fromisoformat(body["day"])
    bbox = tuple(float(x) for x in body["bbox"])  # lat_min, lat_max, lon_min, lon_max
    if bbox[1] - bbox[0] > 1 or bbox[3] - bbox[2] > 1:
        raise HTTPException(400, "box too large")
    key = f"{day}_" + "_".join(f"{x:.3f}" for x in bbox)
    out = DATA / "traces" / f"{key}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() and not out.with_suffix(".running").exists():
        out.with_suffix(".running").touch()

        def run():
            try:
                stream_readsb_archive(_asset_urls(day), out, bbox, log=lambda m: print("traces", m, flush=True))
            except Exception:
                traceback.print_exc()
            finally:
                out.with_suffix(".running").unlink(missing_ok=True)

        threading.Thread(target=run, daemon=True).start()
    return {"key": key}


@app.get("/api/traces/{key}")
def traces_get(key: str, x_password: str | None = Header(None)):
    _check(x_password)
    out = DATA / "traces" / f"{key}.parquet"
    if "/" in key or ".." in key:
        raise HTTPException(404)
    if out.exists():
        return FileResponse(out, media_type="application/octet-stream")
    return JSONResponse({"running": out.with_suffix(".running").exists()}, status_code=202)


# ------------------------------------------------------------------ live recording


def _session(sid):
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s


@app.post("/api/live")
async def live_start(request: Request, x_password: str | None = Header(None)):
    """Body JSON: {lat, lon, radius_km, client_now_ms, audio_start_ms, sample_rate (1000 or 8000)}.
    8000 Hz keeps the whine of jets (up to 4 kHz); 1000 Hz only propeller tones."""
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
    sr = 8000 if int(body.get("sample_rate", 1000)) == 8000 else 1000
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = LiveSession(sid, LIVE / sid, lat, lon, radius, t0, _dem, sr=sr)
    return {"session": sid, "sample_rate": sr, "clock_offset_s": offset}


@app.post("/api/live/{sid}/audio")
async def live_audio(sid: str, request: Request, x_password: str | None = Header(None)):
    """Body: raw little-endian int16 mono PCM at the session's sample rate."""
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
