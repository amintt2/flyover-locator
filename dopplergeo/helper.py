"""Local helper for the Chrome extension: YouTube refuses the server's address, so the
PC fetches the audio (light) and the server does the analysis (heavy).

    python -m dopplergeo.helper            # listens on 127.0.0.1:8765

POST /probe {"url"}  short excerpts spread over the video -> server score
POST /send  {"url"}  whole soundtrack (audio only) -> server scan; progress in GET /jobs
Only YouTube video URLs, only requests from the extension.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVER = os.environ.get("FLYOVER_SERVER", "https://flyover.mciut.fr")
PASSWORD = os.environ.get("FLYOVER_PASSWORD", "")
YTDLP = [os.environ.get("YTDLP", shutil.which("yt-dlp") or "yt-dlp"), "--js-runtimes", "node", "--no-playlist"]
VIDEO = re.compile(r"^https://(www\.|m\.)?(youtube\.com/(watch\?v=|live/|shorts/)|youtu\.be/)[\w-]{11}")
EXCERPTS, EXCERPT_S = 4, 30
INFO_KEYS = ("id", "title", "channel", "channel_follower_count", "duration", "live_status", "upload_date",
             "webpage_url", "view_count")
jobs = {}


def metadata(url):
    r = subprocess.run([*YTDLP, "-J", url], capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError((r.stderr.strip().splitlines() or ["yt-dlp failed"])[-1])
    m = json.loads(r.stdout)
    return {k: m.get(k) for k in INFO_KEYS}


def post_files(path, files, fields):
    """multipart/form-data POST with the standard library."""
    b = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, p in files:
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{p.name}"\r\n'.encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n" + p.read_bytes() + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request(SERVER + path, data=bytes(body), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}",
                                          **({"X-Password": PASSWORD} if PASSWORD else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def probe(url):
    info = metadata(url)
    if info["live_status"] in ("is_live", "is_upcoming"):
        # a live in progress has no fixed length: score it from the title alone
        return post_files("/api/probe", [], {"info": json.dumps(info)}) | {"info": info}
    dur = info["duration"] or 0
    n = EXCERPTS if dur >= EXCERPTS * EXCERPT_S * 2 else 1
    starts = [dur * (k + 0.5) / n - EXCERPT_S / 2 for k in range(n)] if dur else [0]
    d = Path(tempfile.mkdtemp(prefix="flyover_"))
    try:
        cmd = [*YTDLP, "-f", "ba/b", "-o", str(d / "%(section_start)s.%(ext)s")]
        for a in starts:
            cmd += ["--download-sections", f"*{max(a, 0):.0f}-{max(a, 0) + EXCERPT_S:.0f}"]
        r = subprocess.run([*cmd, url], capture_output=True, text=True, timeout=300)
        got = sorted(p for p in d.iterdir() if not p.name.endswith(".part"))
        if not got:
            raise RuntimeError((r.stderr.strip().splitlines() or ["download failed"])[-1])
        return post_files("/api/probe", [("files", p) for p in got], {"info": json.dumps(info)}) | {"info": info}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def send(job, url):
    j = jobs[job]
    d = Path(tempfile.mkdtemp(prefix="flyover_"))
    try:
        j.update(stage="metadata")
        info = metadata(url)
        j.update(info=info, stage="download")
        p = subprocess.Popen([*YTDLP, "-f", "ba/b", "--newline", "-o", str(d / "a.%(ext)s"), url],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = ""
        for line in p.stdout:
            last = line.strip() or last
            m = re.search(r"\[download\]\s+([\d.]+)%", line)
            if m:
                j["progress"] = float(m.group(1))
        if p.wait():
            raise RuntimeError(last or "download failed")
        src = next(q for q in d.iterdir() if q.name.startswith("a.") and not q.name.endswith(".part"))
        j.update(stage="upload", mb=round(src.stat().st_size / 1e6, 1))
        r = post_files("/api/survey/file", [("file", src)], {"info": json.dumps(info)})
        j.update(stage="sent", server_id=r.get("id"), waiting=r.get("waiting", 0))
    except Exception as e:  # noqa: BLE001 - reported to the extension
        j.update(stage="error", error=str(e))
    finally:
        shutil.rmtree(d, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin") or "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _allowed(self):
        # web pages could reach this port too: only the extension may use it
        return (self.headers.get("Origin") or "").startswith("chrome-extension://")

    def do_OPTIONS(self):
        self._reply(204 if self._allowed() else 403, {})

    def do_GET(self):
        if not self._allowed():
            return self._reply(403, {"error": "forbidden"})
        if self.path == "/health":
            return self._reply(200, {"ok": True, "server": SERVER})
        if self.path == "/jobs":
            return self._reply(200, jobs)
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        if not self._allowed():
            return self._reply(403, {"error": "forbidden"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self._reply(400, {"error": "bad json"})
        url = str(body.get("url", ""))
        if not VIDEO.match(url):
            return self._reply(400, {"error": "pas une vidéo YouTube"})
        if self.path == "/probe":
            try:
                return self._reply(200, probe(url))
            except Exception as e:  # noqa: BLE001
                return self._reply(500, {"error": str(e)})
        if self.path == "/send":
            job = uuid.uuid4().hex[:8]
            jobs[job] = {"url": url, "stage": "queued", "progress": 0.0, "started": time.time()}
            threading.Thread(target=send, args=(job, url), daemon=True).start()
            return self._reply(200, {"job": job})
        self._reply(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        sys.stderr.write("helper: " + fmt % args + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    print(f"Flyover helper on http://127.0.0.1:{a.port} -> {SERVER}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
