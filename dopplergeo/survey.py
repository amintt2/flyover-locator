"""Survey: which online videos carry aircraft Doppler curves at all?

Detection only (no ADS-B, no search). Each URL is downloaded (audio only), analysed,
logged as one JSON line, and deleted. A small spectrogram crop of every detected pass
is kept for checking by eye.

    python -m dopplergeo.survey URL [URL ...]            # any page yt-dlp can read
    python -m dopplergeo.survey --search podcast -n 50   # most viewed this week
    python -m dopplergeo.survey --file urls.txt

Results go to survey/results.jsonl (videos already there are skipped) and
survey/review/<id>/*.png.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio import load_audio
from .physics import straight_flyover
from .pipeline import SPEC, SR, WIDE, analyze_audio

# YouTube search filters: sort by view count, uploaded this week / this month / any time
SORTS = {"week": "CAMSAggD", "month": "CAMSAggE", "views": "CAM%3D", "relevance": ""}


def list_urls(target, n, sort="week"):
    """Expand a search query, playlist or channel to video URLs (a single video stays itself)."""
    if not target.startswith("http"):
        target = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(target) + (
            f"&sp={SORTS[sort]}" if SORTS[sort] else "")
    out = subprocess.run(["yt-dlp", "--flat-playlist", "--playlist-end", str(n), "-J", target],
                         capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "yt-dlp failed")
    info = json.loads(out.stdout)
    if info.get("_type") != "playlist":
        return [info.get("webpage_url") or target]
    return [e.get("url") or e.get("webpage_url") for e in info.get("entries") or [] if e]


def download(url, workdir, max_s):
    """Audio-only download; returns (path, metadata) or raises."""
    meta = subprocess.run(["yt-dlp", "--no-playlist", "-J", url], capture_output=True, text=True, timeout=120)
    if meta.returncode:
        raise RuntimeError((meta.stderr.strip().splitlines() or ["metadata failed"])[-1])
    m = json.loads(meta.stdout)
    info = {k: m.get(k) for k in ("id", "title", "channel", "duration", "view_count", "upload_date",
                                  "live_status", "categories", "extractor_key", "webpage_url")}
    info["tags"] = (m.get("tags") or [])[:10]
    if info["live_status"] in ("is_live", "is_upcoming"):
        raise RuntimeError("live stream")
    if info["duration"] and info["duration"] > max_s:
        raise RuntimeError(f"too long ({info['duration'] / 3600:.1f} h)")
    d = Path(tempfile.mkdtemp(dir=workdir))
    # best audio: low bitrates drop faint tones under speech
    r = subprocess.run(["yt-dlp", "--no-playlist", "-f", "ba/b", "-o", str(d / "a.%(ext)s"), url],
                       capture_output=True, text=True, timeout=1800)
    src = next((p for p in d.glob("a.*") if not p.name.endswith(".part")), None)
    if r.returncode or src is None:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError((r.stderr.strip().splitlines() or ["download failed"])[-1])
    info["codec"] = m.get("acodec"), m.get("abr"), m.get("asr")
    return src, info


def audio_stats(x, sr):
    """How 'raw' the soundtrack is: noise gates and suppressors leave stretches of near
    digital silence and a very low floor in the band where propellers sit."""
    blk = x[: len(x) // sr * sr].reshape(-1, sr)
    rms = 10 * np.log10(np.mean(blk ** 2, axis=1) + 1e-20)
    f = np.fft.rfftfreq(sr, 1 / sr)
    win = np.hanning(sr)
    spec = (np.abs(np.fft.rfft(blk * win, axis=1)) * 2 / win.sum()) ** 2  # per-bin power, dBFS scale
    low = 10 * np.log10(spec[:, (f >= 50) & (f <= 300)].mean(axis=1) + 1e-20)
    return {"loud_db": round(float(np.median(rms)), 1),
            "silent_share": round(float(np.mean(rms < -70)), 3),
            "floor_50_300_db": round(float(np.percentile(low, 10)), 1)}


def review_png(a, f, path, c=343.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = getattr(a.spec, "wide", a.spec) if f.get("band") == "wide" else getattr(a.spec, "low", a.spec)
    t0, t1 = f["tc"] - 50, f["tc"] + 50
    lo, hi = f["f0"] * (1 - 1.6 * f["v"] / c), f["f0"] * (1 + 1.6 * f["v"] / c)
    m = (spec.times >= t0) & (spec.times <= t1)
    fm = (spec.freqs >= lo) & (spec.freqs <= hi)
    if m.sum() < 2 or fm.sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=70)
    ax.imshow(spec.z[np.ix_(fm, m)], origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=4,
              extent=[spec.times[m][0], spec.times[m][-1], spec.freqs[fm][0], spec.freqs[fm][-1]])
    tt = np.linspace(*f["window"], 300)
    ax.plot(tt, straight_flyover(tt, f["f0"], f["v"], f["d"], f["tc"], c) * 1.02, "c--", lw=0.8)
    ax.set_title(f"{f['tc']:.0f}s f0 {f['f0']:.0f} Hz v {f['v']:.0f} m/s d {f['d']:.0f} m strength {f['strength']:.1f}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyse(src, info, review_dir):
    t = time.time()
    x, sr = load_audio(src, sr=SR)
    stats = audio_stats(x, sr)
    a = analyze_audio(src, None, log=lambda m: None, plots=False)
    keep = ("tc", "f0", "v", "d", "strength", "rms_hz")
    fits = [{k: round(float(f[k]), 1) for k in keep if k in f} | {"band": f.get("band") or "low"} for f in a.fits]
    if a.fits:
        review_dir.mkdir(parents=True, exist_ok=True)
        for f in a.fits:
            review_png(a, f, review_dir / f"{f['tc']:07.1f}.png")
    others = {}
    for o in a.others or []:
        others[o["label"]] = others.get(o["label"], 0) + 1
    return info | stats | {"fits": fits, "n_fits": len(fits), "others": others,
                           "hum_hz": a.enf.get("nominal"), "analysis_s": round(time.time() - t)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="video, playlist or channel URLs")
    ap.add_argument("--search", action="append", default=[], help="YouTube search query (repeatable)")
    ap.add_argument("--sort", choices=list(SORTS), default="week")
    ap.add_argument("--file", help="text file, one URL or 'search: query' per line")
    ap.add_argument("-n", type=int, default=20, help="videos per search/playlist")
    ap.add_argument("--out", default="survey")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--min-minutes", type=float, default=2.0)
    args = ap.parse_args(argv)

    targets = [(u, None) for u in args.urls] + [(q, q) for q in args.search]
    if args.file:
        for line in Path(args.file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                q = line[7:].strip() if line.startswith("search:") else None
                targets.append((q or line, q))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done |= {r.get("id"), r.get("url")}

    queue = []
    for target, query in targets:
        try:
            for u in list_urls(target, args.n, args.sort):
                queue.append((u, query))
        except Exception as e:  # noqa: BLE001 - keep surveying the other sources
            print(f"! {target}: {e}", flush=True)
    queue = [(u, q) for u, q in dict(queue).items() if u not in done and u.rsplit("=", 1)[-1] not in done]
    print(f"{len(queue)} videos to survey", flush=True)

    work = Path(tempfile.mkdtemp(prefix="survey_"))
    max_s = args.max_hours * 3600

    def fetch(item):
        try:
            return download(item[0], work, max_s), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    with ThreadPoolExecutor(1) as pool, res_path.open("a") as log:
        nxt = pool.submit(fetch, queue[0]) if queue else None
        for i, (url, query) in enumerate(queue):
            got, err = nxt.result()
            nxt = pool.submit(fetch, queue[i + 1]) if i + 1 < len(queue) else None  # download ahead
            row = {"url": url, "query": query, "ts": int(time.time())}
            if got:
                src, info = got
                try:
                    if (info.get("duration") or 1e9) < args.min_minutes * 60:
                        raise RuntimeError("too short")
                    row |= analyse(str(src), info, out / "review" / info["id"])
                except Exception as e:  # noqa: BLE001
                    row |= info | {"error": str(e)}
                finally:
                    shutil.rmtree(src.parent, ignore_errors=True)
            else:
                row["error"] = err
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            if "error" in row:
                print(f"[{i + 1}/{len(queue)}] {url}: {row['error']}", flush=True)
            else:
                print(f"[{i + 1}/{len(queue)}] {row['duration'] or 0:>5.0f}s {row['n_fits']} pass(es) "
                      f"silent {row['silent_share']:.0%} floor {row['floor_50_300_db']} dB  {row['title'][:60]}",
                      flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
