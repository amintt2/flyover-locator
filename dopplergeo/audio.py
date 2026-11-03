"""Audio loading, band-limited high-resolution spectrograms, ridge tracking and
flyover curve fitting."""

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from .physics import straight_flyover


def load_audio(path, sr=2000):
    """Decode any audio/video file to mono float32 at `sr` Hz (via ffmpeg when available)."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.float32).copy(), sr
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != sr:
        g = np.gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
    return x, sr


def media_start_time(path):
    """Recording start (unix s) from container metadata, or None.

    Phones write the capture time (creation_time / com.apple.quicktime.creationdate),
    usually at the *end* or *start* of capture depending on the device, so callers
    should keep a tolerance of about the clip duration.
    """
    if not shutil.which("ffprobe"):
        return None
    import datetime as dt
    import json

    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        info = json.loads(subprocess.run(cmd, check=True, capture_output=True).stdout)
    except (subprocess.CalledProcessError, ValueError):
        return None
    tags = dict(info.get("format", {}).get("tags", {}))
    for s in info.get("streams", []):
        for k, v in s.get("tags", {}).items():
            tags.setdefault(k, v)
    for key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        v = next((tags[k] for k in tags if k.lower() == key), None)
        if not v:
            continue
        try:
            t = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=dt.timezone.utc)
            if t.year >= 2005:
                return t.timestamp()
        except ValueError:
            continue
    return None


@dataclass
class Spectrogram:
    times: np.ndarray  # frame centres, seconds from recording start
    freqs: np.ndarray  # Hz, only the requested band
    db: np.ndarray  # (F, T) power in dB
    z: np.ndarray  # (F, T) background-normalised, lightly smoothed

    @property
    def hop(self):
        return self.times[1] - self.times[0]

    @property
    def df(self):
        return self.freqs[1] - self.freqs[0]

    @property
    def duration(self):
        return float(self.times[-1] + self.hop / 2)

    def lookup(self, t, f):
        """Nearest-bin lookup of z at arrays of (time, freq); 0 outside the map."""
        it = np.rint((t - self.times[0]) / self.hop).astype(np.int64)
        jf = np.rint((f - self.freqs[0]) / self.df).astype(np.int64)
        ok = (it >= 0) & (it < self.z.shape[1]) & (jf >= 0) & (jf < self.z.shape[0])
        out = np.zeros(np.shape(t), np.float32)
        out[ok] = self.z[jf[ok], it[ok]]
        return out


def spectrogram(x, sr, fmin, fmax, win_s=1.0, hop_s=0.1, pad=4, smooth_hz=0.5, smooth_s=0.2):
    """STFT restricted to [fmin, fmax], computed in chunks so long recordings fit in memory.

    z is each frequency row minus its median (removes stationary hum), divided by a
    robust global noise scale, so a faint Doppler line stands out as positive values.
    """
    n = int(round(win_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    nfft = 1 << int(np.ceil(np.log2(n * pad)))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    win = np.hanning(n).astype(np.float32)
    n_frames = 1 + (len(x) - n) // hop
    if n_frames < 2:
        raise ValueError("recording shorter than one analysis window")
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop][:n_frames]
    db = np.empty((band.sum(), n_frames), np.float32)
    for s in range(0, n_frames, 256):
        spec = np.fft.rfft(frames[s : s + 256] * win, nfft)[:, band]
        db[:, s : s + 256] = (10 * np.log10(np.abs(spec) ** 2 + 1e-20)).T
    times = (np.arange(n_frames) * hop + n / 2) / sr
    freqs = freqs[band]

    z = db - np.median(db, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(z)) + 1e-9
    z = z / scale
    sig = (smooth_hz / (freqs[1] - freqs[0]), smooth_s / hop_s)
    z = gaussian_filter(z, sig).astype(np.float32)
    return Spectrogram(times, freqs, db, z)


def track_ridge(spec, t0, t1, fmin, fmax, max_jump_hz=1.5, jump_cost=0.5, min_z=0.5):
    """Follow the brightest continuous line in a time/frequency box (Viterbi).

    Returns (t, f, weight) for frames where the ridge is above `min_z`.
    """
    ti = np.where((spec.times >= t0) & (spec.times <= t1))[0]
    fi = np.where((spec.freqs >= fmin) & (spec.freqs <= fmax))[0]
    z = spec.z[np.ix_(fi, ti)]
    F, T = z.shape
    J = max(1, int(round(max_jump_hz / spec.df)))
    score = z[:, 0].copy()
    back = np.zeros((F, T), np.int32)
    offs = np.arange(-J, J + 1)
    pen = jump_cost * (offs / J) ** 2
    for k in range(1, T):
        cand = np.full((len(offs), F), -np.inf, np.float32)
        for a, o in enumerate(offs):
            src = np.arange(F) - o
            ok = (src >= 0) & (src < F)
            cand[a, ok] = score[src[ok]] - pen[a]
        best = cand.argmax(axis=0)
        back[:, k] = np.arange(F) - offs[best]
        score = cand[best, np.arange(F)] + z[:, k]
    path = np.empty(T, np.int64)
    path[-1] = score.argmax()
    for k in range(T - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    w = z[path, np.arange(T)]
    # parabolic sub-bin refinement
    p = np.clip(path, 1, F - 2)
    a, b, c = z[p - 1, np.arange(T)], z[p, np.arange(T)], z[p + 1, np.arange(T)]
    den = a - 2 * b + c
    delta = np.where(den < 0, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    f = spec.freqs[fi][path] + np.clip(delta, -0.5, 0.5) * spec.df
    keep = w >= min_z
    return spec.times[ti][keep], f[keep], w[keep]


def fit_flyover(t, f, w=None, c=343.0):
    """Fit the straight-flight Doppler model to ridge points.

    Returns dict with f0, v (m/s), d (slant CPA distance, m), tc (heard CPA time, s)
    and 1-sigma uncertainties.
    """
    t, f = np.asarray(t, float), np.asarray(f, float)
    w = np.ones_like(t) if w is None else np.clip(np.asarray(w, float), 1e-3, None)
    hi, lo = np.percentile(f, 95), np.percentile(f, 5)
    f0 = 2 * hi * lo / (hi + lo)
    v = c * (hi - lo) / (hi + lo)
    order = np.argsort(t)
    tc = float(np.interp(f0, f[order][::-1], t[order][::-1]))
    slope = np.gradient(np.convolve(f[order], np.ones(9) / 9, "same"), t[order])
    d = f0 * v * v / (c * max(np.abs(slope[5:-5]).max() if len(slope) > 10 else 1.0, 1e-3))
    x0 = [f0, max(v, 15.0), float(np.clip(d, 200, 15000)), tc]
    sw = np.sqrt(w)

    def resid(p):
        return (straight_flyover(t, p[0], p[1], p[2], p[3], c) - f) * sw

    lb = [0.5 * lo, 5.0, 30.0, t.min() - 30]
    ub = [2 * hi, 0.8 * c, 30000.0, t.max() + 30]
    x0 = np.clip(x0, np.add(lb, 1e-6), np.subtract(ub, 1e-6))
    best = None
    for dscale in (0.5, 1.0, 2.0):
        start = x0.copy()
        start[2] = np.clip(start[2] * dscale, lb[2] + 1, ub[2] - 1)
        r = least_squares(resid, start, bounds=(lb, ub), loss="soft_l1", f_scale=1.0)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    dof = max(1, len(t) - 4)
    s2 = 2 * best.cost / dof
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * s2
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(4, np.nan)
    res = straight_flyover(t, *p, c) - f
    inl = np.abs(res) < 3.0
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("inf")
    return {
        "f0": float(p[0]), "v": float(p[1]), "d": float(p[2]), "tc": float(p[3]),
        "sigma": {"f0": float(sig[0]), "v": float(sig[1]), "d": float(sig[2]), "tc": float(sig[3])},
        "rms_hz": rms, "inliers": inl, "n_points": int(len(t)),
    }


def event_from_fit(fit, window, harmonics=(1,), v_lo=0.65, v_hi=1.6, d_lo=0.4, d_hi=3.0, t_tol=4.0):
    """Turn a fit into a generous search event. With a partly visible curve, speed
    and distance trade off against each other, and turns or climbs bias the
    straight-flight fit, so ranges are deliberately wide; the time offsets,
    silence and fine curve scoring do the real discrimination."""
    s = fit["sigma"]
    return {
        "t_cpa": round(fit["tc"], 2),
        "t_tol": round(max(t_tol, 3 * float(np.nan_to_num(s["tc"], nan=t_tol))), 2),
        "v": [round(max(5.0, fit["v"] * v_lo), 1), round(fit["v"] * v_hi, 1)],
        "d": [round(max(50.0, fit["d"] * d_lo)), round(fit["d"] * d_hi)],
        "f0": round(fit["f0"], 2),
        "harmonics": list(harmonics),
        "window": [round(window[0], 1), round(window[1], 1)],
    }


def plot_spectrogram(spec, path, t0=None, t1=None, fmin=None, fmax=None, curves=(), title=None):
    """Save a PNG of the normalised spectrogram, optionally with overlaid (t, f) curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = spec.times[0] if t0 is None else t0
    t1 = spec.times[-1] if t1 is None else t1
    fmin = spec.freqs[0] if fmin is None else fmin
    fmax = spec.freqs[-1] if fmax is None else fmax
    ti = (spec.times >= t0) & (spec.times <= t1)
    fi = (spec.freqs >= fmin) & (spec.freqs <= fmax)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(spec.z[np.ix_(fi, ti)], origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=6,
              extent=[spec.times[ti][0], spec.times[ti][-1], spec.freqs[fi][0], spec.freqs[fi][-1]])
    for ct, cf, label in curves:
        ax.plot(ct, cf, lw=1.2, alpha=0.8, label=label)
    if curves:
        ax.legend(loc="upper right")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
