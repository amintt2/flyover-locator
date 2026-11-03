"""Two-stage search.

Coarse stage (per 1-degree tile, 200 m grid): keep listener positions that hear
an aircraft matching event 1 (speed + slant CPA distance), then event 2 at the
measured time offset, ..., and hear nothing else nearby during the silences.

Fine stage (100 m grid around survivors): model the exact Doppler curve each
listener would have heard from the real trajectory, and score it against the
spectrogram with a locked time offset between events.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .adsb import speed_runs
from .geo import M_PER_DEG_LAT, ecef, latlon_grid, m_per_deg_lon, offset_latlon
from .physics import heard_curves


@dataclass
class Event:
    t_cpa: float  # heard CPA time in the recording (s)
    v: tuple  # speed range (m/s)
    d: tuple  # slant CPA distance range (m)
    f0: float  # rest frequency of the tracked line (Hz)
    window: tuple  # recording span used for scoring (s)
    t_tol: float = 4.0
    harmonics: list = field(default_factory=lambda: [1])

    @classmethod
    def from_dict(cls, d):
        return cls(d["t_cpa"], tuple(d["v"]), tuple(d["d"]), d["f0"], tuple(d["window"]),
                   d.get("t_tol", 4.0), list(d.get("harmonics", [1])))


@dataclass
class SearchParams:
    c: float = 343.0
    coarse_m: float = 200.0
    fine_m: float = 100.0
    tile_deg: float = 1.0
    hear_radius_m: float = 10000.0
    silence_radius_m: float | None = None  # default: smallest event d_min
    silence_margin_s: float = 60.0
    silence_intervals: list | None = None  # recording-time [a, b]; None = auto
    start_window: tuple | None = None  # (unix_min, unix_max) of recording start
    obs_height_m: float = 1.5
    geoid_offset_m: float = 0.0
    sweep_t: float = 4.0
    step_t: float = 0.5
    sweep_f: float = 0.03
    step_f: float = 0.005
    tau_step: float = 0.5
    refine_top: int = 60
    max_candidates: int = 200000


# ---------------------------------------------------------------- geometry


def find_passes(tree, obs_xyz, t, xyz, r_max, chunk=240, max_cells=2_000_000):
    """All closest approaches (local minima of range) of a trajectory to listeners.

    Returns arrays (obs_idx, t_cpa, r_min, sample_idx). Exact slant ranges in ECEF.
    """
    N = len(t)
    out = ([], [], [], [])
    if N < 3:
        return tuple(np.array(a) for a in out)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    for s in range(1, N - 1, chunk):
        e = min(s + chunk, N - 1)
        lo, hi = s - 1, e + 1
        seg = xyz[lo:hi]
        max_step = steps[lo : hi - 1].max() if hi - 1 > lo else 0.0
        q = max(1, int(r_max / 4 / max(max_step, 1.0)))
        probes = np.vstack([seg[::q], seg[-1:]])
        lists = tree.query_ball_point(probes, r_max + q * max_step + 1.0)
        cand = np.unique(np.fromiter((i for l in lists for i in l), np.int64))
        if cand.size == 0:
            continue
        bs = max(1, max_cells // len(seg))
        for b in range(0, cand.size, bs):
            blk = cand[b : b + bs]
            d = cdist(obs_xyz[blk], seg)
            mid = d[:, 1:-1]
            is_min = (mid < d[:, :-2]) & (mid <= d[:, 2:]) & (mid <= r_max)
            mi, ni = np.nonzero(is_min)
            if mi.size == 0:
                continue
            ni = ni + 1
            r0, r1, r2 = d[mi, ni - 1], d[mi, ni], d[mi, ni + 1]
            den = r0 - 2 * r1 + r2
            delta = np.clip(np.where(den > 0, 0.5 * (r0 - r2) / np.where(den > 0, den, 1), 0.0), -0.5, 0.5)
            gi = lo + ni
            out[0].append(blk[mi])
            out[1].append(t[gi] + delta * (t[gi + 1] - t[gi - 1]) / 2)
            out[2].append(r1 - 0.25 * (r0 - r2) * delta)
            out[3].append(gi)
    if not out[0]:
        return tuple(np.array([], dtype=float) for _ in range(4))
    return tuple(np.concatenate(a) for a in out)


def _passes_table(tree, obs_xyz, segs, r_max, c, r_min=0.0, v=None):
    rows = []
    for s in segs:
        oi, tc, rm, gi = find_passes(tree, obs_xyz, s.t, s.xyz, r_max)
        if len(oi) == 0:
            continue
        keep = rm >= r_min
        sp = s.speed[gi.astype(int)]
        if v is not None:
            keep &= (sp >= v[0]) & (sp <= v[1])
        if keep.any():
            rows.append(pd.DataFrame({"obs": oi[keep].astype(np.int64), "sid": s.sid,
                                      "T": tc[keep] + rm[keep] / c, "r": rm[keep], "v": sp[keep]}))
    if not rows:
        return pd.DataFrame(columns=["obs", "sid", "T", "r", "v"])
    return pd.concat(rows, ignore_index=True)


def _crop(segs, lat0, lat1, lon0, lon1, t0=-np.inf, t1=np.inf):
    out = []
    for s in segs:
        a, b, c_, d = s.bbox()
        if b < lat0 or a > lat1 or d < lon0 or c_ > lon1 or s.t[-1] < t0 or s.t[0] > t1:
            continue
        m = (s.lat >= lat0) & (s.lat <= lat1) & (s.lon >= lon0) & (s.lon <= lon1) & (s.t >= t0) & (s.t <= t1)
        out += s.sub(m)
    return out


def silence_intervals(events, duration, p):
    if p.silence_intervals is not None:
        return [tuple(x) for x in p.silence_intervals]
    busy = sorted((e.t_cpa - p.silence_margin_s, e.t_cpa + p.silence_margin_s) for e in events)
    out, cur = [], 0.0
    for a, b in busy:
        if a - cur >= 10:
            out.append((cur, a))
        cur = max(cur, b)
    if duration - cur >= 10:
        out.append((cur, duration))
    return out


# ---------------------------------------------------------------- coarse stage


def coarse_search(segments, events, area, dem, duration, p, log=print):
    """Return a DataFrame of surviving coarse grid points with matched segment ids."""
    e1 = events[0]
    sil = silence_intervals(events, duration, p)
    # a plane passing closer than the ones we did hear would have been heard too
    r_sil = p.silence_radius_m or min(np.sqrt(e.d[0] * e.d[1]) for e in events)
    t_lo, t_hi = -np.inf, np.inf
    if p.start_window:
        t_lo, t_hi = p.start_window[0] - 120, p.start_window[1] + duration + 120
    results = []
    lat_tiles = np.arange(area["lat_min"], area["lat_max"], p.tile_deg)
    lon_tiles = np.arange(area["lon_min"], area["lon_max"], p.tile_deg)
    for la0 in lat_tiles:
        la1 = min(la0 + p.tile_deg, area["lat_max"])
        buf_lat = p.hear_radius_m / M_PER_DEG_LAT
        buf_lon = p.hear_radius_m / m_per_deg_lon(max(abs(la0), abs(la1)))
        for lo0 in lon_tiles:
            lo1 = min(lo0 + p.tile_deg, area["lon_max"])
            tile = _crop(segments, la0 - buf_lat, la1 + buf_lat, lo0 - buf_lon, lo1 + buf_lon, t_lo, t_hi)
            ev_segs = [[r for s in tile if s.prop for r in speed_runs(s, *e.v)] for e in events]
            if not all(ev_segs):
                continue
            glat, glon = latlon_grid(la0, la1, lo0, lo1, p.coarse_m)
            gh = dem(glat, glon) + p.geoid_offset_m + p.obs_height_m
            gxyz = ecef(glat, glon, gh)
            tree = cKDTree(gxyz)

            P = []
            for k, (e, segs_k) in enumerate(zip(events, ev_segs)):
                tab = _passes_table(tree, gxyz, segs_k, e.d[1], p.c, e.d[0], e.v)
                if p.start_window:
                    tab = tab[(tab["T"] >= p.start_window[0] + e.t_cpa - e.t_tol)
                              & (tab["T"] <= p.start_window[1] + e.t_cpa + e.t_tol)]
                P.append(tab)
                if tab.empty:
                    break
            if any(t.empty for t in P) or len(P) < len(events):
                continue

            cand = P[0].rename(columns={"sid": "sid1", "T": "T1", "r": "r1", "v": "v1"})
            for k in range(1, len(events)):
                e = events[k]
                nxt = P[k].rename(columns={"sid": f"sid{k+1}", "T": f"T{k+1}", "r": f"r{k+1}", "v": f"v{k+1}"})
                cand = cand.merge(nxt, on="obs")
                dt = cand[f"T{k+1}"] - cand["T1"] - (e.t_cpa - e1.t_cpa)
                ok = dt.abs() <= (e1.t_tol + e.t_tol)
                for j in range(1, k + 1):
                    ok &= ~((cand[f"sid{k+1}"] == cand[f"sid{j}"]) & ((cand[f"T{k+1}"] - cand[f"T{j}"]).abs() < 30))
                cand = cand[ok]
                if len(cand) > p.max_candidates:
                    log(f"  tile {la0:.1f},{lo0:.1f}: {len(cand)} candidates, truncating")
                    cand = cand.iloc[: p.max_candidates]
                if cand.empty:
                    break
            if cand.empty:
                continue
            n_time = len(cand)

            if sil:
                obs_u = np.unique(cand["obs"].to_numpy())
                sub_tree = cKDTree(gxyz[obs_u])
                origin = cand["T1"] - e1.t_cpa
                lo_t, hi_t = origin.min() + min(a for a, _ in sil) - 60, origin.max() + max(b for _, b in sil) + 60
                quiet = _passes_table(sub_tree, gxyz[obs_u], _crop(tile, -90, 90, -180, 180, lo_t - 60, hi_t),
                                      r_sil, p.c)
                if not quiet.empty:
                    quiet["obs"] = obs_u[quiet["obs"].to_numpy()]
                    c2 = cand.reset_index(drop=True)
                    c2["row"] = np.arange(len(c2))
                    m = c2.merge(quiet.rename(columns={"sid": "qsid", "T": "qT"})[["obs", "qsid", "qT"]], on="obs")
                    rel = m["qT"] - (m["T1"] - e1.t_cpa)
                    loud = np.zeros(len(m), bool)
                    for a, b in sil:
                        loud |= (rel >= a + e1.t_tol) & (rel <= b - e1.t_tol)
                    for k in range(1, len(events) + 1):  # the matched passes themselves
                        loud &= ~((m["qsid"] == m[f"sid{k}"]) & ((m["qT"] - m[f"T{k}"]).abs() < 30))
                    bad = np.unique(m.loc[loud, "row"].to_numpy())
                    cand = c2.drop(index=bad).drop(columns="row")
            log(f"  tile {la0:.1f},{lo0:.1f}: {n_time} pass time-offset filter, {len(cand)} pass silence filter")
            if cand.empty:
                continue
            cand = cand.copy()
            cand["lat"], cand["lon"] = glat[cand["obs"].to_numpy()], glon[cand["obs"].to_numpy()]
            results.append(cand.drop(columns="obs"))
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------- fine stage


class Trajectory:
    def __init__(self, seg):
        self.t0, self.t1 = seg.t[0], seg.t[-1]
        self.spline = CubicSpline(seg.t, seg.xyz, axis=0)

    def sample(self, a, b, step):
        a, b = max(a, self.t0), min(b, self.t1)
        if b - a < 4 * step:
            return None, None
        tau = np.arange(a, b, step)
        return tau, self.spline(tau)


def score_points(lat, lon, h, sids, events, spec, traj, p, origin_hint, refine=True):
    """Fine-score listener positions for one tuple of matched segments.

    origin_hint: (min, max) of plausible recording-start times (unix s).
    Returns DataFrame with total score, per-event scores, recording start and dt.
    """
    obs = ecef(lat, lon, h)
    e1 = events[0]
    models = []
    for e, sid in zip(events, sids):
        o0, o1 = origin_hint[0] - p.sweep_t - e1.t_tol, origin_hint[1] + p.sweep_t + e1.t_tol
        tau, src = traj[sid].sample(o0 + e.window[0] - p.hear_radius_m / p.c - 5, o1 + e.window[1] + 5,
                                    p.tau_step)
        if tau is None:
            return None
        models.append((tau, src))

    def run(idx, dts, fs):
        o = obs[idx]
        curves = [heard_curves(tau, src, o, e.f0, p.c) for (tau, src), e in zip(models, events)]
        # Recording origin from event 1's modelled delayed-CPA time.
        t_arr1, _, r1 = curves[0]
        j = r1.argmin(axis=1)
        T1 = t_arr1[np.arange(len(o)), j]
        best = np.full(len(o), -np.inf)
        best_dt = np.zeros(len(o))
        best_each = np.zeros((len(o), len(events)))
        for dt in dts:
            origin = T1 - (e1.t_cpa + dt)
            tot = np.zeros(len(o))
            each = np.zeros((len(o), len(events)))
            for k, ((t_arr, f, _), e) in enumerate(zip(curves, events)):
                t_rec = t_arr - origin[:, None]
                inwin = (t_rec >= e.window[0]) & (t_rec <= e.window[1])
                w = np.gradient(t_arr, axis=1) * inwin
                wsum = np.maximum(w.sum(axis=1), 1e-9)
                sk = np.full(len(o), -np.inf)
                for s in fs:
                    acc = sum(spec.lookup(t_rec, f * s * h_) for h_ in e.harmonics) / len(e.harmonics)
                    sk = np.maximum(sk, (acc * w).sum(axis=1) / wsum)
                each[:, k] = sk
                tot += sk
            better = tot > best
            best[better], best_dt[better], best_each[better] = tot[better], dt, each[better]
        return best, best_dt, best_each, T1 - (e1.t_cpa + best_dt)

    def grid(sweep, step):
        return np.arange(-sweep, sweep + step / 2, step)

    n = len(obs)
    tot, dt, each, origin = np.zeros(n), np.zeros(n), np.zeros((n, len(events))), np.zeros(n)
    bs = 256
    coarse_dts, coarse_fs = grid(p.sweep_t, max(1.0, p.step_t)), 1 + grid(p.sweep_f, max(0.01, p.step_f))
    for b in range(0, n, bs):
        idx = np.arange(b, min(b + bs, n))
        tot[idx], dt[idx], each[idx], origin[idx] = run(idx, coarse_dts, coarse_fs)
    if refine:
        top = np.argsort(-tot)[: p.refine_top]
        fine_dts, fine_fs = grid(p.sweep_t, p.step_t), 1 + grid(p.sweep_f, p.step_f)
        for b in range(0, len(top), bs):
            idx = top[b : b + bs]
            tot[idx], dt[idx], each[idx], origin[idx] = run(idx, fine_dts, fine_fs)
    df = pd.DataFrame({"lat": lat, "lon": lon, "elev": h, "score": tot, "dt": dt, "rec_start": origin})
    for k in range(len(events)):
        df[f"score{k+1}"] = each[:, k]
    return df


def fine_search(cands, segments, events, spec, dem, p, log=print):
    """Densify coarse survivors and score them; one row per scored point."""
    traj, by_sid = {}, {s.sid: s for s in segments}
    sid_cols = [f"sid{k+1}" for k in range(len(events))]
    out = []
    groups = cands.groupby(sid_cols)
    log(f"fine scoring {len(cands)} coarse points in {groups.ngroups} segment combinations")
    off = np.arange(-p.coarse_m / 2, p.coarse_m / 2 + 1e-6, p.fine_m)
    ox, oy = np.meshgrid(off, off)
    for gi, (sids, g) in enumerate(groups):
        sids = sids if isinstance(sids, tuple) else (sids,)
        for sid in sids:
            if sid not in traj:
                traj[sid] = Trajectory(by_sid[sid])
        la, lo = offset_latlon(g["lat"].to_numpy()[:, None], g["lon"].to_numpy()[:, None],
                               ox.ravel()[None, :], oy.ravel()[None, :])
        la, lo = la.ravel(), lo.ravel()
        key = np.unique(np.c_[np.rint(la * M_PER_DEG_LAT / p.fine_m), np.rint(lo * 1e5 / p.fine_m * 1.1)],
                        axis=0, return_index=True)[1]
        la, lo = la[key], lo[key]
        h = dem(la, lo) + p.geoid_offset_m + p.obs_height_m
        origin = g["T1"] - events[0].t_cpa
        df = score_points(la, lo, h, sids, events, spec, traj, p, (origin.min(), origin.max()))
        if df is None:
            continue
        for k, sid in enumerate(sids):
            df[f"sid{k+1}"] = sid
        out.append(df)
        if (gi + 1) % 50 == 0:
            log(f"  {gi + 1}/{groups.ngroups} combinations scored")
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).sort_values("score", ascending=False, ignore_index=True)


def best_per_combination(scored, n_events):
    sid_cols = [f"sid{k+1}" for k in range(n_events)]
    idx = scored.groupby(sid_cols)["score"].idxmax()
    return scored.loc[idx].sort_values("score", ascending=False, ignore_index=True)


def model_curve(lat, lon, h, seg, event, rec_start, p):
    """Modelled (t_rec, f) curve for plotting a result over the spectrogram."""
    tr = Trajectory(seg)
    tau, src = tr.sample(rec_start + event.window[0] - 40, rec_start + event.window[1] + 5, 0.25)
    if tau is None:
        return np.array([]), np.array([])
    t_arr, f, _ = heard_curves(tau, src, ecef([lat], [lon], [h]), event.f0, p.c)
    t_rec = t_arr[0] - rec_start
    m = (t_rec >= event.window[0]) & (t_rec <= event.window[1])
    return t_rec[m], f[0][m]
