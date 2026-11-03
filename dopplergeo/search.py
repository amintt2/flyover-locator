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
    clip: int = 0  # edited videos: events in different clips have no known time offset
    any_type: bool = False  # fast high-pitched pass: jets too, not only propeller aircraft

    @classmethod
    def from_dict(cls, d):
        return cls(d["t_cpa"], tuple(d["v"]), tuple(d["d"]), d["f0"], tuple(d["window"]),
                   d.get("t_tol", 4.0), list(d.get("harmonics", [1])), int(d.get("clip", 0)),
                   bool(d.get("any_type", False)))


@dataclass
class SearchParams:
    c: float = 343.0
    coarse_m: float = 200.0
    screen_m: float = 2000.0
    fine_m: float = 100.0
    tile_deg: float = 1.0
    hear_radius_m: float = 10000.0
    silence_radius_m: float | None = None  # default: smallest event d_min
    silence_margin_s: float = 60.0
    silence_intervals: list | None = None  # recording-time [a, b]; None = auto
    start_window: tuple | None = None  # (unix_min, unix_max) of recording start
    clips: list | None = None  # edited video: [(a, b), ...] continuous spans in recording time
    clip_gap_s: float = 6 * 3600.0  # max real time between two clips of one video
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


def anchors(events):
    """Index of the first event of each event's clip: times are only comparable
    inside one continuous clip, so every event is timed against its anchor."""
    first = {}
    return [first.setdefault(e.clip, k) for k, e in enumerate(events)]


def silence_intervals(events, duration, p):
    """Quiet spans [(a, b, anchor)] in recording time; each is timed from its clip's anchor
    event. Clips with no heard flyover cannot be placed in time and are skipped."""
    if p.silence_intervals is not None:
        return [(x[0], x[1], 0) for x in p.silence_intervals]
    anc = anchors(events)
    out = []
    for k, (ca, cb) in enumerate(p.clips or [(0.0, duration)]):
        inside = [i for i, e in enumerate(events) if ca <= e.t_cpa <= cb]
        if not inside:
            continue
        busy = sorted((events[i].t_cpa - p.silence_margin_s, events[i].t_cpa + p.silence_margin_s) for i in inside)
        cur = ca
        for a, b in busy:
            if a - cur >= 10:
                out.append((cur, a, anc[inside[0]]))
            cur = max(cur, b)
        if cb - cur >= 10:
            out.append((cur, cb, anc[inside[0]]))
    return out


# ---------------------------------------------------------------- coarse stage


def _tiles_touched(segs, td, ring=1):
    out = set()
    for s in segs:
        ij = set(zip(np.floor(s.lat[::10] / td).astype(int).tolist(), np.floor(s.lon[::10] / td).astype(int).tolist()))
        for i, j in ij:
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    out.add((i + di, j + dj))
    return out


def _match_events(tree, gxyz, ev_segs, events, p, d_pad=0.0, t_pad=0.0, log=print, label=""):
    """Listener points hearing event 1, then event 2 at the right offset, ..."""
    e1 = events[0]
    anc = anchors(events)
    P = []
    for e, segs_k in zip(events, ev_segs):
        tab = _passes_table(tree, gxyz, segs_k, e.d[1] + d_pad, p.c, max(0.0, e.d[0] - d_pad), e.v)
        if p.start_window:
            tol = e.t_tol + t_pad + (p.clip_gap_s if e.clip != e1.clip else 0.0)
            tab = tab[(tab["T"] >= p.start_window[0] + e.t_cpa - tol) & (tab["T"] <= p.start_window[1] + e.t_cpa + tol)]
        if tab.empty:
            return pd.DataFrame()
        P.append(tab)
    cand = P[0].rename(columns={"sid": "sid1", "T": "T1", "r": "r1", "v": "v1"})
    for k in range(1, len(events)):
        e = events[k]
        nxt = P[k].rename(columns={"sid": f"sid{k+1}", "T": f"T{k+1}", "r": f"r{k+1}", "v": f"v{k+1}"})
        cand = cand.merge(nxt, on="obs")
        a = anc[k]
        if a == k:  # first flyover after a cut: same place, but the time between clips is unknown
            dt = cand[f"T{k+1}"] - cand["T1"] - (e.t_cpa - e1.t_cpa)
            ok = dt.abs() <= p.clip_gap_s
        else:
            ea = events[a]
            dt = cand[f"T{k+1}"] - cand[f"T{a+1}"] - (e.t_cpa - ea.t_cpa)
            ok = dt.abs() <= (ea.t_tol + e.t_tol + 2 * t_pad)
        for j in range(1, k + 1):
            ok &= ~((cand[f"sid{k+1}"] == cand[f"sid{j}"]) & ((cand[f"T{k+1}"] - cand[f"T{j}"]).abs() < 30))
        cand = cand[ok]
        if len(cand) > p.max_candidates:
            log(f"  {label}: {len(cand)} candidates, truncating")
            cand = cand.iloc[: p.max_candidates]
        if cand.empty:
            break
    return cand


def coarse_search(segments, events, area, dem, duration, p, log=print):
    """Return a DataFrame of surviving coarse grid points with matched segment ids.

    Only tiles crossed by aircraft matching every event are visited. Inside a
    tile a cheap screening pass on a `screen_m` grid (sea-level ground,
    tolerances widened for the coarse spacing) discards empty areas before the
    200 m grid, the elevation model and the silence check are used.
    """
    e1 = events[0]
    sil = silence_intervals(events, duration, p)
    # a plane passing closer than the ones we did hear would have been heard too
    r_sil = p.silence_radius_m or min(np.sqrt(e.d[0] * e.d[1]) for e in events)
    t_lo, t_hi = -np.inf, np.inf
    if p.start_window:
        t_lo, t_hi = p.start_window[0] - 120, p.start_window[1] + duration + 120
        if p.clips and len(p.clips) > 1:
            t_lo, t_hi = t_lo - p.clip_gap_s, t_hi + p.clip_gap_s
    segs_t = _crop(segments, area["lat_min"] - 1, area["lat_max"] + 1, area["lon_min"] - 1.5, area["lon_max"] + 1.5,
                   t_lo, t_hi)
    ev_all = [[r for s in segs_t if s.prop or e.any_type for r in speed_runs(s, *e.v)] for e in events]
    if not all(ev_all):
        return pd.DataFrame()
    td = p.tile_deg
    tiles = set.intersection(*(_tiles_touched(runs, td) for runs in ev_all))
    i0, i1 = int(np.floor(area["lat_min"] / td)), int(np.floor((area["lat_max"] - 1e-9) / td))
    j0, j1 = int(np.floor(area["lon_min"] / td)), int(np.floor((area["lon_max"] - 1e-9) / td))
    tiles = sorted((i, j) for i, j in tiles if i0 <= i <= i1 and j0 <= j <= j1)
    log(f"  {len(tiles)} tile(s) crossed by matching aircraft")

    # screening tolerances: unknown ground height (assume up to 3000 m) and coarse spacing
    v_min = max(10.0, min(e.v[0] for e in events))
    d_pad = p.screen_m + 3000.0
    t_pad = (d_pad + 0.71 * p.screen_m * p.c / v_min) / p.c

    results = []
    for i, j in tiles:
        la0, lo0 = max(i * td, area["lat_min"]), max(j * td, area["lon_min"])
        la1, lo1 = min((i + 1) * td, area["lat_max"]), min((j + 1) * td, area["lon_max"])
        buf_lat = p.hear_radius_m / M_PER_DEG_LAT
        buf_lon = p.hear_radius_m / m_per_deg_lon(max(abs(la0), abs(la1)))
        box = (la0 - buf_lat, la1 + buf_lat, lo0 - buf_lon, lo1 + buf_lon)
        ev_segs = [_crop(runs, *box) for runs in ev_all]
        if not all(ev_segs):
            continue
        label = f"tile {la0:.1f},{lo0:.1f}"

        # 1) screening pass
        slat, slon = latlon_grid(la0, la1, lo0, lo1, p.screen_m)
        sxyz = ecef(slat, slon, np.zeros_like(slat))
        scr = _match_events(cKDTree(sxyz), sxyz, ev_segs, events, p, d_pad, t_pad, log, label)
        if scr.empty:
            continue
        keep = np.unique(scr["obs"].to_numpy())

        # 2) 200 m grid only around surviving screening cells
        glat, glon = latlon_grid(la0, la1, lo0, lo1, p.coarse_m)
        kx = m_per_deg_lon((la0 + la1) / 2)
        near = cKDTree(np.c_[slat[keep] * M_PER_DEG_LAT, slon[keep] * kx])
        dist, _ = near.query(np.c_[glat * M_PER_DEG_LAT, glon * kx], distance_upper_bound=p.screen_m)
        sel = np.isfinite(dist)
        glat, glon = glat[sel], glon[sel]
        gh = dem(glat, glon) + p.geoid_offset_m + p.obs_height_m
        gxyz = ecef(glat, glon, gh)
        tree = cKDTree(gxyz)
        cand = _match_events(tree, gxyz, ev_segs, events, p, log=log, label=label)
        if cand.empty:
            continue
        n_time = len(cand)

        if sil:
            tile = _crop(segs_t, *box)
            obs_u = np.unique(cand["obs"].to_numpy())
            sub_tree = cKDTree(gxyz[obs_u])
            origins = {an: cand[f"T{an+1}"] - events[an].t_cpa for an in {x[2] for x in sil}}
            lo_t = min(origins[an].min() + a for a, _, an in sil) - 60
            hi_t = max(origins[an].max() + b for _, b, an in sil) + 60
            quiet = _passes_table(sub_tree, gxyz[obs_u], _crop(tile, -90, 90, -180, 180, lo_t - 60, hi_t),
                                  r_sil, p.c)
            if not quiet.empty:
                quiet["obs"] = obs_u[quiet["obs"].to_numpy()]
                c2 = cand.reset_index(drop=True)
                c2["row"] = np.arange(len(c2))
                m = c2.merge(quiet.rename(columns={"sid": "qsid", "T": "qT"})[["obs", "qsid", "qT"]], on="obs")
                loud = np.zeros(len(m), bool)
                for a, b, an in sil:
                    rel = m["qT"] - (m[f"T{an+1}"] - events[an].t_cpa)
                    loud |= (rel >= a + events[an].t_tol) & (rel <= b - events[an].t_tol)
                for k in range(1, len(events) + 1):  # the matched passes themselves
                    loud &= ~((m["qsid"] == m[f"sid{k}"]) & ((m["qT"] - m[f"T{k}"]).abs() < 30))
                bad = np.unique(m.loc[loud, "row"].to_numpy())
                cand = c2.drop(index=bad).drop(columns="row")
        log(f"  {label}: {len(keep)} screening cells, {n_time} pass time-offset filter, "
            f"{len(cand)} pass silence filter")
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

    origin_hint: {anchor: (min, max)} of plausible clip-start times (unix s); anchor 0
    is the recording start (see `anchors`).
    Returns DataFrame with total score, per-event scores and origins, recording start and dt.
    """
    obs = ecef(lat, lon, h)
    e1 = events[0]
    anc = anchors(events)
    models = []
    for k, (e, sid) in enumerate(zip(events, sids)):
        hint, ea = origin_hint[anc[k]], events[anc[k]]
        o0, o1 = hint[0] - p.sweep_t - ea.t_tol, hint[1] + p.sweep_t + ea.t_tol
        tau, src = traj[sid].sample(o0 + e.window[0] - p.hear_radius_m / p.c - 5, o1 + e.window[1] + 5,
                                    p.tau_step)
        if tau is None:
            return None
        models.append((tau, src))

    def run(idx, dts, fs):
        o = obs[idx]
        curves = [heard_curves(tau, src, o, e.f0, p.c) for (tau, src), e in zip(models, events)]
        # Each clip's origin from its anchor event's modelled delayed-CPA time.
        Ta = {}
        for a in set(anc):
            t_arr_a, _, r_a = curves[a]
            Ta[a] = t_arr_a[np.arange(len(o)), r_a.argmin(axis=1)]
        best = np.full(len(o), -np.inf)
        best_dt = np.zeros(len(o))
        best_each = np.zeros((len(o), len(events)))
        best_org = np.zeros((len(o), len(events)))
        for dt in dts:
            org = np.stack([Ta[anc[k]] - (events[anc[k]].t_cpa + dt) for k in range(len(events))], axis=1)
            tot = np.zeros(len(o))
            each = np.zeros((len(o), len(events)))
            for k, ((t_arr, f, _), e) in enumerate(zip(curves, events)):
                t_rec = t_arr - org[:, k, None]
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
            best[better], best_dt[better], best_each[better], best_org[better] = tot[better], dt, each[better], org[better]
        return best, best_dt, best_each, best_org

    def grid(sweep, step):
        return np.arange(-sweep, sweep + step / 2, step)

    n = len(obs)
    tot, dt, each, origin = np.zeros(n), np.zeros(n), np.zeros((n, len(events))), np.zeros((n, len(events)))
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
    df = pd.DataFrame({"lat": lat, "lon": lon, "elev": h, "score": tot, "dt": dt, "rec_start": origin[:, 0]})
    for k in range(len(events)):
        df[f"score{k+1}"] = each[:, k]
        df[f"origin{k+1}"] = origin[:, k]
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
        hints = {}
        for a in set(anchors(events)):
            o = g[f"T{a+1}"] - events[a].t_cpa
            hints[a] = (o.min(), o.max())
        df = score_points(la, lo, h, sids, events, spec, traj, p, hints)
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
