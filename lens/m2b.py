"""composition-lens M2b: extent (kind), rabatment, golden triangle, golden spiral, perspective (vanishing points). Definitions: PLAN.md, section M2b.
No trained model. Builds on lens.engine (profiles, blob map)."""
from __future__ import annotations
import json, os
import numpy as np, cv2
from scipy.ndimage import map_coordinates, maximum_filter1d
from . import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = E.TOL
PHI = (1 + 5 ** 0.5) / 2
CAL_PATH = os.path.join(HERE, "..", "results", "m2b_calibration.json")

def _cal():
    if os.path.exists(CAL_PATH): return json.load(open(CAL_PATH))
    return {"tau_c": 0.6, "tau_b": 0.01}

# ---------------------------------------------------------------- generic path edge strength
def _sample(M, xs, ys): return map_coordinates(M, [ys, xs], order=1, mode="nearest")

def path_strength(ev, pts, tol_px=None):
    """Mean over the path of the edge strength normal to the path, maximised across +/- tol (5 offsets). pts: Nx2 pixels (x, y)."""
    W, H = ev["W"], ev["H"]; pts = np.asarray(pts, float)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] <= W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= H - 1)
    if ok.sum() < 8: return np.nan, 0.0
    t = np.gradient(pts, axis=0); tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9; t = t / tn; n = np.stack([-t[:, 1], t[:, 0]], 1)
    tol_px = tol_px if tol_px is not None else TOL * ev["short"]; best = np.zeros(len(pts))
    for d in np.linspace(-tol_px, tol_px, 5):
        x = np.clip(pts[:, 0] + d * n[:, 0], 0, W - 1); y = np.clip(pts[:, 1] + d * n[:, 1], 0, H - 1)
        v = np.abs(_sample(ev["gx"], x, y) * n[:, 0] + _sample(ev["gy"], x, y) * n[:, 1]); best = np.maximum(best, v)
    return float(best[ok].mean()), float(ok.mean())

def _tau(ev): return 3.0 * float(np.median(np.abs(ev["gy"]))) + 1e-4

# ---------------------------------------------------------------- extent / kind
def coverage_h(ev, y):
    H = ev["H"]; r = max(1, int(round(TOL * H))); c = int(round(y * (H - 1))); band = np.abs(ev["gy"])[max(0, c - r):c + r + 1].max(0)
    return float((band > 3.0 * np.median(np.abs(ev["gy"])) + 1e-4).mean())
def coverage_v(ev, x):
    W = ev["W"]; r = max(1, int(round(TOL * W))); c = int(round(x * (W - 1))); band = np.abs(ev["gx"])[:, max(0, c - r):c + r + 1].max(1)
    return float((band > 3.0 * np.median(np.abs(ev["gx"])) + 1e-4).mean())
def _diag_line_pts(ev, name, off_short=0.0):
    W, H = ev["W"], ev["H"]; d, n = E._diag_dirs(ev, name); D = float(np.hypot(W, H)); ts = np.arange(-0.5 * D, 0.5 * D, 1.0)
    c = np.array([W / 2.0, H / 2.0]) + off_short * ev["short"] * n; return np.stack([c[0] + ts * d[0], c[1] + ts * d[1]], 1)
def coverage_diag(ev, name):
    """coverage along the best offset of the diagonal family (offset chosen by edge strength in the v0 profile)"""
    offs, prof = E._profile_diag(ev, name); off = float(offs[np.argmax(prof)]); pts = _diag_line_pts(ev, name, off)
    W, H = ev["W"], ev["H"]; d, n = E._diag_dirs(ev, name); ok = (pts[:, 0] >= 0) & (pts[:, 0] <= W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= H - 1); pts = pts[ok]
    r = max(1, int(round(TOL * ev["short"]))); best = np.zeros(len(pts))
    for dd in np.linspace(-r, r, 5):
        x = np.clip(pts[:, 0] + dd * n[0], 0, W - 1); y = np.clip(pts[:, 1] + dd * n[1], 0, H - 1)
        best = np.maximum(best, np.abs(_sample(ev["gx"], x, y) * n[0] + _sample(ev["gy"], x, y) * n[1]))
    return float((best > _tau(ev)).mean()), off

def kind_reading(image, ev=None, cal=None):
    ev = ev or E.evidence(image); cal = cal or _cal(); H, W = ev["H"], ev["W"]
    Ph, Pv = E._profile_h(ev), E._profile_v(ev); ys = (np.arange(H) + .5) / H; xs = (np.arange(W) + .5) / W
    mh = (ys >= .04) & (ys <= .96); mv = (xs >= .04) & (xs <= .96)
    y_s = float(ys[mh][np.argmax(Ph[mh])]); x_s = float(xs[mv][np.argmax(Pv[mv])])
    ch, cv_ = coverage_h(ev, y_s), coverage_v(ev, x_s); cd1, o1 = coverage_diag(ev, "BL-TR"); cd2, o2 = coverage_diag(ev, "TL-BR")
    blob = E._disc_share(ev); gy_ = np.arange(0.12, 0.8801, 0.02); blob_peak = float(max(blob[int(round(y * (H - 1))), int(round(x * (W - 1)))] for y in gy_ for x in gy_))
    cov = {"hline": ch, "vline": cv_, "diag": max(cd1, cd2)}; k = max(cov, key=cov.get)
    kind = k if cov[k] >= cal["tau_c"] else ("point" if blob_peak >= cal["tau_b"] else "none")
    return dict(kind=kind, coverage=cov, coverage_diag={"BL-TR": cd1, "TL-BR": cd2}, blob_peak=blob_peak, y_star=y_s, x_star=x_s)

# ---------------------------------------------------------------- rabatment
def rabatment_template(W, H):
    if abs(W - H) < 1e-6: return None
    if W > H: r = H / W; return dict(id="rabatment", kind="vline", primitives=[r if r < 0.5 else 1 - r, 1 - r if r < 0.5 else r], family="frame")
    r = W / H; return dict(id="rabatment", kind="hline", primitives=[r if r < 0.5 else 1 - r, 1 - r if r < 0.5 else r], family="frame")

# ---------------------------------------------------------------- translated-segment null (golden triangle)
def _segment_z(ev, p0, p1):
    """robust z of a segment's edge strength against parallel translations of itself (beyond 2 TOL), as in the v0 diagonal profile."""
    W, H, short = ev["W"], ev["H"], ev["short"]; p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); L = np.linalg.norm(p1 - p0)
    if L < 4: return np.nan
    u = (p1 - p0) / L; n = np.array([-u[1], u[0]]); ts = np.linspace(0, L, max(16, int(L)))
    step = 0.005 * short; offs = np.arange(-0.45 * short, 0.45 * short + 1e-9, step); vals = []
    for o in offs:
        pts = np.stack([p0[0] + o * n[0] + ts * u[0], p0[1] + o * n[1] + ts * u[1]], 1)
        v, frac = path_strength(ev, pts)
        vals.append(v if frac * L >= 0.4 * L and not np.isnan(v) else np.nan)
    vals = np.array(vals); ok = ~np.isnan(vals); offs, vals = offs[ok], vals[ok]
    if len(vals) < 20: return np.nan
    pw = maximum_filter1d(vals, size=max(1, int(2 * TOL * short / step) + 1), mode="nearest"); obs = float(pw[np.argmin(np.abs(offs))]); nul = pw[np.abs(offs) > 2 * TOL * short]
    med = np.median(nul); sd = 1.4826 * np.median(np.abs(nul - med)) + 1e-12; return float((obs - med) / sd)

def golden_triangle_fits(ev):
    W, H = ev["W"], ev["H"]; out = {}
    for name, (A, C, B, Dc) in {"gt_TLBR": ((0, 0), (W, H), (W, 0), (0, H)), "gt_BLTR": ((0, H), (W, 0), (0, 0), (W, H))}.items():
        A, C, B, Dc = map(lambda p: np.array(p, float), (A, C, B, Dc)); u = (C - A) / np.linalg.norm(C - A)
        FB = A + ((B - A) @ u) * u; FD = A + ((Dc - A) @ u) * u
        zs = [_segment_z(ev, A, C), _segment_z(ev, B, FB), _segment_z(ev, Dc, FD)]
        out[name] = dict(z_segments=zs, score=float(np.nanmin(zs)) if not np.any(np.isnan(zs)) else float("nan"))
    return out

# ---------------------------------------------------------------- golden spiral
def _spiral_base(n_arcs=10, pts_per_arc=24):
    """Fibonacci-tiling clockwise quarter arcs in [0, PHI] x [0, 1] (image coordinates, y down). Returns Nx2."""
    x0, y0, x1, y1 = 0.0, 0.0, PHI, 1.0; out = []; sides = ["L", "T", "R", "B"]
    dirs = {"left": (-1, 0), "up": (0, -1), "right": (1, 0), "down": (0, 1)}
    spec = {"L": ("right_bottom", "left", "up"), "T": ("left_bottom", "up", "right"), "R": ("left_top", "right", "down"), "B": ("right_top", "down", "left")}
    for k in range(n_arcs):
        s = min(x1 - x0, y1 - y0); side = sides[k % 4]
        if side == "L": sq = (x0, y0, x0 + s, y1); x0 += s
        elif side == "T": sq = (x0, y0, x1, y0 + s); y0 += s
        elif side == "R": sq = (x1 - s, y0, x1, y1); x1 -= s
        else: sq = (x0, y1 - s, x1, y1); y1 -= s
        cpos, sd, ed = spec[side]
        cx = sq[2] if cpos.startswith("right") else sq[0]; cy = sq[3] if cpos.endswith("bottom") else sq[1]
        sv = np.array(dirs[sd], float); ev_ = np.array(dirs[ed], float); t = np.linspace(0, np.pi / 2, pts_per_arc)
        arc = np.array([cx, cy]) + s * (np.cos(t)[:, None] * sv + np.sin(t)[:, None] * ev_); out.append(arc)
    return np.vstack(out)

_BASE = None
def spiral_variant(v, W, H):
    """v in 0..7: bit 0 mirror x, bit 1 mirror y, bit 2 transpose; stretched to the frame. Returns Nx2 pixels."""
    global _BASE
    if _BASE is None: _BASE = _spiral_base()
    p = _BASE.copy(); w, h = PHI, 1.0
    if v & 4: p = p[:, ::-1].copy(); w, h = 1.0, PHI
    if v & 1: p[:, 0] = w - p[:, 0]
    if v & 2: p[:, 1] = h - p[:, 1]
    return np.stack([p[:, 0] / w * (W - 1), p[:, 1] / h * (H - 1)], 1)

def _spiral_obs(ev, pts): return path_strength(ev, pts)[0]
def spiral_fits(ev, n_null=120, seed=0):
    W, H, short = ev["W"], ev["H"], ev["short"]; rng = np.random.default_rng(seed); res = {}
    c = np.array([W / 2.0, H / 2.0])
    for v in range(8):
        pts = spiral_variant(v, W, H); obs = _spiral_obs(ev, pts); nul = []
        while len(nul) < n_null:
            dx, dy = rng.uniform(-0.15, 0.15, 2) * np.array([W, H]); s = rng.uniform(0.85, 1.15); th = np.deg2rad(rng.uniform(-10, 10))
            if np.hypot(dx / short, dy / short) < 2 * TOL and abs(s - 1) < 0.03 and abs(th) < np.deg2rad(2): continue
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); q = (pts - c) @ R.T * s + c + np.array([dx, dy]); val = _spiral_obs(ev, q)
            if not np.isnan(val): nul.append(val)
        nul = np.array(nul); med = np.median(nul); sd = 1.4826 * np.median(np.abs(nul - med)) + 1e-12
        res[v] = float((obs - med) / sd)
    best = max(res, key=res.get); return dict(per_variant=res, best_variant=best)
EMAX8 = 1.4236   # expected maximum of 8 standard normals
def spiral_score(res): return res["per_variant"][res["best_variant"]] - EMAX8

# ---------------------------------------------------------------- perspective
def _segments(gray):
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8); lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g8)[0]
    return np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)

def vanishing_reading(image, ev=None, seed=0, nhyp=600, tol_deg=2.5, min_len=0.06):
    ev = ev or E.evidence(image); g, H, W = ev["g"], ev["H"], ev["W"]; diag = float(np.hypot(H, W)); rng = np.random.default_rng(seed)
    seg = _segments(g)
    if len(seg):
        d = seg[:, 2:4] - seg[:, 0:2]; L = np.hypot(d[:, 0], d[:, 1]); keep = L >= min_len * diag; seg, d, L = seg[keep], d[keep], L[keep]
    if len(seg) < 3: return dict(n_finite=0, n_parallel=0, families=[], total_len=float(L.sum()) if len(seg) else 0.0)
    dirs = d / L[:, None]; mid = (seg[:, 0:2] + seg[:, 2:4]) / 2; lines_h = np.cross(np.c_[seg[:, 0:2], np.ones(len(seg))], np.c_[seg[:, 2:4], np.ones(len(seg))])
    sin_tol = np.sin(np.deg2rad(tol_deg)); total = float(L.sum()); alive = np.ones(len(seg), bool); fams = []
    for _ in range(4):
        idx = np.where(alive)[0]
        if len(idx) < 3: break
        p = L[idx] / L[idx].sum(); best = None; bs = -1.0
        for _h in range(nhyp):
            i, j = rng.choice(len(idx), 2, replace=False, p=p); a, b = idx[i], idx[j]; v = np.cross(lines_h[a], lines_h[b])
            if abs(v[2]) < 1e-9 * (abs(v[0]) + abs(v[1]) + 1e-12): vp = np.array([v[0], v[1]]) / (np.hypot(v[0], v[1]) + 1e-12); inf = True
            else: vp = np.array([v[0] / v[2], v[1] / v[2]]); inf = False
            dd = np.broadcast_to(vp, mid[idx].shape) if inf else vp[None, :] - mid[idx]; dn = np.linalg.norm(dd, axis=1) + 1e-9
            sup = np.abs(dirs[idx, 0] * dd[:, 1] - dirs[idx, 1] * dd[:, 0]) / dn <= sin_tol; sc = float(L[idx][sup].sum())
            if sc > bs: bs, best = sc, (vp, inf, idx[sup])
        if best is None or bs < 0.08 * total: break
        vp, inf, members = best; far = (not inf) and (-2 * W <= vp[0] <= 3 * W) and (-2 * H <= vp[1] <= 3 * H)
        fams.append(dict(vp=[float(vp[0] / W), float(vp[1] / H)] if not inf else None, finite=bool(far), support=float(bs / total), n_seg=int(len(members))))
        alive[members] = False
    n_fin = sum(1 for f in fams if f["finite"] and f["support"] >= 0.12 and f["n_seg"] >= 4); n_par = sum(1 for f in fams if not f["finite"] and f["support"] >= 0.12 and f["n_seg"] >= 4)
    return dict(n_finite=n_fin, n_parallel=n_par, families=fams, total_len=total)

def perspective_scores(vr):
    k = vr["n_finite"]; return {"persp_1pt": float(k == 1), "persp_2pt": float(k == 2), "persp_3pt": float(k >= 3)}

# ---------------------------------------------------------------- all together
def analyse(image, spiral=True, triangle=True, persp=True):
    ev = E.evidence(image); out = {"v0": E.readings_and_fits(image), "kind": kind_reading(image, ev)}
    rt = rabatment_template(ev["W"], ev["H"])
    if rt:
        r = E.readings_and_fits(image, templates=[rt]); out["rabatment"] = r["fits"].get("rabatment")
    if triangle: out["golden_triangle"] = golden_triangle_fits(ev)
    if spiral: sp = spiral_fits(ev); sp["score"] = spiral_score(sp); out["golden_spiral"] = sp
    if persp: vr = vanishing_reading(image, ev); out["perspective"] = dict(reading=vr, scores=perspective_scores(vr))
    return out
