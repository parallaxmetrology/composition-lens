"""composition-lens v0 engine: position profiles + mass map -> template fits against a null. See PLAN.md section 3 for the fixed definitions.
No trained model is used. Images are RGB arrays (uint8 or float)."""
from __future__ import annotations
import json, os
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, maximum_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = 384
LUMA = np.array([0.2126, 0.7152, 0.0722])
def load_templates(status="v0"):
    T = json.load(open(os.path.join(HERE, "..", "catalogue", "templates.json")))
    return [t for t in T if t["status"] == status]

def prep(image):
    a = np.asarray(image, float)
    if a.max() > 1.5: a = a / 255.0
    if a.ndim == 3: a = a @ LUMA
    h, w = a.shape; s = LONG / max(h, w)
    if abs(s - 1) > 1e-6: a = cv2.resize(a.astype(np.float32), (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return a.astype(np.float32)

def evidence(image):
    g = prep(image); H, W = g.shape; short = min(H, W)
    gb = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3) / 8.0; gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    M = gaussian_filter(np.abs(g - gaussian_filter(g, 0.15 * short)), 0.02 * short)
    return dict(g=g, gx=gx, gy=gy, M=M, H=H, W=W, short=short)

def _profile_h(ev):   # mean over x of |dL/dy|, smoothed along y
    return gaussian_filter1d(np.abs(ev["gy"]).mean(1), 0.008 * ev["H"])
def _profile_v(ev):
    return gaussian_filter1d(np.abs(ev["gx"]).mean(0), 0.008 * ev["W"])

def _diag_dirs(ev, kind):
    W, H = ev["W"], ev["H"]
    d = np.array([W, H], float) if kind == "TL-BR" else np.array([W, -H], float)   # image coords: y down
    d /= np.linalg.norm(d); n = np.array([-d[1], d[0]]); return d, n
def _profile_diag(ev, kind):
    W, H, short = ev["W"], ev["H"], ev["short"]; d, n = _diag_dirs(ev, kind)
    D = float(np.hypot(W, H)); En = np.abs(ev["gx"] * n[0] + ev["gy"] * n[1])
    step = 0.005 * short; offs = np.arange(-0.5 * D, 0.5 * D + 1e-9, step)
    c = np.array([W / 2.0, H / 2.0]); ts = np.arange(-0.5 * D, 0.5 * D, 1.0)
    vals = np.full(len(offs), np.nan)
    for i, o in enumerate(offs):
        x = c[0] + o * n[0] + ts * d[0]; y = c[1] + o * n[1] + ts * d[1]
        m = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        if m.sum() >= 0.4 * D: vals[i] = map_coordinates(En, [y[m], x[m]], order=1).mean()
    ok = ~np.isnan(vals); offs, vals = offs[ok], vals[ok]
    return offs / short, gaussian_filter1d(vals, 0.008 * short / step)   # offsets as fractions of the short side

def _disc_share(ev):
    """Amendment A1 (20 Nov 2026): the point evidence is a BLOB response, the disc share minus the MAXIMUM disc share on a ring of 8 positions at 14% of the short side (A1b: the first version used the ring mean and still let a line through),
    so a line (high share all along it) no longer scores like a point."""
    short = ev["short"]; r = max(2, int(round(0.07 * short))); k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, -1); share = cv2.filter2D(ev["M"].astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    share = share / max(float(ev["M"].sum()), 1e-9)
    R = int(round(0.14 * short)); pad = cv2.copyMakeBorder(share, R, R, R, R, cv2.BORDER_REPLICATE); H, W = share.shape
    ring = np.full_like(share, -1e9)
    for ang in range(8):
        dx, dy = int(round(R * np.cos(ang * np.pi / 4))), int(round(R * np.sin(ang * np.pi / 4))); ring = np.maximum(ring, pad[R + dy:R + dy + H, R + dx:R + dx + W])
    return share - ring

TOL = 0.015   # position tolerance (fraction of the frame): a template fits if a strong line lies within +/-TOL of its position
def _wmax(prof, frac_len):
    return maximum_filter1d(prof, size=max(1, int(2 * TOL * frac_len) + 1), mode="nearest")
EMAX = {1: 0.0, 2: 0.564, 3: 0.846, 4: 1.029}   # E[max of k standard normals]
def _p(obs, null):
    null = np.asarray(null); return (1 + np.sum(null >= obs)) / (1 + len(null))
def _sidak(pmin, k): return 1 - (1 - pmin) ** k

def readings_and_fits(image, templates=None):
    ev = evidence(image); H, W = ev["H"], ev["W"]
    templates = templates or load_templates()
    Ph, Pv = _profile_h(ev), _profile_v(ev)
    ys = (np.arange(H) + 0.5) / H; xs = (np.arange(W) + 0.5) / W
    mh = (ys >= 0.04) & (ys <= 0.96); mv = (xs >= 0.04) & (xs <= 0.96)
    share = _disc_share(ev); gy_ = np.arange(0.12, 0.8801, 0.02); grid = np.array([[share[int(round(y * (H - 1))), int(round(x * (W - 1)))] for x in gy_] for y in gy_]).ravel()
    dprof = {k: _profile_diag(ev, k) for k in ("BL-TR", "TL-BR")}
    out = {}
    out["readings"] = dict(
        y_star=float(ys[mh][np.argmax(Ph[mh])]), y_strength=float(Ph[mh].max() / (np.median(Ph) + 1e-9)),
        x_star=float(xs[mv][np.argmax(Pv[mv])]), x_strength=float(Pv[mv].max() / (np.median(Pv) + 1e-9)),
        mass_cx=float((ev["M"].sum(0) * xs).sum() / ev["M"].sum()), mass_cy=float((ev["M"].sum(1) * ys).sum() / ev["M"].sum()))
    fits = {}
    Phm, Pvm = _wmax(Ph, H), _wmax(Pv, W)
    gxs, gys = np.meshgrid(gy_, gy_)   # grid of point positions (x right, y down)
    sh_m = grid.reshape(len(gy_), len(gy_))
    for t in templates:
        k = t["kind"]; prims = t["primitives"]
        if k == "hline":
            obs = [np.interp(p, ys, Phm) for p in prims]; far = np.min(np.abs(ys[:, None] - np.array(prims)[None]), 1) > 2 * TOL; nul = Phm[mh & far]
        elif k == "vline":
            obs = [np.interp(p, xs, Pvm) for p in prims]; far = np.min(np.abs(xs[:, None] - np.array(prims)[None]), 1) > 2 * TOL; nul = Pvm[mv & far]
        elif k == "points":
            r = int(round(TOL * ev["short"])); obs = []
            for (px, py) in prims:
                cy_, cx_ = int(round(py * (H - 1))), int(round(px * (W - 1))); obs.append(share[max(0, cy_ - r):cy_ + r + 1, max(0, cx_ - r):cx_ + r + 1].max())
            d = np.min([np.hypot(gxs - px, gys - py) for px, py in prims], 0); nul = sh_m[d > 0.1]
        elif k == "diag":
            offs, prof = dprof[prims[0]]; step = 0.005
            pw = maximum_filter1d(prof, size=max(1, int(2 * TOL / step) + 1), mode="nearest"); obs = [float(pw[np.argmin(np.abs(offs))])]; nul = pw[np.abs(offs) > 2 * TOL]
        else: continue
        ps = [_p(o, nul) for o in obs]; pT = _sidak(min(ps), len(prims))
        nul = np.asarray(nul, float); med = np.median(nul); sd = 1.4826 * np.median(np.abs(nul - med)) + 1e-12
        z = max((o - med) / sd for o in obs) - EMAX.get(len(prims), 1.03)     # Amendment A2: robust z of the best primitive, minus the expected maximum of k normals
        fits[t["id"]] = dict(score=float(z), rank_score=float(-np.log10(pT)), p=float(pT))
    out["fits"] = fits
    return out
