"""M2b synthetic generators: golden triangle, golden spiral, perspective line drawings, strokes. Exact layouts by construction."""
import numpy as np, cv2
from . import synth, m2b
from .synth import ASPECTS, _background, _distractors, CAT_X, CAT_Y

def _canvas(W, H): return np.zeros((H * 2, W * 2), np.float32)
def _draw(canvas, pts, width_px):
    p = (np.asarray(pts, float) * 2).round().astype(np.int32).reshape(-1, 1, 2); cv2.polylines(canvas, [p], False, 1.0, thickness=max(1, int(round(width_px * 2))), lineType=cv2.LINE_AA)
def _finish(rng, W, H, canvas, amp, noise, n_dist=3, avoid=True):
    img = _background(rng, W, H); c = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA); c = cv2.GaussianBlur(c, (0, 0), 0.7)
    img = img + rng.choice([-1, 1]) * amp * np.minimum(1.0, c * 1.6)
    img = _distractors(rng, img, W, H, CAT_X if avoid else [], CAT_Y if avoid else [], rng.integers(2, 4))
    if noise > 0: img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 1); return (np.stack([img] * 3, -1) * 255).astype(np.uint8)

def triangle_segments(W, H, orient):
    A, C, B, D = ((0, 0), (W - 1, H - 1), (W - 1, 0), (0, H - 1)) if orient == "TLBR" else ((0, H - 1), (W - 1, 0), (0, 0), (W - 1, H - 1))
    A, C, B, D = map(lambda p: np.array(p, float), (A, C, B, D)); u = (C - A) / np.linalg.norm(C - A)
    return [(A, C), (B, A + ((B - A) @ u) * u), (D, A + ((D - A) @ u) * u)]

def render_triangle(mode, orient, aspect="4:3", noise=0.02, seed=0, contrast=None):
    rng = np.random.default_rng(seed); W, H = ASPECTS[aspect]; short = min(W, H); can = _canvas(W, H); amp = contrast if contrast is not None else rng.uniform(0.22, 0.38)
    segs = triangle_segments(W, H, orient); w = 0.012 * short
    if mode == "diag_only": segs = segs[:1]
    if mode == "shifted":
        ang = rng.uniform(0, 2 * np.pi); mag = rng.uniform(0.12, 0.25) * short; sh = mag * np.array([np.cos(ang), np.sin(ang)]); segs = [(a + sh, b + sh) for a, b in segs]
    if mode == "random":
        segs = []
        for _ in range(3):
            while True:
                a = rng.uniform([0, 0], [W, H]); b = rng.uniform([0, 0], [W, H])
                if np.linalg.norm(a - b) > 0.4 * np.hypot(W, H): break
            segs.append((a, b))
    jit = rng.normal(0, 0.002 * short, 2)
    for a, b in segs: _draw(can, [a + jit, b + jit], w)
    return _finish(rng, W, H, can, amp, noise)

def render_spiral(variant, aspect="4:3", noise=0.02, seed=0, shift=0.0, perturb=False, straight=False, contrast=None):
    rng = np.random.default_rng(seed); W, H = ASPECTS[aspect]; short = min(W, H); can = _canvas(W, H); amp = contrast if contrast is not None else rng.uniform(0.22, 0.38)
    if straight:
        for _ in range(3):
            while True:
                a = rng.uniform([0, 0], [W, H]); b = rng.uniform([0, 0], [W, H])
                if np.linalg.norm(a - b) > 0.4 * np.hypot(W, H): break
            _draw(can, [a, b], 0.012 * short)
    else:
        pts = m2b.spiral_variant(variant, W, H); c = np.array([W / 2.0, H / 2.0])
        if perturb:
            while True:
                dx, dy = rng.uniform(-0.15, 0.15, 2) * np.array([W, H]); s = rng.uniform(0.85, 1.15); th = np.deg2rad(rng.uniform(-10, 10))
                if np.hypot(dx, dy) / short > 0.06 or abs(s - 1) > 0.05 or abs(th) > np.deg2rad(4): break
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); pts = (pts - c) @ R.T * s + c + np.array([dx, dy])
        elif shift > 0:
            ang = rng.uniform(0, 2 * np.pi); pts = pts + shift * short * np.array([np.cos(ang), np.sin(ang)])
        _draw(can, pts, 0.012 * short)
    return _finish(rng, W, H, can, amp, noise)

def render_perspective(k, aspect="4:3", noise=0.0, seed=0, contrast=0.3):
    rng = np.random.default_rng(seed); W, H = ASPECTS[aspect]; short = min(W, H); can = _canvas(W, H); diag = np.hypot(W, H)
    hy = rng.uniform(0.35, 0.65) * H; vps = []
    if k == 1: vps = [np.array([rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H])]
    else:
        vps = [np.array([rng.uniform(-1.0, -0.3) * W, hy]), np.array([rng.uniform(1.3, 2.0) * W, hy])]
        if k == 3: vps.append(np.array([rng.uniform(0.3, 0.7) * W, rng.choice([rng.uniform(-1.5, -0.8) * H, rng.uniform(1.8, 2.5) * H])]))
    per = {1: 16, 2: 11, 3: 9}[k]; n_lines = 0
    for vp in vps:
        for _ in range(per):
            for _try in range(30):
                P = rng.uniform([0.05 * W, 0.05 * H], [0.95 * W, 0.95 * H]); d = vp - P; dist = np.linalg.norm(d)
                if dist < 0.25 * diag: continue
                u = d / dist; ln = rng.uniform(0.15, 0.4) * diag; Q = P + u * min(ln, 0.7 * dist)
                if np.linalg.norm(Q - P) > 0.1 * diag: break
            _draw(can, [P, Q], 0.008 * short); n_lines += 1
    if k == 1:
        for _ in range(4):                    # axis-aligned parallels (a parallel family, not a finite vanishing point)
            y = rng.uniform(0.1, 0.9) * H; x0 = rng.uniform(0, 0.5) * W; _draw(can, [[x0, y], [x0 + rng.uniform(0.3, 0.5) * W, y]], 0.008 * short)
            x = rng.uniform(0.1, 0.9) * W; y0 = rng.uniform(0, 0.5) * H; _draw(can, [[x, y0], [x, y0 + rng.uniform(0.3, 0.5) * H]], 0.008 * short)
    else:
        for _ in range(6):                    # verticals (two- and three-point drawings)
            if k == 3: break
            x = rng.uniform(0.1, 0.9) * W; y0 = rng.uniform(0.1, 0.5) * H; _draw(can, [[x, y0], [x, y0 + rng.uniform(0.2, 0.45) * H]], 0.008 * short)
    for _ in range(max(2, int(0.25 * n_lines))):   # distractor lines
        a = rng.uniform([0, 0], [W, H]); ang = rng.uniform(0, np.pi); ln = rng.uniform(0.15, 0.35) * diag; _draw(can, [a, a + ln * np.array([np.cos(ang), np.sin(ang)])], 0.008 * short)
    return _finish(rng, W, H, can, contrast, noise), [[float(v[0] / W), float(v[1] / H)] for v in vps]
