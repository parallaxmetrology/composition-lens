"""Exact-layout synthetic images for the recovery tests. A structure is placed exactly on a template position (jitter <= ~0.5% of the frame)."""
import numpy as np
from scipy.ndimage import gaussian_filter
ASPECTS = {"4:3": (512, 384), "3:2": (576, 384), "1:1": (384, 384), "4:5": (307, 384)}   # (W, H)
CAT_X = [1/3, 2/3, .5, .381966, .618034, .2, .4, .6, .8]; CAT_Y = list(CAT_X)

def _background(rng, W, H):
    base = rng.uniform(0.38, 0.62); lf = gaussian_filter(rng.normal(size=(H, W)), 0.1 * min(W, H)); lf = lf / (lf.std() + 1e-9) * 0.04
    grad = (np.linspace(-1, 1, H)[:, None] * rng.uniform(-0.05, 0.05) + np.linspace(-1, 1, W)[None, :] * rng.uniform(-0.05, 0.05))
    return base + lf + grad + rng.normal(0, 0.012, (H, W))

def _soft_step(coord, pos, soft):  # 0..1 step centred at pos (pixels) with gaussian-ish softness
    return 1 / (1 + np.exp(-(coord - pos) / max(soft, 0.3)))

def _distractors(rng, img, W, H, avoid_x, avoid_y, n):
    yy, xx = np.mgrid[0:H, 0:W]; short = min(W, H)
    for _ in range(n):
        for _try in range(20):
            cx, cy = rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)
            if min(abs(cx - a) for a in avoid_x) > 0.04 and min(abs(cy - a) for a in avoid_y) > 0.04: break
        r = rng.uniform(0.03, 0.07) * short; amp = rng.choice([-1, 1]) * rng.uniform(0.04, 0.09)
        img += amp * np.exp(-(((xx - cx * W) / r) ** 2 + ((yy - cy * H) / r) ** 2) / 2)
    return img

def render(kind, pos, aspect="4:3", noise=0.0, seed=0, contrast=None, soft=None):
    """kind: 'hline' (pos=y), 'vline' (pos=x), 'point' (pos=(x,y)), 'diag' (pos='BL-TR'|'TL-BR' with optional offset via tuple (name, offset_frac_of_short))."""
    rng = np.random.default_rng(seed); W, H = ASPECTS[aspect]; short = min(W, H)
    img = _background(rng, W, H); yy, xx = np.mgrid[0:H, 0:W].astype(float)
    c = contrast if contrast is not None else rng.uniform(0.22, 0.38); sf = soft if soft is not None else rng.uniform(0.4, 1.5) * 0.01 * short
    sgn = rng.choice([-1, 1]); jit = rng.normal(0, 0.002)
    if kind == "hline": img += sgn * c * (_soft_step(yy, (pos + jit) * H, sf) - 0.5); ax, ay = [], [pos]
    elif kind == "vline": img += sgn * c * (_soft_step(xx, (pos + jit) * W, sf) - 0.5); ax, ay = [pos], []
    elif kind == "point":
        px, py = pos[0] + jit, pos[1] + rng.normal(0, 0.002); r = 0.05 * short
        img += -abs(c) * np.exp(-(((xx - px * W) ** 2 + (yy - py * H) ** 2) / (2 * (r * 0.6) ** 2)) ** 1.5); ax, ay = [pos[0]], [pos[1]]
    elif kind == "diag":
        name, off = pos if isinstance(pos, tuple) else (pos, 0.0)
        d = np.array([W, H], float) if name == "TL-BR" else np.array([W, -H], float); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
        sd = (xx - W / 2) * n[0] + (yy - H / 2) * n[1]
        img += sgn * c * (_soft_step(sd, (off + jit) * short, sf) - 0.5); ax, ay = [], []
    img = _distractors(rng, img, W, H, CAT_X + ax, CAT_Y + ay, rng.integers(2, 5))
    if noise > 0: img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 1); return (np.stack([img] * 3, -1) * 255).astype(np.uint8)
