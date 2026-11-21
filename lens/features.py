"""Per-image named features for the M3 scorecard (PLAN.md section M3). No trained model."""
import numpy as np
from PIL import Image
from . import engine as E, m2b, facet_style
IDS = [t["id"] for t in E.load_templates("v0")]
KINDS = ["none", "hline", "vline", "diag", "point"]

def extract(path):
    im = np.asarray(Image.open(path).convert("RGB")); r = m2b.analyse(im); ev = E.evidence(im); f = {}
    fits = r["v0"]["fits"]
    for t in IDS: f["z_" + t] = fits[t]["score"]; f["r_" + t] = fits[t]["rank_score"]
    rd = r["v0"]["readings"]; k = r["kind"]
    for a in ("y_star", "y_strength", "x_star", "x_strength", "mass_cx", "mass_cy"): f[a] = rd[a]
    for a in ("hline", "vline", "diag"): f["cov_" + a] = k["coverage"][a]
    f["blob_peak"] = k["blob_peak"]; f["kind"] = KINDS.index(k["kind"])
    rb = r.get("rabatment"); f["z_rabatment"] = rb["score"] if rb else 0.0; f["rabatment_defined"] = float(rb is not None)
    for n_, v in r["golden_triangle"].items(): f["z_" + n_] = 0.0 if np.isnan(v["score"]) else v["score"]
    f["z_spiral"] = r["golden_spiral"]["score"]; f["spiral_variant"] = float(r["golden_spiral"]["best_variant"])
    pr = r["perspective"]["reading"]; fin = sorted([x["support"] for x in pr["families"] if x["finite"]], reverse=True)
    f["persp_n_finite"] = float(pr["n_finite"]); f["persp_n_parallel"] = float(pr["n_parallel"]); f["persp_support_finite"] = float(sum(fin)); f["persp_top1"] = float(fin[0]) if fin else 0.0; f["persp_top2"] = float(fin[1]) if len(fin) > 1 else 0.0
    # new named readings (M3)
    H, W = ev["H"], ev["W"]; blob = E._disc_share(ev); gy = np.arange(0.12, 0.8801, 0.02)
    best = max(((blob[int(round(y * (H - 1))), int(round(x * (W - 1)))], x, y) for y in gy for x in gy)); _, bx, by = best
    th = (1 / 3, 2 / 3); f["blob_x"], f["blob_y"] = float(bx), float(by); f["blob_d_centre"] = float(np.hypot(bx - .5, by - .5)); f["blob_d_power"] = float(min(np.hypot(bx - px, by - py) for px in th for py in th))
    ys, xs = rd["y_star"], rd["x_star"]; f["d_centre_h"] = abs(ys - 0.5); f["d_third_h"] = min(abs(ys - 1 / 3), abs(ys - 2 / 3)); f["d_centre_v"] = abs(xs - 0.5); f["d_third_v"] = min(abs(xs - 1 / 3), abs(xs - 2 / 3))
    f.update(facet_style.scores(im)); f["img_w"], f["img_h"] = float(im.shape[1]), float(im.shape[0])
    return f
