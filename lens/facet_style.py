"""Facet-style thirds / centre heuristic, REIMPLEMENTED from the public description and source reading of Facet's composition analyser
(github.com/ncoevoet/facet, analyzers/composition.py): an edge-based subject box, then the distance of its centre to the thirds lines/points and to the centre.
This is not Facet's code and uses a 384 px long side; it is a comparator only."""
import numpy as np, cv2
from . import engine as E

def subject_box(g):
    h, w = g.shape; g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8); med = float(np.median(g8))
    edges = cv2.Canny(g8, int(max(0, 0.5 * med)), int(min(255, 1.5 * med))); cs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, bs = None, 0.0
    for c in cs:
        a = cv2.contourArea(c)
        if a <= h * w * 0.0001: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]; dx = min(abs(cx - t) for t in (w / 3, 2 * w / 3)) / w; dy = min(abs(cy - t) for t in (h / 3, 2 * h / 3)) / h
        sc = a / (h * w) * (1 + max(0, 1 - (dx + dy)))
        if sc > bs: bs, best = sc, c
    if best is None: return None
    x, y, bw, bh = cv2.boundingRect(best); return [x, y, x + bw, y + bh]

def scores(image):
    g = E.prep(image); h, w = g.shape; box = subject_box(g)
    if box is None: return dict(facet_thirds=5.0, facet_center=7.0, facet_power=5.0, facet_has_box=0.0)
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h; th = (1 / 3, 2 / 3)
    dtx, dty = min(abs(cx - t) for t in th), min(abs(cy - t) for t in th); dcx = abs(cx - 0.5)
    pdist = min(np.hypot(cx - px, cy - py) for px in th for py in th)
    return dict(facet_thirds=max(0.0, 10 - (dtx + dty) * 20), facet_center=max(0.0, 10 - dcx * 20), facet_power=max(0.0, 10 - pdist * 25), facet_has_box=1.0)
