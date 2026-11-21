"""Generates LABBOOK.html for composition-lens from the files in results/ and catalogue/. Run: python3 -I labbook/make_labbook.py
Every number in the text is computed from the results files; nothing is typed in by hand except wording."""
import os, sys, json, hashlib, datetime, base64, io, html, re, random
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS = os.environ.get("LENS_LINKS") == "1"; NOTE_REF = '<a href="index.html">the short note</a>' if LINKS else "the short note (NOTE.html)"
NO_IMAGES = os.environ.get("LENS_NO_IMAGES") == "1"; OUTDIR = os.environ.get("LENS_OUT_DIR", HERE); PICD_ROOT = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
import numpy as np
from PIL import Image
sys.path.insert(0, HERE)
from lens import engine as E, synth, synth2
R = lambda f: os.path.join(HERE, "results", f); J = lambda f: json.load(open(R(f)))
esc = html.escape; f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; pct = lambda x: f"{100 * x:.0f}%"
BASE = PICD_ROOT
rec, m2b, devA, devB, m3b, m3c, m4 = J("recovery.json"), J("m2b.json"), J("m3_devA.json"), J("m3_devB.json"), J("m3b.json"), J("m3c.json"), J("m4.json")
cat = json.load(open(os.path.join(HERE, "catalogue", "templates.json"))); cw = json.load(open(os.path.join(HERE, "catalogue", "crosswalk.json")))
cal = J("m2b_calibration.json"); today = datetime.date.today().strftime("%d %b %Y")
fl = lambda p: [r for r in json.load(open(R(p))) if "error" not in r]
FA, FB, FC = fl("m3_features_devA.json"), fl("m3_features_devB.json"), fl("m3c_features_lockboxb.json")
spec = {k: open(R(f"{k}_spec.sha256")).read().split() for k in ("m3", "m3c", "m4")}
CAD = os.path.join(HERE, "data", "cadb", "CADB_Dataset"); cel = json.load(open(os.path.join(CAD, "composition_elements.json"))); F4 = [r for r in json.load(open(R("m4_features_cadb.json"))) if "error" not in r]; F4id = {r["id"]: r for r in F4}
plan = open(os.path.join(HERE, "PLAN.md")).read()
ABBR = {1: "P-RoT", 2: "S-RoT", 3: "P-Cent", 4: "S-Cent", 5: "P-Dia", 6: "LS-Dia", 7: "P-Hori", 8: "LS-Hori2", 9: "LS-Hori3", 10: "S-Hori", 11: "P-Ver", 12: "L-Ver2", 13: "L-Ver3", 14: "L-VerMul", 15: "P-Tri", 16: "LS-Tri", 17: "LS-C-Cur", 18: "LS-O-Cur", 19: "LS-S-Cur", 20: "LS-Dif", 21: "S-Per", 22: "PL-Den", 23: "PL-Pat", 24: "P-Scat"}
MAPPED = [a for a, d in devB["per_category"].items() if "lens_auc" in d]
PAIRS = ["LS-Hori3 vs LS-Hori2", "L-Ver3 vs L-Ver2", "P-RoT vs P-Cent"]

# ------------------------------------------------------------------ svg kit
def svg(w, h, body, label): return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'
def T(x, y, s, cls="t", anchor="start", extra=""): return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(str(s))}</text>'
def table(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
FIGN = [0]; FNUM = {}; F = {}
ARGS = {}
def add(k, *a, **kw):
    ARGS[k] = (a, kw); F[k] = figure(*a, **kw); FNUM[k] = FIGN[0]
def figure(title, caption, built, reading=""):
    FIGN[0] += 1; n = FIGN[0]; sv, head, rows = built
    rd = f'<p class="read"><b>Reading.</b> {reading}</p>' if reading else ""
    return f'<figure id="fig{n}"><figcaption><b>Figure {n}.</b> {title}</figcaption>{sv}{rd}<p class="cap">{caption}</p><details><summary>Data table for figure {n}</summary>{table([esc(h) for h in head], [[esc(str(c)) for c in r] for r in rows])}</details></figure>'
def axis_x(b, x0, w, ymax, ticks, lo, hi, fmt="{:.1f}", cls="ts"):
    for t in ticks:
        x = x0 + w * (t - lo) / (hi - lo); b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ymax-0}" y2="{ymax+4}" class="gl"/>' + T(x, ymax + 16, fmt.format(t), cls, "middle"))
# ------------------------------------------------------------------ figures
def fig_coverage():
    cats = [ABBR[k] for k in range(1, 25)]; v0 = {a for a, ids in cw["picd"].items() if any(next(t for t in cat if t["id"] == i)["status"] == "v0" for i in ids)}
    anyt = {a for a, ids in cw["picd"].items() if ids}; w, h = 720, 120; b = []; cw_ = 28
    for i, a in enumerate(cats):
        x = 10 + i * cw_; cls = "cv0" if a in v0 else ("cm" if a in anyt else "cn"); b.append(f'<rect x="{x}" y="30" width="{cw_-3}" height="34" rx="3" class="{cls}"/>' + f'<text x="{x+cw_/2-1.5}" y="86" class="ts" text-anchor="end" transform="rotate(-50 {x+cw_/2-1.5} 86)">{esc(a)}</text>')
    b.append(T(10, 18, "PICD categories (label order)", "ts")); b.append('<rect x="330" y="8" width="10" height="10" rx="2" class="cv0"/>' + T(345, 17, "has a v0 template", "ts") + '<rect x="450" y="8" width="10" height="10" rx="2" class="cm"/>' + T(465, 17, "template planned (M2b)", "ts") + '<rect x="590" y="8" width="10" height="10" rx="2" class="cn"/>' + T(605, 17, "no template", "ts"))
    rows = [[a, ", ".join(cw["picd"][a]) or "none", "v0" if a in v0 else ("planned" if a in anyt else "none")] for a in cats]
    return svg(w, h, "".join(b), "Strip of the 24 PICD categories coloured by whether a template covers them"), ["PICD category", "templates", "coverage"], rows
def fig_synth():
    keys = list(rec["R2"].keys()); labs = ["noise 0", "noise 0.02", "noise 0.05", "contrast 0.10", "contrast 0.05"]; w, h = 720, 250; b = []; x0, x1 = 70, 700; ys = lambda v: 200 - 150 * (v - 0.5) / 0.5
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0): b.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="gl"/>' + T(x0 - 8, ys(v) + 4, f"{v:.1f}", "ts", "end"))
    rows = []
    for i, k in enumerate(keys):
        xc = x0 + 70 + i * 120; r2 = rec["R2"][k]["within_kind"]; vals = [("rank", r2["rank_scores"]["mean"], "s1"), ("z", r2["scores"]["mean"], "s3"), ("all templates (z)", rec["R2"][k]["mean_top1"], "s2")]
        for j, (nm, v, cls) in enumerate(vals):
            x = xc + (j - 1) * 30; b.append(f'<path d="M{x-11} {ys(0.5):.1f} V{ys(v)+4:.1f} a4 4 0 0 1 4 -4 H{x+7} a4 4 0 0 1 4 4 V{ys(0.5):.1f} Z" class="{cls}"/>' + T(x, ys(v) - 5, f2(v), "ts", "middle"))
        b.append(T(xc, 224, labs[i], "ts", "middle")); rows.append([labs[i], f3(r2["rank_scores"]["mean"]), f3(r2["scores"]["mean"]), f3(rec["R2"][k]["mean_top1"])])
    b.append(T(x0, 20, "Top-1 template recovery on exact synthetic layouts (13 v0 templates; 4 aspect ratios)", "ts")); b.append('<rect x="170" y="238" width="10" height="10" class="s1"/>' + T(185, 247, "within kind, rank score", "ts") + '<rect x="350" y="238" width="10" height="10" class="s3"/>' + T(365, 247, "within kind, z score", "ts") + '<rect x="500" y="238" width="10" height="10" class="s2"/>' + T(515, 247, "all templates, z score", "ts"))
    return svg(w, h + 14, "".join(b), "Bar chart of template recovery accuracy across five synthetic conditions"), ["condition", "within kind (rank score)", "within kind (z score)", "all templates (z score)"], rows
def fig_sweep():
    sw = rec["R4"]["sweep"]; ps = [float(p) for p in sw]; w, h = 720, 250; x0, x1 = 60, 690; xs = lambda p: x0 + (x1 - x0) * (p - 0.30) / 0.10; ys = lambda v: 200 - 160 * v; b = []
    for v in (0, 0.25, 0.5, 0.75, 1.0): b.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(v)}" y2="{ys(v)}" class="gl"/>' + T(x0 - 8, ys(v) + 4, f"{v:.2f}", "ts", "end"))
    for p in (0.30, 0.32, 0.34, 0.36, 0.38, 0.40): b.append(T(xs(p), 218, f"{p:.2f}", "ts", "middle"))
    for p, lab in ((1 / 3, "thirds 0.333"), (0.381966, "phi 0.382")): b.append(f'<line x1="{xs(p)}" x2="{xs(p)}" y1="30" y2="200" class="chance"/>' + T(xs(p), 26, lab, "ts", "middle"))
    for key, cls, nm in (("thirds_h", "ls2", "thirds_h wins"), ("phi_h", "ls1", "phi_h wins")):
        pts = " ".join(f"{xs(p):.1f},{ys(sw[str(p)]['winner_share'][key]):.1f}" for p in ps); b.append(f'<polyline points="{pts}" class="ln {cls}"/>')
    b.append(T(x0, 246, "position of a single synthetic horizontal line (fraction of the height)", "ts")); b.append('<line x1="470" x2="486" y1="240" y2="240" class="ln ls2"/>' + T(491, 244, "thirds_h wins", "ts") + '<line x1="590" x2="606" y1="240" y2="240" class="ln ls1"/>' + T(611, 244, "phi_h wins", "ts"))
    rows = [[p, f3(v["median_pos_err"]), f2(v["winner_share"]["thirds_h"]), f2(v["winner_share"]["phi_h"]), f2(v["winner_share"]["centre_h"])] for p, v in sw.items()]
    return svg(w, h + 10, "".join(b), "Share of images where each template wins as a single line moves from 0.30 to 0.40"), ["line position", "median position error", "thirds_h wins", "phi_h wins", "centre_h wins"], rows
def fig_confusion():
    c = rec["R2"]["noise0.02_contrastdefault"]["confusion"]; ids = list(c); n = len(ids); cell = 36; ml, mt = 90, 80; w, h = ml + n * cell + 20, mt + n * cell + 30; b = []; rows = []
    for i, a in enumerate(ids):
        tot = sum(c[a].values()); b.append(T(ml - 6, mt + i * cell + 22, a, "ts", "end")); b.append(f'<text x="{ml + i*cell + 20}" y="{mt - 8}" class="ts" text-anchor="start" transform="rotate(-55 {ml + i*cell + 20} {mt - 8})">{esc(ids[i])}</text>')
        for j, bb in enumerate(ids):
            v = c[a][bb] / tot; k = int(round(7 * v)); b.append(f'<rect x="{ml+j*cell}" y="{mt+i*cell}" width="{cell-2}" height="{cell-2}" rx="3" fill="var(--h{k})"><title>true {a} read as {bb}: {v:.2f}</title></rect>')
            if v >= 0.1 and a != bb or (a == bb and v < 0.9): b.append(T(ml + j * cell + cell / 2 - 1, mt + i * cell + 22, f"{v:.2f}".lstrip("0"), "tc" if k >= 4 else "ts", "middle"))
            rows.append([a, bb, f3(v)]) if v > 0 else None
    b.append(T(ml, h - 8, "rows: template the image was built on; columns: template with the highest z score (over all 13)", "ts"))
    return svg(w, h, "".join(b), "Confusion matrix of the 13 position templates on exact synthetic layouts"), ["true template", "read as", "share"], rows
def fig_m2b():
    w, h = 720, 250; b = []; ks = [("default", "default contrast"), ("contrast0.1", "contrast 0.10"), ("contrast0.05", "contrast 0.05")]; x0 = 70; ys = lambda v: 190 - 150 * v; rows = []
    for v in (0, 0.25, 0.5, 0.75, 1.0): b.append(f'<line x1="{x0}" x2="700" y1="{ys(v)}" y2="{ys(v)}" class="gl"/>' + T(x0 - 8, ys(v) + 4, f"{v:.2f}", "ts", "end"))
    for i, (k, lab) in enumerate(ks):
        e = m2b["E1"][k]; xc = x0 + 100 + i * 200
        for j, (nm, v, cls) in enumerate((("kind accuracy", e["kind_accuracy"], "s1"), ("arbitrated top-1", e["arbitrated_top1_rank"], "s3"), ("unarbitrated top-1", e["unarbitrated_top1_rank"], "s2"))):
            x = xc + (j - 1) * 38; b.append(f'<path d="M{x-14} {ys(0):.1f} V{ys(v)+4:.1f} a4 4 0 0 1 4 -4 H{x+10} a4 4 0 0 1 4 4 V{ys(0):.1f} Z" class="{cls}"/>' + T(x, ys(v) - 5, f2(v), "ts", "middle"))
        b.append(T(xc, 210, lab, "ts", "middle")); rows.append([lab, f3(e["kind_accuracy"]), f3(e["arbitrated_top1_rank"]), f3(e["unarbitrated_top1_rank"])])
    b.append('<rect x="170" y="232" width="10" height="10" class="s1"/>' + T(185, 241, "kind (line vs compact) accuracy", "ts") + '<rect x="400" y="232" width="10" height="10" class="s3"/>' + T(415, 241, "top-1 among the detected kind", "ts") + '<rect x="590" y="232" width="10" height="10" class="s2"/>' + T(605, 241, "top-1 over all", "ts"))
    return svg(w, h + 10, "".join(b), "Bar chart of kind-reading accuracy and arbitrated top-1 at three contrast levels"), ["condition", "kind accuracy", "arbitrated top-1", "unarbitrated top-1"], rows
def fig_spiral():
    sh = m2b["E4"]["shift_sweep"]; xs_ = [0.0, 0.01, 0.02, 0.04, 0.08]; auc_ = [m2b["E4"]["auc_vs_nulls"]] + [sh[str(s)]["auc_vs_nulls"] for s in xs_[1:]]; acc = [m2b["E4"]["variant_accuracy"]] + [sh[str(s)]["variant_accuracy"] for s in xs_[1:]]
    w, h = 720, 230; x0, x1 = 70, 690; xs = lambda s: x0 + (x1 - x0) * s / 0.08; ys = lambda v: 170 - 150 * (v - 0.0) / 1.0; b = []
    for v in (0, 0.25, 0.5, 0.75, 1.0): b.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(v)}" y2="{ys(v)}" class="gl"/>' + T(x0 - 8, ys(v) + 4, f"{v:.2f}", "ts", "end"))
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(0.125)}" y2="{ys(0.125)}" class="chance"/>' + T(x1, ys(0.125) - 4, "chance for 8 variants", "ts", "end"))
    for s in xs_: b.append(T(xs(s), 190, f"{100*s:.0f}%", "ts", "middle"))
    for vals, cls in ((auc_, "ls2"), (acc, "ls1")): b.append(f'<polyline points="{" ".join(f"{xs(s):.1f},{ys(v):.1f}" for s, v in zip(xs_, vals))}" class="ln {cls}"/>' + "".join(f'<circle cx="{xs(s):.1f}" cy="{ys(v):.1f}" r="4" class="dot {cls.replace("ls", "s")}"/>' for s, v in zip(xs_, vals)))
    b.append(T(x0, 212, "spiral displaced by this share of the short side", "ts")); b.append('<line x1="470" x2="486" y1="206" y2="206" class="ln ls2"/>' + T(491, 210, "AUC against nulls", "ts") + '<line x1="590" x2="606" y1="206" y2="206" class="ln ls1"/>' + T(611, 210, "variant accuracy", "ts"))
    return svg(w, h + 6, "".join(b), "Line chart of spiral detection and variant accuracy against displacement"), ["displacement (share of short side)", "AUC against nulls", "variant accuracy"], [[f"{100*s:.0f}%", f3(a), f3(c)] for s, a, c in zip(xs_, auc_, acc)]
def auc_dots(rows_, series, width=720, label_w=110, title=None, h_row=26, x_lo=0.2, x_hi=1.0, spread=0):
    n = len(rows_); w = width; h = 60 + h_row * n + 50; x0, x1 = label_w + 10, w - 20; xs = lambda v: x0 + (x1 - x0) * (v - x_lo) / (x_hi - x_lo); b = []
    for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        if x_lo <= v <= x_hi: b.append(f'<line x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="42" y2="{42+h_row*n}" class="gl"/>' + T(xs(v), 42 + h_row * n + 14, f"{v:.1f}", "ts", "middle"))
    b.append(f'<line x1="{xs(0.5):.1f}" x2="{xs(0.5):.1f}" y1="38" y2="{42+h_row*n}" class="chance"/>')
    if title: b.append(T(x0, 18, title, "ts"))
    for i, rw in enumerate(rows_):
        y = 42 + h_row * i + h_row / 2; y0c = y; b.append(T(label_w, y + 4, rw["label"], "t" if not rw.get("bold") else "tb", "end"))
        for si, (key, cls, shape) in enumerate(series):
            if key in rw and rw[key] is not None:
                v = rw[key]; ci = rw.get(key + "_ci"); y = y0c + (si - (len(series) - 1) / 2) * spread
                if ci: b.append(f'<line x1="{xs(ci[0]):.1f}" x2="{xs(ci[1]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="eb {cls}"/>')
                b.append(f'<circle cx="{xs(v):.1f}" cy="{y:.1f}" r="{4.5 if shape == "o" else 3.5}" class="dot {cls}"/>' if shape == "o" else f'<rect x="{xs(v)-3.5:.1f}" y="{y-3.5:.1f}" width="7" height="7" class="dot {cls}"/>')
    return svg(w, h, "".join(b), title or "dot plot"), n
def legend_row(items, y, step=175):
    b = ""
    for i, (cls, lab, shape) in enumerate(items):
        x = 14 + i * step; b += (f'<circle cx="{x}" cy="{y}" r="4.5" class="dot {cls}"/>' if shape == "o" else f'<rect x="{x-3.5}" y="{y-3.5}" width="7" height="7" class="dot {cls}"/>') + T(x + 10, y + 4, lab, "ts")
    return b
def fig_replicate():
    rows_ = []; tab = []
    for a in MAPPED:
        dA, dB = devA["per_category"][a], devB["per_category"][a]; dC = m3c["per_category"][a]
        rows_.append(dict(label=f"{a}  ({dC['template']})", a=dA["lens_auc"], b=dB["lens_auc"], b_ci=dB["lens_ci"], c=dC["lens_auc"], c_ci=dC["lens_ci"], bold=not dC["prediction_met"]))
        tab.append([a, dC["template"], f3(dA["lens_auc"]), f"{f3(dB['lens_auc'])} [{f3(dB['lens_ci'][0])}, {f3(dB['lens_ci'][1])}]", f"{f3(dC['lens_auc'])} [{f3(dC['lens_ci'][0])}, {f3(dC['lens_ci'][1])}]", "met" if dC["prediction_met"] else "not met"])
    sv, n = auc_dots(rows_, [("a", "s7", "s"), ("b", "s1", "o"), ("c", "s2", "o")], label_w=190, title="One-vs-rest AUC of the training-free template score for the 11 mapped categories", x_lo=0.3, x_hi=0.95, spread=5)
    sv = sv.replace("</svg>", legend_row([("s7", "DevA (design)", "s"), ("s1", "DevB (confirmation)", "o"), ("s2", "LockboxB (fresh)", "o")], 60 + 26 * n + 22, 200) + "</svg>")
    return sv, ["category", "template", "DevA", "DevB [95% interval]", "LockboxB [95% interval]", "LockboxB prediction"], tab
def fig_methods():
    rows_ = []; tab = []
    for k in range(1, 25):
        a = ABBR[k]; d = m3c["per_category"][a]
        rows_.append(dict(label=a, lens=d.get("lens_auc"), facet=d.get("facet_style_auc"), fl=d["fitted_lens_auc"], fd=d["fitted_dinov3_auc"], bold=False))
        tab.append([a, f3(d["lens_auc"]) if "lens_auc" in d else "-", f3(d["facet_style_auc"]) if "facet_style_auc" in d else "-", f3(d["fitted_lens_auc"]), f3(d["fitted_dinov3_auc"])])
    sv, n = auc_dots(rows_, [("fd", "s7", "s"), ("fl", "s3", "s"), ("facet", "sm", "o"), ("lens", "s2", "o")], label_w=80, title="LockboxB (fresh images): one-vs-rest AUC for each method, per PICD category", x_lo=0.3, x_hi=1.0, h_row=24, spread=4)
    sv = sv.replace("</svg>", legend_row([("s2", "lens template (training-free)", "o"), ("sm", "Facet-style heuristic", "o"), ("s3", "fitted probe, lens features", "s"), ("s7", "fitted probe, DINOv3", "s")], 60 + 24 * n + 22, 175) + "</svg>")
    return sv, ["category", "lens template", "Facet-style", "fitted lens probe", "fitted DINOv3 probe"], tab
def _hist(b, x0, y0, pw, ph, a1, a2, bins, refs, n1, n2, title):
    h1, _ = np.histogram(a1, bins); h2, _ = np.histogram(a2, bins); mx = max(h1.max(), h2.max(), 1); lo, hi = bins[0], bins[-1]; xs = lambda v: x0 + pw * (v - lo) / (hi - lo)
    b.append(T(x0, y0 - 16, title, "ts") + f'<line x1="{x0}" x2="{x0+pw}" y1="{y0+ph}" y2="{y0+ph}" class="gl"/>')
    for v, lab in refs: b.append(f'<line x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="{y0}" y2="{y0+ph}" class="chance"/>' + T(xs(v), y0 + ph + 14, lab, "ts", "middle"))
    for i in range(len(h1)):
        bw = xs(bins[i + 1]) - xs(bins[i]); b.append(f'<rect x="{xs(bins[i])+0.5:.1f}" y="{y0+ph-ph*h1[i]/mx:.1f}" width="{bw/2-0.5:.1f}" height="{ph*h1[i]/mx:.1f}" class="s1"/><rect x="{xs(bins[i])+bw/2:.1f}" y="{y0+ph-ph*h2[i]/mx:.1f}" width="{bw/2-0.5:.1f}" height="{ph*h2[i]/mx:.1f}" class="s2"/>')
    b.append(f'<circle cx="{x0+8}" cy="{y0+ph+40}" r="5" class="dot s1"/>' + T(x0 + 17, y0 + ph + 44, n1, "ts") + f'<circle cx="{x0+80}" cy="{y0+ph+40}" r="5" class="dot s2"/>' + T(x0 + 89, y0 + ph + 44, n2, "ts"))
def fig_hist(sel=(0, 1, 2)):
    def v(F, key, lab): return np.array([r[key] for r in F if r["label"] == lab])
    P = [("Strongest horizontal line: height", "y_star", 8, 9, "LS-Hori2", "LS-Hori3", np.linspace(0, 1, 21), [(1 / 3, "1/3"), (0.5, "1/2"), (2 / 3, "2/3")]), ("Strongest vertical line: position", "x_star", 12, 13, "L-Ver2", "L-Ver3", np.linspace(0, 1, 21), [(1 / 3, "1/3"), (0.5, "1/2"), (2 / 3, "2/3")]),
         ("Compact object: distance from centre", "blob_d_centre", 3, 1, "P-Cent", "P-RoT", np.linspace(0, 0.6, 16), [(0.0, "centre"), (2 ** 0.5 / 6, "power point")])]
    P = [P[i] for i in sel]; w, h = 720, 235; b = []; rows = []
    for i, (t, key, c1, c2, n1, n2, bins, refs) in enumerate(P):
        a1, a2 = v(FC, key, c1), v(FC, key, c2); _hist(b, 14 + i * 236, 48, 190, 120, a1, a2, bins, refs, n1, n2, t); rows.append([t, n1, f"{np.median(a1):.3f}", n2, f"{np.median(a2):.3f}"])
    return svg(w, h, "".join(b), "Histograms of the dominant line or object position for the two categories of each pair, LockboxB"), ["reading", "category 1", "median", "category 2", "median"], rows
def fig_pairs(keys=None):
    rows_ = []; tab = []
    for k in (keys or PAIRS):
        a, bb, c = devA["pairs"][k], devB["pairs"][k], m3c["pairs"][k]
        rows_.append(dict(label=k, a=a["auc"], b=bb["auc"], b_ci=bb["ci"], c=c["auc"], c_ci=c["ci"], d=bb["fitted_dinov3_pair_auc"]))
        tab.append([k, bb["score"], f3(a["auc"]), f"{f3(bb['auc'])} [{f3(bb['ci'][0])}, {f3(bb['ci'][1])}]", f"{f3(c['auc'])} [{f3(c['ci'][0])}, {f3(c['ci'][1])}]", f3(bb["fitted_dinov3_pair_auc"]), pct(c["detection_rate"])])
    sv, n = auc_dots(rows_, [("d", "sm", "o"), ("a", "s7", "s"), ("b", "s1", "o"), ("c", "s2", "o")], label_w=150, title=("The two line pairs: AUC of the position reading on three sets" if keys and len(keys) == 2 else "The three position pairs: AUC of the position reading on three sets"), x_lo=0.4, x_hi=1.0, h_row=40, spread=6)
    sv = sv.replace("</svg>", legend_row([("sm", "pooled DINOv3 probe (DevB)", "o"), ("s7", "DevA (design)", "s"), ("s1", "DevB (confirmation)", "o"), ("s2", "LockboxB (fresh)", "o")], 60 + 40 * n + 22, 175) + "</svg>")
    return sv, ["pair", "score", "DevA", "DevB [95% interval]", "LockboxB [95% interval]", "pooled DINOv3 probe, DevB", "detection rate, LockboxB"], tab
def fig_curves(keys=None):
    pairs = [("LS-Hori3 vs LS-Hori2", "Horizon: middle vs third"), ("L-Ver3 vs L-Ver2", "Vertical line: middle vs third"), ("P-RoT vs P-Cent", "Compact object: centre vs power point")]; ns = [10, 20, 40, 70, 100]
    series = [("D-pool", "DINOv3 pooled", "ls7"), ("D-grid", "DINOv3 4x4 patch grid", "ls1"), ("D-norm", "DINOv3 patch-norm map", "ls3"), ("G-bright", "brightness grid", "lsm"), ("G-edge", "edge-strength grid", "ls2")]
    w, h = 720, 300; b = []; pw, ph = 190, 150; rows = []
    for pi, (key, title) in enumerate([p_ for p_ in pairs if keys is None or p_[0] in keys]):
        x0, y0 = 40 + pi * 232, 40; d = m3b["P2"][key]; xs = lambda n: x0 + pw * n / 140.0; ys = lambda v: y0 + ph * (1.0 - (v - 0.5) / 0.5)
        b.append(T(x0 - 22, 22, title, "ts"))
        for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0): b.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{ys(v)}" y2="{ys(v)}" class="gl"/>' + (T(x0 - 6, ys(v) + 4, f"{v:.1f}", "ts", "end") if pi == 0 else ""))
        for n in ns + [138]: b.append(T(xs(n), y0 + ph + 14, str(n), "ts", "middle"))
        rd = d["100"]["reading"][0]; b.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{ys(rd)}" y2="{ys(rd)}" class="lsk"/>')
        for k, lab, cls in series:
            pts = " ".join(f"{xs(n):.1f},{ys(d[str(n)][k][0]):.1f}" for n in ns); b.append(f'<polyline points="{pts}" class="ln {cls}"/>' + "".join(f'<circle cx="{xs(n):.1f}" cy="{ys(d[str(n)][k][0]):.1f}" r="3" class="dot dd {cls}"/>' for n in ns))
            fv = m3c["baselines"][key][k]["auc"]; b.append(f'<rect x="{xs(138)-4:.1f}" y="{ys(fv)-4:.1f}" width="8" height="8" class="dot dd {cls}" transform="rotate(45 {xs(138):.1f} {ys(fv):.1f})"/>')
            rows.append([key, k, *[f"{d[str(n)][k][0]:.3f} +/- {d[str(n)][k][1]:.3f}" for n in ns], f3(fv)])
        rows.append([key, "position reading", *[f"{d[str(n)]['reading'][0]:.3f} +/- {d[str(n)]['reading'][1]:.3f}" for n in ns], f3(m3c["pairs"][key]["auc"])])
        b.append(T(x0 + pw / 2, y0 + ph + 30, "labelled training images per category", "ts", "middle"))
    ly = 242
    for i, (k, lab, cls) in enumerate(series): x = 24 + (i % 3) * 232; y = ly + (i // 3) * 18; b.append(f'<line x1="{x}" x2="{x+16}" y1="{y}" y2="{y}" class="ln {cls}"/>' + T(x + 22, y + 4, lab, "ts"))
    b.append(f'<line x1="488" x2="504" y1="{ly+18}" y2="{ly+18}" class="lsk"/>' + T(510, ly + 22, "position reading (no training)", "ts")); b.append(T(24, ly + 42, "Lines: mean of 10 random splits on the pool (test 38 per category). Diamonds at 138: trained on the whole pool, tested on LockboxB (40 per category).", "ts"))
    return svg(w, h + 20, "".join(b), "Learning curves for five trained probes against the position reading, with LockboxB points"), ["pair", "method", "10", "20", "40", "70", "100", "LockboxB (train 138)"], rows
def fig_overlays(plan=((8, "h"), (9, "h"), (12, "v"), (13, "v"), (3, "p"), (1, "p"))):
    if NO_IMAGES: return '<p class="small">Example photographs are omitted in this edition (they are shown, with credit, in the edition on parallaxmetrology.com/composition-lens/).</p>', ["category", "image id", "reading"], []
    rng = random.Random(7); pk = json.load(open(f"{BASE}/data/pick_lockboxb.json")); byid = {r["id"]: r for r in FC}; out = []; tab = []
    for cat_id, kind in plan:
        ids = [p["img_id"] for p in pk if p["cat"] == cat_id]; iid = rng.choice(ids); r = byid[iid]; im = Image.open(f"{BASE}/data/lockboxb/{iid}").convert("RGB"); im.thumbnail((300, 300)); bf = io.BytesIO(); im.save(bf, "JPEG", quality=72); W, H = im.size
        src = "data:image/jpeg;base64," + base64.b64encode(bf.getvalue()).decode(); ov = ""
        refs_h = "".join(f'<line x1="0" x2="{W}" y1="{H*t:.1f}" y2="{H*t:.1f}" class="ovr"/>' for t in (1 / 3, 0.5, 2 / 3)); refs_v = "".join(f'<line y1="0" y2="{H}" x1="{W*t:.1f}" x2="{W*t:.1f}" class="ovr"/>' for t in (1 / 3, 0.5, 2 / 3))
        if kind == "h":
            y = r["y_star"]; ov = refs_h + f'<line x1="0" x2="{W}" y1="{H*y:.1f}" y2="{H*y:.1f}" class="ovl"/>'; nearest = min((abs(y - t), n) for t, n in ((1 / 3, "the upper third"), (0.5, "the middle"), (2 / 3, "the lower third")))
            txt = f"Strongest horizontal line at {y:.2f} of the height: {nearest[0]:.2f} from {nearest[1]}."
        elif kind == "v":
            x = r["x_star"]; ov = refs_v + f'<line y1="0" y2="{H}" x1="{W*x:.1f}" x2="{W*x:.1f}" class="ovl"/>'; nearest = min((abs(x - t), n) for t, n in ((1 / 3, "the left third"), (0.5, "the middle"), (2 / 3, "the right third")))
            txt = f"Strongest vertical line at {x:.2f} of the width: {nearest[0]:.2f} from {nearest[1]}."
        else:
            bx, by = r["blob_x"], r["blob_y"]; ov = refs_h + refs_v + f'<circle cx="{W*bx:.1f}" cy="{H*by:.1f}" r="{0.07*min(W,H):.1f}" class="ovc"/>'
            txt = f"Most compact object at ({bx:.2f}, {by:.2f}): {r['blob_d_centre']:.2f} from the centre, {r['blob_d_power']:.2f} from the nearest power point."
        out.append(f'<div class="ovcard"><svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Photograph with measured position drawn over it"><image href="{src}" width="{W}" height="{H}"/>{ov}</svg><div class="ovt"><b>{ABBR[cat_id]}</b> <span class="ids">{iid[:-4]}</span><br>{txt}</div></div>'); tab.append([ABBR[cat_id], iid[:-4], txt])
    return '<div class="ovgrid">' + "".join(out) + "</div>", ["category", "image id", "reading"], tab


def _b64(arr_or_im, px=230, q=74):
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(arr_or_im); im = im.convert("RGB"); im.thumbnail((px, px)); b = io.BytesIO(); im.save(b, "JPEG", quality=q); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), im.size
def fig_how():
    if NO_IMAGES:
        iid = "synthetic"; im = Image.fromarray(synth.render("hline", 0.37, "4:3", 0.02, 21)); arr = np.asarray(im)
    else:
        pk = json.load(open(f"{BASE}/data/pick_lockboxb.json")); iid = random.Random(11).choice([p["img_id"] for p in pk if p["cat"] == 9]); im = Image.open(f"{BASE}/data/lockboxb/{iid}").convert("RGB"); arr = np.asarray(im)
    ev = E.evidence(arr); Ph = E._profile_h(ev); H = len(Ph); ys = (np.arange(H) + 0.5) / H; fits = E.readings_and_fits(arr)
    src, (W, Hh) = _b64(im, px=300); ph = 260; iw = ph * W / Hh; x0 = 14; px0 = x0 + iw + 24; pw = 720 - px0 - 44; yoff = 20; mx = Ph.max(); b = [f'<image href="{src}" x="{x0}" y="{yoff}" width="{iw:.1f}" height="{ph}"/>']
    for t, lab in ((1 / 3, "thirds_h / phi_h region"), (0.5, "centre_h"), (2 / 3, "thirds_h")):
        y = yoff + ph * t; b.append(f'<line x1="{x0}" x2="{px0+pw}" y1="{y:.1f}" y2="{y:.1f}" class="chance"/>' + T(px0 + pw + 4, y + 4, {1 / 3: "1/3", 0.5: "1/2", 2 / 3: "2/3"}[t], "ts"))
        b.append(f'<rect x="{px0}" y="{y-ph*E.TOL:.1f}" width="{pw}" height="{2*ph*E.TOL:.1f}" class="band"/>')
    pts = " ".join(f"{px0 + pw * Ph[i] / mx:.1f},{yoff + ph * ys[i]:.1f}" for i in range(H)); b.append(f'<polyline points="{pts}" class="ln ls1"/>')
    ystar = fits["readings"]["y_star"]; b.append(f'<line x1="{x0}" x2="{px0+pw}" y1="{yoff+ph*ystar:.1f}" y2="{yoff+ph*ystar:.1f}" class="ovl2"/>' + T(px0 + pw - 4, yoff + ph * ystar - 5, f"y* = {ystar:.2f}", "tb", "end"))
    b.append(T(px0, 14, "row profile: mean |vertical gradient| of each row", "ts")); b.append(T(x0, yoff + ph + 18, ("synthetic image (a tone split at 0.37 of the height); " if NO_IMAGES else f"LS-Hori3 image {iid[:-4]} (seeded draw); ") + "shaded bands: +/- 1.5% tolerance around the template positions", "ts"))
    fz = fits["fits"]; txt = "  ".join(f"{k} z = {fz[k]['score']:.1f}" for k in ("thirds_h", "centre_h", "phi_h", "fifths_h"))
    b.append(T(x0, yoff + ph + 36, "template fits for this image: " + txt, "ts"))
    rows = [[k, f"{fz[k]['score']:.2f}", f"{fz[k]['rank_score']:.2f}"] for k in IDS_H] + [["y* (dominant horizontal line)", f"{ystar:.3f}", ""]]
    global HOWTXT; bk = max(IDS_H, key=lambda k: fz[k]["score"]); HOWTXT = ("In this edition the example is a synthetic image. " if NO_IMAGES else "") + f"One measurement (the profile) serves every horizontal template; the thirds, centre, phi and fifths scores differ only in which positions they read. Here the strongest line sits at {ystar:.2f} of the height and {bk} scores highest (z {fz[bk]['score']:.1f}); a line near the lower or upper third can sit nearer the phi position than the thirds one, which are only 4.9% of the frame apart (see the thirds-to-phi sweep below)."
    return svg(720, yoff + ph + 48, "".join(b), "A photograph beside its row profile with the template positions marked"), ["template / reading", "z score", "rank score"], rows
IDS_H = ["thirds_h", "centre_h", "phi_h", "fifths_h"]
def fig_gallery():
    items = [("horizontal line at 1/3", synth.render("hline", 1 / 3, "4:3", 0.02, 1)), ("vertical line at 0.382 (phi)", synth.render("vline", 0.381966, "4:3", 0.02, 2)), ("compact object on a power point", synth.render("point", (2 / 3, 1 / 3), "4:3", 0.02, 3)),
             ("diagonal tone split (TL-BR)", synth.render("diag", "TL-BR", "4:3", 0.02, 4)), ("golden-triangle strokes", synth2.render_triangle("gt", "TLBR", "4:3", 0.02, 5)), ("golden spiral, variant 5", synth2.render_spiral(5, "4:3", 0.02, 6)),
             ("two-point perspective drawing", synth2.render_perspective(2, "4:3", 0.0, 7, 0.3)[0]), ("point at contrast 0.05 (hard)", synth.render("point", (0.5, 0.5), "4:3", 0.02, 8, contrast=0.05))]
    out = []; rows = []
    for lab, a in items:
        src, (w_, h_) = _b64(a, px=240); out.append(f'<div class="ovcard"><img src="{src}" alt="{esc(lab)}" style="width:100%;border-radius:4px"><div class="ovt">{esc(lab)}</div></div>'); rows.append([lab])
    return '<div class="ovgrid">' + "".join(out) + "</div>", ["synthetic layout"], rows


def fig_cadb_classes():
    order = ["center", "horizontal", "diagonal", "vertical", "rule_of_thirds", "triangle", "symmetric", "golden_ratio", "curved", "fill_the_frame", "vanishing_point", "pattern", "radial"]; rows_ = []; tab = []
    for c in order:
        d = m4["classes"][c]; rows_.append(dict(label=f"{c}  ({d['n_test_pos']})", lens=d.get("lens_auc_test"), lens_ci=d.get("lens_ci_test"), facet=d.get("facet_auc_test"), fl=d.get("fitted_lens_auc"), fd=d.get("fitted_dinov3_auc"), bold=("met_test" in d and not d["met_test"])))
        tab.append([c, d["n_test_pos"], d.get("score", "-"), (f"{d['lens_auc_test']:.3f} [{d['lens_ci_test'][0]:.3f}, {d['lens_ci_test'][1]:.3f}]" if "lens_auc_test" in d else "-"), (f"{d['lens_auc_all']:.3f}" if "lens_auc_all" in d else "-"), (f"{d['facet_auc_test']:.3f}" if "facet_auc_test" in d else "-"), (f"{d['fitted_lens_auc']:.3f}" if "fitted_lens_auc" in d else "-"), (f"{d['fitted_dinov3_auc']:.3f}" if "fitted_dinov3_auc" in d else "-")])
    sv, n = auc_dots(rows_, [("fd", "s7", "s"), ("fl", "s3", "s"), ("facet", "sm", "o"), ("lens", "s2", "o")], label_w=170, title="CADB test split (950 images): one-vs-rest AUC per composition class (positives in brackets)", x_lo=0.4, x_hi=1.0, h_row=26, spread=4)
    sv = sv.replace("</svg>", legend_row([("s2", "lens template (95% interval)", "o"), ("sm", "Facet-style heuristic", "o"), ("s3", "fitted probe, lens features", "s"), ("s7", "fitted probe, DINOv3", "s")], 60 + 26 * n + 22, 178) + "</svg>")
    return sv, ["class", "positives (test)", "lens score", "lens AUC test [95% interval]", "lens AUC all 9,497", "Facet-style", "fitted lens probe", "fitted DINOv3 probe"], tab
def fig_cadb_agree():
    def pts(cls, key, dim):
        xs, ys = [], []
        for i, d in cel.items():
            if len(d.get(cls, [])) == 1:
                r = F4id.get(i); g = d[cls][0]; xs.append(((g[1] + g[3]) / 2 / r["img_h"]) if dim == "h" else ((g[0] + g[2]) / 2 / r["img_w"])); ys.append(r[key])
        return np.array(xs), np.array(ys)
    w, h = 720, 345; b = []; rows = []
    for pi, (cls, key, dim, ttl) in enumerate((("horizontal", "y_star", "h", "Horizontal: lens y* against the marked line"), ("vertical", "x_star", "w", "Vertical: lens x* against the marked line"))):
        xm, yl = pts(cls, key, dim); x0, y0, S = 40 + pi * 350, 40, 280; sx = lambda v: x0 + S * v; sy = lambda v: y0 + S * (1 - v)
        b.append(T(x0, 22, ttl, "ts") + f'<rect x="{x0}" y="{y0}" width="{S}" height="{S}" class="fr2"/><line x1="{sx(0)}" x2="{sx(1)}" y1="{sy(0)}" y2="{sy(1)}" class="chance"/>')
        b.extend(f'<circle cx="{sx(a):.1f}" cy="{sy(c):.1f}" r="1.6" class="s1" opacity="0.28"/>' for a, c in zip(xm, yl))
        e = np.abs(xm - yl); b.append(T(x0 + S / 2, y0 + S + 16, f"annotators' mark ({'height' if dim == 'h' else 'width'} fraction), n = {len(xm)}", "ts", "middle")); b.append(T(x0 + S / 2, y0 + S + 31, f"median error {np.median(e):.3f}; {100*np.mean(e <= 0.05):.0f}% within 0.05", "ts", "middle"))
        rows.append([cls, len(xm), f"{np.median(e):.3f}", f"{100*np.mean(e <= 0.05):.0f}%", f"{m4['T2']['all'][cls]['null_median_error']:.3f}"])
    return svg(w, h + 14, "".join(b), "Scatter plots of the lens line position against the annotators' marked line position"), ["class", "images with one mark", "median error", "within 0.05", "median error, permuted marks"], rows

# ------------------------------------------------------------------ numbers used in the text
r1 = rec["R1"]["noise0.0_contrastdefault"]; r2 = rec["R2"]["noise0.02_contrastdefault"]; r2l = rec["R2"]["noise0.02_contrast0.05"]
met = lambda d: sum(v.get("prediction_met", False) for v in d["per_category"].values())
d_ahead = sum(v["fitted_dinov3_auc"] > v["fitted_lens_auc"] for v in m3c["per_category"].values())
sc_pairs = {k: (devA["pairs"][k]["auc"], devB["pairs"][k]["auc"], m3c["pairs"][k]["auc"]) for k in PAIRS}
bl = m3c["baselines"]; best_bl = {k: max(bl[k], key=lambda b: bl[k][b]["auc"]) for k in bl}
v0n = sum(t["status"] == "v0" for t in cat); st = lambda s: sum(t["status"] == s for t in cat)
n_pool = devA["n"] + devB["n"]
# ------------------------------------------------------------------ tables
tmpl_t = table(["id", "template", "kind", "positions / geometry", "PICD", "CADB", "status"], [[f"<code>{t['id']}</code>", esc(t["name"]), t["kind"], esc(", ".join(f"{p:.3f}" if isinstance(p, float) else str(p) for p in t["primitives"])[:60]) if t["primitives"] else "-", esc(", ".join(t["picd"]) or "-"), esc(", ".join(t["cadb"]) or "-"), t["status"]] for t in cat])
# ------------------------------------------------------------------ build figures in order
add("cov", "Which PICD categories a template covers.", "Of the 24 PICD categories, " + f"{24 - len(cw['picd_without_v0_template'])} have a v0 template; the other {len(cw['picd_without_v0_template'])} are arrangements, textures, curves or perspective that no overlay in the catalogue describes. CADB classes without any template: pattern, fill the frame.", fig_coverage(), "The overlay vocabulary covers well under half of PICD.")
add("how", "How a template fit is computed on a photograph.", "A real photograph (PICD; dataset credit under the overlay figure in section 5) beside its row profile. The strongest horizontal line sets y*; each template reads the profile within the tolerance of its position and is scored against the rest of the profile.", fig_how(), HOWTXT)
add("gallery", "What the synthetic images look like.", "Exact layouts placed on template positions with jitter up to 0.5% of the frame, a smooth background, low-contrast distractors and noise. Generated by lens/synth.py and lens/synth2.py; seeds in the figure code.", fig_gallery(), "The synthetic tests are the easy case: clean strokes, tone splits and drawn lines. The hard point at contrast 0.05 shows why compact objects fail first.")
add("synth", "Template recovery on exact synthetic layouts.", "10,020 images across four aspect ratios and five conditions. 'Within kind' restricts the choice to templates of the same kind (horizontal, vertical, point, diagonal).", fig_synth(), "Noise barely matters; contrast does. Choosing among all templates at once is worse than within a kind because a compact object on a power point also makes the line profiles peak (see the confusion matrix below).")
add("sweep", "Where thirds hands over to phi.", "A single synthetic horizontal line, 20 images per position, 4:3, noise 0.02.", fig_sweep(), "The hand-over sits at 0.355, almost exactly the midpoint (0.358) of the two template positions.")
add("conf", "Which template is read for which synthetic layout.", "Default contrast, noise 0.02, scored over all 13 templates with the z score. Cells with at least 0.10 off the diagonal, or a diagonal below 0.90, are labelled.", fig_confusion(), "The mistakes are geometric: compact objects on a power point or the centre are read as the line templates that pass through them, and fifths lines as the nearby phi lines.")
add("m2b", "Kind (line vs compact) reading and arbitrated recovery.", "Thresholds tau_c = " + f"{cal['tau_c']:.2f}, tau_b = {cal['tau_b']:.4f}" + " calibrated on a separate 1,200-image set, then frozen.", fig_m2b(), "Compact points are the casualty at low contrast; lines and diagonals survive.")
add("spiral", "Golden spiral: tolerance to displacement.", "Eight variants; AUC against perturbed spirals and straight strokes, and eight-way variant accuracy, as the exact spiral is displaced.", fig_spiral(), "The spiral can be placed to about 2% of the short side; beyond 4% it is lost.")
add("rep", "The 11 training-free crosswalk predictions on three sets.", "Circles carry 95% bootstrap intervals; squares are the design set. Bold labels are predictions not met on LockboxB. Template scores are the z scores fixed in PLAN.md.", fig_replicate(), f"{met(m3c)} of 11 met on each set; the same three fail on all three.")
add("meth", "Methods compared on fresh images, per category.", "Fitted probes train on the whole pool (3,312 images); the lens template score and the Facet-style heuristic need no training.", fig_methods(), "The fitted DINOv3 probe leads almost everywhere. Between the two training-free scores, the lens template is above the Facet-style heuristic on the line, diagonal and point categories and below it on S-Cent.")
add("ovl", "What the lens reports on a photograph.", "Six images drawn at random (seeded) from LockboxB, one per category of the three position pairs. White dashed lines mark 1/3, 1/2 and 2/3; the solid line or circle is the measured position. Images from PICD (Zhao et al., CVPR 2025), the part released for redistribution, shown reduced and uncropped under CC BY-NC-SA 4.0 and the dataset's Terms of Use; the dataset does not record photographer or platform per image; ids shown.", fig_overlays())
add("hist", "Where the dominant line or object sits in each category (LockboxB).", "40 images per category.", fig_hist(), "Middle categories cluster at 1/2 (or at the centre); third categories sit at 1/3 and 2/3 (or a power point).")
add("pairs", "The three position pairs on three sets.", "AUC of the pre-stated position reading; the pooled DINOv3 probe uses about 69 training images per category.", fig_pairs(), "All three pairs separate on all three sets.")
add("curves", "Do learned probes recover the separation?", "Probes use cross-validated regularisation; 38 test images per category in the pool experiment.", fig_curves(), "Position-aware features catch up with the reading at about 20 to 40 labelled images for the vertical and compact pairs; the pooled embedding does not.")
add("cadb", "A second dataset: the lens on CADB composition classes.", "CADB (BMVC 2021; AADB Flickr images; research use only; local copy, no images shown). The lens spec is the frozen M3 spec: zero-shot transfer, no design on CADB. Multi-label images count as positives for each of their classes. Fitted probes train on the official train split (8,547); the test split has 950 images. Bold labels are classes whose lens score does not clear 0.5 with its interval (diagonal was a pre-stated prediction; golden ratio, triangle and curved had none; vanishing point was predicted not to).", fig_cadb_classes(), "Four of five predicted classes are above chance, but only the line-coverage readings are strong. A fitted DINOv3 probe leads on every class.")
add("agree", "Does the lens agree with where the annotators marked the line?", "Images with exactly one annotated horizontal (n = " + str(m4["T2"]["all"]["horizontal"]["n"]) + ") or vertical (n = " + str(m4["T2"]["all"]["vertical"]["n"]) + ") segment, all 9,497 images; the dashed line is perfect agreement. Annotation coordinates are in each image's own pixel frame.", fig_cadb_agree(), "The dominant-line readings agree with the human-marked lines; permuting the marks across images gives errors four to six times larger.")

claims = [
 ("C1", "solid (synthetic)", "On exact synthetic layouts the 13 position templates are located and recovered.", f"Position error: median {r1['hline']['median']*100:.2f}% (horizontal) and {r1['vline']['median']*100:.2f}% (vertical) of the frame, 95th percentile about {r1['hline']['p95']*100:.1f}%. Top-1 recovery within a kind, rank score: {f2(r2['within_kind']['rank_scores']['mean'])} at default contrast, {f2(rec['R2']['noise0.02_contrast0.1']['within_kind']['rank_scores']['mean'])} at contrast 0.10, {f2(r2l['within_kind']['rank_scores']['mean'])} at 0.05. Lines and diagonals keep AUC at least {min(v for k, v in rec['R3']['noise0.02_contrast0.05'].items() if not k.endswith('pts') and not k.endswith('_pt')):.2f} against nulls down to contrast 0.05.", f"Figures {FNUM['synth']}, {FNUM['sweep']}", "Synthetic only; strokes and tone splits on smooth backgrounds are the easy case."),
 ("C2", "solid (synthetic)", "Some templates cannot be told apart by geometry, and the engine reports where.", "Thirds and phi lines (4.9% of the frame apart) hand over at 0.355 on a single-line sweep. Fifths against phi (1.8% apart) is confused: fifths recovered " + f"{f2(r2['within_kind']['rank_scores']['per_template']['fifths_h'])} and {f2(r2['within_kind']['rank_scores']['per_template']['fifths_v'])}. Rabatment equals the thirds lines on 3:2 and the fifths end lines on 4:5.", f"Figures {FNUM['sweep']}, {FNUM['conf']}", "Identifiability is a property of the templates and the tolerance (1.5% of the frame), not a defect."),
 ("C3", "solid (synthetic)", "An extent reading (line vs compact) removes most cross-kind confusion, and compact structures need much more contrast to be seen.", f"Kind accuracy {f2(m2b['E1']['default']['kind_accuracy'])} at default contrast, {f2(m2b['E1']['contrast0.1']['kind_accuracy'])} at 0.10, {f2(m2b['E1']['contrast0.05']['kind_accuracy'])} at 0.05; top-1 among the detected kind {f2(m2b['E1']['default']['arbitrated_top1_rank'])} against {f2(m2b['E1']['default']['unarbitrated_top1_rank'])} unarbitrated.", f"Figure {FNUM['m2b']}", "The arbitrated rule abstains when it sees nothing."),
 ("C4", "solid (synthetic, easy case)", "Golden triangle, golden spiral and perspective readings pass their gates on drawn strokes and line drawings.", f"Triangle AUC {f2(m2b['E2']['auc_vs_nulls'])} against shifted and random nulls; spiral AUC {f2(m2b['E4']['auc_vs_nulls'])}, eight-way variant accuracy {f2(m2b['E4']['variant_accuracy'])}, tolerance about 2% of the short side; perspective exact count {f2(m2b['E5']['noise0.02_contrast0.3']['mean_exact'])} (one-point scenes weakest, {f2(m2b['E5']['noise0.02_contrast0.3']['1']['exact_count'])}).", f"Figure {FNUM['spiral']}", "Not validated on any real photograph: PICD has no category for them."),
 ("C5", "replicated: confirmation and fresh", "The training-free template scores rank the categories they map to above the rest (AUC above 0.5), modestly.", f"Predictions met (AUC above 0.5, interval above 0.5): {met(devA)} of 11 on DevA (design), {met(devB)} of 11 on DevB (confirmation, spec frozen), {met(m3c)} of 11 on LockboxB (fresh, spec hashed first): LS-Hori2, LS-Hori3, L-Ver2, L-Ver3, LS-Dia, P-Dia, P-RoT, P-Cent. LockboxB AUCs {f2(min(m3c['per_category'][a]['lens_auc'] for a in ['P-RoT','P-Cent','P-Dia','LS-Dia','LS-Hori2','LS-Hori3','L-Ver2','L-Ver3']))} to {f2(max(m3c['per_category'][a]['lens_auc'] for a in ['P-RoT','P-Cent','P-Dia','LS-Dia','LS-Hori2','LS-Hori3','L-Ver2','L-Ver3']))}.", f"Figure {FNUM['rep']}", "Modest AUCs; the claim is direction, not accuracy."),
("C6", "replicated: confirmation and fresh", "A line-position reading separates two of the five PICD category pairs that no earlier whole-image method separated: horizon in the middle vs a third, and a vertical line in the middle vs a third.", "AUC on DevA / DevB / LockboxB: " + "; ".join(f"{k} {f2(a)} / {f2(b)} / {f2(c)}" for k, (a, b, c) in list(sc_pairs.items())[:2]) + ". Fitted DINOv3 probe on the pooled embedding for the same pairs: " + ", ".join(f2(devB['pairs'][k]['fitted_dinov3_pair_auc']) for k in PAIRS[:2]) + " (DevB, about 69 images per category).", f"Figures {FNUM['hist']}, {FNUM['pairs']}", "The categories differ in position as their definitions say. Separation is partial. The same readings agree with human-marked lines on CADB (C9)."),
 ("C6b", "held back: PICD only", "A compact-object reading separates P-RoT from P-Cent on PICD.", f"AUC {f2(sc_pairs['P-RoT vs P-Cent'][0])} / {f2(sc_pairs['P-RoT vs P-Cent'][1])} / {f2(sc_pairs['P-RoT vs P-Cent'][2])} on DevA / DevB / LockboxB, against {f2(devB['pairs']['P-RoT vs P-Cent']['fitted_dinov3_pair_auc'])} for a pooled DINOv3 probe. But on CADB the same blob location does not follow the annotated subject boxes (C10).", f"Figure {FNUM['pairs']}", "Not treated as established: one dataset, and the reading failed its second-dataset check."),
 ("C7", "partly replicated", "Position-aware learned probes match or pass the named reading once they have labelled images; the pooled DINOv3 embedding does not.", f"Pool learning curves (10 to 100 per category): pooled embedding 0.53 to 0.68; edge-strength grid {f2(m3b['P2'][PAIRS[0]]['100']['G-edge'][0])} and {f2(m3b['P2'][PAIRS[1]]['100']['G-edge'][0])} on the horizon and vertical pairs; patch-norm map {f2(m3b['P2'][PAIRS[2]]['100']['D-norm'][0])} on the compact pair. LockboxB (train 138 per category): best baselines {', '.join(f'{k}: {best_bl[k]} {f2(bl[k][best_bl[k]]['auc'])}' for k in bl)}; pooled embedding {', '.join(f2(bl[k]['D-pool']['auc']) for k in bl)}. Four of six pre-stated baseline sub-predictions met; both misses are on the horizon pair (the pooled embedding reached 0.74 against a cap of 0.70; the best baseline was 0.04 above the reading against 0.03 allowed).", f"Figure {FNUM['curves']}", "The named reading's advantage is that it needs no labelled images, not a higher ceiling."),
 ("C9", "replicated on a second dataset", "The dominant horizontal and vertical line readings agree with where CADB annotators marked the line.", f"Images with one marked segment, all 9,497 images: horizontal n = {m4['T2']['all']['horizontal']['n']}, median |y* - mark| {m4['T2']['all']['horizontal']['median_error']:.3f} of the height against {m4['T2']['all']['horizontal']['null_median_error']:.3f} for permuted marks, {pct(m4['T2']['all']['horizontal']['within_0_05'])} within 0.05; vertical n = {m4['T2']['all']['vertical']['n']}, median {m4['T2']['all']['vertical']['median_error']:.3f} against {m4['T2']['all']['vertical']['null_median_error']:.3f}, {pct(m4['T2']['all']['vertical']['within_0_05'])} within 0.05. Line-coverage scores rank the CADB horizontal and vertical classes above the rest: AUC {f2(m4['classes']['horizontal']['lens_auc_test'])} and {f2(m4['classes']['vertical']['lens_auc_test'])} on the test split.", f"Figures {FNUM['cadb']}, {FNUM['agree']}", "Zero-shot: the lens spec was frozen before CADB was read. Different annotators and class definitions from PICD."),
 ("C10", "not supported on the second dataset", "The compact-object, thirds-point, centre-point, diagonal, golden, spiral and perspective scores.", f"CADB test AUC: rule_of_thirds {f2(m4['classes']['rule_of_thirds']['lens_auc_test'])}, center {f2(m4['classes']['center']['lens_auc_test'])} (above chance, barely), diagonal {f2(m4['classes']['diagonal']['lens_auc_test'])}, golden_ratio {f2(m4['classes']['golden_ratio']['lens_auc_test'])}, triangle {f2(m4['classes']['triangle']['lens_auc_test'])}, curved {f2(m4['classes']['curved']['lens_auc_test'])}, vanishing_point {f2(m4['classes']['vanishing_point']['lens_auc_test'])}. The blob location does not track the annotated subject boxes: median distance {f2(m4['T2']['test']['center_box']['median_error'])} to the centre-class boxes against {f2(m4['T2']['test']['center_box']['null_median_error'])} for permuted boxes. The Facet-style thirds heuristic scores {f2(m4['classes']['rule_of_thirds']['facet_auc_test'])} on rule_of_thirds, above the lens.", f"Figure {FNUM['cadb']}", "The line readings carry the cross-dataset result; the compact-object reading carries the PICD P-RoT vs P-Cent result and is not validated outside PICD."),
 ("C8", "descriptive", "As a classifier of PICD categories the lens is weaker than a fitted learned probe; its value is that every number has a name.", f"Fitted DINOv3 probe ahead of the fitted lens-feature probe on {d_ahead} of 24 categories on LockboxB; Facet-style heuristic (reimplemented) {f2(min(v['facet_style_auc'] for v in m3c['per_category'].values() if 'facet_style_auc' in v))} to {f2(max(v['facet_style_auc'] for v in m3c['per_category'].values() if 'facet_style_auc' in v))}.", f"Figure {FNUM['meth']}", "Fitted layers are named as fitted wherever they appear."),
 ("E1", "exploratory", "Single lens features other than the pre-stated ones carry signal for several categories (for example line coverage for the horizon categories, AUC 0.81 to 0.85 on DevA and DevB).", "Best single feature per category is listed in results/m3_devA.json and m3_devB.json; 62 features per category, no multiplicity correction.", "-", "Not a claim; a place to look."),
 ("N1", "not claimed", "That these templates describe how people see composition, or that fit to a template means a photograph is well composed.", "Nothing here measures quality or perception.", "", ""),
 ("N2", "not claimed", "Anything about the golden triangle, golden spiral, phi, fifths or rabatment templates on real photographs.", "Tested on synthetic layouts only; PICD has no labelled category for them.", "", ""),
 ("N3", "not claimed", "Anything about the P-Dia vs P-Hori and P-Dia vs P-Ver pairs.", "They are arrangements of several points and the lens has no reading for them.", "", ""),
 ("N4", "not claimed", "That the pattern holds in other datasets beyond the line readings checked on CADB, in other photographic collections, or at other resolutions.", "Two datasets (PICD, CADB), both Flickr-derived or web photographs, 384-pixel readings.", "", ""),
 ("N6", "not claimed", "Anything about the ADadras thirds classifier, PhotoFramer, or Facet itself.", "Only a reimplementation of Facet's thirds and centre heuristic was compared; the other two were not run.", "", ""),
 ("N5", "not claimed", "That the remaining overlap between labelled categories is label noise, or that annotators would agree.", "No annotator-agreement data on these pairs.", "", ""),
]
ORDER = ["C9", "C6", "C7", "C6b", "C10", "C5", "C1", "C2", "C3", "C4", "C8", "E1", "N1", "N2", "N3", "N4", "N6", "N5"]; claims = sorted(claims, key=lambda c: ORDER.index(c[0]) if c[0] in ORDER else 99)
pill = lambda s: f'<span class="pill {"not" if s.startswith("not") else ("rep" if s.startswith("replicated") or s.startswith("partly") else ("exp" if s.startswith("expl") or s.startswith("desc") or s.startswith("held") else "sol"))}">{esc(s)}</span>'
claims_t = table(["id", "status", "claim", "evidence", "where", "note"], [[c[0], pill(c[1]), esc(c[2]), esc(c[3]), c[4], esc(c[5])] for c in claims])
json.dump([dict(zip(("id", "status", "claim", "evidence", "where", "note"), c)) for c in claims], open(R("claims_ledger.json"), "w"), indent=1)
fallen = [("Rank-only template score", "It saturated: any strong line or noise bump could reach rank 1, so a flat profile in the other orientation beat the true line (A2).", "Score against the null by magnitude (robust z), keep the rank score for choosing within a kind."),
          ("Raw disc share as point evidence", "A line passes through the power points and scored like a point (A1).", "Use a blob response: share minus the maximum on a ring around it."),
          ("Noise as the stress test", "Gaussian noise up to 0.05 barely moved any result (A3).", "Contrast against distractors is the stressor; compact structures fail first."),
          ("Top-1 over all templates", f"A compact object on a power point also makes the line profiles peak; top-1 over all templates is {f2(m2b['E1']['default']['unarbitrated_top1_rank'])} at best.", "Add the extent reading and choose among templates of the detected kind."),
          ("Single-shape and perspective templates on real photographs", f"S-RoT {f2(m3c['per_category']['S-RoT']['lens_auc'])}, S-Cent {f2(m3c['per_category']['S-Cent']['lens_auc'])}, S-Per {f2(m3c['per_category']['S-Per']['lens_auc'])} on LockboxB (S-Per significantly below chance).", "A point template is not a shape template; vanishing points are everywhere in textured scenes. Operating window: lines and compact points, not shapes or perspective."),
          ("Facet-style thirds and centre score", f"{f2(m3c['per_category']['LS-Hori2']['facet_style_auc'])} and {f2(m3c['per_category']['LS-Hori3']['facet_style_auc'])} on the two horizon categories (LockboxB).", "A subject-box score ignores a horizon; the lens line readings do not."),
          ("Blob-distance reading for P-RoT vs P-Cent", f"AUC {f2(m3c['pairs']['P-RoT vs P-Cent']['auc'])} on LockboxB against {f2(bl['P-RoT vs P-Cent']['D-norm']['auc'])} for a patch-norm probe; blob detection fires on about 30% of images.", "The information is in the image; a better compact-object reading is open."),
          ("'The reading stays ahead on the horizon pair'", "Held on the pool (0.85 against 0.81 at 100 per category) and not on LockboxB (0.78 against 0.82 for an edge grid).", "Differences of 0.05 are inside the noise at 40 images per category; the claim is dropped."),
          ("'Pooled embeddings discard position' as a general statement", f"The pooled DINOv3 embedding reached {f2(bl['LS-Hori3 vs LS-Hori2']['D-pool']['auc'])} on the horizon pair when trained on 138 images per category.", "True for the vertical and compact pairs, weaker for horizons."),
          ("Compact-object and thirds/centre point scores on CADB", f"CADB test AUC {f2(m4['classes']['rule_of_thirds']['lens_auc_test'])} for rule_of_thirds and {f2(m4['classes']['center']['lens_auc_test'])} for center; the blob location does not track the annotated boxes (median distance {f2(m4['T2']['test']['center_box']['median_error'])}, permuted {f2(m4['T2']['test']['center_box']['null_median_error'])}).", "The compact-object reading works on PICD's centre-versus-power-point pair but is not validated by the annotators' boxes; subject placement needs a better subject detector (a trained saliency model would be labelled as such)."),
          ("Diagonal, golden and perspective scores on CADB", f"diagonal {f2(m4['classes']['diagonal']['lens_auc_test'])}, golden_ratio {f2(m4['classes']['golden_ratio']['lens_auc_test'])}, triangle {f2(m4['classes']['triangle']['lens_auc_test'])}, curved {f2(m4['classes']['curved']['lens_auc_test'])}, vanishing_point {f2(m4['classes']['vanishing_point']['lens_auc_test'])}.", "No evidence for these templates on real photographs; the synthetic results are not enough."),
          ("Overlay templates beyond thirds, centre and diagonals", "Golden triangle, spiral, phi, fifths and rabatment have no PICD category and no real-photograph test.", "Their value is the synthetic identifiability map, not a validated detector.")]
fallen_t = table(["what was tried or expected", "what happened", "what it taught"], [[esc(a), esc(b), esc(c)] for a, b, c in fallen])
limits = ["CADB classes are multi-label and imbalanced (centre 4,804 images, radial 42); several classes have few test positives, so their intervals are wide. CADB images come from AADB (Flickr, research use only) and are not shown in this book.", "The first M4 extraction hung and produced nothing; the extraction script was made crash-safe and the spec re-frozen before the rerun (amendment A6). The measurements are unchanged.", "Eleven one-sided predictions were tested without multiplicity correction; the intervals are descriptive. DevA, DevB and LockboxB are all PICD images from the same sources and labelling scheme, so agreement across them is replication within one dataset, not independence of provenance.", "Position tolerance (1.5% of the frame) and the 384-pixel working size are design choices fixed before the tests, not tuned.",
          "Everything real was measured on one dataset (PICD) whose labels are the dataset's own; the crosswalk from templates to categories was made by name, not validated independently.",
          "Synthetic tests use drawn strokes and tone splits on smooth backgrounds; they show the engine does what it is defined to do, not that photographs behave that way.",
          "With 40 images per category on LockboxB, one AUC carries an interval of about +/- 0.1, so claims are about direction and ordering.",
          "The three position pairs were chosen because they were the pairs that earlier instruments failed on; the pre-stated rules and the confirmation and fresh sets limit tuning, not that selection.",
          "The DINOv3 probes use one model, whole-frame squashing to the model's input size, and logistic regression with cross-validated strength; other pooling or models could do better.",
          "The Facet-style comparator is a reimplementation from the public source, not Facet's code, and runs at 384 pixels.",
          "Thresholds tau_c and tau_b were calibrated on synthetic images; their transfer to photographs is untested except through the pair results, which do not depend on tau_b.",
          "Fitted comparators (the logistic probes) are fitted layers and are named so; the lens readings and templates are training-free."]
amend = re.findall(r"^- \*\*(A\d[^*]*)\*\*(.*)$", plan[plan.index("## Amendments"):plan.index("## Result log")], re.M)
amend = sorted(amend, key=lambda x: x[0]); amend_t = table(["amendment", "what and why"], [[esc(a.strip(" .")), esc(b.strip().replace("`", ""))] for a, b in amend])
roles = table(["set", "images", "role", "state"], [["PICD held-out pool, DevA", "1,656", "design: the lens may be changed against it (amendments dated)", "used"], ["PICD held-out pool, DevB", "1,656", "confirmation: read once under the frozen spec (hash " + spec["m3"][0][:16] + ")", "spent"], ["Six-category subset (both halves)", "828", "baseline training and learning curves (M3b)", "used"], ["LockboxB", "960", "fresh test, read once under the M3c spec (hash " + spec["m3c"][0][:16] + ")", "spent"], ["CADB (all 9,497; test split 950)", "9,497", "second dataset, zero-shot transfer under the M4 spec (hash " + spec["m4"][0][:16] + "); annotators' own marks used for T2", "spent"], ["Synthetic images", "10,020 + 4,884 + 1,200", "recovery, M2b tests, calibration (seeds fixed in the scripts)", "regenerable"]])
repro = """# 0. catalogue (templates as data, crosswalk by name)
python3 -I catalogue/build_catalogue.py
# 1. M2: exact synthetic layouts, position templates
python3 -I tests/run_recovery.py ; python3 -I tests/analyse_recovery.py
# 2. M2b: extent, golden triangle, rabatment, spiral, perspective (calibrate on separate seeds, then test)
python3 -I tests/run_m2b.py cal ; python3 -I tests/run_m2b.py test ; python3 -I tests/analyse_m2b.py
# 3. M3: PICD pool (read-only images in composition-benchmark/data); design on DevA, freeze, confirm once on DevB
python3 -I tests/m3_extract.py devA ; python3 -I tests/m3_analyse.py devA
python3 -I tests/m3_freeze.py ; python3 -I tests/m3_extract.py devB ; python3 -I tests/m3_analyse.py devB      # devB refuses a second run or a changed spec
# 4. M3b: baselines on the three pairs (DINOv3 in an environment with torch and timm, offline)
HF_HUB_OFFLINE=1 python tests/m3b_extract_dino.py ; python3 -I tests/m3b_analyse.py
# 5. M3c: fresh LockboxB test. The spec was frozen by writing spec_hash() of tests/m3c_extract.py to results/m3c_spec.sha256, then:
python3 -I tests/m3c_extract.py ; HF_HUB_OFFLINE=1 python tests/m3c_extract_dino.py ; python3 -I tests/m3c_analyse.py
# 6. M4: CADB (download CADB_Dataset.zip from the authors' Dropbox link into data/cadb_zip, unzip into data/cadb); spec frozen as for M3c, then
python3 -I tests/m4_extract.py ; HF_HUB_OFFLINE=1 python tests/m4_extract_dino.py ; python3 -I tests/m4_analyse.py
# 7. this book
python3 -I labbook/make_labbook.py"""
env_t = table(["environment", "python", "libraries", "used for"], [["system", "3.14.3", "numpy 2.4.2, scipy, opencv 4.13.0, Pillow, scikit-learn 1.8.0", "everything except DINOv3"], ["second environment with torch and timm (the original used a local venv)", "3.14.3", "torch (nightly), timm 1.0.30", "DINOv3 embeddings (offline, cached weights)"]])
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
mani = [f for f in sorted(os.listdir(R(""))) if f.endswith((".json", ".sha256", ".npz", ".npy")) and not f.endswith("_raw.json") and not f.startswith("m3_features") and not f.startswith("m4_features") and not f.startswith("m3c_features")]
mani_t = table(["file", "sha256 (first 16)", "last modified"], [[f"<code>results/{f}</code>", f"<code>{sha(R(f))}</code>", datetime.datetime.fromtimestamp(os.path.getmtime(R(f))).strftime("%d %b %Y %H:%M")] for f in mani])
spec_t = table(["spec", "hash", "frozen", "covers"], [["M3 (DevB confirmation)", f"<code>{spec['m3'][0]}</code>", " ".join(spec["m3"][2:4]), "lens/*.py, catalogue, tests/m3_*.py, calibration, plan snapshot"], ["M3c (LockboxB)", f"<code>{spec['m3c'][0]}</code>", " ".join(spec["m3c"][2:4]), "the same plus tests/m3b_analyse.py, tests/m3c_*.py, both plan snapshots"], ["M4 (CADB)", f"<code>{spec['m4'][0]}</code>", " ".join(spec["m4"][2:4]), "the same plus tests/m4_*.py and the M4 plan snapshot (re-frozen after amendment A6)"]])
gloss = """<div class="key"><b>How to read this book.</b> <b>PICD</b> is a public benchmark of 24 photographic composition categories (Zhao et al., CVPR 2025); each image carries one expert label. A <b>template</b> is a named geometry (for example, lines at 1/3 and 2/3 of the frame). The lens measures how well an image's lines or objects sit on each template, in numbers that are named and can be inspected. <b>AUC</b> (one-vs-rest) is the chance that a random image of the target category scores higher than a random image of any other category: 0.5 is no separation, 1 is perfect. Pair AUCs compare two categories only; the <b>pair accuracy</b> reported in an earlier benchmark study (unpublished) is a different statistic, so they are not compared directly. <b>Training-free</b> means no model is fitted to labelled images (two detection thresholds were calibrated on synthetic images); a <b>fitted layer</b> is a logistic regression trained on labelled images, and is named as such wherever it appears. <b>Design / confirmation / fresh</b>: DevA (1,656 images) was used to design, DevB (1,656) was read once after the spec was frozen, LockboxB (960) was read once under a spec hashed beforehand. <b>z and rank scores</b>: how far a template's position stands above the rest of the row profile, by magnitude and by rank.</div>"""
related = table(["work", "what it does", "what we did with it"], [["ADadras: Rule-of-Thirds detection (ACM SUMAC 2025, MIT)", "22 named geometric features and a logistic or boosted classifier for thirds", "not run (listed as a future comparator)"], ["PhotoFramer-Assessment (CVPR 2026, MIT)", "a 7B vision-language model fine-tuned for composition score and class", "not run"], ["Facet (analyzers/composition.py)", "subject box scored against thirds points and the centre", "its heuristic reimplemented from the public source and compared (Figure COMP); Facet itself not run"], ["CADB (BMVC 2021; AADB images)", "9,497 photographs with composition scores, 13 composition classes and elements", "used as the second dataset (Figures CADB, AGREE): class crosswalk, annotators' marks, fitted comparators; zero-shot, local copy only"], ["Grid Maker Pro overlay list", "82 drawing and design overlays, about 27 of them photographic compositions", "names of templates only; its popularity claims are unsourced"], ["DeepGaze, UMSI", "gaze saliency and design importance models", "not used"]])
# ------------------------------------------------------------------ page
CSS = open(os.path.join(HERE, "labbook", "style.css")).read()
nav = '<nav><a href="#summary">Summary</a><a href="#aims">Aims</a><a href="#method">Method</a><a href="#synthetic">Synthetic tests</a><a href="#real">Real photographs</a><a href="#pairs">The five pairs</a><a href="#cadb">CADB</a><a href="#fallen">Fallen</a><a href="#limits">Limits</a><a href="#repro">Reproduce</a></nav>'
body = f"""<h1>composition-lens: named readings of composition, checked against human marks</h1>
<p class="sub">Lab book, generated {today} by <code>labbook/make_labbook.py</code> from the files in <code>results/</code> and <code>catalogue/</code>. Everything is local.</p>{nav}
<h2 id="summary">1. Summary and claims ledger</h2>
<div class="key"><b>The claim, in short.</b> The height of the strongest horizontal line and the position of the strongest vertical line in a photograph can be read by a plain, named measurement (no model trained on photographs), and that measurement agrees with people. On CADB (9,497 photographs, read once) the horizon reading lies within 0.05 of the image height of the line the annotators drew for {pct(m4['T2']['all']['horizontal']['within_0_05'])} of images (median error {m4['T2']['all']['horizontal']['median_error']:.3f}, against {m4['T2']['all']['horizontal']['null_median_error']:.3f} when the marks are shuffled), and the vertical reading agrees similarly. The same readings separate PICD's horizon-in-the-middle from horizon-on-a-third categories (AUC {f2(sc_pairs[PAIRS[0]][2])} on fresh images) and the vertical-line pair ({f2(sc_pairs[PAIRS[1]][2])}), where whole-image methods stayed near chance. <b>Not established:</b> subject placement. The compact-object reading did not follow the annotators' subject boxes on CADB, the thirds-point and centre-point scores are barely above chance there, and the diagonal, golden, spiral and perspective scores are not supported on real photographs.</div>
<p><b>Background for a new reader.</b> Photographers describe composition with named patterns: a horizon in the middle or on a third of the frame, a subject at the centre or at a rule-of-thirds point, a diagonal. Two public datasets label photographs this way. <b>PICD</b> (Zhao et al., CVPR 2025) gives each photograph one of 24 expert-assigned categories. <b>CADB</b> (Zhang, Niu and Zhang, BMVC 2021) gives each photograph one or more of 13 composition classes and the annotators' own drawn marks (lines, boxes). A <b>reading</b> here is a named number computed from an image with no training, such as "the strongest horizontal line sits at 0.37 of the height". The work asks whether such readings track what people mark and label, where they do, and where they don't. Every stage was fixed in a written plan before it ran, and the real-photograph tests were read once under a hashed specification.</p>
<p><b>Question.</b> Photographers and artists use a small set of named compositions (thirds, phi, diagonals, golden triangle, rabatment, spiral, one- to three-point perspective). As far as a limited GitHub search found (it was rate-limited), public tools score one or two of them with a heuristic and none reports, per template, how far the measurement can be trusted. Can a named, inspectable measurement of each template be built, and where does it work?</p>
<p><b>Where it stands.</b> <i>Established (two datasets).</i> The dominant horizontal and vertical line readings agree with the annotators' marked lines on CADB, rank the CADB horizontal and vertical classes above the rest (AUC {f2(m4['classes']['horizontal']['lens_auc_test'])} and {f2(m4['classes']['vertical']['lens_auc_test'])}), and separate the PICD horizon and vertical-line pairs on three sets. A trained probe with position-aware features matches or passes them given labelled images, so their advantage is that they need none. <i>Held back.</i> The compact-object reading separates P-RoT from P-Cent on PICD (AUC {f2(sc_pairs['P-RoT vs P-Cent'][2])} on fresh images) but did not follow CADB's annotated subject boxes. <i>Modest.</i> {met(m3c)} of 11 crosswalk predictions are met on each PICD set, with AUCs of {f2(min(m3c['per_category'][a]['lens_auc'] for a in ['P-RoT','P-Cent','P-Dia','LS-Dia','LS-Hori2','LS-Hori3','L-Ver2','L-Ver3']))} to {f2(max(m3c['per_category'][a]['lens_auc'] for a in ['P-RoT','P-Cent','P-Dia','LS-Dia','LS-Hori2','LS-Hori3','L-Ver2','L-Ver3']))}; on CADB thirds and centre are {f2(m4['classes']['rule_of_thirds']['lens_auc_test'])} and {f2(m4['classes']['center']['lens_auc_test'])}. <i>Not supported on real photographs.</i> Diagonal, golden ratio, golden triangle, spiral and perspective scores. <i>Synthetic.</i> On exact layouts the catalogue of {len(cat)} templates ({v0n} position templates implemented, the rest built or planned) is located to about {r1['hline']['median']*100:.1f}% of the frame, and the engine reports which templates it cannot separate. As a classifier of dataset categories the lens is weaker than a fitted DINOv3 probe, which leads on {d_ahead} of 24 PICD categories.</p>
{gloss}{claims_t}
<h2 id="aims">2. Aims, and what changed</h2>
<p>The work started from an earlier, unpublished benchmark study of PICD, where five category pairs resisted every method and the data could not say why. The idea: build detectors for the named templates themselves, give each a null and a recovery test, and validate against labelled data. The plan (PLAN.md) fixes definitions and tests before each stage runs; changes after a result are dated amendments.</p>
<h3>Plan changes after seeing results</h3>{amend_t}
<h3>Data roles</h3>{roles}{spec_t}<h3>Related public work</h3>{related.replace('Figure COMP', 'Figure ' + str(FNUM['meth'])).replace('Figures CADB, AGREE', 'Figures ' + str(FNUM['cadb']) + ', ' + str(FNUM['agree']))}
<h2 id="method">3. Templates and method</h2>
<p>Each template is a geometry spec (lines, points, paths) in <code>catalogue/templates.json</code>; the crosswalk to PICD and CADB is by name and marked draft. Only templates whose geometry is unambiguous in standard sources are implemented.</p>{F['cov']}{tmpl_t}{F['how']}
<p><b>Evidence.</b> At a long side of 384 px: a position profile of line strength against position for horizontal, vertical and the two diagonal orientations; a mass map and a blob response for compact objects; coverage of the dominant line (how much of its length has an edge). <b>Fit.</b> A template's fit is the profile value within a tolerance (1.5% of the frame) of its positions, ranked against the rest of the profile (rank score) and scored by magnitude against the profile's own spread (robust z, corrected for the number of primitives). No trained model is used anywhere in the lens. Full definitions: PLAN.md.</p>
<h2 id="synthetic">4. Synthetic tests</h2>
<p>All stages were tested first on images whose layout is exact by construction. Sanity gates were fixed beforehand and all passed (position error under 1% of the frame, AUC at least 0.9 against nulls, kind accuracy at least 0.9, spiral AUC at least 0.9, perspective count accuracy at least 0.8).</p>{F['gallery']}{F['synth']}{F['sweep']}{F['conf']}{F['m2b']}{F['spiral']}
<p><b>Triangle, rabatment, perspective.</b> The golden triangle separates from a bare diagonal at AUC {f2(m2b['E2']['auc_vs_diag_only'])}; rabatment is the top template on 4:3 ({m2b['E3']['4:3']['winners']}) and ties with thirds on 3:2 and with the fifths end lines on 4:5; the exact vanishing-point count is {f2(m2b['E5']['noise0.02_contrast0.3']['mean_exact'])} on clean line drawings (one-point scenes {f2(m2b['E5']['noise0.02_contrast0.3']['1']['exact_count'])}, median vanishing-point error {m2b['E5']['noise0.02_contrast0.3']['1']['vp_err_median']:.3f} of the frame for one-point scenes).</p>
<h2 id="real">5. Real photographs (PICD)</h2>
<p>Design on DevA; spec frozen and hashed; one confirmation run on DevB; a last fresh test on LockboxB with the plan and predictions fixed before it was read. The comparators are a reimplementation of Facet's thirds and centre heuristic, a fitted logistic probe on the lens features (a fitted layer) and a fitted logistic probe on DINOv3 embeddings (a fitted layer over a trained model).</p>{F['rep']}{F['meth']}
<h3>What the lens reports</h3>{F['ovl']}
<h2 id="pairs">6. The five pairs that no whole-image method separated</h2>
<p>In an earlier benchmark study (unpublished) five category pairs scored 0.49 to 0.57 pair accuracy for every method. Three of them have a position reading. Two use the line readings (LS-Hori2 vs LS-Hori3, L-Ver2 vs L-Ver3), which section 7 validates against human-marked lines on a second dataset. The third (P-RoT vs P-Cent) uses the compact-object reading, which that check did not support, so it is shown here but held back. The two P-Dia pairs need an arrangement reading the lens does not have. The reading is the distance of the dominant horizontal line, vertical line or compact blob from the centre; the interpretation rule was fixed before any real image was read.</p>{F['hist']}{F['pairs']}{F['curves']}
<p><b>Reading the three figures together.</b> The labelled categories do differ in position as defined, and a named reading shows it with no labelled images. Position-aware probes show the information is also easy to learn, so the separation is cleaner than the reading alone suggested (AUC up to {f2(bl['P-RoT vs P-Cent']['D-norm']['auc'])} on LockboxB for the compact pair, where a patch-norm probe does much better than the blob reading). The pooled embedding discards much of it, though less on horizons ({f2(bl['LS-Hori3 vs LS-Hori2']['D-pool']['auc'])}) than on the other two pairs ({f2(bl['L-Ver3 vs L-Ver2']['D-pool']['auc'])}, {f2(bl['P-RoT vs P-Cent']['D-pool']['auc'])}).</p>
<h2 id="cadb">7. A second dataset: CADB</h2><p>CADB (Zhang, Niu and Zhang, BMVC 2021) has 9,497 photographs, 13 composition classes per image (multi-label) and the annotators' own marks for each class. It was downloaded after the plan (PLAN.md, section M4) was written and the spec hashed, and read once, with no design step: the lens is the same frozen lens. Two questions: do the template scores rank the CADB classes they map to, and do the lens's line positions agree with where the annotators drew the lines?</p>{F['cadb']}{F['agree']}<p><b>Reading.</b> The pre-stated rule (at least four of five predicted classes above chance) reads \"supports generalisation\", but the result is lopsided. The line-coverage readings are strong (horizontal {f2(m4['classes']['horizontal']['lens_auc_test'])}, vertical {f2(m4['classes']['vertical']['lens_auc_test'])}) and agree with the annotators' marks; the thirds and centre scores are barely above chance ({f2(m4['classes']['rule_of_thirds']['lens_auc_test'])}, {f2(m4['classes']['center']['lens_auc_test'])}); the diagonal, golden, spiral and perspective scores are not supported. The Facet-style thirds heuristic ({f2(m4['classes']['rule_of_thirds']['facet_auc_test'])}) is above the lens on rule_of_thirds. A fitted DINOv3 probe leads on every class.</p><h2 id="fallen">8. Fallen ideas and operating windows</h2>{fallen_t}
<h2 id="limits">9. Limits and threats to validity</h2><ul>{''.join(f'<li>{esc(l)}</li>' for l in limits)}</ul>
<h2 id="repro">10. Reproduction</h2><p>Commands in the order they were run. Seeds are fixed in every script; the guards refuse a second DevB or LockboxB run or a changed spec.</p><pre>{esc(repro)}</pre>{env_t}<h3>Results manifest (captured when this book was built)</h3>{mani_t}
<p class="small">Source notes: the template vocabulary comes from a tool vendor's overlay list (Grid Maker Pro; unsourced popularity claims, so used as a list of names only). Datasets: PICD (CC BY-NC-SA 4.0 with Terms of Use). Comparators: Facet (analyzers/composition.py, read for the public description), DINOv3 (timm weights, offline). See also {NOTE_REF}.</p>"""
doc = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>composition-lens lab book</title><style>{CSS}</style></head><body><main>{body}</main></body></html>'
open(os.path.join(OUTDIR, "LABBOOK.html"), "w").write(doc); print("wrote LABBOOK.html", len(doc) // 1024, "KB;", FIGN[0], "figures;", len(claims), "ledger entries")
