"""M3c analysis on LockboxB (PLAN.md section M3c). python3 -I tests/m3c_analyse.py -> results/m3c.json. Reads LockboxB features that exist only after the one run."""
import os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, cv2
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
ABBR = {1: "P-RoT", 2: "S-RoT", 3: "P-Cent", 4: "S-Cent", 5: "P-Dia", 6: "LS-Dia", 7: "P-Hori", 8: "LS-Hori2", 9: "LS-Hori3", 10: "S-Hori", 11: "P-Ver", 12: "L-Ver2", 13: "L-Ver3", 14: "L-VerMul", 15: "P-Tri", 16: "LS-Tri", 17: "LS-C-Cur", 18: "LS-O-Cur", 19: "LS-S-Cur", 20: "LS-Dif", 21: "S-Per", 22: "PL-Den", 23: "PL-Pat", 24: "P-Scat"}
def auc(s, p):
    s = np.asarray(s, float); p = np.asarray(p, bool); r = rankdata(s); n1 = p.sum(); n0 = len(p) - n1; return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
def boot(s, p, B=1000, seed=0):
    s = np.asarray(s, float); p = np.asarray(p, bool); rng = np.random.default_rng(seed); i1, i0 = np.where(p)[0], np.where(~p)[0]; v = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); v.append(auc(s[ii], p[ii]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
# ---- lockboxb features
rs = [r for r in json.load(open(os.path.join(HERE, "results", "m3c_features_lockboxb.json"))) if "error" not in r]; names = [k for k in rs[0] if k not in ("id", "label")]
X = np.nan_to_num(np.array([[r[k] for k in names] for r in rs], float), nan=0.0, posinf=0.0, neginf=0.0); y = np.array([r["label"] for r in rs]); lid = [r["id"] for r in rs]
col = {n: i for i, n in enumerate(names)}; f = lambda n: X[:, col[n]]; cal = json.load(open(os.path.join(HERE, "results", "m2b_calibration.json")))
pick = json.load(open(f"{BASE}/data/pick_lockboxb.json")); assert [p["img_id"] for p in pick] == lid
Dlb = np.load(f"{BASE}/results/fitted_dino_lockboxb.npy")
# ---- pool (training) data
sp = json.load(open(f"{BASE}/results/split.json")); emb = np.load(f"{BASE}/results/emb_dinov3_web_l16.npy"); row = {i: k for k, i in enumerate(sp["ids"])}
pf = []
for w in ("devA", "devB"): pf += [r for r in json.load(open(os.path.join(HERE, "results", f"m3_features_{w}.json"))) if "error" not in r]
Xp = np.nan_to_num(np.array([[r[k] for k in names] for r in pf], float), nan=0.0, posinf=0.0, neginf=0.0); yp = np.array([r["label"] for r in pf]); Dp = emb[[row[r["id"]] for r in pf]]
def fit_scores(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=800).fit(sc.transform(Xtr), ytr); return m.decision_function(sc.transform(Xte)), m.classes_
SL, cl = fit_scores(Xp, yp, X, 0.1); SD, _ = fit_scores(Dp, yp, Dlb, 0.02)
TF = {"P-RoT": f("z_thirds_pts"), "S-RoT": f("z_thirds_pts"), "P-Cent": f("z_centre_pt"), "S-Cent": f("z_centre_pt"), "LS-Hori2": f("z_centre_h"), "LS-Hori3": f("z_thirds_h"), "L-Ver2": f("z_centre_v"), "L-Ver3": f("z_thirds_v"),
      "LS-Dia": np.maximum(f("z_diag_baroque"), f("z_diag_sinister")), "P-Dia": np.maximum(f("z_diag_baroque"), f("z_diag_sinister")), "S-Per": f("persp_support_finite")}
TFNAME = {"P-RoT": "thirds_pts", "S-RoT": "thirds_pts", "P-Cent": "centre_pt", "S-Cent": "centre_pt", "LS-Hori2": "centre_h", "LS-Hori3": "thirds_h", "L-Ver2": "centre_v", "L-Ver3": "thirds_v", "LS-Dia": "diag (max of two)", "P-Dia": "diag (max of two)", "S-Per": "finite VP support"}
FACET = {"P-RoT": f("facet_power"), "S-RoT": f("facet_power"), "P-Cent": f("facet_center"), "S-Cent": f("facet_center"), "LS-Hori3": f("facet_thirds"), "L-Ver3": f("facet_thirds"), "LS-Hori2": f("facet_center"), "L-Ver2": f("facet_center")}
out = {"n": int(len(y)), "per_category": {}, "pairs": {}, "baselines": {}}
for k in range(1, 25):
    a = ABBR[k]; pos = y == k; d = dict(n_pos=int(pos.sum())); j = list(cl).index(k); d["fitted_lens_auc"] = auc(SL[:, j], pos); d["fitted_dinov3_auc"] = auc(SD[:, j], pos)
    if a in TF: d["template"] = TFNAME[a]; d["lens_auc"] = auc(TF[a], pos); d["lens_ci"] = boot(TF[a], pos); d["prediction_met"] = bool(d["lens_ci"][0] > 0.5)
    if a in FACET: d["facet_style_auc"] = auc(FACET[a], pos)
    out["per_category"][a] = d
# ---- three pair position tests
PAIRS = {"LS-Hori3 vs LS-Hori2": (9, 8, f("d_centre_h"), f("cov_hline") >= cal["tau_c"], "|y*-0.5|"), "L-Ver3 vs L-Ver2": (13, 12, f("d_centre_v"), f("cov_vline") >= cal["tau_c"], "|x*-0.5|"), "P-RoT vs P-Cent": (1, 3, f("blob_d_centre"), f("blob_peak") >= cal["tau_b"], "blob distance to centre")}
for k, (pk_, nk_, sc_, det, lab) in PAIRS.items():
    m = (y == pk_) | (y == nk_); s = sc_[m]; p = y[m] == pk_; r = dict(auc=auc(s, p), ci=boot(s, p), n_pos=int(p.sum()), n_neg=int((~p).sum()), detection_rate=float(det[m].mean()), score=lab, median_pos=float(np.median(s[p])), median_neg=float(np.median(s[~p])))
    r["reading_rule_met"] = bool(r["auc"] >= 0.65 and r["ci"][0] > 0.5); out["pairs"][k] = r
# ---- baselines on the three pairs, trained on the whole pool (138 per class), tested on LockboxB
Z = np.load(os.path.join(HERE, "results", "m3b_dino.npz")); zid = [str(i) for i in Z["ids"]]; zl = Z["labels"]; ZL = np.load(os.path.join(HERE, "results", "m3c_dino_lockboxb.npz")); lid6 = [str(i) for i in ZL["ids"]]; ll = ZL["labels"]
def grids(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255; g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA); b = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    e = cv2.resize(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (16, 16), interpolation=cv2.INTER_AREA).ravel(); return b, e
GP = [grids(f"{BASE}/data/images/{i}") for i in zid]; GL = [grids(f"{BASE}/data/lockboxb/{i}") for i in lid6]
TR = {"D-pool": Z["pooled"], "D-grid": Z["grid4"], "D-norm": Z["norm14"], "G-bright": np.array([g[0] for g in GP]), "G-edge": np.array([g[1] for g in GP])}
TE = {"D-pool": ZL["pooled"], "D-grid": ZL["grid4"], "D-norm": ZL["norm14"], "G-bright": np.array([g[0] for g in GL]), "G-edge": np.array([g[1] for g in GL])}
CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
for k, (pk_, nk_, sc_, det, lab) in PAIRS.items():
    mtr = (zl == pk_) | (zl == nk_); mte = (ll == pk_) | (ll == nk_); ytr = (zl[mtr] == pk_).astype(float); yte = ll[mte] == pk_; d = {}
    for b in TR:
        mdl = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=3, max_iter=300, scoring="roc_auc")).fit(TR[b][mtr], ytr); s = mdl.decision_function(TE[b][mte]); d[b] = dict(auc=auc(s, yte), ci=boot(s, yte))
    out["baselines"][k] = d
# ---- predictions
PA8 = ["P-RoT", "P-Cent", "P-Dia", "LS-Dia", "LS-Hori2", "LS-Hori3", "L-Ver2", "L-Ver3"]; PA3 = ["S-RoT", "S-Cent", "S-Per"]
out["predictions"] = {"PA_eight_met": {a: out["per_category"][a]["prediction_met"] for a in PA8}, "PA_three_unmet_expected": {a: out["per_category"][a]["prediction_met"] for a in PA3},
    "PB_all_three_pairs_read_positions_differ": {k: v["reading_rule_met"] for k, v in out["pairs"].items()}}
B = out["baselines"]; R = {k: v["auc"] for k, v in out["pairs"].items()}
out["predictions"]["PC"] = {"pooled_embedding_at_most_0.70_all_pairs": {k: B[k]["D-pool"]["auc"] <= 0.70 for k in B},
    "Ver_and_RoT_best_of_Dnorm_Gedge_matches_or_passes_reading": {k: max(B[k]["D-norm"]["auc"], B[k]["G-edge"]["auc"]) >= R[k] for k in ("L-Ver3 vs L-Ver2", "P-RoT vs P-Cent")},
    "Hori_best_baseline_no_more_than_0.03_above_reading": max(B["LS-Hori3 vs LS-Hori2"][b]["auc"] for b in TR) <= R["LS-Hori3 vs LS-Hori2"] + 0.03}
json.dump(out, open(os.path.join(HERE, "results", "m3c.json"), "w"), indent=1)
print(f"M3c LockboxB (n={len(y)})\n{'category':10s} {'template':18s} {'lens AUC [95% CI]':22s} {'facet':>7s} {'fit lens':>9s} {'fit DINOv3':>11s} met")
for k in range(1, 25):
    a = ABBR[k]; d = out["per_category"][a]
    print(f"{a:10s} {d.get('template', '-'):18s} {('%.3f [%.3f, %.3f]' % (d['lens_auc'], *d['lens_ci'])) if 'lens_auc' in d else '-':22s} {('%.3f' % d['facet_style_auc']) if 'facet_style_auc' in d else '-':>7s} {d['fitted_lens_auc']:9.3f} {d['fitted_dinov3_auc']:11.3f} {d.get('prediction_met', '')}")
print("pairs:")
for k, v in out["pairs"].items(): print("  ", k, "AUC %.3f [%.3f, %.3f]" % (v["auc"], *v["ci"]), "det %.2f" % v["detection_rate"], "rule met", v["reading_rule_met"], "| medians", round(v["median_pos"], 3), round(v["median_neg"], 3))
print("baselines (train pool 138/class, test LockboxB 40/class):")
for k, d in out["baselines"].items(): print("  ", k, {b: round(v["auc"], 3) for b, v in d.items()})
print("predictions:", json.dumps(out["predictions"], indent=1))
