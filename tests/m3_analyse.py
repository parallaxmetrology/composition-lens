"""M3 scorecard (PLAN.md section M3). python3 -I tests/m3_analyse.py devA | devB -> results/m3_<split>.json"""
import sys, os, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
ABBR = {1: "P-RoT", 2: "S-RoT", 3: "P-Cent", 4: "S-Cent", 5: "P-Dia", 6: "LS-Dia", 7: "P-Hori", 8: "LS-Hori2", 9: "LS-Hori3", 10: "S-Hori", 11: "P-Ver", 12: "L-Ver2", 13: "L-Ver3", 14: "L-VerMul", 15: "P-Tri", 16: "LS-Tri", 17: "LS-C-Cur", 18: "LS-O-Cur", 19: "LS-S-Cur", 20: "LS-Dif", 21: "S-Per", 22: "PL-Den", 23: "PL-Pat", 24: "P-Scat"}
which = sys.argv[1]; other = "devB" if which == "devA" else "devA"
sp = json.load(open(f"{BASE}/results/split.json")); emb = np.load(f"{BASE}/results/emb_dinov3_web_l16.npy"); row = {i: k for k, i in enumerate(sp["ids"])}
def load(w):
    rs = [r for r in json.load(open(os.path.join(HERE, "results", f"m3_features_{w}.json"))) if "error" not in r]; names = [k for k in rs[0] if k not in ("id", "label")]
    X = np.array([[r[k] for k in names] for r in rs], float); X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return rs, names, X, np.array([r["label"] for r in rs]), emb[[row[r["id"]] for r in rs]]
rs, names, X, y, D = load(which); col = {n: i for i, n in enumerate(names)}; cal = json.load(open(os.path.join(HERE, "results", "m2b_calibration.json")))
def auc(score, pos):
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
def boot(score, pos, B=1000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(pos)[0], np.where(~pos)[0]; v = []
    for _ in range(B):
        s = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); p = np.concatenate([np.ones(len(i1), bool), np.zeros(len(i0), bool)]); v.append(auc(score[s], p))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
f = lambda n: X[:, col[n]]
TF = {"P-RoT": f("z_thirds_pts"), "S-RoT": f("z_thirds_pts"), "P-Cent": f("z_centre_pt"), "S-Cent": f("z_centre_pt"), "LS-Hori2": f("z_centre_h"), "LS-Hori3": f("z_thirds_h"),
      "L-Ver2": f("z_centre_v"), "L-Ver3": f("z_thirds_v"), "LS-Dia": np.maximum(f("z_diag_baroque"), f("z_diag_sinister")), "P-Dia": np.maximum(f("z_diag_baroque"), f("z_diag_sinister")), "S-Per": f("persp_support_finite")}
TFNAME = {"P-RoT": "thirds_pts", "S-RoT": "thirds_pts", "P-Cent": "centre_pt", "S-Cent": "centre_pt", "LS-Hori2": "centre_h", "LS-Hori3": "thirds_h", "L-Ver2": "centre_v", "L-Ver3": "thirds_v", "LS-Dia": "diag (max of two)", "P-Dia": "diag (max of two)", "S-Per": "finite VP support"}
FACET = {"P-RoT": f("facet_power"), "S-RoT": f("facet_power"), "P-Cent": f("facet_center"), "S-Cent": f("facet_center"), "LS-Hori3": f("facet_thirds"), "L-Ver3": f("facet_thirds"), "LS-Hori2": f("facet_center"), "L-Ver2": f("facet_center")}
# fitted comparators: train on the other half
# other-half embeddings and labels come from the benchmark's split file (no lens features of the other half are read in the design phase)
oth = [i for i, sv in zip(sp["ids"], sp["split"]) if sv == (1 if which == "devA" else 0)]; yo = np.array([l for l, sv in zip(sp["labels"], sp["split"]) if sv == (1 if which == "devA" else 0)]); Do = emb[[row[i] for i in oth]]
def fit_scores(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=800).fit(sc.transform(Xtr), ytr); return m.decision_function(sc.transform(Xte)), m.classes_
SD, cl = fit_scores(Do, yo, D, 0.02)
if which == "devA":   # design phase: fitted lens probe by 5-fold cross-validation inside DevA (DevB lens features are not read)
    from sklearn.model_selection import StratifiedKFold
    SL = np.zeros((len(y), len(cl)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y): SL[te], _ = fit_scores(X[tr], y[tr], X[te], 0.1)
else:                 # confirmation: train on all of DevA's lens features
    _, _, Xa, ya, _ = load("devA"); SL, _ = fit_scores(Xa, ya, X, 0.1)
out = {"split": which, "n": int(len(y)), "calibration": cal, "per_category": {}}
for k in range(1, 25):
    a = ABBR[k]; pos = y == k; d = dict(n_pos=int(pos.sum())); j = list(cl).index(k)
    d["fitted_lens_auc"] = float(auc(SL[:, j], pos)); d["fitted_dinov3_auc"] = float(auc(SD[:, j], pos))
    if a in TF:
        d["template"] = TFNAME[a]; d["lens_auc"] = float(auc(TF[a], pos)); d["lens_ci"] = boot(TF[a], pos); d["prediction_met"] = bool(d["lens_ci"][0] > 0.5)
    if a in FACET: d["facet_style_auc"] = float(auc(FACET[a], pos))
    ex = sorted(((float(auc(X[:, i], pos)), n) for i, n in enumerate(names) if X[:, i].std() > 0), reverse=True)
    d["best_single_feature"] = ex[0]; d["worst_single_feature"] = ex[-1]
    out["per_category"][a] = d
# pairwise position tests
def pair(pos_k, neg_k, score, det, label):
    m = (y == pos_k) | (y == neg_k); s = score[m]; p = (y[m] == pos_k); r = dict(auc=float(auc(s, p)), ci=boot(s, p), n_pos=int(p.sum()), n_neg=int((~p).sum()), detection_rate=float(det[m].mean()))
    mo = (yo == pos_k) | (yo == neg_k); sc = StandardScaler().fit(Do[mo]); mdl = LogisticRegression(C=0.02, max_iter=800).fit(sc.transform(Do[mo]), yo[mo] == pos_k)
    r["fitted_dinov3_pair_auc"] = float(auc(mdl.decision_function(sc.transform(D[m])), p)); r["score"] = label
    r["median_score_pos"] = float(np.median(s[p])); r["median_score_neg"] = float(np.median(s[~p])); return r
out["pairs"] = {"LS-Hori3 vs LS-Hori2": pair(9, 8, f("d_centre_h"), f("cov_hline") >= cal["tau_c"], "|y*-0.5|"),
                "L-Ver3 vs L-Ver2": pair(13, 12, f("d_centre_v"), f("cov_vline") >= cal["tau_c"], "|x*-0.5|"),
                "P-RoT vs P-Cent": pair(1, 3, f("blob_d_centre"), f("blob_peak") >= cal["tau_b"], "blob distance to centre")}
for k, v in out["pairs"].items():
    a, lo = v["auc"], v["ci"][0]; v["reading"] = ("positions differ as defined (AUC>=0.65, CI>0.5)" if a >= 0.65 and lo > 0.5 else ("no difference in this reading" if (v["ci"][0] >= 0.45 and v["ci"][1] <= 0.55) and v["detection_rate"] >= 0.30 else ("inconclusive (detection rate < 30%)" if v["detection_rate"] < 0.30 else "intermediate / unclear")))
# kind distribution and y* position summary
out["kind_share"] = {n: float((f("kind") == i).mean()) for i, n in enumerate(["none", "hline", "vline", "diag", "point"])}
out["y_star_median_by_cat"] = {ABBR[k]: float(np.median(f("y_star")[y == k])) for k in (8, 9, 1, 3)}
json.dump(out, open(os.path.join(HERE, "results", f"m3_{which}.json"), "w"), indent=1)
print(f"M3 scorecard {which} (n={len(y)})\n{'category':10s} {'template':18s} {'lens AUC [95% CI]':22s} {'facet-style':>11s} {'fitted lens':>12s} {'fitted DINOv3':>14s}  met")
for k in range(1, 25):
    a = ABBR[k]; d = out["per_category"][a]
    print(f"{a:10s} {d.get('template', '-'):18s} {('%.3f [%.3f, %.3f]' % (d['lens_auc'], *d['lens_ci'])) if 'lens_auc' in d else '-':22s} {('%.3f' % d['facet_style_auc']) if 'facet_style_auc' in d else '-':>11s} {d['fitted_lens_auc']:12.3f} {d['fitted_dinov3_auc']:14.3f}  {d.get('prediction_met', '')}")
met = [v["prediction_met"] for v in out["per_category"].values() if "prediction_met" in v]; print("predictions met:", sum(met), "of", len(met))
print("best single feature per category (exploratory):", {a: (round(d["best_single_feature"][0], 3), d["best_single_feature"][1]) for a, d in out["per_category"].items()})
for k, v in out["pairs"].items(): print("PAIR", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("kind share", out["kind_share"], "y* medians", out["y_star_median_by_cat"])
