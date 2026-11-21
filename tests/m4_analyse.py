"""M4 analysis on CADB (PLAN.md section M4). python3 -I tests/m4_analyse.py -> results/m4.json"""
import os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
CAD = os.path.join(HERE, "data", "cadb", "CADB_Dataset"); el = json.load(open(os.path.join(CAD, "composition_elements.json"))); sp = json.load(open(os.path.join(CAD, "split.json"))); sc = json.load(open(os.path.join(CAD, "scene_categories.json")))
CLASSES = ["center", "horizontal", "diagonal", "vertical", "rule_of_thirds", "triangle", "symmetric", "golden_ratio", "curved", "fill_the_frame", "vanishing_point", "pattern", "radial", "none"]
rs = [r for r in json.load(open(os.path.join(HERE, "results", "m4_features_cadb.json"))) if "error" not in r]; names = [k for k in rs[0] if k != "id"]; ids = [r["id"] for r in rs]
assert ids == sp["all"][:len(ids)] and len(ids) == len(sp["all"])
X = np.nan_to_num(np.array([[r[k] for k in names] for r in rs], float), nan=0.0, posinf=0.0, neginf=0.0); col = {n: i for i, n in enumerate(names)}; f = lambda n: X[:, col[n]]
D = np.load(os.path.join(HERE, "results", "m4_dino_cadb.npy")); test = np.array([i in set(sp["test"]) for i in ids]); train = np.array([i in set(sp["train"]) for i in ids])
Y = {c: np.array([c in el[i] for i in ids]) for c in CLASSES}
def auc(s, p):
    s = np.asarray(s, float); p = np.asarray(p, bool); r = rankdata(s); n1 = p.sum(); n0 = len(p) - n1; return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
def boot(s, p, B=1000, seed=0):
    s = np.asarray(s, float); p = np.asarray(p, bool); rng = np.random.default_rng(seed); i1, i0 = np.where(p)[0], np.where(~p)[0]; v = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); v.append(auc(s[ii], p[ii]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
mx = lambda *ns: np.max([f(n) for n in ns], 0)
TF = {"rule_of_thirds": (mx("z_thirds_h", "z_thirds_v", "z_thirds_pts"), "max thirds_h/v/pts"), "center": (mx("z_centre_h", "z_centre_v", "z_centre_pt"), "max centre_h/v/pt"), "horizontal": (f("cov_hline"), "line coverage (h)"), "vertical": (f("cov_vline"), "line coverage (v)"),
      "diagonal": (mx("z_diag_baroque", "z_diag_sinister"), "max of two diagonals"), "golden_ratio": (mx("z_phi_h", "z_phi_v", "z_phi_pts"), "max phi_h/v/pts"), "triangle": (mx("z_gt_TLBR", "z_gt_BLTR"), "max golden triangle"), "curved": (f("z_spiral"), "golden spiral"), "vanishing_point": (f("persp_support_finite"), "finite VP support")}
FACET = {"rule_of_thirds": f("facet_power"), "center": f("facet_center")}
def fit(Xtr, ytr, Xte, C):
    sc_ = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=800, class_weight="balanced").fit(sc_.transform(Xtr), ytr); return m.decision_function(sc_.transform(Xte))
out = {"n": len(ids), "n_test": int(test.sum()), "n_train": int(train.sum()), "classes": {}, "T2": {}}
for c in CLASSES:
    d = dict(n_test_pos=int(Y[c][test].sum()), n_all_pos=int(Y[c].sum()))
    if d["n_test_pos"] >= 5:
        d["fitted_lens_auc"] = auc(fit(X[train], Y[c][train], X[test], 0.1), Y[c][test]); d["fitted_dinov3_auc"] = auc(fit(D[train], Y[c][train], D[test], 0.02), Y[c][test])
    if c in TF:
        s, nm = TF[c]; d["score"] = nm; d["lens_auc_test"] = auc(s[test], Y[c][test]); d["lens_ci_test"] = boot(s[test], Y[c][test]); d["lens_auc_all"] = auc(s, Y[c]); d["lens_ci_all"] = boot(s, Y[c]); d["met_test"] = bool(d["lens_ci_test"][0] > 0.5)
    if c in FACET: d["facet_auc_test"] = auc(FACET[c][test], Y[c][test]); d["facet_auc_all"] = auc(FACET[c], Y[c])
    out["classes"][c] = d
# ---- T2: agreement with the annotators' own marks
def perm_test(obs_err, lens_v, marks, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); meds = [np.median(fn(lens_v, marks[rng.permutation(len(marks))])) for _ in range(B)]; return float(np.mean(np.array(meds) <= np.median(obs_err)))
def t2(mask, label):
    res = {}
    for cls, key, lens_key, dim, tol in (("horizontal", "y_star", None, "h", 0.05), ("vertical", "x_star", None, "w", 0.05)):
        sel = [k for k, i in enumerate(ids) if mask[k] and len(el[i].get(cls, [])) == 1]
        if len(sel) < 20: continue
        marks = np.array([((el[ids[k]][cls][0][1] + el[ids[k]][cls][0][3]) / 2 / rs[k]["img_h"]) if dim == "h" else ((el[ids[k]][cls][0][0] + el[ids[k]][cls][0][2]) / 2 / rs[k]["img_w"]) for k in sel]); lv = np.array([rs[k][key] for k in sel])
        err = np.abs(lv - marks); res[cls] = dict(n=len(sel), median_error=float(np.median(err)), within_0_05=float(np.mean(err <= tol)), spearman=float(spearmanr(lv, marks)[0]), p_vs_permuted=perm_test(err, lv, marks, lambda a, b: np.abs(a - b)), null_median_error=float(np.median(np.abs(lv - marks[np.random.default_rng(1).permutation(len(marks))]))))
    for cls in ("center", "rule_of_thirds"):
        sel = [k for k, i in enumerate(ids) if mask[k] and len(el[i].get(cls, [])) == 1 and len(el[i][cls][0]) == 4]
        if len(sel) < 20: continue
        mk = np.array([[(el[ids[k]][cls][0][0] + el[ids[k]][cls][0][2]) / 2 / rs[k]["img_w"], (el[ids[k]][cls][0][1] + el[ids[k]][cls][0][3]) / 2 / rs[k]["img_h"]] for k in sel]); lv = np.array([[rs[k]["blob_x"], rs[k]["blob_y"]] for k in sel]); err = np.hypot(*(lv - mk).T)
        res[cls + "_box"] = dict(n=len(sel), median_error=float(np.median(err)), within_0_10=float(np.mean(err <= 0.10)), p_vs_permuted=perm_test(err, lv, mk, lambda a, b: np.hypot(*(a - b).T)), null_median_error=float(np.median(np.hypot(*(lv - mk[np.random.default_rng(1).permutation(len(mk))]).T))))
    return res
out["T2"]["test"] = t2(test, "test"); out["T2"]["all"] = t2(np.ones(len(ids), bool), "all")
P1 = ["rule_of_thirds", "center", "horizontal", "vertical", "diagonal"]; n1 = sum(out["classes"][c]["met_test"] for c in P1)
out["predictions"] = {"P1_met": {c: out["classes"][c]["met_test"] for c in P1}, "P1_count": n1, "P1_reading": "supports generalisation" if n1 >= 4 else ("mixed" if n1 == 3 else "crosswalk specific to PICD"), "P2_reported": {c: out["classes"][c]["lens_auc_test"] for c in ("golden_ratio", "triangle", "curved")},
    "P3_vanishing_point_met": out["classes"]["vanishing_point"]["met_test"], "T2_a_b_beat_permutation": {s: {c: (out["T2"][s][c]["p_vs_permuted"] < 0.05) for c in ("horizontal", "vertical") if c in out["T2"][s]} for s in out["T2"]}}
json.dump(out, open(os.path.join(HERE, "results", "m4.json"), "w"), indent=1)
print(f"M4 CADB (n={len(ids)}, test {int(test.sum())})\n{'class':15s} {'score':20s} {'AUC test [95% CI]':22s} {'AUC all':>8s} {'facet':>7s} {'fit lens':>9s} {'fit DINOv3':>11s} met  (pos test/all)")
for c in CLASSES:
    d = out["classes"][c]; print(f"{c:15s} {d.get('score', '-'):20s} {('%.3f [%.3f, %.3f]' % (d['lens_auc_test'], *d['lens_ci_test'])) if 'lens_auc_test' in d else '-':22s} {('%.3f' % d['lens_auc_all']) if 'lens_auc_all' in d else '-':>8s} {('%.3f' % d['facet_auc_test']) if 'facet_auc_test' in d else '-':>7s} {('%.3f' % d['fitted_lens_auc']) if 'fitted_lens_auc' in d else '-':>9s} {('%.3f' % d['fitted_dinov3_auc']) if 'fitted_dinov3_auc' in d else '-':>11s} {d.get('met_test', '')!s:5s} ({d['n_test_pos']}/{d['n_all_pos']})")
print("T2:", json.dumps(out["T2"], indent=1)); print("predictions:", json.dumps(out["predictions"], indent=1))
