"""Recomputes the real-photograph numbers from derived/ and compares them with the recorded results.
Tier 1 (always): the PICD crosswalk AUCs, the three position pairs and the Facet-style comparator on DevA, DevB and LockboxB, from derived/picd_features_*.json.
Tier 2 (if CADB is present at data/cadb/CADB_Dataset or CADB_ROOT): the CADB class AUCs and the agreement with the annotators' marks, from derived/cadb_features.json.
Not recomputed here: the fitted probes (they need the DINOv3 embeddings), the learning curves, and the synthetic tests (see README for the commands).
Run: python reproduce.py"""
import json, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = lambda *p: os.path.join(HERE, *p)
checks = []; bad = []
def check(name, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) <= tol; checks.append(ok)
    if not ok: bad.append((name, got, want))
def auc(s, p):
    s = np.asarray(s, float); p = np.asarray(p, bool); r = rankdata(s); n1 = p.sum(); n0 = len(p) - n1; return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
def boot(s, p, B=1000, seed=0):
    s = np.asarray(s, float); p = np.asarray(p, bool); rng = np.random.default_rng(seed); i1, i0 = np.where(p)[0], np.where(~p)[0]; v = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); v.append(auc(s[ii], p[ii]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
ABBR = {1: "P-RoT", 2: "S-RoT", 3: "P-Cent", 4: "S-Cent", 5: "P-Dia", 6: "LS-Dia", 7: "P-Hori", 8: "LS-Hori2", 9: "LS-Hori3", 10: "S-Hori", 11: "P-Ver", 12: "L-Ver2", 13: "L-Ver3", 14: "L-VerMul", 15: "P-Tri", 16: "LS-Tri", 17: "LS-C-Cur", 18: "LS-O-Cur", 19: "LS-S-Cur", 20: "LS-Dif", 21: "S-Per", 22: "PL-Den", 23: "PL-Pat", 24: "P-Scat"}
cal = json.load(open(R("results", "m2b_calibration.json")))
# ------------------------------------------------------------------ tier 1: PICD
for tag, ff, rf, key in (("DevA", "picd_features_devA.json", "m3_devA.json", "lens_auc"), ("DevB", "picd_features_devB.json", "m3_devB.json", "lens_auc"), ("LockboxB", "picd_features_lockboxb.json", "m3c.json", "lens_auc")):
    rs = [r for r in json.load(open(R("derived", ff))) if "error" not in r]; names = [k for k in rs[0] if k not in ("id", "label")]; res = json.load(open(R("results", rf)))
    X = np.nan_to_num(np.array([[r[k] for k in names] for r in rs], float), nan=0.0, posinf=0.0, neginf=0.0); y = np.array([r["label"] for r in rs]); col = {n: i for i, n in enumerate(names)}; f = lambda n: X[:, col[n]]
    TF = {"P-RoT": f("z_thirds_pts"), "S-RoT": f("z_thirds_pts"), "P-Cent": f("z_centre_pt"), "S-Cent": f("z_centre_pt"), "LS-Hori2": f("z_centre_h"), "LS-Hori3": f("z_thirds_h"), "L-Ver2": f("z_centre_v"), "L-Ver3": f("z_thirds_v"),
          "LS-Dia": np.maximum(f("z_diag_baroque"), f("z_diag_sinister")), "P-Dia": np.maximum(f("z_diag_baroque"), f("z_diag_sinister")), "S-Per": f("persp_support_finite")}
    FACET = {"P-RoT": f("facet_power"), "S-RoT": f("facet_power"), "P-Cent": f("facet_center"), "S-Cent": f("facet_center"), "LS-Hori3": f("facet_thirds"), "L-Ver3": f("facet_thirds"), "LS-Hori2": f("facet_center"), "L-Ver2": f("facet_center")}
    for k in range(1, 25):
        a = ABBR[k]; pos = y == k; d = res["per_category"][a]
        if a in TF:
            check(f"{tag} {a} lens AUC", auc(TF[a], pos), d["lens_auc"]); ci = boot(TF[a], pos); check(f"{tag} {a} CI low", ci[0], d["lens_ci"][0]); check(f"{tag} {a} CI high", ci[1], d["lens_ci"][1])
        if a in FACET: check(f"{tag} {a} Facet-style AUC", auc(FACET[a], pos), d["facet_style_auc"])
    for pk, (pos_k, neg_k, sc_, det) in {"LS-Hori3 vs LS-Hori2": (9, 8, f("d_centre_h"), f("cov_hline") >= cal["tau_c"]), "L-Ver3 vs L-Ver2": (13, 12, f("d_centre_v"), f("cov_vline") >= cal["tau_c"]), "P-RoT vs P-Cent": (1, 3, f("blob_d_centre"), f("blob_peak") >= cal["tau_b"])}.items():
        m = (y == pos_k) | (y == neg_k); s = sc_[m]; p = y[m] == pos_k; want = res["pairs"][pk]; check(f"{tag} {pk} AUC", auc(s, p), want["auc"]); ci = boot(s, p); check(f"{tag} {pk} CI low", ci[0], want["ci"][0]); check(f"{tag} {pk} CI high", ci[1], want["ci"][1]); check(f"{tag} {pk} detection rate", det[m].mean(), want["detection_rate"])
    print(f"PICD {tag}: {len(rs)} images checked")
# ------------------------------------------------------------------ tier 2: CADB
cad = os.environ.get("CADB_ROOT", R("data", "cadb", "CADB_Dataset"))
if os.path.exists(os.path.join(cad, "composition_elements.json")):
    el = json.load(open(os.path.join(cad, "composition_elements.json"))); sp = json.load(open(os.path.join(cad, "split.json"))); rs = json.load(open(R("derived", "cadb_features.json"))); m4 = json.load(open(R("results", "m4.json")))
    ids = [r["id"] for r in rs]; assert ids == sp["all"]; test = np.array([i in set(sp["test"]) for i in ids]); col = lambda n: np.array([r[n] for r in rs], float); mx = lambda *ns: np.max([col(n) for n in ns], 0)
    TF = {"rule_of_thirds": mx("z_thirds_h", "z_thirds_v", "z_thirds_pts"), "center": mx("z_centre_h", "z_centre_v", "z_centre_pt"), "horizontal": col("cov_hline"), "vertical": col("cov_vline"), "diagonal": mx("z_diag_baroque", "z_diag_sinister"),
          "golden_ratio": mx("z_phi_h", "z_phi_v", "z_phi_pts"), "triangle": mx("z_gt_TLBR", "z_gt_BLTR"), "curved": col("z_spiral"), "vanishing_point": col("persp_support_finite")}
    for c, sc_ in TF.items():
        Y = np.array([c in el[i] for i in ids]); d = m4["classes"][c]; check(f"CADB {c} AUC (test)", auc(sc_[test], Y[test]), d["lens_auc_test"]); ci = boot(sc_[test], Y[test]); check(f"CADB {c} CI low", ci[0], d["lens_ci_test"][0]); check(f"CADB {c} CI high", ci[1], d["lens_ci_test"][1]); check(f"CADB {c} AUC (all)", auc(sc_, Y), d["lens_auc_all"])
    def perm(err, lens_v, marks, fn, B=1000, seed=0):
        rng = np.random.default_rng(seed); meds = [np.median(fn(lens_v, marks[rng.permutation(len(marks))])) for _ in range(B)]; return float(np.mean(np.array(meds) <= np.median(err)))
    for tagm, mask in (("test", test), ("all", np.ones(len(ids), bool))):
        for cls, key, dim in (("horizontal", "y_star", "h"), ("vertical", "x_star", "w")):
            sel = [k for k, i in enumerate(ids) if mask[k] and len(el[i].get(cls, [])) == 1]; want = m4["T2"][tagm][cls]
            marks = np.array([((el[ids[k]][cls][0][1] + el[ids[k]][cls][0][3]) / 2 / rs[k]["img_h"]) if dim == "h" else ((el[ids[k]][cls][0][0] + el[ids[k]][cls][0][2]) / 2 / rs[k]["img_w"]) for k in sel]); lv = np.array([rs[k][key] for k in sel]); err = np.abs(lv - marks)
            check(f"CADB {tagm} {cls} n", len(sel), want["n"]); check(f"CADB {tagm} {cls} median error", np.median(err), want["median_error"]); check(f"CADB {tagm} {cls} within 0.05", np.mean(err <= 0.05), want["within_0_05"]); check(f"CADB {tagm} {cls} spearman", spearmanr(lv, marks)[0], want["spearman"], 1e-9); check(f"CADB {tagm} {cls} p vs permuted", perm(err, lv, marks, lambda a, b: np.abs(a - b)), want["p_vs_permuted"])
        for cls in ("center", "rule_of_thirds"):
            sel = [k for k, i in enumerate(ids) if mask[k] and len(el[i].get(cls, [])) == 1 and len(el[i][cls][0]) == 4]; want = m4["T2"][tagm][cls + "_box"]
            mk = np.array([[(el[ids[k]][cls][0][0] + el[ids[k]][cls][0][2]) / 2 / rs[k]["img_w"], (el[ids[k]][cls][0][1] + el[ids[k]][cls][0][3]) / 2 / rs[k]["img_h"]] for k in sel]); lv = np.array([[rs[k]["blob_x"], rs[k]["blob_y"]] for k in sel]); err = np.hypot(*(lv - mk).T)
            check(f"CADB {tagm} {cls} box n", len(sel), want["n"]); check(f"CADB {tagm} {cls} box median error", np.median(err), want["median_error"]); check(f"CADB {tagm} {cls} box p vs permuted", perm(err, lv, mk, lambda a, b: np.hypot(*(a - b).T)), want["p_vs_permuted"])
    print(f"CADB: {len(ids)} images checked")
else:
    print("CADB tier skipped: unzip CADB_Dataset.zip into data/cadb/ (or set CADB_ROOT) to also recompute the CADB numbers.")
print(f"\n{sum(checks)} of {len(checks)} checks match" + ("" if not bad else f"; {len(bad)} MISMATCH:"))
for b in bad[:20]: print("  ", b)
sys.exit(1 if bad else 0)
