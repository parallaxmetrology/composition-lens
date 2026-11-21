"""Reads results/m2b_raw.json -> results/m2b.json (E1 to E5 of PLAN.md section M2b). Run: python3 -I tests/analyse_m2b.py"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from lens import engine as E
T0 = E.load_templates("v0"); TK = {t["id"]: {"hline": "hline", "vline": "vline", "points": "point", "diag": "diag"}[t["kind"]] for t in T0}
rows = json.load(open(os.path.join(HERE, "results", "m2b_raw.json"))); cal = json.load(open(os.path.join(HERE, "results", "m2b_calibration.json")))
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))
out = {"calibration": cal}; typ = lambda r: r["job"][0]
# ---- E1
e1 = [r for r in rows if typ(r) == "e1"]; e1n = [r for r in rows if typ(r) == "e1none"]; out["E1"] = {}
for ct in (None, 0.1, 0.05):
    key = "default" if ct is None else f"contrast{ct}"; conf = collections.defaultdict(collections.Counter); arb = []; unarb = []; unarb_z = []
    for r in e1:
        if r["job"][4] != ct: continue
        tid = r["job"][1]; tk = TK[tid]; conf[tk][r["kind"]] += 1
        cand = [b for b in TK if TK[b] == r["kind"]]; arb.append((max(cand, key=lambda b: r["rank_scores"][b]) == tid) if cand else False)
        unarb.append(max(r["rank_scores"], key=r["rank_scores"].get) == tid); unarb_z.append(max(r["scores"], key=r["scores"].get) == tid)
    for r in e1n:
        if r["job"][2] == ct: conf["none"][r["kind"]] += 1
    acc = sum(c[k] for k, c in conf.items()) / sum(sum(c.values()) for c in conf.values())
    out["E1"][key] = dict(kind_accuracy=float(acc), confusion={k: dict(v) for k, v in conf.items()}, arbitrated_top1_rank=float(np.mean(arb)), unarbitrated_top1_rank=float(np.mean(unarb)), unarbitrated_top1_z=float(np.mean(unarb_z)))
# ---- E2
e2 = [r for r in rows if typ(r) == "e2"]
def own(r): return r["scores"]["gt_TLBR" if r["job"][2] == "TLBR" else "gt_BLTR"]
def best(r): return max(r["scores"].values())
pos = [own(r) for r in e2 if r["job"][1] == "gt"]; sets = {m: [best(r) for r in e2 if r["job"][1] == m] for m in ("diag_only", "shifted", "random")}
out["E2"] = {"auc_vs_" + m: auc(pos, v) for m, v in sets.items()}; out["E2"]["auc_vs_nulls"] = auc(pos, sets["shifted"] + sets["random"])
out["E2"]["median_score"] = {"gt": float(np.median(pos)), **{m: float(np.median(v)) for m, v in sets.items()}}
# ---- E3
e3 = [r for r in rows if typ(r) == "e3"]; out["E3"] = {}
for a in ("4:3", "3:2", "4:5"):
    rs = [r for r in e3 if r["job"][1] == a]; rk = rs[0]["rk"]; cand_kinds = [t["id"] for t in T0 if t["kind"] == rk] + ["rabatment"]
    win = collections.Counter(max(cand_kinds, key=lambda b: r["rank_scores"][b]) for r in rs); out["E3"][a] = dict(orientation=rk, winners=dict(win), n=len(rs), rabatment_top1=win["rabatment"] / len(rs))
# ---- E4
e4 = [r for r in rows if typ(r) == "e4"]; tr = [r for r in e4 if r["job"][1] == "true"]; nul = [r for r in e4 if r["job"][1] in ("perturb", "straight")]
out["E4"] = dict(auc_vs_nulls=auc([r["score"] for r in tr], [r["score"] for r in nul]), auc_vs_perturbed=auc([r["score"] for r in tr], [r["score"] for r in e4 if r["job"][1] == "perturb"]), auc_vs_straight=auc([r["score"] for r in tr], [r["score"] for r in e4 if r["job"][1] == "straight"]),
                 variant_accuracy=float(np.mean([r["best"] == r["job"][2] for r in tr])), median_score_true=float(np.median([r["score"] for r in tr])), median_score_null=float(np.median([r["score"] for r in nul])))
sh = {}
for s in (0.01, 0.02, 0.04, 0.08):
    rs = [r for r in e4 if r["job"][1] == "shift" and abs(r["job"][4] - s) < 1e-9]; sh[str(s)] = dict(median_score=float(np.median([r["score"] for r in rs])), variant_accuracy=float(np.mean([r["best"] == r["job"][2] for r in rs])), auc_vs_nulls=auc([r["score"] for r in rs], [r["score"] for r in nul]))
out["E4"]["shift_sweep"] = sh
# ---- E5
e5 = [r for r in rows if typ(r) == "e5"]; out["E5"] = {}
for (nz, ct) in ((0.0, 0.3), (0.02, 0.3), (0.02, 0.1)):
    key = f"noise{nz}_contrast{ct}"; d = {}
    for k in (1, 2, 3):
        rs = [r for r in e5 if r["job"][1] == k and r["job"][3] == nz and r["job"][4] == ct]; nf = collections.Counter(min(r["n_finite"], 3) for r in rs)
        errs = [r["vp_err"] for r in rs if r["vp_err"] is not None]
        d[str(k)] = dict(n=len(rs), exact_count=nf[k] / len(rs), counts=dict(nf), vp_err_median=float(np.median(errs)) if errs else None, vp_err_p90=float(np.percentile(errs, 90)) if errs else None)
    d["mean_exact"] = float(np.mean([d[str(k)]["exact_count"] for k in (1, 2, 3)])); out["E5"][key] = d
out["gates"] = dict(E1_kind_acc_default_ge_0_9=out["E1"]["default"]["kind_accuracy"] >= 0.9, E2_auc_nulls_ge_0_9=out["E2"]["auc_vs_nulls"] >= 0.9, E4_auc_ge_0_9=out["E4"]["auc_vs_nulls"] >= 0.9, E5_count_ge_0_8_default=out["E5"]["noise0.02_contrast0.3"]["mean_exact"] >= 0.8)
json.dump(out, open(os.path.join(HERE, "results", "m2b.json"), "w"), indent=1)
for k, v in out["E1"].items(): print("E1", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "confusion"}); print("    ", v["confusion"])
print("E2", out["E2"]); print("E3", {a: (v["orientation"], v["winners"]) for a, v in out["E3"].items()}); print("E4", {k: v for k, v in out["E4"].items() if k != "shift_sweep"}); print("   shift", out["E4"]["shift_sweep"])
for k, v in out["E5"].items(): print("E5", k, {a: (b if a == "mean_exact" else (round(b["exact_count"], 2), b["counts"], b["vp_err_median"] and round(b["vp_err_median"], 3))) for a, b in v.items()})
print("gates", out["gates"])
