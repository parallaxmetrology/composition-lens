"""Reads results/recovery_raw.json -> results/recovery.json (R1 to R4 of PLAN.md section 4). Run: python3 -I tests/analyse_recovery.py"""
import os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from lens import engine
T = engine.load_templates("v0"); IDS = [t["id"] for t in T]; FAM = {t["id"]: t["family"] for t in T}; KIND = {t["id"]: t["kind"] for t in T}
rows = json.load(open(os.path.join(HERE, "results", "recovery_raw.json")))
CONDS = [(0.0, None), (0.02, None), (0.05, None), (0.02, 0.10), (0.02, 0.05)]; KEY = lambda c: f'noise{c[0]}_contrast{c[1] if c[1] is not None else "default"}'
def cond(r): return (r['job'][3], r['job'][5])
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); 
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))
out = {"R1": {}, "R2": {}, "R3": {}, "R4": {}}
exact = [r for r in rows if r["tag"][0] == "exact"]; null = [r for r in rows if r["tag"][0] == "null"]
# R1 position recovery
for cd in CONDS:
    e = {"hline": [], "vline": []}
    for r in exact:
        if cond(r) != cd: continue
        k = KIND[r["tag"][1]]
        if k == "hline": e["hline"].append(abs(r["readings"]["y_star"] - r["tag"][2]))
        if k == "vline": e["vline"].append(abs(r["readings"]["x_star"] - r["tag"][2]))
    out["R1"][KEY(cd)] = {k: dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in e.items()}
# R2 top-1 and confusion
for cd in CONDS:
    conf = {a: {b: 0 for b in IDS} for a in IDS}; n = {a: 0 for a in IDS}
    for r in exact:
        if cond(r) != cd: continue
        tid = r["tag"][1]; pred = max(r["scores"], key=r["scores"].get); conf[tid][pred] += 1; n[tid] += 1
    top1 = {a: conf[a][a] / n[a] for a in IDS}; fam = {a: sum(v for b, v in conf[a].items() if FAM[b] == FAM[a]) / n[a] for a in IDS}
    out["R2"][KEY(cd)] = dict(top1=top1, family=fam, confusion=conf, mean_top1=float(np.mean(list(top1.values()))), mean_family=float(np.mean(list(fam.values()))))
# R2b within-kind top-1 (exploratory, added after seeing R2: cross-kind ranking is confounded because a point on a power point also makes line profiles peak)
for cd in CONDS:
    d = {}
    for stat in ("scores", "rank_scores"):
        acc = {}
        for a in IDS:
            rs = [r for r in exact if cond(r) == cd and r["tag"][1] == a]
            cand = [b for b in IDS if KIND[b] == KIND[a]]
            acc[a] = float(np.mean([max(cand, key=lambda b: r[stat][b]) == a for r in rs]))
        d[stat] = dict(per_template=acc, mean=float(np.mean(list(acc.values()))))
    out["R2"][KEY(cd)]["within_kind"] = d
# R3 AUC vs null
for cd in CONDS:
    neg = {a: [r["scores"][a] for r in null if cond(r) == cd] for a in IDS}
    out["R3"][KEY(cd)] = {a: auc([r["scores"][a] for r in exact if cond(r) == cd and r["tag"][1] == a], neg[a]) for a in IDS}
# R4 resolution sweep and pairwise separation
sw = {}
for r in rows:
    if r["tag"][0] == "sweep": sw.setdefault(r["tag"][2], []).append(r)
out["R4"]["sweep"] = {str(p): dict(median_pos_err=float(np.median([abs(x["readings"]["y_star"] - p) for x in v])),
                                   winner_share={w: float(np.mean([max(x["scores"], key=x["scores"].get) == w for x in v])) for w in ("thirds_h", "phi_h", "centre_h")}) for p, v in sorted(sw.items())}
pr = {}
for r in rows:
    if r["tag"][0] == "pair": pr.setdefault(r["tag"][1], []).append(r)
def sep(a, b, ta, tb):
    sa = [r["scores"][ta] - r["scores"][tb] for r in pr[a]]; sb = [r["scores"][ta] - r["scores"][tb] for r in pr[b]]
    return float((np.mean(np.array(sa) > 0) + np.mean(np.array(sb) < 0)) / 2)
out["R4"]["pairwise_accuracy"] = {"thirds_h vs phi_h on lines at 1/3 vs 0.382": sep("1/3", "phi", "thirds_h", "phi_h"),
                                   "phi_h vs centre_h on lines at 0.382 vs 1/2": sep("phi", "1/2", "phi_h", "centre_h"),
                                   "thirds_h vs centre_h on lines at 1/3 vs 1/2": sep("1/3", "1/2", "thirds_h", "centre_h")}
# gates
g = {"R1_median_under_1pct_at_zero_noise": all(v["median"] < 0.01 for v in out["R1"][KEY(CONDS[0])].values()),
     "R3_AUC_at_least_0.9_at_zero_noise": {a: v >= 0.9 for a, v in out["R3"][KEY(CONDS[0])].items()}}
out["gates"] = g
json.dump(out, open(os.path.join(HERE, "results", "recovery.json"), "w"), indent=1)
print("R1 (error, fraction of frame):"); [print(" noise", k, {a: (round(b["median"], 4), round(b["p95"], 4)) for a, b in v.items()}) for k, v in out["R1"].items()]
for cd in CONDS:
    r2 = out["R2"][KEY(cd)]; print(f"R2 {KEY(cd)}: mean top-1 {r2['mean_top1']:.2f}, mean family {r2['mean_family']:.2f}"); print("   ", {a: round(v, 2) for a, v in r2["top1"].items()})
for cd in CONDS: print(f"R3 AUC {KEY(cd)}:", {a: round(v, 2) for a, v in out["R3"][KEY(cd)].items()})
for cd in CONDS:
    w = out["R2"][KEY(cd)]["within_kind"]; print(f"R2b within-kind {KEY(cd)}: z-score {w['scores']['mean']:.2f}, rank-score {w['rank_scores']['mean']:.2f}", {a: (round(w['scores']['per_template'][a], 2), round(w['rank_scores']['per_template'][a], 2)) for a in IDS} if cd == CONDS[1] else "")
print("R4 pairwise:", out["R4"]["pairwise_accuracy"]); print("gates:", g)
