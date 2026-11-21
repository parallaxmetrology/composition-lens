"""M2 recovery tests on exact synthetic layouts (PLAN.md section 4). Writes results/recovery.json. Run: python3 -I tests/run_recovery.py"""
import sys, os, json, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
from lens import engine, synth
T = engine.load_templates("v0"); IDS = [t["id"] for t in T]; FAM = {t["id"]: t["family"] for t in T}
CONDS = [(0.0, None), (0.02, None), (0.05, None), (0.02, 0.10), (0.02, 0.05)]; ASP = list(synth.ASPECTS); N_EACH = 30; N_NULL = 15
CAT = synth.CAT_X
def _task(job):
    kind, pos, aspect, noise, seed, kw = job; ct = kw.get('contrast')
    r = engine.readings_and_fits(synth.render(kind, pos, aspect, noise, seed, **kw))
    return dict(job=[kind, pos if not isinstance(pos, tuple) else list(pos), aspect, noise, seed, ct], readings=r["readings"], scores={k: v["score"] for k, v in r["fits"].items()}, rank_scores={k: v["rank_score"] for k, v in r["fits"].items()})
def place(t, prim):
    k = t["kind"]
    return ("hline", prim) if k == "hline" else ("vline", prim) if k == "vline" else ("point", tuple(prim)) if k == "points" else ("diag", prim)
def null_jobs(rng):
    jobs = []
    for ai, a in enumerate(ASP):
        for ci, (nz, ct) in enumerate(CONDS):
            kw0 = {} if ct is None else {"contrast": ct}
            for rep in range(N_NULL):
                s = 900000 + ai * 10000 + ci * 1000 + rep * 10
                def off():
                    while True:
                        v = rng.uniform(0.12, 0.88)
                        if min(abs(v - c) for c in CAT) > 0.04: return v
                jobs.append(("hline", off(), a, nz, s, dict(kw0))); jobs.append(("vline", off(), a, nz, s + 1, dict(kw0)))
                while True:
                    px, py = rng.uniform(0.12, 0.88, 2)
                    if min(np.hypot(px - x, py - y) for x in CAT for y in CAT) > 0.06: break
                jobs.append(("point", (px, py), a, nz, s + 2, dict(kw0)))
                jobs.append(("diag", (rng.choice(["BL-TR", "TL-BR"]), float(rng.choice([-1, 1]) * rng.uniform(0.12, 0.35))), a, nz, s + 3, dict(kw0)))
                jobs.append(("hline", 0.5, a, nz, s + 4, {"contrast": 0.0}))   # texture only
    return jobs
def main():
    t0 = time.time(); rng = np.random.default_rng(12345); jobs, tags = [], []
    for ti, t in enumerate(T):
        for ai, a in enumerate(ASP):
            for ni, (nz, ct) in enumerate(CONDS):
                for rep in range(N_EACH):
                    prim = t["primitives"][rep % len(t["primitives"])]; kind, pos = place(t, prim)
                    jobs.append((kind, pos, a, nz, 100000 + ti * 10000 + ai * 1000 + ni * 100 + rep, {} if ct is None else {"contrast": ct})); tags.append(("exact", t["id"], prim))
    nj = null_jobs(rng); jobs += nj; tags += [("null", "", None)] * len(nj)
    # resolution sweep (4:3, noise 0.02): single horizontal line at p
    sweep = [round(p, 3) for p in np.arange(0.30, 0.4001, 0.005)]
    for pi, p in enumerate(sweep):
        for rep in range(20): jobs.append(("hline", p, "4:3", 0.02, 500000 + pi * 100 + rep, {})); tags.append(("sweep", "", p))
    # pairwise separation tasks (4:3, noise 0.02)
    pairs = {"1/3": 1 / 3, "phi": 0.381966, "1/2": 0.5}
    for name, p in pairs.items():
        for rep in range(100): jobs.append(("hline", p, "4:3", 0.02, 700000 + int(p * 1000) * 1000 + rep, {})); tags.append(("pair", name, p))
    print(len(jobs), "images", flush=True)
    with Pool(max(1, os.cpu_count() - 1)) as pool: res = pool.map(_task, jobs, chunksize=40)
    rows = [dict(tag=list(tg), **r) for tg, r in zip(tags, res)]
    json.dump(rows, open(os.path.join(HERE, "results", "recovery_raw.json"), "w"))
    print("done in %.0fs" % (time.time() - t0))
if __name__ == "__main__": main()
