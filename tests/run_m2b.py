"""M2b tests (PLAN.md section M2b). Usage: python3 -I tests/run_m2b.py cal | test. Writes results/m2b_calibration.json | results/m2b_raw.json."""
import sys, os, json, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
from lens import engine as E, m2b, synth, synth2
T0 = E.load_templates("v0"); IDS = [t["id"] for t in T0]; ASP = list(synth.ASPECTS)
KCONDS = [(0.02, None), (0.02, 0.10), (0.02, 0.05)]
def kw(ct): return {} if ct is None else {"contrast": ct}
def place(t, prim):
    k = t["kind"]; return ("hline", prim) if k == "hline" else ("vline", prim) if k == "vline" else ("point", tuple(prim)) if k == "points" else ("diag", prim)
TKIND = {"hline": "hline", "vline": "vline", "points": "point", "diag": "diag"}

def job_fn(job):
    typ = job[0]
    if typ == "cal":
        _, kind, aspect, ct, seed = job; pos = {"hline": 1 / 3, "vline": 0.5, "point": (2 / 3, 1 / 3), "diag": "TL-BR", "none": 0.5}[kind]
        rng = np.random.default_rng(seed); 
        if kind == "hline": pos = float(rng.choice([1 / 3, 0.5, 0.381966, 0.6, 0.2]))
        if kind == "vline": pos = float(rng.choice([1 / 3, 0.5, 0.381966, 0.6, 0.8]))
        if kind == "point": pos = (float(rng.choice([1 / 3, 2 / 3, 0.5])), float(rng.choice([1 / 3, 2 / 3, 0.5])))
        if kind == "diag": pos = str(rng.choice(["TL-BR", "BL-TR"]))
        k2 = "hline" if kind == "none" else kind; kwargs = {"contrast": 0.0} if kind == "none" else kw(ct)
        r = m2b.kind_reading(synth.render(k2, pos, aspect, 0.02, seed, **kwargs), cal={"tau_c": 0.5, "tau_b": 0.0})
        return dict(job=list(job), cov=r["coverage"], blob=r["blob_peak"])
    if typ == "e1":
        _, tid, prim, aspect, ct, seed = job; t = next(x for x in T0 if x["id"] == tid); kind, pos = place(t, prim)
        im = synth.render(kind, pos, aspect, 0.02, seed, **kw(ct)); kr = m2b.kind_reading(im); r = E.readings_and_fits(im)
        return dict(job=list(job[:4]) + [ct, seed], kind=kr["kind"], cov=kr["coverage"], blob=kr["blob_peak"], scores={k: v["score"] for k, v in r["fits"].items()}, rank_scores={k: v["rank_score"] for k, v in r["fits"].items()})
    if typ == "e1none":
        _, aspect, ct, seed = job; im = synth.render("hline", 0.5, aspect, 0.02, seed, contrast=0.0); kr = m2b.kind_reading(im)
        return dict(job=list(job), kind=kr["kind"], cov=kr["coverage"], blob=kr["blob_peak"])
    if typ == "e2":
        _, mode, orient, aspect, seed = job; im = synth2.render_triangle(mode, orient, aspect, 0.02, seed); ev = E.evidence(im); g = m2b.golden_triangle_fits(ev)
        return dict(job=list(job), scores={k: v["score"] for k, v in g.items()}, z={k: v["z_segments"] for k, v in g.items()})
    if typ == "e3":
        _, aspect, prim, seed = job; W, H = synth.ASPECTS[aspect]; rt = m2b.rabatment_template(W, H); kind = "vline" if rt["kind"] == "vline" else "hline"
        im = synth.render(kind, prim, aspect, 0.02, seed); tl = T0 + [rt]; r = E.readings_and_fits(im, templates=tl)
        return dict(job=list(job), rk=rt["kind"], rank_scores={k: v["rank_score"] for k, v in r["fits"].items()}, scores={k: v["score"] for k, v in r["fits"].items()})
    if typ == "e4":
        _, mode, variant, aspect, extra, seed = job
        if mode == "true": im = synth2.render_spiral(variant, aspect, 0.02, seed)
        elif mode == "shift": im = synth2.render_spiral(variant, aspect, 0.02, seed, shift=extra)
        elif mode == "perturb": im = synth2.render_spiral(variant, aspect, 0.02, seed, perturb=True)
        else: im = synth2.render_spiral(variant, aspect, 0.02, seed, straight=True)
        ev = E.evidence(im); sp = m2b.spiral_fits(ev); return dict(job=list(job), best=sp["best_variant"], score=m2b.spiral_score(sp), per_variant=sp["per_variant"])
    if typ == "e5":
        _, k, aspect, noise, ct, seed = job; im, vps = synth2.render_perspective(k, aspect, noise, seed, ct); vr = m2b.vanishing_reading(im)
        fam = [f for f in vr["families"] if f["finite"]]; err = None
        if fam:
            f0 = max(fam, key=lambda f: f["support"]); err = float(min(np.hypot(f0["vp"][0] - v[0], f0["vp"][1] - v[1]) for v in vps))
        return dict(job=list(job), n_finite=vr["n_finite"], n_parallel=vr["n_parallel"], vp_err=err, supports=[f["support"] for f in vr["families"]])

def cal_jobs():
    J = []
    for kind in ("hline", "vline", "diag", "point", "none"):
        for ai, a in enumerate(ASP):
            for ci, (nz, ct) in enumerate(KCONDS):
                for rep in range(20): J.append(("cal", kind, a, ct, 2000000 + hash(kind) % 1000 * 0 + ai * 10000 + ci * 1000 + rep * 7 + {"hline": 1, "vline": 2, "diag": 3, "point": 4, "none": 5}[kind] * 100000))
    return J
def test_jobs():
    J = []
    for ti, t in enumerate(T0):
        for ai, a in enumerate(ASP):
            for ci, (nz, ct) in enumerate(KCONDS):
                for rep in range(15):
                    prim = t["primitives"][rep % len(t["primitives"])]; J.append(("e1", t["id"], prim, a, ct, 3000000 + ti * 10000 + ai * 1000 + ci * 100 + rep))
    for ai, a in enumerate(ASP):
        for ci, (nz, ct) in enumerate(KCONDS):
            for rep in range(30): J.append(("e1none", a, ct, 3900000 + ai * 1000 + ci * 100 + rep))
    for mode in ("gt", "diag_only", "shifted", "random"):
        for orient in ("TLBR", "BLTR"):
            for ai, a in enumerate(ASP):
                for rep in range(20): J.append(("e2", mode, orient, a, 4000000 + ai * 1000 + rep * 3 + (1 if orient == "TLBR" else 2) * 100000 + {"gt": 1, "diag_only": 2, "shifted": 3, "random": 4}[mode] * 1000000))
    for aspect, prims in (("4:3", [0.25, 0.75]), ("3:2", [1 / 3, 2 / 3]), ("4:5", [0.2, 0.8])):
        for rep in range(40): J.append(("e3", aspect, prims[rep % 2], 5000000 + rep * 11 + int(len(aspect) * 1000)))
    for v in range(8):
        for ai, a in enumerate(ASP):
            for rep in range(6): J.append(("e4", "true", v, a, 0.0, 6000000 + v * 1000 + ai * 100 + rep))
    for sh in (0.01, 0.02, 0.04, 0.08):
        for v in range(8):
            for rep in range(4): J.append(("e4", "shift", v, "4:3", sh, 6100000 + int(sh * 1000) * 1000 + v * 100 + rep))
    for ai, a in enumerate(ASP):
        for rep in range(48): J.append(("e4", "perturb", rep % 8, a, 0.0, 6200000 + ai * 1000 + rep))
        for rep in range(48): J.append(("e4", "straight", 0, a, 0.0, 6300000 + ai * 1000 + rep))
    for k in (1, 2, 3):
        for ai, a in enumerate(ASP):
            for ci, (nz, ct) in enumerate([(0.0, 0.3), (0.02, 0.3), (0.02, 0.1)]):
                for rep in range(20): J.append(("e5", k, a, nz, ct, 7000000 + k * 100000 + ai * 10000 + ci * 1000 + rep))
    return J
def main():
    mode = sys.argv[1]; t0 = time.time()
    if mode == "cal":
        jobs = cal_jobs(); print(len(jobs), "calibration images", flush=True)
        with Pool(max(1, os.cpu_count() - 1)) as p: res = p.map(job_fn, jobs, chunksize=20)
        json.dump(res, open(os.path.join(HERE, "results", "m2b_cal_raw.json"), "w"))
        ks = [r["job"][1] for r in res]; cov = [r["cov"] for r in res]; blob = np.array([r["blob"] for r in res])
        best = (-1, None)
        for tc in np.arange(0.30, 0.951, 0.05):
            for tb in np.unique(np.round(np.quantile(blob, np.linspace(0.02, 0.98, 49)), 5)):
                acc = 0
                for k, c, b in zip(ks, cov, blob):
                    m = max(c, key=c.get); pred = m if c[m] >= tc else ("point" if b >= tb else "none"); acc += pred == k
                acc /= len(ks)
                if acc > best[0] + 1e-9: best = (acc, [(tc, tb)])
                elif abs(acc - best[0]) <= 1e-9: best[1].append((tc, tb))
        tc = float(np.median([x[0] for x in best[1]])); tb = float(np.median([x[1] for x in best[1]]))
        json.dump(dict(tau_c=tc, tau_b=tb, calibration_accuracy=best[0], n=len(ks), n_optimal_settings=len(best[1])), open(os.path.join(HERE, "results", "m2b_calibration.json"), "w"), indent=1)
        print("calibration:", tc, tb, "accuracy", best[0], "in %.0fs" % (time.time() - t0))
    else:
        jobs = test_jobs(); print(len(jobs), "test images", flush=True)
        with Pool(max(1, os.cpu_count() - 1)) as p: res = p.map(job_fn, jobs, chunksize=8)
        json.dump(res, open(os.path.join(HERE, "results", "m2b_raw.json"), "w")); print("done in %.0fs" % (time.time() - t0))
if __name__ == "__main__": main()
