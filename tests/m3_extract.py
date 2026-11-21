"""Extract lens features for a PICD split. python3 -I tests/m3_extract.py devA | devB. devB refuses to run unless the spec is frozen and unchanged, and runs only once."""
import sys, os, json, time, hashlib
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from multiprocessing import Pool
BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def spec_files():
    fs = [os.path.join("lens", f) for f in sorted(os.listdir(os.path.join(HERE, "lens"))) if f.endswith(".py")] + ["catalogue/templates.json", "catalogue/crosswalk.json", "tests/m3_extract.py", "tests/m3_analyse.py", "plans/PLAN_frozen_m3.md", "results/m2b_calibration.json"]
    return fs
def spec_hash():
    h = hashlib.sha256()
    for f in spec_files(): h.update(f.encode()); h.update(sha(os.path.join(HERE, f)).encode())
    return h.hexdigest()
def work(item):
    from lens import features
    i, lab = item
    try: f = features.extract(f"{BASE}/data/images/{i}")
    except Exception as e: return dict(id=i, label=lab, error=repr(e)[:120])
    f["id"], f["label"] = i, lab; return f
def main():
    which = sys.argv[1]; sp = json.load(open(f"{BASE}/results/split.json")); want = 0 if which == "devA" else 1
    items = [(i, l) for i, l, s in zip(sp["ids"], sp["labels"], sp["split"]) if s == want]; out = os.path.join(HERE, "results", f"m3_features_{which}.json")
    if which == "devB":
        sf = os.path.join(HERE, "results", "m3_spec.sha256")
        if not os.path.exists(sf): sys.exit("REFUSED: spec not frozen (results/m3_spec.sha256 missing)")
        if open(sf).read().split()[0] != spec_hash(): sys.exit("REFUSED: a spec file changed since the freeze")
        if os.path.exists(out): sys.exit("REFUSED: devB has already been run once")
    print(which, len(items), "images", flush=True); t0 = time.time()
    with Pool(max(1, os.cpu_count() - 1)) as p: res = p.map(work, items, chunksize=8)
    errs = [r for r in res if "error" in r]; json.dump(res, open(out, "w")); print("done in %.0fs" % (time.time() - t0), "errors", len(errs), errs[:2], flush=True)
if __name__ == "__main__": main()
