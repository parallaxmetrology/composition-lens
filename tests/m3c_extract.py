"""M3c: lens features for LockboxB. python3 -I tests/m3c_extract.py. Refuses to run unless the M3c spec is frozen and unchanged, and runs only once."""
import sys, os, json, time, hashlib
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tests"))
from multiprocessing import Pool
BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def spec_files():
    return [os.path.join("lens", f) for f in sorted(os.listdir(os.path.join(HERE, "lens"))) if f.endswith(".py")] + ["catalogue/templates.json", "catalogue/crosswalk.json", "tests/m3_extract.py", "tests/m3_analyse.py", "tests/m3b_analyse.py", "tests/m3c_extract.py", "tests/m3c_extract_dino.py", "tests/m3c_analyse.py", "plans/PLAN_frozen_m3c.md", "plans/PLAN_frozen_m3.md", "results/m2b_calibration.json", "results/m3_spec.sha256"]
def spec_hash():
    h = hashlib.sha256()
    for f in spec_files(): h.update(f.encode()); h.update(sha(os.path.join(HERE, f)).encode())
    return h.hexdigest()
def work(item):
    from lens import features
    i, lab = item
    try: f = features.extract(f"{BASE}/data/lockboxb/{i}")
    except Exception as e: return dict(id=i, label=lab, error=repr(e)[:120])
    f["id"], f["label"] = i, lab; return f
def main():
    pk = json.load(open(f"{BASE}/data/pick_lockboxb.json")); items = [(p["img_id"], p["cat"]) for p in pk]; out = os.path.join(HERE, "results", "m3c_features_lockboxb.json")
    sf = os.path.join(HERE, "results", "m3c_spec.sha256")
    if not os.path.exists(sf): sys.exit("REFUSED: M3c spec not frozen")
    if open(sf).read().split()[0] != spec_hash(): sys.exit("REFUSED: a spec file changed since the freeze")
    if os.path.exists(out): sys.exit("REFUSED: LockboxB has already been read once")
    print("lockboxb", len(items), "images", flush=True); t0 = time.time()
    with Pool(max(1, os.cpu_count() - 1)) as p: res = p.map(work, items, chunksize=8)
    errs = [r for r in res if "error" in r]; json.dump(res, open(out, "w")); print("done in %.0fs" % (time.time() - t0), "errors", len(errs), errs[:2], flush=True)
if __name__ == "__main__": main()
