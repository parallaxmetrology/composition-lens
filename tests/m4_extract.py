"""M4: lens features for CADB images. python3 -I tests/m4_extract.py. Refuses to run unless the M4 spec is frozen and unchanged, and runs only once."""
import sys, os, json, time, hashlib
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from multiprocessing import Pool
CAD = os.path.join(HERE, "data", "cadb", "CADB_Dataset")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def spec_files():
    return [os.path.join("lens", f) for f in sorted(os.listdir(os.path.join(HERE, "lens"))) if f.endswith(".py")] + ["catalogue/templates.json", "catalogue/crosswalk.json", "tests/m3_extract.py", "tests/m3_analyse.py", "tests/m4_extract.py", "tests/m4_extract_dino.py", "tests/m4_analyse.py", "plans/PLAN_frozen_m4.md", "plans/PLAN_frozen_m3c.md", "plans/PLAN_frozen_m3.md", "results/m2b_calibration.json", "results/m3_spec.sha256", "results/m3c_spec.sha256"]
def spec_hash():
    h = hashlib.sha256()
    for f in spec_files(): h.update(f.encode()); h.update(sha(os.path.join(HERE, f)).encode())
    return h.hexdigest()
def work(i):
    from lens import features
    try: f = features.extract(os.path.join(CAD, "images", i))
    except Exception as e: return dict(id=i, error=repr(e)[:120])
    f["id"] = i; return f
def chunk_main(a, b):
    ids = json.load(open(os.path.join(CAD, "split.json")))["all"][a:b]
    with Pool(max(1, os.cpu_count() - 1)) as p: res = p.map(work, ids, chunksize=4)
    json.dump(res, open(os.path.join(HERE, "results", "m4_chunks", f"chunk_{a:05d}.json"), "w"))
def run_sub(args, timeout):
    import subprocess
    try: return subprocess.run([sys.executable, "-I", os.path.abspath(__file__)] + args, timeout=timeout, capture_output=True).returncode == 0
    except subprocess.TimeoutExpired: return False
def main():
    ids = json.load(open(os.path.join(CAD, "split.json")))["all"]; out = os.path.join(HERE, "results", "m4_features_cadb.json"); sf = os.path.join(HERE, "results", "m4_spec.sha256")
    if not os.path.exists(sf): sys.exit("REFUSED: M4 spec not frozen")
    if open(sf).read().split()[0] != spec_hash(): sys.exit("REFUSED: a spec file changed since the freeze")
    if os.path.exists(out): sys.exit("REFUSED: CADB has already been read once")
    os.makedirs(os.path.join(HERE, "results", "m4_chunks"), exist_ok=True); print("cadb", len(ids), "images", flush=True); t0 = time.time(); CH = 200
    for a in range(0, len(ids), CH):
        cf = os.path.join(HERE, "results", "m4_chunks", f"chunk_{a:05d}.json")
        if os.path.exists(cf): continue
        if not run_sub(["--chunk", str(a), str(min(a + CH, len(ids)))], 900):       # a crashed or hung chunk is redone image by image, each in its own process
            res = []
            for i in ids[a:a + CH]:
                tf = os.path.join(HERE, "results", "m4_chunks", f"one_{i}.json")
                ok = run_sub(["--one", i, tf], 120) and os.path.exists(tf); res.append(json.load(open(tf)) if ok else dict(id=i, error="crashed or timed out in a worker"))
            json.dump(res, open(cf, "w"))
        print(a, round(time.time() - t0), "s", flush=True)
    res = []
    for a in range(0, len(ids), CH): res += json.load(open(os.path.join(HERE, "results", "m4_chunks", f"chunk_{a:05d}.json")))
    errs = [r for r in res if "error" in r]; json.dump(res, open(out, "w")); print("done in %.0fs" % (time.time() - t0), "errors", len(errs), [e["id"] for e in errs][:10], flush=True)
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--chunk": chunk_main(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--one": json.dump(work(sys.argv[2]), open(sys.argv[3], "w"))
    else: main()
