"""M3b baselines for the three position pairs (PLAN.md section M3b). python3 -I tests/m3b_analyse.py -> results/m3b.json"""
import os, sys, json, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, cv2
from multiprocessing import Pool
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
Z = np.load(os.path.join(HERE, "results", "m3b_dino.npz")); ids = [str(i) for i in Z["ids"]]; lab = Z["labels"]; half = Z["halves"]
def grids(i):
    g = cv2.imread(f"{BASE}/data/images/{i}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255; g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    b = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).ravel(); gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1); e = cv2.resize(np.hypot(gx, gy), (16, 16), interpolation=cv2.INTER_AREA).ravel(); return b, e
GB = [grids(i) for i in ids]; FEATS = {"D-pool": Z["pooled"], "D-grid": Z["grid4"], "D-norm": Z["norm14"], "G-bright": np.array([x[0] for x in GB]), "G-edge": np.array([x[1] for x in GB])}
LF = {}
for w in ("devA", "devB"):
    for r in json.load(open(os.path.join(HERE, "results", f"m3_features_{w}.json"))):
        if "error" not in r: LF[r["id"]] = r
PAIRS = {"LS-Hori3 vs LS-Hori2": (9, 8, lambda r: r["d_centre_h"], lambda r: abs(r["mass_cy"] - 0.5)),
         "L-Ver3 vs L-Ver2": (13, 12, lambda r: r["d_centre_v"], lambda r: abs(r["mass_cx"] - 0.5)),
         "P-RoT vs P-Cent": (1, 3, lambda r: r["blob_d_centre"], lambda r: float(np.hypot(r["mass_cx"] - 0.5, r["mass_cy"] - 0.5)))}
def auc(s, p):
    s = np.asarray(s, float); p = np.asarray(p, bool); r = rankdata(s); n1 = p.sum(); n0 = len(p) - n1; return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
def probe(Xtr, ytr, Xte):
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=3, max_iter=300, scoring="roc_auc")).fit(Xtr, ytr); return m.decision_function(Xte)
def task(a):
    pair, split_seed, n = a; pos, neg, rd, sd = PAIRS[pair]; rng = np.random.default_rng(1000 * split_seed + pos)
    ip = np.where(lab == pos)[0]; ineg = np.where(lab == neg)[0]; rng.shuffle(ip); rng.shuffle(ineg); te = np.r_[ip[:38], ineg[:38]]; tr = np.r_[ip[38:38 + n], ineg[38:38 + n]]
    ytr = np.r_[np.ones(n), np.zeros(n)]; yte = np.r_[np.ones(38), np.zeros(38)].astype(bool); out = dict(pair=pair, split=split_seed, n=n)
    for k, X in FEATS.items(): out[k] = auc(probe(X[tr], ytr, X[te]), yte)
    out["reading"] = auc([rd(LF[ids[i]]) for i in te], yte); out["simple"] = auc([sd(LF[ids[i]]) for i in te], yte); return out
def p1(pair):
    pos, neg, rd, sd = PAIRS[pair]; trm = ((lab == pos) | (lab == neg)) & (half == 0); tem = ((lab == pos) | (lab == neg)) & (half == 1); out = dict(pair=pair, n_train=int(trm.sum()), n_test=int(tem.sum()))
    for k, X in FEATS.items(): out[k] = auc(probe(X[trm], (lab[trm] == pos).astype(float), X[tem]), lab[tem] == pos)
    out["reading"] = auc([rd(LF[ids[i]]) for i in np.where(tem)[0]], lab[tem] == pos); out["simple"] = auc([sd(LF[ids[i]]) for i in np.where(tem)[0]], lab[tem] == pos); return out
if __name__ == "__main__":
    t0 = time.time(); NS = [10, 20, 40, 70, 100]; SPL = 10
    jobs = [(p, s, n) for p in PAIRS for s in range(SPL) for n in NS]
    with Pool(max(1, os.cpu_count() - 1)) as pool: res = pool.map(task, jobs, chunksize=2); r1 = pool.map(p1, list(PAIRS))
    out = {"P1": r1, "P2": {}}
    for p in PAIRS:
        out["P2"][p] = {}
        for n in NS:
            rs = [r for r in res if r["pair"] == p and r["n"] == n]; out["P2"][p][str(n)] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in list(FEATS) + ["reading", "simple"]}
    json.dump(out, open(os.path.join(HERE, "results", "m3b.json"), "w"), indent=1)
    print("P1 (train DevA 69/class, test DevB):")
    for r in r1: print("  ", r["pair"], {k: round(v, 3) for k, v in r.items() if k not in ("pair", "n_train", "n_test")})
    for p in PAIRS:
        print("P2", p)
        for n in NS: print("   n=%3d" % n, {k: round(v[0], 3) for k, v in out["P2"][p][str(n)].items()})
    print("done in %.0fs" % (time.time() - t0))
