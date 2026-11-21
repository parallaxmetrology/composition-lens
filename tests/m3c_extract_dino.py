"""M3c: DINOv3 pooled / patch-grid / patch-norm features for LockboxB's six pair categories (240 images). Same code path as m3b_extract_dino.py.
HF_HUB_OFFLINE=1 python tests/m3c_extract_dino.py"""
import json, os, sys, time, hashlib
import numpy as np, torch, timm
from PIL import Image
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
CATS = {8, 9, 12, 13, 1, 3}
sf = os.path.join(HERE, "results", "m3c_spec.sha256"); out = os.path.join(HERE, "results", "m3c_dino_lockboxb.npz")
if not os.path.exists(sf): sys.exit("REFUSED: M3c spec not frozen")
sys.path.insert(0, os.path.join(HERE, "tests"))
import importlib.util; spec = importlib.util.spec_from_file_location("m3c_extract", os.path.join(HERE, "tests", "m3c_extract.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
if open(sf).read().split()[0] != m.spec_hash(): sys.exit("REFUSED: a spec file changed since the freeze")
if os.path.exists(out): sys.exit("REFUSED: LockboxB DINOv3 features already extracted")
pk = json.load(open(f"{BASE}/data/pick_lockboxb.json")); sel = [k for k, p in enumerate(pk) if p["cat"] in CATS]; ids = [pk[k]["img_id"] for k in sel]; labels = np.array([pk[k]["cat"] for k in sel])
ref = np.load(f"{BASE}/results/fitted_dino_lockboxb.npy")[sel]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
net = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval().to(dev); cfg = net.pretrained_cfg; s = cfg["input_size"][-1]
mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1); npref = net.num_prefix_tokens; pooled, grid4, norm14 = [], [], []
for i in range(0, len(ids), 16):
    x = np.stack([np.asarray(Image.open(f"{BASE}/data/lockboxb/{f}").convert("RGB").resize((s, s), Image.BICUBIC), np.float32) / 255 for f in ids[i:i + 16]]); x = (torch.from_numpy(x).permute(0, 3, 1, 2) - mean) / std
    with torch.no_grad():
        tok = net.forward_features(x.to(dev)); pooled.append(net.forward_head(tok, pre_logits=True).float().cpu().numpy()); pt = tok[:, npref:].float(); g = int(round(pt.shape[1] ** 0.5)); pt = pt.reshape(len(x), g, g, -1).permute(0, 3, 1, 2)
        grid4.append(torch.nn.functional.adaptive_avg_pool2d(pt, 4).permute(0, 2, 3, 1).reshape(len(x), -1).cpu().numpy()); norm14.append(pt.norm(dim=1).reshape(len(x), -1).cpu().numpy())
pooled, grid4, norm14 = map(np.concatenate, (pooled, grid4, norm14)); cos = (pooled * ref).sum(1) / (np.linalg.norm(pooled, axis=1) * np.linalg.norm(ref, axis=1)); print("pooled vs saved embedding cosine: min %.4f" % cos.min())
np.savez_compressed(out, ids=np.array(ids), labels=labels, pooled=pooled, grid4=grid4, norm14=norm14); print("saved", pooled.shape, grid4.shape, norm14.shape)
