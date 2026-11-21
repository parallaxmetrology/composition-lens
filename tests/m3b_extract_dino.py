"""M3b: DINOv3 pooled / patch-grid / patch-norm features for the 6 categories of the three position pairs. Run with the original project's Python (timm), read-only, offline:
HF_HUB_OFFLINE=1 python tests/m3b_extract_dino.py"""
import json, os, time
import numpy as np, torch, timm
from PIL import Image
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BASE = os.environ.get("PICD_ROOT", os.path.join(HERE, "data", "picd"))
CATS = {8, 9, 12, 13, 1, 3}
sp = json.load(open(f"{BASE}/results/split.json")); emb = np.load(f"{BASE}/results/emb_dinov3_web_l16.npy")
idx = [k for k, l in enumerate(sp["labels"]) if l in CATS]; ids = [sp["ids"][k] for k in idx]; labels = np.array([sp["labels"][k] for k in idx]); halves = np.array([sp["split"][k] for k in idx])
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval().to(dev); cfg = m.pretrained_cfg; s = cfg["input_size"][-1]
mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1); npref = m.num_prefix_tokens
pooled, grid4, norm14 = [], [], []; t0 = time.time()
for i in range(0, len(ids), 16):
    x = np.stack([np.asarray(Image.open(f"{BASE}/data/images/{f}").convert("RGB").resize((s, s), Image.BICUBIC), np.float32) / 255 for f in ids[i:i + 16]])
    x = (torch.from_numpy(x).permute(0, 3, 1, 2) - mean) / std
    with torch.no_grad():
        tok = m.forward_features(x.to(dev)); pooled.append(m.forward_head(tok, pre_logits=True).float().cpu().numpy()); pt = tok[:, npref:].float()
        g = int(round(pt.shape[1] ** 0.5)); pt = pt.reshape(len(x), g, g, -1).permute(0, 3, 1, 2)
        grid4.append(torch.nn.functional.adaptive_avg_pool2d(pt, 4).permute(0, 2, 3, 1).reshape(len(x), -1).cpu().numpy()); norm14.append(pt.norm(dim=1).reshape(len(x), -1).cpu().numpy())
pooled, grid4, norm14 = map(np.concatenate, (pooled, grid4, norm14))
ref = emb[idx]; cos = (pooled * ref).sum(1) / (np.linalg.norm(pooled, axis=1) * np.linalg.norm(ref, axis=1)); print("pooled vs saved embedding cosine: min %.4f median %.4f" % (cos.min(), np.median(cos)))
np.savez_compressed(os.path.join(HERE, "results", "m3b_dino.npz"), ids=np.array(ids), labels=labels, halves=halves, pooled=pooled, grid4=grid4, norm14=norm14)
print("saved", pooled.shape, grid4.shape, norm14.shape, round(time.time() - t0), "s on", dev)
