"""M4: pooled DINOv3 embeddings for all CADB images, same model path as the benchmark (whole frame squashed, no crop).
HF_HUB_OFFLINE=1 python tests/m4_extract_dino.py"""
import json, os, sys, time
import numpy as np, torch, timm
from PIL import Image
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CAD = os.path.join(HERE, "data", "cadb", "CADB_Dataset"); out = os.path.join(HERE, "results", "m4_dino_cadb.npy"); sf = os.path.join(HERE, "results", "m4_spec.sha256")
if not os.path.exists(sf): sys.exit("REFUSED: M4 spec not frozen")
import importlib.util; spec = importlib.util.spec_from_file_location("m4_extract", os.path.join(HERE, "tests", "m4_extract.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
if open(sf).read().split()[0] != m.spec_hash(): sys.exit("REFUSED: a spec file changed since the freeze")
if os.path.exists(out): sys.exit("REFUSED: CADB DINOv3 features already extracted")
ids = json.load(open(os.path.join(CAD, "split.json")))["all"]; dev = "mps" if torch.backends.mps.is_available() else "cpu"
net = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval().to(dev); cfg = net.pretrained_cfg; s = cfg["input_size"][-1]
mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1); feats = []; t0 = time.time()
for i in range(0, len(ids), 16):
    x = np.stack([np.asarray(Image.open(os.path.join(CAD, "images", f)).convert("RGB").resize((s, s), Image.BICUBIC), np.float32) / 255 for f in ids[i:i + 16]]); x = (torch.from_numpy(x).permute(0, 3, 1, 2) - mean) / std
    with torch.no_grad(): feats.append(net(x.to(dev)).float().cpu().numpy())
    if (i // 16) % 100 == 0: print(i, round(time.time() - t0), "s", flush=True)
np.save(out, np.concatenate(feats)); print("saved", len(ids), round(time.time() - t0), "s on", dev)
