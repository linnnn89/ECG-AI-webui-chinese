# code/tools/calc_thresholds.py
import os, json, numpy as np, torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from code.datasets.ptbxl import PTBXLWaveform
from code.models.inception_time import InceptionTime

def collect_probs(ptb_root, ckpt_path, batch=64, amp=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = PTBXLWaveform(ptb_root, split="val", cache=True)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]; c_out = len(classes)
    model = InceptionTime(c_in=12, c_out=c_out).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    yp, yt = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp and device=='cuda', dtype=torch.float16):
        for xb, yb in dl:
            xb = xb.to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            yp.append(prob); yt.append(yb.numpy())
    return classes, np.concatenate(yt,0), np.concatenate(yp,0)

def best_thresholds(y_true, y_prob):
    thr_grid = np.linspace(0.05,0.95,19)
    J = y_true.shape[1]; best = {}
    for j in range(J):
        yt = y_true[:,j]
        if yt.max()==0 or yt.min()==1:  # 单类全正/全负，退化
            best[j] = 0.5; continue
        f1s = [f1_score(yt, (y_prob[:,j]>=t).astype(int), zero_division=0) for t in thr_grid]
        best[j] = float(thr_grid[int(np.argmax(f1s))])
    return best

def main(ptb_root, ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    classes, yt, yp = collect_probs(ptb_root, ckpt_path)
    mapping = best_thresholds(yt, yp)
    thr = {classes[j]: mapping[j] for j in range(len(classes))}
    out = os.path.join(out_dir, "thresholds_5cls.json")
    with open(out,"w",encoding="utf-8") as f: json.dump(thr, f, ensure_ascii=False, indent=2)
    print("SAVED", out); print(thr)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--ckpt", default="O:/ECG AI project/models/inception_5cls_best.pt")
    ap.add_argument("--out",  default="O:/ECG AI project/models")
    a = ap.parse_args()
    main(a.ptb, a.ckpt, a.out)

