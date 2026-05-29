# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score
from contextlib import nullcontext
from code.datasets.ptbxl import PTBXLWaveform
from code.models.inception_time import InceptionTime

def sigmoid(x): return 1/(1+np.exp(-x))

def find_temperature(logits, y_true, device, T_grid=np.linspace(0.5,3.0,26)):
    import torch.nn.functional as F
    z = torch.tensor(logits, dtype=torch.float32, device=device)
    y = torch.tensor(y_true, dtype=torch.float32, device=device)
    best_T, best_loss = 1.0, 1e9
    for T in T_grid:
        p = torch.sigmoid(z / T)
        loss = F.binary_cross_entropy(p, y).item()
        if loss < best_loss: best_loss, best_T = loss, float(T)
    return best_T

def grid_thresholds(y_true, prob, classes, mode, pfloor=None):
    thr={}
    grid = np.linspace(0.05,0.95,19)
    for i,c in enumerate(classes):
        yt = y_true[:,i]; pr = prob[:,i]
        best_t, best_f1 = 0.5, -1.0
        if mode=="precision" and pfloor and c in pfloor:
            keep=[]
            for t in grid:
                yp = (pr>=t).astype(int)
                if yp.sum()==0: continue
                prec = precision_score(yt, yp, zero_division=0)
                f1   = f1_score(yt, yp, zero_division=0)
                if prec >= pfloor[c]: keep.append((f1,t))
            if keep: best_f1, best_t = max(keep, key=lambda x:x[0])
            else:
                for t in grid:
                    f1 = f1_score(yt, (pr>=t).astype(int), zero_division=0)
                    if f1>best_f1: best_f1, best_t = f1, t
        else:
            for t in grid:
                f1 = f1_score(yt, (pr>=t).astype(int), zero_division=0)
                if f1>best_f1: best_f1, best_t = f1, t
        thr[c]=float(best_t)
    return thr

def parse_pfloor(s):
    if not s: return {}
    out={}
    for kv in s.split(","):
        kv=kv.strip()
        if not kv: continue
        k,v=kv.split("="); out[k.strip()]=float(v)
    return out

def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PTBXLWaveform(a.ptb, split="val", band="hr", cache=True)
    classes = ds.classes

    model = InceptionTime(12, len(classes))
    # 兼容 {'model': state_dict} / {'state_dict': ...} / 纯 state_dict
    ckpt = torch.load(a.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
        sd = ckpt.get("model", ckpt.get("state_dict"))
        if isinstance(sd, torch.nn.Module): sd = sd.state_dict()
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    use_amp = (device.type=="cuda")
    amp_cm = torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16) if use_amp else nullcontext()

    all_logits, all_y = [], []
    with torch.no_grad(), amp_cm:
        for xb, yb in dl:
            xb=xb.to(device); yb=yb.to(device)
            all_logits.append(model(xb).float().cpu().numpy())
            all_y.append(yb.cpu().numpy())
    logits = np.concatenate(all_logits,0)
    y_true = np.concatenate(all_y,0)

    T = find_temperature(logits, y_true, device)
    prob = sigmoid(logits / T)

    pfloor = parse_pfloor(a.pfloor) if a.mode=="precision" else None
    thr = grid_thresholds(y_true, prob, classes, a.mode, pfloor)

    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, "thresholds_5cls.json")
    json.dump(thr, open(outp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("TEMP", T)
    print("SAVED", outp)
    print(thr)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["f1","precision"], default="f1")
    ap.add_argument("--pfloor", type=str, default="")
    a=ap.parse_args(); main(a)
