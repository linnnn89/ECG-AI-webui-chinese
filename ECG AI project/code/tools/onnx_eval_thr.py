# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from code.datasets.ptbxl import PTBXLWaveform
import onnxruntime as ort

def safe_macro_auc(y_true, prob):
    aucs=[]
    C = y_true.shape[1]
    for i in range(C):
        y = y_true[:, i]
        p = prob[:, i]
        if len(np.unique(y)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y, p))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")

def main(a):
    ds = PTBXLWaveform(a.ptb, split=a.split, band="hr", cache=True)
    classes = ds.classes
    thr = json.load(open(a.thr, encoding="utf-8"))
    prov = ["CUDAExecutionProvider","CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(a.onnx, providers=prov)
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    probs, ys = [], []
    for xb, yb in dl:
        xnp = xb.numpy().astype(np.float32)   # [B,12,5000]
        p = sess.run([oname], {iname: xnp})[0]  # 期望为概率
        probs.append(p); ys.append(yb.numpy())
    prob = np.concatenate(probs, 0)
    y_true = np.concatenate(ys, 0)

    macro_auc = safe_macro_auc(y_true, prob)

    # 应用逐类阈值
    y_pred = np.zeros_like(prob, dtype=np.int32)
    for i, c in enumerate(classes):
        t = float(thr.get(c, 0.5))
        y_pred[:, i] = (prob[:, i] >= t).astype(np.int32)

    f1s = [f1_score(y_true[:,i], y_pred[:,i], zero_division=0) for i in range(len(classes))]
    macro_f1 = float(np.mean(f1s))

    print(f"macro_auc {macro_auc:.4f}")
    print(f"macro_f1  {macro_f1:.4f}")
    # 如需类级F1，解除下一行注释：
    # print(dict(zip(classes, [round(x,4) for x in f1s])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--thr", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(); main(a)
