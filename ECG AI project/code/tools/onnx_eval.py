# code/tools/onnx_eval.py
import os, json, argparse, numpy as np, onnxruntime as ort, torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader
from code.datasets.ptbxl import PTBXLWaveform
from code.models.inception_time import InceptionTime

def macro_auc(y_true, y_prob):
    aucs=[]
    for j in range(y_true.shape[1]):
        yt=y_true[:,j]; yp=y_prob[:,j]
        if yt.max()>0 and yt.min()<1:
            try: aucs.append(roc_auc_score(yt, yp))
            except: pass
    return float(np.mean(aucs)) if aucs else float("nan")

def main(ptb, onnx_path, ckpt, split="val", batch=64):
    # 数据
    ds = PTBXLWaveform(ptb, split=split, cache=True); C=len(ds.classes)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    # PyTorch 基线
    ck = torch.load(ckpt, map_location="cpu")
    m = InceptionTime(12, C).eval(); m.load_state_dict(ck["model"])
    # ORT 会话（CPU）
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ypt_list=[]; yrt_list=[]; yt_list=[]
    for xb, yb in dl:
        with torch.no_grad():
            logits = m(xb).numpy()
            ypt = 1/(1+np.exp(-logits))
        inp = {"x": xb.numpy()}
        yrt = 1/(1+np.exp(-sess.run(None, inp)[0]))
        ypt_list.append(ypt); yrt_list.append(yrt); yt_list.append(yb.numpy())
    ypt = np.concatenate(ypt_list,0); yrt = np.concatenate(yrt_list,0); yt = np.concatenate(yt_list,0)
    # 差异与指标
    mse = float(np.mean((ypt-yrt)**2))
    auc = macro_auc(yt, yrt)
    f1  = f1_score(yt, (yrt>=0.5).astype(int), average="macro", zero_division=0)
    print("onnx_mse_vs_torch", f"{mse:.6e}")
    print("onnx_macro_auc", f"{auc:.4f}")
    print("onnx_macro_f1 ", f"{f1:.4f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--onnx", default="O:/ECG AI project/models/inception_5cls.onnx")
    ap.add_argument("--ckpt", default="O:/ECG AI project/models/inception_5cls_best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=64)
    a=ap.parse_args(); main(a.ptb, a.onnx, a.ckpt, a.split, a.batch)
