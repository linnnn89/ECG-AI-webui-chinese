# code/tools/infer_and_report.py
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from code.datasets.ptbxl import PTBXLWaveform
from code.models.inception_time import InceptionTime

ZH = {"NORM":"正常心电图","MI":"心肌梗死相关改变","STTC":"ST-T异常","LVH":"左心室肥厚","LBBB":"左束支传导阻滞","RBBB":"右束支传导阻滞","1AVB":"一度房室传导阻滞","2AVB":"二度房室传导阻滞","3AVB":"三度房室传导阻滞","WPW":"预激综合征（WPW）"}

def macro_auc(y_true, y_prob):
    aucs=[]
    for j in range(y_true.shape[1]):
        yt=y_true[:,j]; yp=y_prob[:,j]
        if yt.max()>0 and yt.min()<1:
            try: aucs.append(roc_auc_score(yt, yp))
            except: pass
    return float(np.mean(aucs)) if aucs else float("nan")

def macro_f1(y_true, y_prob, thr):
    y_pred=(y_prob>=thr[None,:]).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def load_thr(path, classes):
    if path and os.path.exists(path):
        d=json.load(open(path,"r",encoding="utf-8"))
        return np.array([d.get(c,0.5) for c in classes], dtype="float32")
    return np.full(len(classes), 0.5, dtype="float32")

def gen_report(rec_id, classes, yhat, thr):
    pos=[c for c,p,t in zip(classes, yhat, thr) if p>=t]
    if not pos: pos=["NORM"]
    zh=[ZH.get(c,c) for c in pos]
    lines=[
      f"心电图自动分析报告（试验版）",
      f"记录: {rec_id}",
      f"结论: { '、'.join(zh) }",
      "说明: 本报告基于研究模型，需结合临床。"
    ]
    return "\n".join(lines)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PTBXLWaveform(args.ptb, split=args.split, cache=True)
    classes = ds.classes; C=len(classes)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = InceptionTime(c_in=12, c_out=C).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    thr = load_thr(args.thr, classes)
    os.makedirs(args.out, exist_ok=True); rep_dir=os.path.join(args.out,"reports"); os.makedirs(rep_dir, exist_ok=True)

    y_prob=[]; y_true=[]; recs=[]
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device=="cuda")):
        for i,(xb,yb) in enumerate(dl):
            xb=xb.to(device); prob=torch.sigmoid(model(xb)).cpu().numpy()
            y_prob.append(prob); y_true.append(yb.numpy())
            # 写报告
            for k in range(prob.shape[0]):
                rec = ds.meta.iloc[i*args.batch+k]["record"]
                rec_id = os.path.basename(rec)
                txt = gen_report(rec_id, classes, prob[k], thr)
                open(os.path.join(rep_dir, f"{rec_id}.txt"),"w",encoding="utf-8").write(txt)

    y_prob = np.concatenate(y_prob,0); y_true=np.concatenate(y_true,0)
    auc = macro_auc(y_true, y_prob)
    f1  = macro_f1(y_true, y_prob, thr)
    # 保存CSV
    import pandas as pd
    df = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in classes])
    df.insert(0,"record", ds.meta["record"].values)
    df.to_csv(os.path.join(args.out, f"preds_{args.split}.csv"), index=False, encoding="utf-8")
    print("classes", classes)
    print("macro_auc", f"{auc:.4f}")
    print("macro_f1 ", f"{f1:.4f}")
    print("reports ->", rep_dir)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--ckpt", default="O:/ECG AI project/models/inception_5cls_best.pt")
    ap.add_argument("--thr",  default="O:/ECG AI project/models/thresholds_5cls.json")
    ap.add_argument("--out",  default="O:/ECG AI project/outputs")
    ap.add_argument("--split", choices=["val","test","train"], default="val")
    ap.add_argument("--batch", type=int, default=64)
    a=ap.parse_args(); main(a)

