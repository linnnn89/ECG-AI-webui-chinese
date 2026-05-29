# code/tools/ptbxl_prepare_labels.py
import os, ast, argparse, pandas as pd

CLASSES = ["NORM","MI","STTC","HYP","CD"]  # PTB-XL 诊断超类
def parse_codes(s): 
    try: return {k:float(v) for k,v in ast.literal_eval(s).items()}
    except: return {}

def main(ptb_root, band="hr"):
    db  = pd.read_csv(os.path.join(ptb_root,"ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(ptb_root,"scp_statements.csv"))
    scp = scp.set_index(scp.columns[0])     # 索引为SCP缩写，如 'NORM'
    # 仅保留 diagnostic=1 的语句，并映射到 diagnostic_class（五大类）
    diag_map = scp[scp["diagnostic"]==1]["diagnostic_class"].to_dict()

    # 选择文件列
    fn_col = "filename_hr" if band=="hr" and "filename_hr" in db.columns else "filename_lr"
    db["codes_dict"] = db["scp_codes"].astype(str).map(parse_codes)
    # 生成多标签
    for c in CLASSES: db[c] = 0
    for i, row in db.iterrows():
        pos = [k for k,w in row["codes_dict"].items() if w>0 and k in diag_map]
        supers = sorted(set(diag_map[k] for k in pos if diag_map.get(k) in CLASSES))
        for c in supers: db.at[i,c]=1
    # 丢弃无诊断超类的样本
    db = db[db[CLASSES].sum(1)>0].copy()

    # 官方10折：1–8训练，9验证，10测试
    def split_of(fold):
        if fold in range(1,9): return "train"
        if fold==9: return "val"
        return "test"
    db["split"] = db["strat_fold"].map(split_of)

    # 仅保留已下载到 wfdb/ 的记录
    wfdb_root = os.path.join(ptb_root,"wfdb")
    keep = []
    for rel in db[fn_col].astype(str):
        rec = os.path.splitext(os.path.join(wfdb_root, rel))[0]
        if os.path.exists(rec+".dat") and os.path.exists(rec+".hea"): keep.append(True)
        else: keep.append(False)
    db = db[keep].copy()

    out = db[["ecg_id","patient_id","strat_fold","split",fn_col]+CLASSES].rename(columns={fn_col:"record"})
    out_path = os.path.join(ptb_root,"labels_5cls.csv")
    out.to_csv(out_path, index=False)
    print("SAVED", out.shape, "->", out_path)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--band", choices=["hr","lr"], default="hr")
    a=ap.parse_args()
    main(a.ptb, a.band)
