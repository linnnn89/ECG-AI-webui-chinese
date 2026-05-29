# code/tools/ptbxl_prepare_labels_fine.py
import os, ast, argparse, pandas as pd

# 细粒度10类
TARGETS = ["NORM","MI","STTC","LVH","LBBB","RBBB","1AVB","2AVB","3AVB","WPW"]

# 将 scp 语句映射到上述10类
def build_map(scp_df):
    scp_df = scp_df.set_index(scp_df.columns[0])  # 索引为SCP代码
    diag = scp_df[scp_df["diagnostic"]==1]
    mp = {k: set() for k in TARGETS}
    for code, row in diag.iterrows():
        cls = str(row.get("diagnostic_class",""))
        # 直接命中
        if code in ["NORM","LVH","LBBB","RBBB","1AVB","2AVB","3AVB","WPW"]:
            mp[code].add(code)
        # LBBB/RBBB的变体
        if code in ["CLBBB","ILBBB"]: mp["LBBB"].add(code)
        if code in ["CRBBB","IRBBB"]: mp["RBBB"].add(code)
        # 聚合类
        if cls=="MI":   mp["MI"].add(code)
        if cls=="STTC": mp["STTC"].add(code)
    return mp

def parse_codes(s):
    try: return {k:float(v) for k,v in ast.literal_eval(s).items()}
    except: return {}

def main(ptb_root, band="hr"):
    db  = pd.read_csv(os.path.join(ptb_root,"ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(ptb_root,"scp_statements.csv"))
    mp  = build_map(scp)

    fn_col = "filename_hr" if band=="hr" and "filename_hr" in db.columns else "filename_lr"
    db["codes_dict"] = db["scp_codes"].astype(str).map(parse_codes)

    # 逐行打标
    for t in TARGETS: db[t]=0
    for i,row in db.iterrows():
        pos = set(k for k,w in row["codes_dict"].items() if w>0)
        for t in TARGETS:
            if pos & mp[t]: db.at[i,t]=1
    # 若完全无命中的且报告为NORM，可兜底NORM
    db.loc[(db[TARGETS].sum(1)==0) & db["report"].fillna("").str.contains("norm", case=False), "NORM"]=1

    # 官方10折 → 1–8 train, 9 val, 10 test
    def split_of(f): return "train" if f in range(1,9) else ("val" if f==9 else "test")
    db["split"] = db["strat_fold"].map(split_of)

    # 仅保留已下载到 wfdb 的记录
    wfdb_root = os.path.join(ptb_root,"wfdb")
    keep=[]
    for rel in db[fn_col].astype(str):
        rec = os.path.splitext(os.path.join(wfdb_root, rel))[0]
        keep.append(os.path.exists(rec+".dat") and os.path.exists(rec+".hea"))
    db = db[keep].copy()

    cols = ["ecg_id","patient_id","strat_fold","split",fn_col] + TARGETS
    out = db[cols].rename(columns={fn_col:"record"})
    out.to_csv(os.path.join(ptb_root,"labels_fine.csv"), index=False)
    print("SAVED", out.shape, "->", os.path.join(ptb_root,"labels_fine.csv"))
    print("classes", TARGETS)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)
    ap.add_argument("--band", choices=["hr","lr"], default="hr")
    a=ap.parse_args(); main(a.ptb, a.band)
