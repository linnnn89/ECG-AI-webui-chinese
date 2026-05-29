# code/tools/ptbxl_find_bad_records.py
import argparse, os, wfdb, pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--ptb", default=os.path.join("data", "ptbxl"))
args = parser.parse_args()

ptb = args.ptb
wf  = os.path.join(ptb, "wfdb")
meta= pd.read_csv(os.path.join(ptb,"labels_fine.csv"))
bad = []
for rec_rel in meta["record"].astype(str):
    dat = os.path.join(wf, rec_rel + ".dat"); hea = os.path.join(wf, rec_rel + ".hea")
    if not (os.path.exists(dat) and os.path.exists(hea)):
        bad.append(rec_rel); continue
    try:
        rec = os.path.splitext(os.path.join(wf, rec_rel))[0]
        sig, info = wfdb.rdsamp(rec)   # (L, C)
        # 粗校验：12导联且长度>=4000
        if sig.shape[1] != 12 or sig.shape[0] < 4000:
            bad.append(rec_rel)
    except Exception:
        bad.append(rec_rel)
open(os.path.join(ptb,"bad_hr.txt"),"w",encoding="utf-8").write("\n".join(bad))
print("BAD", len(bad), "saved at", os.path.join(ptb,"bad_hr.txt"))
