# code/tools/ptbxl_fetch.py
import argparse, os, requests, pandas as pd
from tqdm import tqdm

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"

def pick_filename_col(df):
    for c in ["filename_hr","filename_lr","filename"]:
        if c in df.columns: return c
    raise ValueError("filename column not found")

def download_one(relpath, out_root):
    # relpath like records500/00000/00001
    for ext in (".hea",".dat"):
        url = BASE + relpath + ext
        dst = os.path.join(out_root, relpath + ext)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst): continue
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=relpath+ext, leave=False) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk); pbar.update(len(chunk))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True)                 # e.g., data/ptbxl
    ap.add_argument("--subset", type=int, default=200)      # how many records to fetch
    ap.add_argument("--band", choices=["hr","lr"], default="hr")  # 500Hz or 100Hz
    args = ap.parse_args()

    csv = os.path.join(args.ptb, "ptbxl_database.csv")
    df  = pd.read_csv(csv)
    col = pick_filename_col(df)
    if args.band=="hr" and "filename_hr" in df.columns: col="filename_hr"
    if args.band=="lr" and "filename_lr" in df.columns: col="filename_lr"

    paths = df[col].astype(str).head(args.subset).tolist()
    out_root = os.path.join(args.ptb, "wfdb")
    for p in paths:
        download_one(p, out_root)

    print("DONE", len(paths), "records to", out_root)

if __name__ == "__main__":
    main()
