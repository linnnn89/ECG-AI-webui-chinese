# code/datasets/ptbxl.py 
import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
import wfdb

BASE_COLS = {"ecg_id","patient_id","strat_fold","split","record"}

class PTBXLWaveform(Dataset):
    def __init__(self, ptb_root, split="train", cache=True, band="hr", normalize="z"):
        self.ptb_root = ptb_root
        self.wfdb_root = os.path.join(ptb_root,"wfdb")
        # 优先细粒度，其次5类
        csv_f = "labels_fine.csv"
        if not os.path.exists(os.path.join(ptb_root,csv_f)):
            csv_f = "labels_5cls.csv"
        self.meta = pd.read_csv(os.path.join(ptb_root,csv_f))
        self.meta = self.meta[self.meta["split"]==split].reset_index(drop=True)
        # 自动识别标签列
        self.classes = [c for c in self.meta.columns if c not in BASE_COLS]
        self.cache = cache
        self.cache_dir = os.path.join(ptb_root, f"cache_{band}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.normalize = normalize

    def __len__(self): return len(self.meta)

    def _load_wfdb(self, rec_rel):
        rec = os.path.splitext(os.path.join(self.wfdb_root, rec_rel))[0]
        npy = os.path.join(self.cache_dir, rec_rel.replace("/","_")+".npy")
        if self.cache and os.path.exists(npy):
            x = np.load(npy, mmap_mode="r")
        else:
            sig, info = wfdb.rdsamp(rec)   # (L,C)
            x = sig.astype("float32").T    # (C,L)
            if self.cache:
                os.makedirs(os.path.dirname(npy), exist_ok=True)
                np.save(npy, x)
        return x

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        x = self._load_wfdb(row["record"])
        if self.normalize=="z":
            m = x.mean(axis=1, keepdims=True); s = x.std(axis=1, keepdims=True)+1e-6
            x = (x - m)/s
        y = row[self.classes].to_numpy("float32")
        return torch.from_numpy(x), torch.from_numpy(y)
