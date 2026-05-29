import os, argparse, json, numpy as np, onnxruntime as ort, pandas as pd

LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
ZH = {"NORM":"正常心电图","MI":"心肌梗死相关改变","STTC":"ST-T异常","HYP":"心室肥厚","CD":"传导阻滞/异常"}

def find_input(p):
    if os.path.isdir(p):
        for ext in ('.csv','.npy','.npz'):
            for r,_,fs in os.walk(p):
                for f in fs:
                    if f.lower().endswith(ext): return os.path.join(r,f)
    return p

def load_signal(fp):
    ext=os.path.splitext(fp)[1].lower()
    if ext=='.csv':
        df=pd.read_csv(fp)
        cols=[c for c in LEADS if c in df.columns]
        if len(cols)==12: x=df[cols].to_numpy('float32').T
        else: x=df.select_dtypes(include=['number']).iloc[:,:12].to_numpy('float32').T
    elif ext=='.npy':
        a=np.load(fp, allow_pickle=True).astype('float32')
        if a.ndim!=2: raise ValueError(f"npy维度{a.ndim}不支持")
        x=a.T if a.shape[1]==12 else a
    elif ext=='.npz':
        d=np.load(fp)
        x=None
        for k in d.files:
            a=d[k]
            if a.ndim==2 and (12 in a.shape):
                x=a.astype('float32'); break
        if x is None: raise ValueError("npz未发现12导联矩阵")
        if x.shape[1]==12: x=x.T
    else:
        raise ValueError(f"不支持的文件: {ext}")
    if x.shape[0]!=12 and x.shape[1]==12: x=x.T
    if x.shape[0]!=12: raise ValueError(f"形状{tuple(x.shape)}不为(12,L)")
    return x

def resample_to_len(x, L=5000):
    C,n=x.shape
    if n==L: return x
    idx=np.linspace(0,n-1,L); base=np.arange(n,dtype=float)
    y=np.zeros((C,L),dtype='float32')
    for c in range(C): y[c]=np.interp(idx, base, x[c])
    return y

def zscore(x):
    m=x.mean(1,keepdims=True); s=x.std(1,keepdims=True)+1e-6
    return (x-m)/s

def main(a):
    fp=find_input(a.input)
    x=load_signal(fp)
    x=resample_to_len(zscore(x), 5000)
    classes=json.load(open(a.classes,'r',encoding='utf-8'))
    thrmap=json.load(open(a.thresholds,'r',encoding='utf-8'))
    thr=np.array([thrmap.get(c,0.5) for c in classes],dtype='float32')
    sess=ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    logits=sess.run(None, {"x": x[None,...]})[0][0]
    prob=1/(1+np.exp(-logits))
    pos=[c for c,p,t in zip(classes,prob,thr) if p>=t] or ['NORM']
    zh=[ZH.get(c,c) for c in pos]
    print("file", fp); print("shape", x.shape)
    print("labels", pos); print("labels_zh", zh)
    outdir=os.path.join(a.out,"reports_img"); os.makedirs(outdir,exist_ok=True)
    rec=os.path.splitext(os.path.basename(fp))[0]
    open(os.path.join(outdir,rec+".txt"),"w",encoding="utf-8").write(
        f"心电图自动分析报告（图像数字化）\n来源文件: {rec}\n结论: {'、'.join(zh)}\n说明: 实验流程：图像→数字化→ONNX诊断。")
    print("saved", os.path.join(outdir,rec+".txt"))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Digitiser输出的单个文件或其目录")
    ap.add_argument("--onnx", default=os.path.join("models", "inception_5cls.onnx"))
    ap.add_argument("--classes", default=os.path.join("models", "classes_5cls.json"))
    ap.add_argument("--thresholds", default=os.path.join("models", "thresholds_5cls.json"))
    ap.add_argument("--out", default="outputs")
    a=ap.parse_args(); main(a)
