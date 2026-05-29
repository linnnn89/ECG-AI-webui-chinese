# -*- coding: utf-8 -*-
import cv2, numpy as np, json

def _dedup(xs, tol):
    xs = sorted(xs)
    out = []
    for x in xs:
        if not out or abs(x - out[-1]) > tol:
            out.append(x)
    return out

def auto_detect(img_bgr):
    """返回: (layout, meta)
    layout ∈ {"3x4","6x2","unknown"}
    meta: {"sep_count":..., "W":..., "H":..., "rhythm_score":...}
    规则：统计长竖分隔线数量。≈3条→4列(3×4)，≈5条→6列(6×2)。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 缩放到 ~1600 的长边，提速
    scale = 1600.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        img = gray.copy()
        scale = 1.0
    H, W = img.shape

    # 边缘与直线
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(0.5*H),
                            minLineLength=int(0.6*H), maxLineGap=10)

    xs = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if abs(x1 - x2) <= max(2, int(0.005*W)):  # 垂直线
                xs.append((x1+x2)/2.0)

    # 合并近邻竖线
    xs = _dedup(xs, tol=0.03*W)
    sep_count = len(xs)

    # 底部“节律条”启发式：底部20%边缘密度
    bottom = edges[int(0.80*H):,:]
    rhythm_score = float(np.mean(bottom) / (np.mean(edges)+1e-6))

    layout = "unknown"
    if sep_count >= 4:      # ~5条 → 6列
        layout = "6x2"
    elif sep_count >= 3:    # ~3条 → 4列
        layout = "3x4"

    meta = {"sep_count": int(sep_count), "W": int(W), "H": int(H), "rhythm_score": rhythm_score}
    return layout, meta

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    a = ap.parse_args()
    img = cv2.imread(a.img)
    layout, meta = auto_detect(img)
    print(json.dumps({"layout": layout, "meta": meta}, ensure_ascii=False, indent=2))
