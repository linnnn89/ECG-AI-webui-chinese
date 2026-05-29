# -*- coding: utf-8 -*-
"""
layout.py
最小可用的版式判别器：
- 先支持 "3x4"（3行×4列）与 "6x6"（2页或单页6+6）的粗判；
- 通过霍夫变换统计长直立分隔线数量来估计列数；
- 判不出则返回 unknown。
后续要扩展：网格线估计、节律条定位、OCR导联名等。
"""
from enum import Enum
import cv2, numpy as np

class Layout(str, Enum):
    L3x4 = "3x4"
    L6x6 = "6x6"
    UNK  = "unknown"

def _count_vertical_separators(img):
    h, w = img.shape[:2]
    # 统一缩放，边缘检测
    scale = 1600.0 / max(h, w)
    img_r = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray  = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    # 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=int(0.6*img_r.shape[0]), maxLineGap=10)
    xs = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            # 近似垂直：dx很小，dy很大
            if abs(x1-x2) < 6 and abs(y1-y2) > img_r.shape[0]*0.5:
                xs.append(x1)
    if not xs: return 0
    # 合并相近x坐标
    xs = sorted(xs)
    merged = [xs[0]]
    for x in xs[1:]:
        if abs(x-merged[-1]) > 15:
            merged.append(x)
    return len(merged)

def detect_layout(img_path:str) -> Layout:
    img = cv2.imread(img_path)
    if img is None: return Layout.UNK
    k = _count_vertical_separators(img)
    # 粗判：3×4有3条竖分隔（→4列），6×6常见有5条（→6列）
    if 2 <= k <= 4: return Layout.L3x4
    if 5 <= k <= 7: return Layout.L6x6
    return Layout.UNK
