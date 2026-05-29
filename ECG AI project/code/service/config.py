# -*- coding: utf-8 -*-
"""
config.py
统一配置。后续如切换模型目录、开关Digitiser、调整阈值来源，改这里即可。
"""
import os

# 项目根
ROOT = os.environ.get(
    "ECG_AI_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

# 模型目录：公开默认只查找可公开复现的候选目录。
# 使用医院数据训练/微调的模型权重不随 GitHub 发布；内部本地运行时用 ECG_MODEL_DIR 显式指定。
_MODEL_DIR_CANDIDATES = [
    os.path.join(ROOT, "models_fine_chapman_ft"),
    os.path.join(ROOT, "models_fine"),
]
_DEFAULT_MODEL_DIR = next((path for path in _MODEL_DIR_CANDIDATES if os.path.isdir(path)), _MODEL_DIR_CANDIDATES[-1])
MODEL_DIR = os.environ.get("ECG_MODEL_DIR", _DEFAULT_MODEL_DIR)

# ONNX与阈值、类别文件
ONNX_PATH = os.path.join(MODEL_DIR, "inception_5cls.onnx")
CLASSES_JSON = os.path.join(MODEL_DIR, 'classes_5cls.json')
THRESHOLDS_JSON = os.path.join(MODEL_DIR, 'thresholds_5cls.json')

# 图片数字化后端。默认关闭，避免第三方依赖导致服务无法启动。
# 可选值：none | felix | ahus
DIGITIZER_BACKEND = os.environ.get("ECG_DIGITIZER_BACKEND", "none").strip().lower()

# FelixKrones ECG-Digitiser：适合高质量扫描件/3x4版式，第三方源码保持只读。
FELIX_ROOT = os.environ.get(
    "ECG_FELIX_ROOT",
    os.path.join(ROOT, "third_party", "ECG-Digitiser"),
)
FELIX_PY = os.environ.get(
    "ECG_FELIX_PY",
    os.path.join(FELIX_ROOT, ".ecgdig", "Scripts", "python.exe"),
)
FELIX_MODEL_DIR = os.environ.get(
    "ECG_FELIX_MODEL_DIR",
    os.path.join(FELIX_ROOT, "models", "M3"),
)

# Ahus-AIM Open-ECG-Digitizer：照片级ECG数字化候选后端，独立目录接入。
AHUS_ROOT = os.environ.get(
    "ECG_AHUS_ROOT",
    os.path.join(ROOT, "third_party", "Open-ECG-Digitizer"),
)
_DEFAULT_AHUS_VENV = ".venv310"
if not os.path.exists(os.path.join(AHUS_ROOT, _DEFAULT_AHUS_VENV, "Scripts", "python.exe")):
    _DEFAULT_AHUS_VENV = ".venv"
AHUS_PY = os.environ.get(
    "ECG_AHUS_PY",
    os.path.join(AHUS_ROOT, _DEFAULT_AHUS_VENV, "Scripts", "python.exe"),
)
AHUS_MODEL_DIR = os.environ.get(
    "ECG_AHUS_MODEL_DIR",
    os.path.join(AHUS_ROOT, "models"),
)
AHUS_WEIGHTS_DIR = os.environ.get(
    "ECG_AHUS_WEIGHTS_DIR",
    os.path.join(AHUS_ROOT, "weights"),
)
AHUS_LAYOUT_CONFIG = os.environ.get(
    "ECG_AHUS_LAYOUT_CONFIG",
    "src/config/lead_layouts_all.yml",
)

# 兼容旧命名，避免历史脚本直接引用时崩溃。
DIGITISER_PY = FELIX_PY
DIGITISER_MODEL_DIR = FELIX_MODEL_DIR

# 图片数字化评估目录
IMAGE_DIGITIZER_EVAL_DIR = os.path.join(ROOT, "data", "image_digitizer_eval")
IMAGE_DIGITIZER_INPUT_DIR = os.path.join(IMAGE_DIGITIZER_EVAL_DIR, "input_images")
IMAGE_DIGITIZER_OUTPUT_DIR = os.path.join(IMAGE_DIGITIZER_EVAL_DIR, "digitized_outputs")

# 目标长度（与你训练时一致）
TARGET_LEN = 5000

# CaseBank 全量向量索引与统一运行缓存。
# 缓存只保存运行时可再生成的 PNG/临时文件；清理后不会影响模型或原始数据。
CASEBANK_DIR = os.environ.get(
    "ECG_CASEBANK_DIR",
    os.path.join(ROOT, "data", "casebank_vector_index"),
)
CASEBANK_CACHE_DIR = os.environ.get(
    "ECG_CASEBANK_CACHE_DIR",
    os.path.join(ROOT, "casebank_cache"),
)
CASEBANK_TOP_K = int(os.environ.get("ECG_CASEBANK_TOP_K", "10"))
CASEBANK_SCORE_THRESHOLD = float(os.environ.get("ECG_CASEBANK_SCORE_THRESHOLD", "0.0"))

# 图片数字化质量门槛：低于该有限值比例时不进入 ONNX 诊断。
MIN_DIGITIZED_FINITE_FRACTION = float(os.environ.get("ECG_MIN_DIGITIZED_FINITE_FRACTION", "0.40"))
MIN_DIGITIZED_LEAD_FINITE_FRACTION = float(
    os.environ.get("ECG_MIN_DIGITIZED_LEAD_FINITE_FRACTION", "0.18")
)
MIN_DIGITIZED_LEAD_COUNT = int(os.environ.get("ECG_MIN_DIGITIZED_LEAD_COUNT", "12"))

# 低置信度安全拒判层：不改变各类别诊断阈值，只在全部类别均未触发时追加提示。
# 当最高概率低于该值时，标记为“所有类别置信度均低”。
UNCERTAINTY_ENABLED = os.environ.get("ECG_UNCERTAINTY_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
UNCERTAINTY_MAX_PROB = float(os.environ.get("ECG_UNCERTAINTY_MAX_PROB", "0.30"))
UNCERTAINTY_LABEL = "UNKNOWN"
UNCERTAINTY_MESSAGE = os.environ.get(
    "ECG_UNCERTAINTY_MESSAGE",
    "结果高度不确定，建议进一步参考人工评估或完善检查资料。",
)
