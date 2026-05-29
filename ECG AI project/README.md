# ECG AI Project（本地部署）
心电图 **WFDB/XML/图片→波形→多标签诊断→中文报告**。可离线运行。

---

## 一、系统架构

**输入路径：**
1. **WFDB 文件上传**：`/infer_wfdb_files`
2. **ECG XML 上传**：`/infer_xml`，服务端临时转换为 WFDB 后进入 ONNX 推理
3. **图片上传**（安全骨架阶段）：`/infer_image`
   - 手动启动服务时默认不启用图像数字化后端；未配置时返回 501 提示，不影响 WFDB 推理服务
   - 后端预留：`none | felix | ahus`
4. **PTB-XL 相对路径记录**（兼容接口）：`/infer_record`

**核心模型：** InceptionTime-style 1D CNN（PyTorch 训练，ONNX 推理）

**模型架构引用与许可边界：**
- 本项目的心电图分类器采用 InceptionTime 风格的一维卷积时间序列分类架构，论文引用：Ismail Fawaz H, Lucas B, Forestier G, Pelletier C, Schmidt DF, Weber J, et al. InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery*. 2020;34:1936-1962. doi: `10.1007/s10618-020-00710-y`.
- 原作者官方 companion GitHub 仓库为 `hfawaz/InceptionTime`，许可证为 GPL-3.0。
- 当前 `code/models/inception_time.py` 是本项目按论文思路维护的 PyTorch 简化实现，用于本地 ECG 多标签分类、ONNX 导出和服务推理；不应表述为本项目原创提出 InceptionTime 架构。
- 若未来直接复制、移植或改写原作者官方仓库代码，公开发布时必须同步保留并遵守其 GPL-3.0 许可证要求。

**标签方案：**
- 演示 5 类：`NORM, MI, STTC, HYP, CD`
- 细粒度 10 类（当前默认）：`NORM, MI, STTC, LVH, LBBB, RBBB, 1AVB, 2AVB, 3AVB, WPW`

**阈值：** 每类单独标定（`thresholds_5cls.json` 含10类名，历史命名保留）

**安全拒判层：** 服务端默认启用低置信度不确定输出，不改变原诊断阈值。当 10 类均未达到各自阈值，且模型不能给出可靠阳性/正常结论时，返回 `UNKNOWN`，并提示“结果高度不确定，建议进一步参考人工评估或完善检查资料”。其中最高类别概率低于 `ECG_UNCERTAINTY_MAX_PROB`（默认 `0.30`）时，标记为 `low_confidence_all_classes`。

**服务：** FastAPI + onnxruntime
**前端：** 纯静态页面 `/ui/`

---

## 二、目录结构

下列结构包含公开代码、建议数据目录和本地生成产物。GitHub 仓库不附带原始数据、模型权重、真实 CaseBank 索引或第三方 digitizer 运行环境。

```text
ECG AI project/
├─ code
│ ├─ datasets
│ │ └─ ptbxl.py # WFDB→Tensor 数据集（自动识别 5/10 类标签CSV）
│ ├─ models
│ │ └─ inception_time.py # Torch 2.8 兼容的 InceptionTime 实现
│ ├─ service
│ │ ├─ app.py # FastAPI：/health /infer_record /infer_wfdb_files /infer_xml /infer_image + /ui
│ │ ├─ config.py # 路径与开关（模型、Digitiser、目标长度等）
│ │ ├─ layout.py # 轻量版式检测（3×4/6×2/unknown，占位可替换）
│ │ ├─ digitize.py # 图片→波形 适配层（调用 ECG-Digitiser）
│ │ ├─ xml_ecg.py # ECG XML→WFDB→[12,5000] 适配层
│ │ └─ ui\index.html # 本地Web UI
│ ├─ tools
│ │ ├─ ptbxl_fetch.py # 抽样下载 PTB-XL WFDB 记录
│ │ ├─ ptbxl_prepare_labels.py # 5类标签（保留）
│ │ ├─ ptbxl_prepare_labels_fine.py # 10类细粒度标签（当前默认）
│ │ ├─ calc_thresholds.py # 验证集阈值标定
│ │ ├─ infer_and_report.py # 批量推理→CSV+中文报告
│ │ ├─ export_onnx.py # 导出 ONNX（含权重）
│ │ ├─ onnx_eval.py # ONNX⇄PyTorch 一致性+指标
│ │ └─ infer_digitiser_output.py # 直接跑 Digitiser 输出的桥接脚本
│ └─ train_5cls.py # 训练脚本（自动适配 5/10 类）
├─ data # 本地数据目录；公共数据需自行下载
│ ├─ ptbxl
│ │ ├─ ptbxl_database.csv / scp_statements.csv / RECORDS
│ │ ├─ wfdb\records500...*.hea|*.dat
│ │ ├─ labels_5cls.csv（可选）
│ │ └─ labels_fine.csv（细粒度默认）
│ ├─ chapman / chapman_converted
│ ├─ challenge_2020
│ └─ README.md # 数据目录说明；医院 ECG 数据不随 GitHub 公开发布
├─ models* # 本地模型目录；权重文件不随 GitHub 发布
├─ ecg_xml_to_wfdb.py # 根目录 XML 转 WFDB / NPY 工具
├─ outputs / outputs_fine # 本地推理输出与报告，不随 GitHub 发布
├─ third_party\... # 可选第三方 digitizer 本地目录，不随 GitHub 发布
└─ README.md
```

---

## 三、运行环境与版本

- Windows + PowerShell
- Python 3.10.x（项目级 venv）
- 已验证：`torch 2.8.0+cu128 / torchvision 0.23.0+cu128 / torchaudio 2.8.0+cu128`
- CUDA 12.8 可用（如不可用自动切 CPU）

---

## 四、先建立模型与数据库

公开仓库不包含医院数据、医院训练模型权重或真实 CaseBank 向量索引。新用户应先在本地准备合规数据，训练或放置自己的模型，生成阈值与 ONNX，再按需构建本地 CaseBank 数据库。完成这些步骤后，再启动 WebUI。

推荐顺序：

1. 准备公共或自有合规 ECG 数据。
2. 生成标签 CSV。
3. 训练模型、标定阈值并导出 ONNX。
4. 可选：用自己的合规数据和模型构建 CaseBank 向量数据库。
5. 最后启动 FastAPI/WebUI。

### 4.1 准备 PTB-XL 数据

```powershell
# 元数据（注意 ?download 直链）
iwr "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv?download" -OutFile .\data\ptbxl\ptbxl_database.csv
iwr "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv?download"  -OutFile .\data\ptbxl\scp_statements.csv
iwr "https://physionet.org/files/ptb-xl/1.0.3/RECORDS?download"            -OutFile .\data\ptbxl\RECORDS

# 抽样下载 500Hz 记录示例
python .\code\tools\ptbxl_fetch.py --ptb .\data\ptbxl --subset 200 --band hr
```

### 4.2 生成标签并训练模型

本项目支持 5 类（演示）与 10 类（细粒度）自动适配。训练脚本支持 OneCycleLR 调度器与 AMP 混合精度，可根据本机显存和验证集表现调整 batch size、epoch 和学习率。

1. 准备标签地图 (CSV)
根据需求运行对应的脚本生成“地图文件”，程序将自动指向 500Hz 高频数据：

细粒度 10 类（推荐）：

```powershell
python .\code\tools\ptbxl_prepare_labels_fine.py --ptb .\data\ptbxl --band hr
```

演示 5 类：

```powershell
python .\code\tools\ptbxl_prepare_labels.py --ptb .\data\ptbxl --band hr
```

2. 启动模型训练

训练参数需要按显存和数据规模调整。可以从 `batch=32` 或 `batch=64`、`epochs=15` 起步，再根据验证集指标继续调参。

```powershell
# --amp: 开启混合精度加速；batch 可按显存调整
python .\code\train_5cls.py --ptb ".\data\ptbxl" --out ".\models_fine" --epochs 15 --batch 64 --amp
```

注：Windows 环境下请勿开启 torch.compile，以避免 Triton 兼容性报错。

3. 自动化后续流程
训练完成后，依次执行以下命令完成模型标定与转换：

```powershell
# 1. 标定最佳诊断阈值
python -m code.tools.calc_thresholds --ptb ".\data\ptbxl" --ckpt ".\models_fine\inception_5cls_best.pt" --out ".\models_fine"

# 2. 导出 ONNX 模型用于推理服务
python -m code.tools.export_onnx --ckpt ".\models_fine\inception_5cls_best.pt" --out ".\models_fine"

# 3. 验证 ONNX 模型一致性
python -m code.tools.onnx_eval --ptb ".\data\ptbxl" --onnx ".\models_fine\inception_5cls.onnx" --ckpt ".\models_fine\inception_5cls_best.pt"
```

### 4.3 构建本地 CaseBank 数据库（可选）

公开仓库只提供 `data/casebank_empty_shell/` 作为空壳结构示例，不包含真实病例、概率、向量或渲染图片。若需要 WebUI 返回相似病例，请用自己的合规 ECG 数据、标签和模型在本地生成真实索引：

```powershell
# 1. 先准备自己的 display index SQLite。
#    公开仓库不提供医院字段示例；私有/临床来源数据不得提交到 GitHub。
$env:YOUR_CASE_DISPLAY_SQLITE = "path\to\your_case_display_index.sqlite"

# 2. 用自己的模型和 display index 生成 CaseBank 向量数据库。
python -m code.tools.build_casebank_vector_index `
  --display_sqlite $env:YOUR_CASE_DISPLAY_SQLITE `
  --model_dir ".\models_fine" `
  --out_dir ".\data\casebank_vector_index" `
  --cache_dir ".\casebank_cache" `
  --include_cpsc 0 `
  --overwrite 1

# 3. 运行服务时指向本地真实索引。
$env:ECG_CASEBANK_DIR = ".\data\casebank_vector_index"
```

`data/casebank_vector_index/`、`data/casebank_display_assets/`、`casebank_cache/` 和模型目录均为本地产物；若包含私有或医院来源信息，不应提交到 GitHub。

## 五、启动服务与 WebUI

完成模型、阈值、ONNX 和可选 CaseBank 数据库准备后，再启动服务：

```powershell
# 激活
.\.venv\Scripts\Activate.ps1

# 启动服务（GitHub 公开版需自行配置可公开使用的模型权重）
uvicorn code.service.app:APP --host 127.0.0.1 --port 8000

# 访问
#   UI:   http://127.0.0.1:8000/ui/
#   文档: http://127.0.0.1:8000/docs
```

### API 约定

- `GET /health` -> `{status, classes, model_dir, digitizer}`
- `GET /digitizer_status` -> `{configured_backend, felix, ahus, ...}`
- `POST /infer_record` -> `{"record":"records500/00000/00001_hr"}`
- `POST /infer_wfdb_files` -> multipart form, same-record `.hea` plus `.dat/.mat`
- `POST /infer_xml` -> multipart form, HL7 AnnotatedECG XML with 12-lead waveform
- `POST /infer_image` -> multipart form, ECG image `png/jpg/bmp/tiff`, optional `layout=auto|3x4|6x2`

当前 `START.BAT` 默认设置 `ECG_DIGITIZER_BACKEND=ahus`，因此从启动脚本打开服务时图片入口会进入 Ahus/Open-ECG-Digitizer 自动版式识别流程。若手动启动 uvicorn 且未设置该环境变量，则默认仍为 `none`，图片入口会返回 501 配置提示。

### UI 状态

- `/ui/` 已整理为双栏本地工作台：左侧输入，右侧结果。
- 顶部显示服务、模型目录、阈值文件和低置信度拒判状态。
- 输入区提供 `接入WFDB文件`、`图片上传`、`接入XML文件` 三个入口。
- 推理结果显示中文结论、模型目录、输入指标、概率/阈值/触发表格和概率条。
- 后端和WebUI最终输出均强制执行 `NORM` 与异常诊断互斥。
- 图片识别失败时显示用户提示、建议、错误代码和技术细节。

### 统一返回示例

```json
{
  "source": "wfdb|wfdb_upload|xml|image",
  "layout": "n/a|3x4|6x2|unknown",
  "record": "<basename>",
  "classes": [...],
  "prob": {"NORM":0.12, ...},
  "thresholds": {"NORM":0.50, ...},
  "labels": [...],
  "labels_zh": [...],
  "uncertain": false,
  "uncertainty_reason": null,
  "low_confidence": false,
  "max_prob": 0.81,
  "max_prob_class": "STTC",
  "uncertainty_message": null,
  "triggered": {"NORM": false, "STTC": true, ...},
  "triggered_raw": {"NORM": true, "STTC": true, ...},
  "norm_exclusion_applied": true,
  "input_metrics": {"n_leads":12, "model_shape":[12,5000], ...},
  "text": "中文报告..."
}
```

不确定输出规则：

```powershell
# 默认启用
$env:ECG_UNCERTAINTY_ENABLED = "1"

# 当所有类别均未过阈值，且最高概率低于该值时，视为“所有类别置信度均低”
$env:ECG_UNCERTAINTY_MAX_PROB = "0.30"
```

解释口径：该层用于服务安全和后续科研记录，不等于新增第 11 个训练标签，也不参与当前 10 类阈值标定。AFIB、PAC、PVC 等暂未纳入 10 类体系的明确异常数据，后续可作为 out-of-scope abnormal 验证集，用于校准不确定输出阈值。

图片链路当前状态：`START.BAT` 默认启用 Ahus；Felix 保留为实验/备选后端。图片数字化必须先通过完整度与质量门槛，低质量或导联不完整时返回结构化错误，不进入 ONNX 诊断。历史图片链路测试记录保存在内部研究日志中，不随 GitHub README 发布。

图片识别失败时，`/infer_image` 返回面向用户的结构化错误，而不是只返回工程日志。例如：

```json
{
  "detail": {
    "error_code": "IMAGE_DIGITIZATION_LOW_QUALITY",
    "message": "心电图图片无法正确识别，可能由于图片清晰度不足、拍摄角度偏斜、导联显示不完整、版式不符合要求或扫描质量较差导致。",
    "suggestion": "请重新上传更清晰、完整的 12 导联心电图图片，或改用 WFDB/数字 ECG 文件。",
    "technical_detail": {
      "finite_fraction": 0.2351,
      "required_finite_fraction": 0.4
    }
  }
}
```

前端 UI 会显示 `message`、`suggestion` 和 `error_code`；`technical_detail` 保留给调试与科研记录。

当前公开 README 只描述图片入口的安全行为。具体本地测试图片、失败比例、wrapper 试错过程和后端复测结果保存在内部研究日志中。

## 六、推理与报告（批量）

```powershell
# 验证集批量推理并生成 CSV 与文本报告
python -m code.tools.infer_and_report --ptb ".\data\ptbxl" --ckpt ".\models_fine\inception_5cls_best.pt" --thr ".\models_fine\thresholds_5cls.json" --out ".\outputs_fine" --split val
```

## 七、图片上传流水线（安全骨架阶段）
入口：POST /infer_image 或 UI 第二栏

当前原则：图片数字化是可选能力，不能影响 `/health`、WFDB 上传和 XML 上传。`START.BAT` 默认启用 Ahus；手动启动时可用环境变量切换。

统一流程：

图片 → 版式检测（layout.py） → 可选 digitizer 后端（digitize.py） → 12导联矩阵 `(12, L)` → 归一化/重采样 → ONNX → 阈值 → 中文报告

配置：`code\service\config.py`

```powershell
$env:ECG_DIGITIZER_BACKEND = "none"   # 手动关闭，图片入口返回 501
$env:ECG_DIGITIZER_BACKEND = "felix"  # third_party/ECG-Digitiser
$env:ECG_DIGITIZER_BACKEND = "ahus"   # third_party/Open-ECG-Digitizer，当前 START.BAT 默认
```

后端状态：

- `none`：默认状态。服务可启动，WFDB 推理可用，图片入口返回配置提示。
- `felix`：封装现有 `third_party/ECG-Digitiser`，只读调用其 CLI，适合作为高质量扫描件/3×4 候选。
- `ahus`：Ahus-AIM Open-ECG-Digitizer，可在本地配置为图片数字化后端；若依赖、权重或虚拟环境未准备好，图片入口会返回结构化错误。

历史图片链路测试、layout 修复、Felix/Ahus 对比和本地测试图片结果保存在内部研究日志中。公开 README 只保留当前接口、后端配置方式和安全拒判原则。

## 八、配置切换

编辑 code\service\config.py：

切换模型目录：通过 `ECG_MODEL_DIR` 或 `MODEL_DIR` 指向目标目录，不移动、不覆盖模型文件。

修改阈值/类别文件：CLASSES_JSON / THRESHOLDS_JSON

关闭图片数字化：`ECG_DIGITIZER_BACKEND=none`

## 九、指标解释

Macro AUROC：每类 AUROC 平均。0.5 随机，1.0 完美。阈值无关。

Macro F1：阈值后每类 F1 的平均。受类别不平衡与阈值影响大。

ONNX vs Torch MSE：两端概率差的均方误差。≈0 代表导出一致。

## 十、常见问题

ParserError 读 CSV：PhysioNet 链接必须使用 ...?download。

ModuleNotFoundError: code...：用包方式运行 python -m code.xxx，或设置 PYTHONPATH。

tsai 与 torch 2.8 冲突：不安装 tsai。如需 tsai 则建副环境。

Digitiser未配置：默认返回 501；检查 `ECG_DIGITIZER_BACKEND`、Felix/Ahus 路径与本 README 的图片上传流水线部分。

中文乱码/语法错误：确保源文件 UTF-8 保存。PowerShell 用 -Encoding UTF8。

torch._inductor.exc.TritonMissing：Windows 环境下 torch.compile 兼容性较差。若遇到此报错，请在 train_5cls.py 中注释掉 torch.compile 相关代码。

训练第一轮很慢：程序正在将 WFDB 记录转换为缓存文件（.npy），第二轮起速度会大幅提升。

## 十一、路线图

- 继续完善 OneCycleLR、AMP、数据增强和长训配置。
- 完善 6×2 图片版式训练与版式识别。
- 增加 ResNet1D 等基线模型并支持切换。
- 增加 AUPRC、类别级灵敏度/特异度和置信区间汇报。
- 增加 NSSM/Windows 服务或托盘启动等打包部署方案。
- 完善图片合规模块，包括遮挡检测、分辨率校验和走纸参数识别。
- Challenge 2020 / CPSC 数据转换与训练需由读者按 `data/README.md` 中的官方来源在本地完成下载、QC 和转换。

## 十二、本地病例检索与 LLM 报告层（设计与当前状态）

目标：在 InceptionTime/ONNX 主诊断链路之外，提供相似病例检索和可选本地 LLM 报告层。InceptionTime/ONNX 继续承担主要诊断判别职责；LLM 只消费结构化结果和检索摘要，不直接接管 12 导联原始波形判断。

推荐架构：

WFDB / 纯数字 ECG
→ 归一化与重采样
→ InceptionTime / ONNX 推理
→ 类别概率、阈值、阳性标签、置信度
→ 本地病例检索模块（RAG）
→ 本地 LLM 生成中文报告、解释与问答

设计原则：

1. 不让 LLM 自由遍历项目文件夹或直接读取任意文件。医学/科研推理链路应保持可控、可复现。
2. 由 Python 程序负责检索病例库，LLM 只接收检索后的少量结构化结果。
3. 近期不要让 Qwen/Gemma/DeepSeek 直接取代 InceptionTime。若未来要让 LLM 理解 ECG 波形，应采用 ECG Encoder + Projector/Adapter + LLM 的多模态路线。
4. 所有 LLM 输出必须保留“研究模型、非临床诊断”的声明，并尽量引用概率、阈值、相似病例等证据。

本地 LLM 接口预案：

1. 近期预留为 LM Studio 本地服务模式：用户手动启动 LM Studio，加载本地模型，并开放本地 HTTP 端口。
2. 项目侧只访问 `127.0.0.1` 本地端口，不依赖云端 API；默认按 OpenAI-compatible Chat Completions 接口封装。
3. 后续在 `code/service/config.py` 中预留类似配置：

```python
LLM_ENABLED = False
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL = "local-model"
LLM_TIMEOUT_SEC = 60
```

4. 推理服务中可新增独立适配层，例如 `code/service/llm_report.py`，只负责把结构化诊断 JSON 发送到 LM Studio，并返回中文报告文本。
5. 若 LM Studio 未启动、端口不可用或模型未加载，服务应自动回退到现有规则模板报告，不影响 `/infer_record` 的基础诊断。

病例库原型：

当前 CaseBank 原型只做相似病例检索层，不改变 10 类主诊断标签，不替代 InceptionTime/ONNX。工程实现使用 SQLite + NPY：SQLite 保存病例元数据，NPY 保存概率、阈值 margin、基础波形特征和检索向量。

MVP-0 检索向量：

```text
zscore(probabilities) + zscore(threshold margins) + zscore(basic waveform features)
```

CaseBank 当前运行原则是纯向量检索：只用 85 维检索向量的 cosine similarity 排序，不使用模型预测标签进行候选过滤或重排序。结果默认使用 `ECG_CASEBANK_SCORE_THRESHOLD=0.0` 返回最近 top-k；不足 10 个时不硬凑，并在 JSON `warnings` 中说明。MVP-1 再考虑 `forward_features()`、ONNX embedding 输出和 embedding 主导的检索，但仍需保持“标签不参与检索排序”的边界，除非另设对照实验。

CaseBank MVP 调试命令、早期 CSV 草案、public-dataset reference 构建数量和 smoke evaluation 结果保存在内部研究日志中。公开 README 只保留当前接口边界；公开发布时不包含任何医院来源 CaseBank 索引、医院 ECG 渲染图或医院训练模型输出。

推理时流程：

1. 当前 ECG 经 ONNX 得到概率、阈值 margin 和基础波形特征。
2. 程序拼接并标准化为 CaseBank 检索向量，按纯向量 cosine similarity 检索 top-k 相似病例。
3. 从 CSV 或数据库读取相似病例的标准标签、报告摘要、关键概率。
4. 拼成结构化 JSON 输入本地 LLM。
5. LLM 生成中文报告、异常解释、复查建议和置信度提示。

LLM 输入示例：

```json
{
  "prediction": {
    "labels": ["STTC"],
    "prob": {"NORM": 0.12, "STTC": 0.81},
    "thresholds": {"NORM": 0.50, "STTC": 0.42}
  },
  "similar_cases": [
    {
      "case_id": "PTBXL-00001",
      "labels": ["STTC"],
      "summary": "ST-T异常，建议结合临床与既往心电图比较。"
    }
  ],
  "output_requirements": {
    "language": "zh-CN",
    "format": "structured_report",
    "medical_disclaimer": true
  }
}
```

阶段路线：

1. 第一阶段：InceptionTime/ONNX + 规则模板报告。
2. 第二阶段：InceptionTime/ONNX + 本地 LLM 报告润色。
3. 第三阶段：加入 CSV/NPY 病例检索，形成轻量 RAG。
4. 第四阶段：升级向量索引与病例数据库，支持相似病例解释。
5. 第五阶段：在有足够报告数据后，对本地小模型做 LoRA 微调。
6. 远期阶段：探索 ECG Encoder + LLM Adapter 的真正多模态 ECG-LLM。

## 十三、模型训练扩展与公开边界

公开仓库提供模型代码、公共数据适配脚本和本地训练流程示例；不附带训练好的权重、内部实验输出、医院数据或医院来源模型产物。完整历史实验记录、逐项指标表和工程命令保存在内部研究文档中，不进入公开 README。

可复现扩展路径：

1. 使用 PTB-XL 等公共数据生成标签 CSV，并训练当前 10 类输出头。
2. 若加入 Chapman-Shaoxing 等额外公共数据，应先转换为与模型一致的 12 导联、固定长度张量格式。
3. 只纳入能明确映射到当前 10 类标签体系的样本；`AFIB / AFLT / PAC / PVC` 等未覆盖标签不应被当作 10 类全阴性样本。
4. 每次更换训练数据或输出头后，都需要重新标定阈值、导出 ONNX，并验证 Torch/ONNX 推理一致性。

公开说明边界：

1. 本仓库说明的是工程实现和本地复现流程，不把单次本地训练结果表述为论文级因果结论。
2. 若用于论文，应清楚区分 baseline 模型、额外公共数据微调模型和任何本地私有数据模型。
3. `2AVB / 3AVB / WPW` 等低支持类别的类级指标可能波动较大，正式研究应提供外部验证、置信区间或重复实验。
4. 后续若扩展 `AFIB / AFLT / PAC / PVC` 等新诊断，需要重新定义类别表、输出头、阈值和评估协议。

## 十四、许可与声明

本项目公开代码整体采用 GPL-3.0 许可证，详见根目录 `LICENSE`。

`hfawaz/InceptionTime` 官方仓库同样采用 GPL-3.0；本项目不应表述为原创提出 InceptionTime 架构，引用与许可边界以论文、原仓库和本仓库 `LICENSE` 为准。

本项目引用 InceptionTime 架构思想，但不得表述为原创提出 InceptionTime。论文和公开材料应引用：Ismail Fawaz H, Lucas B, Forestier G, Pelletier C, Schmidt DF, Weber J, et al. InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery*. 2020;34:1936-1962. doi: `10.1007/s10618-020-00710-y`.

仅科研与内部评估用途，不直接用于临床决策。

公开代码许可证不等于公开所有数据或模型权重。PTB-XL、Chapman、CPSC2018 等公共数据仍需遵守其原始数据许可；医院 ECG 数据、病例级标签、私有回链文件、医院来源模型权重和 CaseBank 索引不随 GitHub 公开发布。

## 十五、医院 ECG 数据公开边界

当前 10 类为：

- `NORM`, `MI`, `STTC`, `LVH`, `LBBB`, `RBBB`, `1AVB`, `2AVB`, `3AVB`, `WPW`

本项目使用的医院 ECG 数据为伦理审批范围内的脱敏研究数据。公开仓库只保留模型代码、公共数据适配逻辑和可复现实验框架；不上传医院原始数据、脱敏数据库、病例级标签表、私有回链文件、脱敏过程的具体操作说明，且不发布任何使用医院心电图数据训练或微调得到的最终模型/候选模型权重。

标签是多标签体系，不是互斥单标签。`NORM` 与异常诊断互斥；异常标签按当前 10 类定义输出。

## 十六、CaseBank 向量索引与 WebUI 检索公开边界

CaseBank 向量索引属于模型相关的本地生成产物。若索引使用医院 ECG 数据、医院训练/微调模型、医院病例诊断或医院 ECG 渲染图片，则只允许保留在伦理审批范围内的本地研究环境，不随 GitHub 公开发布。

公开仓库只保留可复现代码和缓存目录约定：

- 构建脚本：`code/tools/build_casebank_vector_index.py`
- 本地索引目录：`data/casebank_vector_index`（生成产物，按公开边界决定是否保留）
- 统一运行缓存：`casebank_cache`
- WebUI/service 可读取本地索引，但公开版本不附带医院来源索引、医院渲染图片或医院训练/微调模型权重。

公开仓库中的 `data/casebank_empty_shell/` 只是数据库空壳，不是真实 CaseBank 数据库。该目录只包含零行 SQLite schema、零行 `.npy` 占位数组和非数据派生的占位配置，用于展示文件结构和让代码知道 CaseBank 目录应长什么样；其中没有真实 ECG、病例标签、患者级标识、模型概率、相似病例向量或渲染图片。

读者若要创建自己的 CaseBank 数据库，需要在本地准备自己的合规 ECG 数据、标签和模型后重新构建：

```powershell
# 1. 先在本地准备合规的 display index SQLite。
#    公开仓库不提供医院字段示例；若数据来自私有/临床来源，
#    请只在本地适配字段和构建脚本，不要提交数据、回链表或病例级标签表。
$env:YOUR_CASE_DISPLAY_SQLITE = "path\to\your_case_display_index.sqlite"

# 2. 用自己的模型和 display index 生成检索向量数据库。
python -m code.tools.build_casebank_vector_index `
  --display_sqlite $env:YOUR_CASE_DISPLAY_SQLITE `
  --model_dir path\to\your_model_dir `
  --out_dir data\casebank_vector_index `
  --cache_dir casebank_cache `
  --include_cpsc 0 `
  --overwrite 1

# 3. 运行服务时指向本地真实索引。
$env:ECG_CASEBANK_DIR = "data\casebank_vector_index"
```

`path\to\your_model_dir` 至少需要包含可运行的 ONNX 模型、类别文件和阈值文件，例如 `inception_5cls.onnx`、`classes_*.json`、`thresholds_*.json`。若使用 CPSC2018，可将 `--include_cpsc 0` 改为 `--include_cpsc 1`，并准备 `data\challenge_2020\cpsc_2018_manifest\cpsc2018_manifest.csv`。生成后的 `data/casebank_vector_index/`、`data/casebank_display_assets/`、`casebank_cache/` 和模型目录仍然属于本地产物，不应提交到 GitHub。

当前检索向量为 85 维：

```text
zscore(probabilities) + zscore(threshold_margins) + zscore(basic_wave_features)
```

说明：

1. WebUI 每次推理后会返回 CaseBank 最近 10 例，展示真实诊断、标签范围、模型预测标签和渲染 ECG 图片。
2. 图片按需渲染到根目录 `casebank_cache`，清理该缓存不会删除索引或源 ECG 数据。
3. 非 10 类、混合 out-of-scope、unmapped ECG 已纳入检索展示，但不会被当成 10 类全阴性训练样本。
4. `NORM` 与异常诊断的互斥规则已在 CaseBank 预测标签和 WebUI/service 结果中执行。
5. 若后续更换最终 INCEP 模型或嵌入层，必须重建 `data/casebank_vector_index`。
