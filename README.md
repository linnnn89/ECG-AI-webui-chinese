# ECG-AI WebUI Chinese

**English** | [中文](#中文说明)

ECG-AI WebUI Chinese is a locally deployable research software workflow for 12-lead electrocardiogram (ECG) interpretation. It combines multi-label ECG classification, ONNX-based local inference, FastAPI WebUI deployment, ECG XML/WFDB input support, experimental ECG image digitization, Chinese report generation, uncertainty handling, and similar-case retrieval through a local CaseBank design.

This repository is prepared as the public software companion for a manuscript on a locally deployable ECG interpretation workflow combining multi-label classification and case-based retrieval. The project is intended for research, reproducibility, technical evaluation, and local workflow exploration. It is **not a medical device**, **not a substitute for clinician interpretation**, and **not intended for unsupervised clinical diagnosis**.

> Detailed Chinese documentation is available at: [ECG AI project/README.md](ECG%20AI%20project/README.md)

## Repository description for GitHub About

**Recommended short description:**

> Local ECG AI WebUI for WFDB/XML/image-to-waveform inference, 10-label ONNX ECG classification, Chinese reports, uncertainty handling, and CaseBank similar-case retrieval.

**Suggested topics:**

`ecg`, `electrocardiogram`, `medical-ai`, `clinical-decision-support`, `deep-learning`, `pytorch`, `onnx`, `onnxruntime`, `fastapi`, `wfdb`, `case-based-retrieval`, `ecg-classification`, `health-informatics`, `local-deployment`, `chinese`

## Key features

- **Local deployment:** FastAPI service and WebUI designed for offline or local-network use.
- **Digital ECG inputs:** WFDB record upload, PTB-XL-style record inference, and HL7 Annotated ECG XML conversion.
- **Image pathway:** Experimental ECG image input with pluggable digitizer backends; image-to-waveform performance depends on local digitizer configuration and image quality.
- **10-label ECG classifier:** `NORM`, `MI`, `STTC`, `LVH`, `LBBB`, `RBBB`, `1AVB`, `2AVB`, `3AVB`, and `WPW`.
- **ONNX inference:** PyTorch-trained InceptionTime-style 1D CNN exported to ONNX for local inference.
- **Threshold-based post-processing:** Per-class thresholds, `NORM`–abnormal mutual exclusion, and indeterminate `UNKNOWN` handling for low-confidence outputs.
- **Chinese report generation:** Model outputs are converted into readable Chinese ECG interpretation text for clinician review.
- **CaseBank retrieval:** Local similar-case retrieval design using classifier-informed and waveform-summary features.
- **Privacy-oriented release:** Public code only; hospital ECG data, patient-level files, private CaseBank indices, and non-public model weights are not included.

## Current research status

The corresponding retrospective technical evaluation used public ECG datasets and de-identified hospital ECG data to evaluate:

1. a 10-label ECG classifier;
2. a locally deployable FastAPI/WebUI workflow;
3. an expanded CaseBank retrieval module for auditable similar-case review;
4. an auxiliary reader-impact substudy among junior physicians.

The equity relevance of the project is implementation-oriented: local deployability, auditability, and support for non-specialist review in settings where specialist ECG interpretation may be limited. The software release does **not** claim demonstrated patient-level equity benefit or clinical outcome improvement.

## Data and model availability

This public repository does **not** include:

- raw hospital ECG XML files;
- patient identifiers or private linkage files;
- hospital label tables;
- private CaseBank indices or rendered case assets;
- model weights trained or fine-tuned using non-public hospital ECG data;
- third-party ECG digitizer runtime directories.

Users should prepare their own legally usable ECG datasets and local model artifacts. Public ECG datasets such as PTB-XL, Chapman-Shaoxing/Ningbo, and CPSC/Challenge 2020 can be used according to their original licenses and access terms.

## Quick start

The detailed setup guide is maintained in Chinese in the project subdirectory. A typical Windows/PowerShell workflow is:

```powershell
cd "ECG AI project"
.\.venv\Scripts\Activate.ps1
uvicorn code.service.app:APP --host 127.0.0.1 --port 8000
```

After startup:

- WebUI: `http://127.0.0.1:8000/ui/`
- API documentation: `http://127.0.0.1:8000/docs`
- Health check: `GET /health`

Core API endpoints include:

- `POST /infer_record` for PTB-XL-style relative records;
- `POST /infer_wfdb_files` for uploaded `.hea` plus signal files;
- `POST /infer_xml` for HL7 Annotated ECG XML files;
- `POST /infer_image` for ECG image upload with optional `layout=auto|3x4|6x2`.

## Recommended citation

If you use this repository in academic work, please cite the archived software release DOI when available and cite the associated manuscript after publication. The `CITATION.cff` file is provided for GitHub citation metadata.

## License

This project is released under the GNU General Public License v3.0. See the `LICENSE` file for details. Third-party tools, datasets, and model components remain governed by their own licenses and terms.

---

# 中文说明

ECG-AI WebUI Chinese 是一个面向本地部署和科研复现的 12 导联心电图分析软件工作流。项目将多标签心电图分类、ONNX 本地推理、FastAPI WebUI、ECG XML/WFDB 输入、实验性图片数字化、中文报告生成、不确定性处理和本地 CaseBank 相似病例检索组合在一起。

本仓库作为论文配套软件公开，服务于“本地部署的心电图多标签分类与相似病例检索工作流”的技术复现、审稿检查和后续研究。该软件**不是医疗器械**，**不能替代医生判读**，也**不应用于无人监督的临床诊断**。

详细中文使用说明见：[ECG AI project/README.md](ECG%20AI%20project/README.md)

## GitHub 搜索卡片建议

**仓库简介建议：**

> Local ECG AI WebUI for WFDB/XML/image-to-waveform inference, 10-label ONNX ECG classification, Chinese reports, uncertainty handling, and CaseBank similar-case retrieval.

**关键词建议：**

`ecg`, `electrocardiogram`, `medical-ai`, `clinical-decision-support`, `deep-learning`, `pytorch`, `onnx`, `onnxruntime`, `fastapi`, `wfdb`, `case-based-retrieval`, `ecg-classification`, `health-informatics`, `local-deployment`, `chinese`

## 主要功能

- **本地部署：** 基于 FastAPI 和静态 WebUI，可在本机或院内局域网环境运行。
- **数字心电输入：** 支持 WFDB 文件上传、PTB-XL 相对路径记录推理和 HL7 Annotated ECG XML 转换。
- **图片入口：** 保留 ECG 图片上传与数字化后端适配层；实际效果依赖本地 digitizer 配置和图片质量。
- **10 类心电图分类：** `NORM`, `MI`, `STTC`, `LVH`, `LBBB`, `RBBB`, `1AVB`, `2AVB`, `3AVB`, `WPW`。
- **ONNX 本地推理：** 使用 PyTorch 训练的 InceptionTime-style 1D CNN，并导出为 ONNX 进行服务推理。
- **阈值与后处理：** 支持逐类阈值、`NORM` 与异常诊断互斥、低置信度 `UNKNOWN` 输出。
- **中文报告：** 将概率、阈值触发和最终标签转化为中文心电图报告文本。
- **CaseBank 相似病例检索：** 使用分类器概率、阈值 margin 和基础波形特征构建本地相似 ECG 检索流程。
- **隐私友好公开：** GitHub 仅公开代码、配置模板和文档，不公开院内 ECG、患者级数据、私有 CaseBank 索引或非公开模型权重。

## 当前研究定位

配套论文为回顾性技术评估研究，主要评价：

1. 10 标签 ECG 分类器；
2. 本地 FastAPI/WebUI 工作流；
3. 扩展 CaseBank 对可审计相似病例的检索能力；
4. 面向初级医生的辅助读图探索性子研究。

本项目与 health informatics equity 的关系属于 implementation-oriented equity contribution：重点是本地可部署、可审计、可由非专科医生辅助使用。当前研究不声称已经证明患者层面的公平性获益或临床结局改善。

## 数据和模型说明

本公开仓库不包含：

- 医院原始 ECG XML 文件；
- 患者标识符或私有链接表；
- 医院标签表；
- 私有 CaseBank 索引和病例渲染图；
- 使用非公开医院数据训练或微调得到的模型权重；
- 第三方 ECG digitizer 运行目录。

使用者需要自行准备合规 ECG 数据、本地模型和阈值文件。PTB-XL、Chapman-Shaoxing/Ningbo、CPSC/Challenge 2020 等公共数据集应按其原始许可和访问条款使用。

## 快速启动

```powershell
cd "ECG AI project"
.\.venv\Scripts\Activate.ps1
uvicorn code.service.app:APP --host 127.0.0.1 --port 8000
```

启动后访问：

- WebUI：`http://127.0.0.1:8000/ui/`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`GET /health`

核心接口：

- `POST /infer_record`：PTB-XL 风格相对路径记录；
- `POST /infer_wfdb_files`：上传 `.hea` 和对应信号文件；
- `POST /infer_xml`：上传 HL7 Annotated ECG XML；
- `POST /infer_image`：上传 ECG 图片，可选 `layout=auto|3x4|6x2`。

## 引用

如在论文、报告或软件项目中使用本仓库，请优先引用 Zenodo 归档版本 DOI；论文发表后请同时引用对应论文。仓库已提供 `CITATION.cff` 用于 GitHub 引用信息。

## 许可证

本项目以 GNU General Public License v3.0 开源。第三方工具、公共数据集和外部模型组件仍遵循其各自许可证和使用条款。
