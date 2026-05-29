# ECG AI Project Data Directory

Last updated: 2026-05-29

This folder documents public datasets, public conversion outputs, and local generated artifacts. Hospital ECG data and any hospital-derived databases are not part of the public GitHub release.

## Quick Rules

- Raw hospital ECG XML/CSV files, private linkage materials, case-level hospital label tables, and de-identification implementation details must not be uploaded to GitHub.
- Any approved hospital de-identified data are used only inside the local ethics-approved research environment.
- Final model vectors/probabilities are model-dependent generated artifacts and must be regenerated after the final INCEP model is selected.
- Case display images are rendered on demand and cached; generated cache contents are not public source files.

## Main Public Datasets

| Folder | Role | Current use |
|---|---|---|
| `ptbxl` | PTB-XL source data and prepared labels. Contains `labels_fine.csv` and 500-Hz WFDB records. | Public model training/evaluation source and CaseBank/display source. |
| `chapman` | Raw Chapman dataset. | Source archive; usually use `chapman_converted` instead. |
| `chapman_converted` | Converted Chapman NPY signals plus `ground_truth.csv`. | Public model augmentation, CaseBank/display source. Out-of-scope labels such as AFIB/AFLT/PAC/PVC must be flagged, not treated as all-negative. |
| `challenge_2020` | PhysioNet/CinC Challenge 2020 downloads. Currently contains `cpsc_2018` and `cpsc_2018_extra`. | Main `cpsc_2018` download completeness/QC and conservative current-10-class conversion passed on 2026-05-29. `cpsc_2018_extra` still needs separate QC. |

## Preparing Public ECG Training Sources

The public repository does not include ECG waveform databases. Readers should download public ECG datasets themselves and keep them under the ignored local `data/` subfolders.

Use these public sources as the baseline training pool:

| Dataset | Official source | Local target | Role in this project |
|---|---|---|---|
| PTB-XL v1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | `data/ptbxl/` | Primary 12-lead ECG training and validation source. |
| Chapman-Shaoxing / large-scale 12-lead arrhythmia database | https://physionet.org/content/ecg-arrhythmia/1.0.0/ | raw: `data/chapman/`; converted: `data/chapman_converted/` | Public augmentation source for compatible labels. |
| PhysioNet/CinC Challenge 2020 CPSC2018 subset | https://physionet.org/content/challenge-2020/1.0.2/ | raw: `data/challenge_2020/cpsc_2018/`; converted: `data/challenge_2020/cpsc_2018_converted_current10/` | External public ECG source after QC and conservative label mapping. |

Example download commands:

```powershell
# PTB-XL. AWS CLI is usually faster for the full dataset.
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptbxl

# Chapman-Shaoxing / 12-lead arrhythmia database.
aws s3 sync --no-sign-request s3://physionet-open/ecg-arrhythmia/1.0.0/ data/chapman

# PhysioNet/CinC Challenge 2020. This downloads the full Challenge 2020 tree.
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/
```

After downloading Challenge 2020, place the CPSC2018 training records at:

```text
data/challenge_2020/cpsc_2018/
```

Then build the CPSC2018 manifest and conservative current-10-class conversion:

```powershell
python -m code.tools.build_cpsc2018_manifest `
  --cpsc-root data\challenge_2020\cpsc_2018 `
  --out-dir data\challenge_2020\cpsc_2018_manifest

python -m code.tools.build_cpsc2018_converted `
  --manifest-csv data\challenge_2020\cpsc_2018_manifest\cpsc2018_manifest.csv `
  --out-dir data\challenge_2020\cpsc_2018_converted_current10
```

Expected prepared structure:

```text
data/ptbxl/
  ptbxl_database.csv
  scp_statements.csv
  records500/

data/chapman_converted/
  ground_truth.csv
  signals_npy/

data/challenge_2020/cpsc_2018_converted_current10/
  ground_truth.csv
  signals_npy/
```

The base model should be trained first on PTB-XL. Chapman-Shaoxing and CPSC2018 can then be added as public augmentation/external-check sources only after label mapping, signal-shape QC, and split policy are fixed. Any generated model weights, converted waveform arrays, manifests, and evaluation outputs remain local artifacts and should not be committed to GitHub.

Current CPSC2018 candidate manifest:

- Script: `code/tools/build_cpsc2018_manifest.py`
- Output folder: `challenge_2020/cpsc_2018_manifest`
- Main `cpsc_2018` file counts: 6,877 `.hea`; 6,877 `.mat`
- Manifest rows: 6,877
- Format QC: 12 leads and 500 Hz for all records; `.mat` key `val`; no header-vs-MAT shape mismatch
- Caveat: 10 records are shorter than 10 seconds and require explicit exclusion or padding policy before fixed `[12, 5000]` model input
- Current status: candidate manifest plus conservative converted subset; integration into any internal mixed model or retrieval index is outside the public data release

Current CPSC2018 conservative conversion:

- Script: `code/tools/build_cpsc2018_converted.py`
- Output folder: `challenge_2020/cpsc_2018_converted_current10`
- Converted records: 4,349 `.npy` files under `signals_npy`
- Shape/dtype: `[12, 5000]`, `float32`
- Label source: `label_scope=current_10class` rows from `cpsc2018_manifest.csv`
- Main label table: `challenge_2020/cpsc_2018_converted_current10/ground_truth.csv`
- Exclusion audit: `challenge_2020/cpsc_2018_converted_current10/cpsc2018_conversion_excluded.csv`
- Exclusions: 391 mixed current-plus-out-of-scope rows, 2,131 out-of-scope-only rows, and 6 current-10-class rows shorter than 10 seconds
- Current status: public-dataset conversion output only; any hospital-data mixed-model training or retrieval build remains local and is not part of the GitHub release

## Internal Mixed-Model Note

Local candidate models may be trained inside the approved research environment using public datasets plus approved de-identified hospital ECG data.

The public GitHub release does not include:

- hospital-trained or hospital-fine-tuned model weights;
- internal training/validation/holdout split tables;
- case-level hospital labels;
- hospital holdout outputs;
- detailed de-identification implementation records.

Public documentation should not treat local mixed-model metrics as public manuscript evidence unless they are separately exported into approved study materials after ethics/privacy review.

## Hospital ECG Data

Hospital ECG data are not part of the public GitHub release.

Unified public wording:

- The hospital ECG component of this study was approved by the Ethics Committee of Xiamen Susong Hospital.
- After batch extraction of ECG XML files, the project immediately used a de-identified numbering system instead of personal information.
- The internal research database contains only study numbers, standardized ECG signals, and the corresponding true diagnostic conclusions.
- Raw hospital XML/CSV files, private linkage materials, case-level hospital tables, and de-identification implementation details must not be uploaded to GitHub.
- Internal hospital splits may be used only inside the approved research environment and are not distributed with this repository.

## CaseBank and Display Indexes

| Folder | Public-release status | Use |
|---|---|---|
| `casebank_public_reference` | Generated local artifact unless an explicitly public build is selected. | Public-dataset-only retrieval experiments. Model-dependent vectors must be rebuilt after final model selection. |
| `casebank_ptbxl_chapman` | Generated local artifact. | Historical PTB-XL + Chapman engineering reference. |
| `casebank_display_assets` | Not included when it contains hospital-derived display metadata or rendered hospital ECG images. | Local UI/display base only. |
| `casebank_vector_index` | Not included when built with hospital ECG data or hospital-trained model weights. | Local WebUI/service nearest-case retrieval only. |

Generated CaseBank indexes, model probabilities, embeddings, rendered ECG images, and runtime cache files are local artifacts. Do not upload any CaseBank build that contains hospital-derived records, hospital-rendered images, private linkage information, or hospital-trained model outputs.

Runtime render/cache folder: `casebank_cache/`. Clearing this cache does not delete source ECG data or index definitions.

## Image Digitizer Evaluation

| Folder/File | Role |
|---|---|
| `image_digitizer_eval` | Local image digitizer backend evaluation outputs; not included in the public GitHub release. |
| Local image-digitizer QA scripts | Local-only utilities; not included in the public GitHub release because they depend on local images, backend configuration, and generated outputs. |

Image digitizer work is separate from the digital waveform CaseBank pipeline.

## Scripts Stored in `data`

Some local project scripts may exist in this data folder, but they are not part of the public GitHub release:

| File | Role |
|---|---|
| Incremental training and model-comparison scripts | Local-only research utilities. They depend on ignored datasets/model directories and are not uploaded. |
| Mixed hospital/CPSC candidate-model scripts | Local-only research utilities. Any hospital-data use is limited to the approved internal research environment and is not uploaded. |
| Dataset download helpers, if present locally | Local convenience scripts only. Public readers should follow the official dataset pages and license terms above. |
| Legacy processing notes, if present locally | Local notes only; not part of the public release. |

## What To Use For The Next Model Work

Public-repository model work should use public datasets and public conversion outputs only:

- `ptbxl/labels_fine.csv`
- `chapman_converted/ground_truth.csv`
- `challenge_2020/cpsc_2018_converted_current10/ground_truth.csv` only through the documented conservative split/inclusion policy

Approved hospital reviewed data may be used only inside the local ethics-approved research environment. Do not upload hospital split tables, case-level hospital records, hospital-derived CaseBank indexes, or hospital-trained model weights to GitHub.

Do not use these as training inputs:

- raw hospital XML/CSV folders
- private linkage folders
- hospital de-identified databases or case-level hospital label tables in any public GitHub release
- removed dry-run folders
- removed debug CaseBank folders
- raw `challenge_2020/cpsc_2018` records directly; use the converted CPSC folder only through documented split/inclusion rules

## Cleanup Log

Local cleanup logs that mention internal hospital folders or de-identification implementation details are kept outside the public GitHub documentation. Generated debug/dry-run artifacts should remain ignored.

## Documentation Pointers

Detailed internal worklogs are kept outside the public GitHub release. Public documentation should not expose hospital data folders, private linkage paths, case-level hospital tables, or de-identification implementation details.

## Service Input Utilities

The WebUI/service now supports direct WFDB-file upload and ECG XML upload for inference.

- Root XML converter: `ecg_xml_to_wfdb.py`
- Service XML utility: `code/service/xml_ecg.py`
- XML upload conversion path: XML -> centered 10-second 500 Hz WFDB -> ONNX inference
- This is an inference/input utility only. It does not add uploaded XML files to training data, CaseBank display assets, or retrieval vectors.
