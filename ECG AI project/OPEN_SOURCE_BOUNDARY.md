# Open Source Boundary

This repository is intended to publish code, configuration examples, and public-dataset adapters only.

Hospital ECG data are not included in the public GitHub release.

The hospital ECG component of this study was approved by the Ethics Committee of Xiamen Susong Hospital.

Unified wording for the hospital ECG dataset:

- After batch extraction of ECG XML files, the project immediately performed de-identification.
- A study numbering system was used instead of personal information.
- The internal research database contains only study numbers, standardized ECG signals, and corresponding true diagnostic conclusions.
- The project does not publish, expose, or retrospectively reconstruct any real personal privacy information or fields.

Do not upload to GitHub:

- raw hospital ECG XML/CSV files;
- hospital de-identified databases or case-level hospital label tables;
- private linkage files or any materials that could reconnect study numbers to original records;
- rendered hospital ECG images unless separately approved for public release;
- final or candidate model weights trained or fine-tuned with hospital ECG data;
- implementation details of the de-identification process.

CaseBank upload boundary:

- Do not upload real CaseBank index/database directories:
  - `data/casebank_vector_index/`
  - `data/casebank_display_assets/`
  - `data/casebank_public_reference/`
  - `data/casebank_ptbxl_chapman/`
- The public repository may include only:
  - `code/casebank/`
  - CaseBank build/search/evaluation scripts under `code/tools/`
  - lightweight tests under `tests/`
  - `data/casebank_empty_shell/`, which contains zero real cases and only the empty schema/file layout.
