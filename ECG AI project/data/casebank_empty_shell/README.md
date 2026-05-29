# CaseBank Empty Shell

This directory is safe to publish with the open-source repository.

It contains only the CaseBank file layout and empty SQLite schemas. It does not contain real ECG records, labels, patient-level identifiers, model probabilities, retrieval vectors, rendered ECG images, or private/local paths.

Use it as a structural placeholder only. A real CaseBank must be rebuilt locally with the project tools and kept out of GitHub.

Runtime override example:

```powershell
$env:ECG_CASEBANK_DIR = "data/casebank_empty_shell"
```

Expected real, non-public CaseBank locations:

- `data/casebank_vector_index/`
- `data/casebank_display_assets/`
- `data/casebank_public_reference/`
- `data/casebank_ptbxl_chapman/`

Files in this shell:

- `case_index.sqlite`: zero-row runtime CaseBank index shell.
- `case_display_index.sqlite`: zero-row display-index shell.
- `case_*.npy`: zero-row placeholder arrays with the expected shapes.
- `build_config.json` and `vector_stats.json`: neutral, non-data-derived placeholder metadata.
- `*_schema.sql`: readable SQLite schema dumps for review.
