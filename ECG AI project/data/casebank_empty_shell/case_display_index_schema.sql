CREATE TABLE cases (case_id TEXT, source TEXT, source_record_id TEXT, split TEXT, diagnosis_text TEXT, source_labels_raw TEXT, labels_10class TEXT, out_of_scope_labels TEXT, label_scope TEXT, record_format TEXT, record_path TEXT, npy_path TEXT, wfdb_record TEXT, header_path TEXT, image_cache_path TEXT, render_status TEXT, render_policy TEXT, metadata_source TEXT, stable_metadata_version TEXT, patient_id_hash TEXT, PRIMARY KEY(case_id));

CREATE INDEX idx_cases_label_scope ON cases(label_scope);

CREATE INDEX idx_cases_source ON cases(source);

CREATE INDEX idx_cases_split ON cases(split);
