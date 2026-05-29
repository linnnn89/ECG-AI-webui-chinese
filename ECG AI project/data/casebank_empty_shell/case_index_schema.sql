CREATE TABLE IF NOT EXISTS cases (
    row_id INTEGER PRIMARY KEY,
    case_id TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    source_record_id TEXT,
    record_path TEXT,
    header_path TEXT,
    image_path TEXT,
    patient_id_hash TEXT,
    split TEXT,
    labels_json TEXT,
    predicted_labels_json TEXT,
    probabilities_json TEXT,
    margins_json TEXT,
    signal_quality_json TEXT,
    has_embedding INTEGER DEFAULT 0,
    build_version TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);
CREATE INDEX IF NOT EXISTS idx_cases_source ON cases(source);
CREATE INDEX IF NOT EXISTS idx_cases_split ON cases(split);
CREATE INDEX IF NOT EXISTS idx_cases_patient ON cases(patient_id_hash);
