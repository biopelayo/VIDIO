-- VIDIO - Vision-Integrated Diagnostic Imaging Orchestrator
-- Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- USERS & AUTHENTICATION
-- ============================================================

CREATE TABLE "user" (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    surname TEXT,
    role TEXT DEFAULT 'analyst',
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE user_token (
    id_user UUID PRIMARY KEY REFERENCES "user"(id),
    token TEXT NOT NULL,
    expires TIMESTAMP NOT NULL,
    created TIMESTAMP NOT NULL
);

-- ============================================================
-- CLINICAL DATA HIERARCHY (DICOM-compatible)
-- ============================================================

CREATE TABLE patient (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    medical_record_number TEXT UNIQUE,
    name TEXT,
    date_of_birth DATE,
    sex CHAR(1),
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE study (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_patient UUID REFERENCES patient(id) ON DELETE CASCADE,
    study_date TIMESTAMP,
    modality TEXT NOT NULL,
    institution TEXT,
    protocol TEXT,
    description TEXT,
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE series (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_study UUID REFERENCES study(id) ON DELETE CASCADE,
    series_number INTEGER,
    description TEXT,
    body_part TEXT,
    modality_subtype TEXT,
    manufacturer TEXT,
    equipment_model TEXT,
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE image (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_series UUID REFERENCES series(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    file_format TEXT,
    file_size_bytes BIGINT,
    pixel_spacing JSONB,
    dimensions JSONB,
    sop_instance_uid TEXT,
    info JSONB,
    selected BOOLEAN DEFAULT TRUE,
    deleted BOOLEAN DEFAULT FALSE
);

-- ============================================================
-- ANNOTATIONS & FINDINGS
-- ============================================================

CREATE TABLE annotation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_image UUID REFERENCES image(id) ON DELETE CASCADE,
    id_user UUID REFERENCES "user"(id),
    annotation_type TEXT NOT NULL,
    label TEXT,
    geometry JSONB NOT NULL,
    confidence TEXT,
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE finding (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_image UUID REFERENCES image(id) ON DELETE CASCADE,
    id_study UUID REFERENCES study(id),
    id_process UUID,
    finding_type TEXT NOT NULL,
    disease_category TEXT,
    severity TEXT,
    confidence REAL,
    ml_model_id UUID,
    info JSONB NOT NULL,
    reviewed BOOLEAN DEFAULT FALSE,
    id_reviewer UUID REFERENCES "user"(id),
    review_notes TEXT,
    deleted BOOLEAN DEFAULT FALSE
);

-- ============================================================
-- TCGA INTEGRATION
-- ============================================================

CREATE TABLE tcga_case (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_patient UUID REFERENCES patient(id),
    case_barcode TEXT UNIQUE,
    project TEXT,
    disease_type TEXT,
    primary_diagnosis TEXT,
    tumor_stage TEXT,
    gdc_case_id TEXT,
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE tcga_slide (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_tcga_case UUID REFERENCES tcga_case(id) ON DELETE CASCADE,
    id_image UUID REFERENCES image(id),
    slide_barcode TEXT,
    gdc_file_id TEXT,
    tissue_type TEXT,
    stain_type TEXT,
    magnification TEXT,
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

-- ============================================================
-- SPATIAL TRANSCRIPTOMICS
-- ============================================================

CREATE TABLE spatial_experiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_study UUID REFERENCES study(id) ON DELETE CASCADE,
    id_image UUID REFERENCES image(id),
    platform TEXT,
    h5ad_path TEXT,
    n_spots INTEGER,
    n_genes INTEGER,
    info JSONB,
    deleted BOOLEAN DEFAULT FALSE
);

-- ============================================================
-- ML MODEL REGISTRY
-- ============================================================

CREATE TABLE ml_model (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    modality TEXT NOT NULL,
    architecture TEXT,
    task TEXT,
    weights_path TEXT,
    info JSONB,
    UNIQUE(name, version)
);

-- ============================================================
-- ASYNC PROCESS TRACKING
-- ============================================================

CREATE TABLE process (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_user UUID REFERENCES "user"(id),
    id_study UUID REFERENCES study(id),
    type TEXT NOT NULL,
    status TEXT DEFAULT 'PENDING',
    progress INTEGER DEFAULT 0,
    pid INTEGER,
    parameters JSONB,
    result JSONB,
    time_start TIMESTAMP,
    time_end TIMESTAMP,
    error_message TEXT
);

-- ============================================================
-- FILE TRACKING
-- ============================================================

CREATE TABLE file (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_user UUID REFERENCES "user"(id),
    name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    content_type TEXT,
    info JSONB,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- AUDIT LOG
-- ============================================================

CREATE TABLE log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v1(),
    id_user UUID,
    type TEXT,
    date TIMESTAMP DEFAULT NOW(),
    message TEXT,
    info JSONB
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX user_username_idx ON "user" USING btree(username);
CREATE INDEX study_patient_idx ON study(id_patient);
CREATE INDEX study_modality_idx ON study(modality);
CREATE INDEX series_study_idx ON series(id_study);
CREATE INDEX image_series_idx ON image(id_series);
CREATE INDEX image_format_idx ON image(file_format);
CREATE INDEX annotation_image_idx ON annotation(id_image);
CREATE INDEX annotation_label_idx ON annotation(label);
CREATE INDEX finding_image_idx ON finding(id_image);
CREATE INDEX finding_study_idx ON finding(id_study);
CREATE INDEX finding_type_idx ON finding(finding_type);
CREATE INDEX finding_severity_idx ON finding(severity);
CREATE INDEX finding_disease_idx ON finding(disease_category);
CREATE INDEX process_status_idx ON process(status);
CREATE INDEX process_study_idx ON process(id_study);
CREATE INDEX process_type_idx ON process(type);
CREATE INDEX tcga_case_barcode_idx ON tcga_case(case_barcode);
CREATE INDEX tcga_case_project_idx ON tcga_case(project);
CREATE INDEX tcga_slide_case_idx ON tcga_slide(id_tcga_case);
CREATE INDEX spatial_study_idx ON spatial_experiment(id_study);
CREATE INDEX log_date_idx ON log(date);
CREATE INDEX log_type_idx ON log(type);
