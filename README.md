# VIDIO — Vision-Integrated Diagnostic Imaging Orchestrator

**A multi-modal biomedical image analysis platform for automated detection, segmentation, and classification of pathological findings across retinal, histopathology, radiology, and spatial transcriptomics imaging modalities.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Supported Modalities](#supported-modalities)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Running the Server](#running-the-server)
- [CLI Tool](#cli-tool)
- [API Reference](#api-reference)
- [Processing Pipelines](#processing-pipelines)
- [Core Engine](#core-engine)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Overview

VIDIO provides a complete pipeline for biomedical image ingestion, preprocessing, segmentation, statistical analysis, anomaly detection, and clinical findings management through a REST API. It is designed for research and clinical decision support across four imaging modalities.

### Key Capabilities

- **Universal image I/O** with auto-format detection (PNG, DICOM, NIfTI, SVS/WSI, H5AD)
- **4 specialised analysis pipelines** following a consistent 5-stage pattern
- **REST API** with JWT authentication for programmatic and UI access
- **Async processing** with progress tracking and persistent findings
- **DICOM-compatible data model** (Patient → Study → Series → Image → Finding)
- **Severity classification** (CRITICAL / HIGH / MEDIUM / LOW) with confidence scores
- **Audit logging** for all data modifications
- **ML model registry** for versioned model management

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        REST API (Falcon)                      │
│  /patients  /studies  /series  /images  /findings  /analysis  │
│  Auth (JWT)  │  CORS  │  Error Handling  │  Multipart Upload  │
├──────────────┼────────┼──────────────────┼────────────────────┤
│              │   Data Access Layer (DB.py + SQLAlchemy ORM)    │
│              │   PostgreSQL with JSONB metadata columns        │
├──────────────┼────────────────────────────────────────────────┤
│              │          Processing Pipelines                   │
│   Retinal    │  Histology  │  Radiology  │  Spatial Tx        │
│   (fundus,   │  (WSI, TCGA │  (CT, MRI,  │  (10x Visium,     │
│    OCT,      │   H&E, IHC) │   NIfTI)    │   MERFISH)        │
│    slit-lamp)│             │             │                    │
├──────────────┴─────────────┴─────────────┴────────────────────┤
│                   Core Image Processing Engine                 │
│  image_io │ intensity │ filtering │ edges │ morphology        │
│  contours │ clustering │ statistics │ classification │ shapes  │
└──────────────────────────────────────────────────────────────┘
```

### 5-Stage Pipeline Pattern

Every VIDIO pipeline follows the same sequential processing stages:

```
Load → Preprocess → Segment → Statistics → Detect Anomalies
```

1. **Load** — Read image from disk via the universal reader
2. **Preprocess** — Normalise intensities, enhance contrast, apply filters
3. **Segment** — Identify regions of interest (ROIs)
4. **Statistics** — Compute per-ROI intensity and geometric descriptors
5. **Detect** — Compare ROIs against baselines, classify severity

---

## Supported Modalities

### Retinal Imaging
- **Input**: Fundus photography (colour), OCT cross-sections, slit-lamp images
- **Analysis**: Vessel segmentation, optic disc/macula detection, lesion classification
- **Clinical targets**: Diabetic retinopathy, AMD, glaucoma, Alzheimer's lens biomarkers
- **Models**: EfficientNet B4 (classification), U-Net (vessel segmentation)

### Histopathology (WSI)
- **Input**: Whole-slide images (SVS, NDPI, MRXS via OpenSlide), standard images
- **Analysis**: Stain normalisation (Macenko), tile extraction, tumour region detection
- **Clinical targets**: Breast/lung/brain cancer detection and grading
- **Integration**: TCGA/GDC API for public cancer slide retrieval
- **Models**: ResNet50 (tile classification)

### Radiology (CT/MRI)
- **Input**: DICOM files/series, NIfTI volumes
- **Analysis**: HU windowing, skull stripping, isotropic resampling, MNI registration, volumetric analysis
- **Clinical targets**: Brain atrophy (Alzheimer's), glioblastoma volumetrics, stroke lesions
- **Models**: 3D U-Net (MONAI) for volumetric segmentation

### Spatial Transcriptomics
- **Input**: H5AD files (AnnData format from 10x Visium, MERFISH, Slide-seq)
- **Analysis**: QC filtering, normalisation, HVG selection, PCA, Leiden clustering, Moran's I spatial autocorrelation, neighbourhood enrichment
- **Clinical targets**: Tumour microenvironment, spatial gene expression patterns
- **Tools**: scanpy, squidpy

---

## Project Structure

```
D:\VIDIO\
├── app.py                          # Falcon WSGI entry point
├── cfg.json                        # Main configuration
├── VidioTool.py                    # CLI admin tool
├── ProcessManagement.py            # Async pipeline orchestration
│
├── api/                            # REST API layer
│   ├── Cfg.py                      # Global config singleton
│   ├── CORS.py                     # CORS middleware
│   ├── Authentication.py           # JWT + bcrypt auth
│   ├── util/
│   │   ├── VidioException.py       # Exception hierarchy
│   │   └── ImageUtils.py           # Storage path utilities
│   ├── db/
│   │   ├── DB.py                   # Data access layer (52 methods)
│   │   └── model/                  # SQLAlchemy ORM models (15 files)
│   └── resources/                  # Falcon resource handlers (11 files)
│
├── core/                           # Shared image processing engine
│   ├── image_io.py                 # Universal image reader (6 formats)
│   ├── intensity.py                # Normalisation, CLAHE, HU windowing, stain norm
│   ├── filtering.py                # Bilateral, Gaussian, median, unsharp, NLM
│   ├── edges.py                    # Laplacian, Canny, Sobel
│   ├── morphology.py               # Open, close, dilate, erode, fill, remove
│   ├── contours.py                 # Contour detection & geometric analysis
│   ├── clustering.py               # K-Means (adaptive), Mean Shift, HDBSCAN
│   ├── statistics.py               # ROI stats (9 descriptors), z-score comparison
│   ├── classification.py           # Severity scoring & finding builder
│   └── shapes.py                   # Shape detection & Hu moments
│
├── pipelines/                      # Modality-specific analysis pipelines
│   ├── base.py                     # Abstract BasePipeline (5-stage pattern)
│   ├── retinal/                    # Fundus, OCT, slit-lamp analysis
│   ├── histology/                  # WSI tiling, tumour detection, TCGA
│   ├── radiology/                  # DICOM/NIfTI, 3D segmentation
│   └── spatial/                    # Spatial transcriptomics (scanpy/squidpy)
│
├── SQL/
│   └── create_db_vidio.sql         # Full PostgreSQL schema (16 tables)
│
├── tests/                          # pytest test suite
│   ├── test_core/                  # Core engine unit tests
│   ├── test_pipelines/             # Pipeline integration tests
│   └── test_api/                   # API endpoint tests
│
├── requirements.txt                # Python dependencies
└── setup.py                        # Package configuration
```

---

## Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- OpenCV system libraries
- OpenSlide native libraries (for WSI support)

### Install

```bash
# Clone the repository
git clone <repo-url> D:\VIDIO
cd D:\VIDIO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install VIDIO as package (optional, for CLI)
pip install -e .
```

### Minimal Installation (without heavy ML dependencies)

```bash
# Core only (no torch, no scanpy, no MONAI)
pip install falcon falcon-auth2 falcon-cors falcon-multipart \
    PyJWT bcrypt SQLAlchemy psycopg2-binary \
    opencv-python numpy scipy scikit-image Pillow matplotlib \
    scikit-learn requests xlsxwriter python-dateutil pytz tqdm
```

---

## Configuration

Edit `cfg.json`:

```json
{
    "db": {
        "host": "localhost",
        "port": 5432,
        "db": "vidio",
        "user": "admin_vidio",
        "password": "YOUR_DB_PASSWORD"
    },
    "auth": {
        "secret_key": "YOUR_RANDOM_SECRET_KEY",
        "algorithm": "HS256",
        "expiration_days": 7
    },
    "server": {
        "host": "127.0.0.1",
        "port": 7070
    },
    "repository": {
        "location": "/data/repo-vidio",
        "location_windows": "d:/data/repo-vidio"
    },
    "pipelines": {
        "retinal": { "models_dir": "/data/models/retinal" },
        "histology": { "tile_size": 256, "tissue_threshold": 0.5 },
        "radiology": { "default_voxel_spacing": [1.0, 1.0, 1.0] },
        "spatial": { "min_genes_per_spot": 200, "n_top_genes": 2000 }
    }
}
```

---

## Database Setup

```bash
# Create database and user
psql -U postgres -c "CREATE USER admin_vidio WITH PASSWORD 'YOUR_PASSWORD';"
psql -U postgres -c "CREATE DATABASE vidio OWNER admin_vidio;"

# Run schema creation
psql -U admin_vidio -d vidio -f SQL/create_db_vidio.sql

# Create first admin user
python VidioTool.py -t add_user --username admin --password admin123 --name Administrator --role admin
```

### Database Schema Overview

| Table | Description |
|-------|-------------|
| `user` / `user_token` | Platform users and JWT tokens |
| `patient` | Patient demographics |
| `study` | Imaging studies (one per modality session) |
| `series` | Image series within a study |
| `image` | Individual image files (all formats) |
| `annotation` | Manual/auto annotations (BBOX, POLYGON, MASK) |
| `finding` | Detected anomalies with severity and confidence |
| `tcga_case` / `tcga_slide` | TCGA integration metadata |
| `spatial_experiment` | Spatial transcriptomics metadata |
| `ml_model` | ML model registry |
| `process` | Async job tracking |
| `log` | Audit trail |
| `file` | Uploaded file tracking |

---

## Running the Server

### Development

```bash
python app.py
# Server starts on http://127.0.0.1:7070
```

### Production (gunicorn)

```bash
gunicorn app:application -w 4 -b 0.0.0.0:7070 --timeout 300
```

---

## CLI Tool

```bash
# User management
python VidioTool.py -t add_user --username analyst1 --password pass --name "Dr. Smith" --role analyst

# Patient registration
python VidioTool.py -t add_patient --name "Jane Doe" --mrn MRN001 --sex F --dob 1965-03-15

# Create study
python VidioTool.py -t add_study --patient-id <UUID> --modality RETINAL --description "Fundus screening"

# Import DICOM directory
python VidioTool.py -t import_dicom --study-id <UUID> --directory /data/dicom/patient001/

# Run analysis (synchronous)
python VidioTool.py -t run_analysis --study-id <UUID> --modality retinal

# View results
python VidioTool.py -t list_findings --study-id <UUID>
```

---

## API Reference

### Authentication

```bash
# Login
curl -X POST http://localhost:7070/auth \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
# Returns: {"token": "<JWT>"}

# Use token in subsequent requests
curl -H "Authorization: Bearer <JWT>" http://localhost:7070/patients
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth` | Login, get JWT token |
| `GET/POST` | `/patients` | List/create patients |
| `GET/POST/DELETE` | `/patients/{id}` | CRUD single patient |
| `GET/POST` | `/studies` | List/create studies |
| `GET/POST/DELETE` | `/studies/{id}` | CRUD single study |
| `GET` | `/studies/{id}/series` | Series within a study |
| `GET` | `/studies/{id}/findings` | Findings for a study |
| `GET/POST` | `/series` | List/create series |
| `GET` | `/series/{id}/images` | Images within a series |
| `GET/POST` | `/images` | List/create image records |
| `GET/POST` | `/annotations` | List/create annotations |
| `GET` | `/findings` | List findings (filterable) |
| `POST` | `/findings/{id}` | Review a finding |
| `POST` | `/uploads` | Upload files (multipart) |
| `GET` | `/processes` | List async jobs |
| `GET` | `/processes/{id}` | Job status and progress |

### Analysis Endpoints

```bash
# Trigger retinal analysis
curl -X POST http://localhost:7070/analysis/retinal \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"id_study": "<UUID>", "parameters": {}}'
# Returns: {"process_id": "<UUID>", "status": "PENDING"}

# Same pattern for:
# POST /analysis/histology
# POST /analysis/radiology
# POST /analysis/spatial
```

---

## Processing Pipelines

### Pipeline Execution Flow

```
API Request → Create Process Record → Launch Background Thread
           → Pipeline.run() → For each image:
                Load → Preprocess → Segment → Stats → Detect
           → Save findings to DB → Update process status
```

### Severity Classification

Findings are classified into four severity levels based on two metrics:

| Severity | Min Gradient | Min Confidence |
|----------|-------------|----------------|
| CRITICAL | 80.0 | 0.85 |
| HIGH | 50.0 | 0.70 |
| MEDIUM | 25.0 | 0.50 |
| LOW | 0.0 | 0.00 |

- **Gradient**: Difference between max and mean intensity in the ROI (higher = stronger anomaly)
- **Confidence**: Algorithm confidence score in [0, 1]

### Finding Structure

Each finding stored in the database contains:

```json
{
    "finding_type": "LESION",
    "disease_category": "DIABETIC_RETINOPATHY",
    "severity": "HIGH",
    "confidence": 0.82,
    "info": {
        "statistics": {"mean": 145, "std": 23, "gradient": 62, "z_score": 3.1},
        "geometric_properties": {"area": 850, "solidity": 0.91, "circularity": 0.78},
        "location": {"x": 234, "y": 156, "w": 45, "h": 42}
    }
}
```

---

## Core Engine

The `core/` package provides shared building blocks used by all pipelines:

| Module | Functions | Purpose |
|--------|-----------|---------|
| `image_io` | `read_image()`, `read_dicom()`, `read_wsi()`, `read_nifti()`, `read_h5ad()` | Universal format reader |
| `intensity` | `normalize_minmax()`, `clahe_enhance()`, `hu_windowing()`, `stain_normalize()` | Intensity normalisation |
| `filtering` | `bilateral_filter()`, `gaussian_filter()`, `median_filter()`, `unsharp_mask()` | Noise reduction |
| `edges` | `laplacian_edges()`, `canny_edges()`, `sobel_edges()` | Edge detection with stats |
| `morphology` | `morph_open()`, `morph_close()`, `fill_holes()`, `remove_small_objects()` | Binary mask cleanup |
| `contours` | `find_contours()`, `contour_stats()`, `draw_contours()` | Contour geometry analysis |
| `clustering` | `adaptive_kmeans()`, `kmeans_segment()`, `hdbscan_cluster()` | Feature-space clustering |
| `statistics` | `roi_stats()`, `region_stats()`, `compare_regions()` | ROI descriptors (9 metrics) |
| `classification` | `classify_severity()`, `build_finding()` | Severity scoring |
| `shapes` | `detect_shape()`, `shape_features()`, `is_circular()` | Shape classification |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run core tests only
pytest tests/test_core/ -v

# Run pipeline tests only
pytest tests/test_pipelines/ -v

# Run with coverage
pytest tests/ --cov=core --cov=pipelines -v
```

### Test Categories

- **Core unit tests** (`test_core/`): Test each image processing function with synthetic data. No database or external files required.
- **Pipeline integration tests** (`test_pipelines/`): Test pipeline stages with synthetic images. Spatial tests require scanpy.
- **API tests** (`test_api/`): REST endpoint tests (requires running database).

---

## Dependencies

### Core (required)

| Package | Purpose |
|---------|---------|
| falcon 3.1 | WSGI web framework |
| SQLAlchemy 2.0 | ORM and database access |
| psycopg2-binary | PostgreSQL adapter |
| opencv-python | Image processing |
| numpy, scipy | Numerical computation |
| scikit-learn | Clustering (silhouette, Mean Shift) |
| PyJWT, bcrypt | Authentication |

### Medical Imaging (optional)

| Package | Purpose |
|---------|---------|
| pydicom | DICOM file reading |
| SimpleITK | DICOM series, registration, resampling |
| nibabel | NIfTI file reading |
| openslide-python | Whole-slide image reading |
| scanpy, squidpy, anndata | Spatial transcriptomics |

### Deep Learning (optional)

| Package | Purpose |
|---------|---------|
| torch, torchvision | Neural network inference |
| monai | Medical imaging DL models (3D U-Net) |
| hdbscan | Density-based clustering |

All optional dependencies are wrapped in `try/except ImportError` blocks.
Missing dependencies produce informative error messages only when the
specific functionality is requested.

---

## License

Proprietary. All rights reserved.
