# VIDIO — Quick Start Guide

## The Fastest Way to See It Work (30 seconds)

No database, no server, no configuration needed:

```bash
# 1. Install dependencies
pip install opencv-python-headless numpy bcrypt pyjwt falcon sqlalchemy

# 2. Run the demo
python demo.py
```

This will:
- Generate 3 synthetic biomedical images (retinal fundus, histology H&E, CT slice)
- Run the full analysis pipeline on each
- Save annotated results + JSON findings to `demo_output/`
- Open the output folder automatically (Windows)

**Output you'll see:**

```
RETINAL FUNDUS ANALYSIS
  Green channel extracted → CLAHE enhanced → Canny edges → K-means (4 clusters)
  Found 18 structures

HISTOLOGY H&E ANALYSIS
  Tissue detected: 64.5% → K-means (3 clusters) → Tumor: 16.2%
  Finding: tumor_region, severity=LOW, confidence=0.66

RADIOLOGY CT ANALYSIS
  Body segmented → Bright lesion candidates: 3603 pixels
  Lesion 0: area=3162px, z=1.41, severity=MEDIUM, shape=circle
  Lesion 2: area=848px, z=1.80, severity=MEDIUM, shape=circle

Total findings: 4 | Time: ~2 seconds
```

**Output files in `demo_output/`:**

| File | What it shows |
|------|---------------|
| `retinal_input.png` | Synthetic fundus photograph |
| `retinal_segmented.png` | Vessels (green), optic disc, lesions with severity legend |
| `histology_input.png` | Synthetic H&E tissue tile |
| `histology_segmented.png` | Tumor region outlined in red with % overlay |
| `radiology_input.png` | Synthetic CT axial slice |
| `radiology_segmented.png` | Lesions labeled L0/L1/L2 with severity colors |
| `*_findings.json` | Machine-readable findings per modality |
| `summary_report.txt` | Human-readable summary |

---

## Run Just One Modality

```bash
python demo.py retinal       # only retinal analysis
python demo.py histology     # only histology analysis
python demo.py radiology     # only radiology analysis
python demo.py --no-open     # don't auto-open output folder
```

---

## Full Platform Setup (with Database + GUI)

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- pip

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create the Database

```sql
-- In psql:
CREATE DATABASE vidio;
CREATE USER admin_vidio WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE vidio TO admin_vidio;

-- Then run the schema:
\c vidio
\i SQL/create_db_vidio.sql
```

### Step 3: Configure

Edit `cfg.json`:

```json
{
  "db": {
    "host": "localhost",
    "port": 5432,
    "db": "vidio",
    "user": "admin_vidio",
    "password": "your_password"
  },
  "repository": {
    "location": "/path/to/image/storage",
    "location_windows": "D:/data/repo-vidio"
  },
  "auth": {
    "secret_key": "change-this-to-a-random-string"
  }
}
```

### Step 4: Create Your First User (CLI)

```bash
python VidioTool.py -t add_user -a username=admin -a password=admin123 -a name=Admin -a role=admin
```

### Step 5: Start the Server

```bash
python app.py
```

Server starts at `http://127.0.0.1:7070`

### Step 6: Open the GUI

Navigate to `http://127.0.0.1:7070` in your browser.

---

## How to Add Images (Step by Step)

### Via the GUI

```
1. LOGIN
   → Enter username + password → Click "Sign In"

2. ADD A PATIENT
   → Sidebar → "Patients" → Click "+ Add Patient"
   → Fill: Name, DOB, Gender → Click "Create"

3. ADD A STUDY
   → Sidebar → "Studies" → Click "+ Add Study"
   → Select the patient you just created
   → Enter description (e.g., "Fundus Exam Feb 2025")
   → Select modality: Retinal / Histology / Radiology / Spatial
   → Click "Create"

4. UPLOAD IMAGES
   → Sidebar → "Upload"
   → Drag & drop your image files onto the upload zone
     (or click to browse)
   → Supported: PNG, JPG, DICOM (.dcm), NIfTI (.nii/.nii.gz),
     SVS, TIFF, H5AD
   → Click "Upload Files"

5. RUN ANALYSIS
   → Sidebar → "Run Analysis"
   → Select your study from the dropdown
   → Select the pipeline (must match the modality)
   → Optionally add parameters as JSON
   → Click "Run Analysis"

6. MONITOR PROGRESS
   → Sidebar → "Processes"
   → Watch status: PENDING → RUNNING → COMPLETED
   → Click "Refresh" to update

7. VIEW FINDINGS
   → Sidebar → "Findings"
   → See all results with severity color coding:
     🔴 CRITICAL  🟠 HIGH  🟡 MEDIUM  🟢 LOW
   → Click "Detail" to see full statistics + JSON data
```

### Via the CLI

```bash
# Add a patient
python VidioTool.py -t add_patient -a name="John Doe" -a dob=1985-03-15 -a sex=M

# Add a study
python VidioTool.py -t add_study -a id_patient=<patient_uuid> -a modality=retinal -a description="Fundus Exam"

# Import DICOM files
python VidioTool.py -t import_dicom -a path="D:/images/patient001/"

# Run analysis
python VidioTool.py -t run_analysis -a id_study=<study_uuid> -a modality=retinal

# List findings
python VidioTool.py -t list_findings
```

### Via the API (curl)

```bash
# Login
TOKEN=$(curl -s -X POST http://127.0.0.1:7070/auth \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json; print(json.load(sys.stdin)['token'])")

# Create patient
curl -X POST http://127.0.0.1:7070/patients \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Jane Doe","date_of_birth":"1990-05-20","sex":"F"}'

# Upload image
curl -X POST http://127.0.0.1:7070/uploads \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@fundus_photo.png"

# Launch analysis
curl -X POST http://127.0.0.1:7070/analysis/retinal \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id_study":"<study_uuid>"}'

# Get findings
curl http://127.0.0.1:7070/findings \
  -H "Authorization: Bearer $TOKEN"
```

---

## Understanding the Results

### Severity Levels

| Level | Criteria | Color | What it means |
|-------|----------|-------|---------------|
| **CRITICAL** | gradient ≥ 80, confidence ≥ 0.85 | 🔴 Red | Urgent attention needed |
| **HIGH** | gradient ≥ 50, confidence ≥ 0.70 | 🟠 Orange | Significant anomaly |
| **MEDIUM** | gradient ≥ 25, confidence ≥ 0.50 | 🟡 Yellow | Moderate deviation |
| **LOW** | below thresholds | 🟢 Green | Minor or normal variant |

### Finding JSON Structure

```json
{
  "finding_type": "lesion_0",
  "severity": "MEDIUM",
  "confidence": 0.67,
  "statistics": {
    "mean_intensity": 196.5,
    "area_px": 848,
    "circularity": 0.85,
    "zscore": 1.80
  },
  "geometric_properties": {
    "centroid": [348.2, 193.5],
    "shape": "circle"
  },
  "created_at": "2025-02-20T..."
}
```

### What Each Pipeline Detects

| Pipeline | Input | Detects |
|----------|-------|---------|
| **Retinal** | Fundus photos, OCT scans | Vessels, optic disc, macula, exudates, hemorrhages, lesions |
| **Histology** | H&E tissue slides, WSI | Tumor regions, stroma, tissue ratios, cell clusters |
| **Radiology** | CT slices, MRI volumes | Body structures, lesions, calcifications, atrophy |
| **Spatial** | 10x Visium H5AD files | Gene clusters, spatially variable genes, tissue domains |

---

## Verify the Install Works

```bash
python verify.py
```

Expected output:
```
  15/15 CHECKS PASSED — VIDIO FULLY VERIFIED
```

This tests all core modules without needing the database.
