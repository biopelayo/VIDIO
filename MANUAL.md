# VIDIO User Manual

**Vision-Integrated Diagnostic Imaging Orchestrator**

*Version 1.0 -- Comprehensive User and Reference Guide*

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [First-Time Setup](#2-first-time-setup)
3. [GUI User Manual](#3-gui-user-manual)
4. [CLI Reference](#4-cli-reference)
5. [API Reference](#5-api-reference)
6. [Use Cases](#6-use-cases)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Getting Started

### 1.1 Platform Overview

VIDIO (Vision-Integrated Diagnostic Imaging Orchestrator) is a web-based biomedical image analysis platform designed for clinical and research environments. It provides a unified interface for managing medical images across four imaging modalities -- retinal, histology, radiology, and spatial transcriptomics -- with automated analysis pipelines that detect anomalies and generate diagnostic findings.

The platform follows a DICOM-compatible data hierarchy:

```
Patient --> Study --> Series --> Image --> Finding
```

**Architecture summary:**

- **Backend:** Python Falcon WSGI framework serving a REST API on port 7070.
- **Frontend:** Vanilla JavaScript single-page application (SPA) with hash-based routing, served as static files from the same server.
- **Database:** PostgreSQL with UUID primary keys and JSONB metadata columns.
- **Authentication:** JWT tokens (HS256) with bcrypt password hashing.
- **Processing:** Asynchronous analysis pipelines running in daemon threads.

### 1.2 Prerequisites

Before installing VIDIO, ensure the following software is available on the target system:

| Requirement         | Minimum Version | Notes                                      |
|---------------------|-----------------|---------------------------------------------|
| Python              | 3.9+            | 3.10 or 3.11 recommended                   |
| PostgreSQL          | 12+             | 14 or 15 recommended                       |
| pip                 | 21+             | For installing Python packages              |
| Git                 | 2.30+           | For cloning the repository                  |
| CUDA Toolkit        | 11.8+           | Optional; required for GPU-accelerated ML   |

**Operating system:** VIDIO runs on Windows 10/11 and Linux. The configuration file includes separate repository paths for Windows (`location_windows`) and Linux (`location`).

### 1.3 Python Dependencies

All Python packages are declared in `requirements.txt`. The major dependency groups are:

- **Web framework:** falcon, falcon-cors, PyJWT, bcrypt
- **Database:** SQLAlchemy 2.x, psycopg2-binary
- **Image processing (core):** opencv-python, numpy, scipy, scikit-image, Pillow, matplotlib
- **Image processing (medical):** pydicom, SimpleITK, nibabel, openslide-python
- **Machine learning:** torch, torchvision, monai, scikit-learn, hdbscan
- **Spatial transcriptomics:** scanpy, squidpy, anndata
- **Utilities:** python-dateutil, pytz, xlsxwriter, requests, tqdm

### 1.4 Installation Steps

**Step 1 -- Clone the repository:**

```bash
git clone <repository-url> vidio
cd vidio
```

**Step 2 -- Create and activate a virtual environment:**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Step 3 -- Install Python dependencies:**

```bash
pip install -r requirements.txt
```

If GPU support is needed for PyTorch, install the CUDA-enabled version first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Step 4 -- Install OpenSlide (required for histology WSI support):**

- **Linux (Debian/Ubuntu):** `sudo apt-get install openslide-tools`
- **Windows:** Download the OpenSlide binaries from the OpenSlide website and add the `bin` directory to your system PATH.

### 1.5 Database Setup

**Step 1 -- Create the PostgreSQL database and user:**

```sql
-- Connect to PostgreSQL as a superuser (e.g., postgres)
CREATE USER admin_vidio WITH PASSWORD 'your_secure_password';
CREATE DATABASE vidio OWNER admin_vidio;
GRANT ALL PRIVILEGES ON DATABASE vidio TO admin_vidio;
```

**Step 2 -- Run the schema creation script:**

```bash
psql -U admin_vidio -d vidio -f SQL/create_db_vidio.sql
```

This script creates all required tables, including:

- `user`, `user_token` -- Authentication and user management
- `patient`, `study`, `series`, `image` -- DICOM-compatible clinical data hierarchy
- `annotation`, `finding` -- Analysis results and manual annotations
- `tcga_case`, `tcga_slide` -- TCGA integration tables
- `spatial_experiment` -- Spatial transcriptomics experiment metadata
- `ml_model` -- Machine learning model registry
- `process` -- Asynchronous job tracking
- `file` -- Upload tracking
- `log` -- Audit log

The script also creates indexes on frequently queried columns for performance.

### 1.6 Configuration

All configuration is stored in `cfg.json` in the application root directory. Edit this file before first launch.

```json
{
    "db": {
        "host": "localhost",
        "port": 5432,
        "db": "vidio",
        "user": "admin_vidio",
        "password": "CHANGE_ME"
    },
    "repository": {
        "location": "/data/repo-vidio",
        "location_windows": "d:/data/repo-vidio"
    },
    "server": {
        "host": "127.0.0.1",
        "port": 7070
    },
    "auth": {
        "secret_key": "CHANGE_ME_TO_RANDOM_SECRET",
        "algorithm": "HS256",
        "expiration_days": 7
    },
    "pipelines": {
        "retinal": {
            "models_dir": "/data/models/retinal",
            "default_model": "efficientnet_b4_retinal_v1"
        },
        "histology": {
            "models_dir": "/data/models/histology",
            "tile_size": 256,
            "tissue_threshold": 0.5
        },
        "radiology": {
            "models_dir": "/data/models/radiology",
            "default_voxel_spacing": [1.0, 1.0, 1.0]
        },
        "spatial": {
            "min_genes_per_spot": 200,
            "min_spots_per_gene": 10,
            "n_top_genes": 2000
        }
    },
    "processing": {
        "max_concurrent_jobs": 4,
        "gpu_device": "cuda:0"
    }
}
```

**Critical settings to change before first launch:**

| Key                     | Action Required                                                           |
|-------------------------|---------------------------------------------------------------------------|
| `db.password`           | Set to the actual PostgreSQL password for `admin_vidio`.                  |
| `auth.secret_key`       | Replace with a long random string (64+ characters). This signs all JWTs. |
| `repository.location`   | Set to the directory where uploaded files and images will be stored.      |
| `repository.location_windows` | Set to the Windows path equivalent if running on Windows.           |

Create the repository directory before starting the server:

```bash
# Linux
mkdir -p /data/repo-vidio/uploads

# Windows
mkdir d:\data\repo-vidio\uploads
```

### 1.7 Starting the Server

**Development mode** (single-threaded, auto-reload not included):

```bash
python app.py
```

The server starts on the host and port specified in `cfg.json` (default: `127.0.0.1:7070`).

**Production mode** (multi-worker, Linux only):

```bash
gunicorn app:application -w 4 -b 0.0.0.0:7070
```

Open a browser and navigate to `http://localhost:7070` to access the web interface.

---

## 2. First-Time Setup

### 2.1 Creating the First Admin User via CLI

Before you can log into the web interface, at least one user account must exist. The very first user must be created using the CLI tool, since the web interface requires authentication to access the user management page.

```bash
python VidioTool.py -t add_user \
    --username admin \
    --password your_secure_password \
    --name "System Administrator" \
    --surname "Admin" \
    --role admin
```

Expected output:

```
2026-02-20 10:00:00 [INFO] VidioTool: User created: a1b2c3d4-... (admin)
```

Available roles:

| Role       | Description                                                     |
|------------|-----------------------------------------------------------------|
| `admin`    | Full access to all features including user management.          |
| `analyst`  | Can manage patients, studies, images, run analyses, and review findings. |
| `clinician`| Can view data and review/approve findings.                      |
| `viewer`   | Read-only access to patients, studies, images, and findings.    |

### 2.2 Logging into the Web GUI

1. Open a browser and navigate to `http://localhost:7070`.
2. You will be automatically redirected to the login page (`#/login`).
3. Enter the username and password created in section 2.1.
4. Click **Sign In** (or press Enter while in the password field).
5. On successful authentication, you are redirected to the Dashboard.

The JWT token is stored in the browser's `localStorage` under the key `vidio_token` and is valid for the number of days specified in `cfg.json` (`auth.expiration_days`, default 7). After expiration, you will be redirected to the login page.

---

## 3. GUI User Manual

The VIDIO web interface is a dark-themed single-page application with a persistent sidebar for navigation. This section walks through every screen and interaction in detail.

### 3.1 Login Screen

**URL:** `#/login`

The login screen is the entry point for all users. It displays the VIDIO logo, the full platform name ("Vision-Integrated Diagnostic Imaging Orchestrator"), and a login form with two fields:

- **Username** -- Your platform username. This field is auto-focused on page load.
- **Password** -- Your password. Pressing Enter in this field triggers the login action.

**How to log in:**

1. Type your username in the Username field.
2. Press Tab or click the Password field.
3. Type your password.
4. Click the **Sign In** button or press Enter.

If authentication fails, an error message appears above the form in red text (e.g., "Invalid credentials"). The button reverts from "Signing in..." back to "Sign In" so you can retry.

**Logout:** From any page, click the power icon button at the bottom of the sidebar, next to your user profile. This clears the token and redirects to the login page.

### 3.2 Dashboard

**URL:** `#/dashboard`
**Sidebar:** Overview > Dashboard

The dashboard provides an at-a-glance summary of all data in the system. It loads automatically after login.

**Summary cards (top row):**

| Card       | Description                                    | Visual Indicator                              |
|------------|------------------------------------------------|-----------------------------------------------|
| Patients   | Total number of patient records in the system. | Blue icon.                                    |
| Studies    | Total number of imaging studies.               | Green icon.                                   |
| Findings   | Total findings across all studies.             | Orange icon. Red badge if critical findings exist. |
| Processes  | Total analysis processes.                      | Blue icon if processes are running; green if idle. Blue badge shows running count. |

**Bottom panels (two-column layout):**

- **Recent Findings** (left) -- Shows the 8 most recent findings, each with a colored severity indicator bar, the finding label/type, a severity badge, and a date. Click any finding to navigate to its detail view.
- **Active Processes** (right) -- Shows the 5 most recent processes with animated spinner for running jobs, a checkmark for completed, or an X for failed. Displays the process type, status badge, and start time.

### 3.3 Managing Patients

**URL:** `#/patients`
**Sidebar:** Data > Patients

The Patients page displays all registered patient records in a table with columns: ID (truncated UUID), Name, DOB, Gender, and Created date. Each row has View and Delete action buttons.

#### Adding a Patient

1. Click the **+ Add Patient** button in the page header.
2. A modal dialog opens with the following fields:
   - **Patient Name** (required) -- The patient's full name.
   - **Date of Birth** -- A date picker for selecting the DOB.
   - **Gender** -- A dropdown with options: Male, Female, Other.
   - **Notes** -- A free-text area for additional patient notes.
3. Fill in the required fields. At minimum, the Name field must be provided.
4. Click **Create** to save the patient record.
5. A green toast notification confirms "Patient created" and the table refreshes automatically.

To cancel, click **Cancel** or click outside the modal overlay.

#### Viewing a Patient

Click the **View** button on any patient row. This navigates to the Patient Detail page (`#/patients/<uuid>`) which displays:

- Patient name, full UUID, gender, date of birth, creation date, and any notes.

#### Deleting a Patient

1. Click the **Del** button on the patient row.
2. A browser confirmation dialog appears: "Delete this patient?"
3. Click OK to confirm deletion, or Cancel to abort.
4. On successful deletion, a toast notification confirms "Patient deleted."

**Warning:** Deleting a patient cascades to all associated studies, series, images, and findings due to the database's `ON DELETE CASCADE` constraints.

### 3.4 Managing Studies

**URL:** `#/studies`
**Sidebar:** Data > Studies

The Studies page lists all imaging studies in a table with columns: ID, Description, Modality, Patient (truncated UUID), and Created date.

#### Creating a Study

1. Click the **+ Add Study** button.
2. A modal opens with:
   - **Patient** (required) -- A dropdown populated with all registered patients. Select the patient this study belongs to.
   - **Description** -- A free-text description (e.g., "Fundus Exam 2026", "Brain MRI Follow-up").
   - **Modality** -- A dropdown with four options:
     - Retinal (for fundus photography, OCT, slit-lamp images)
     - Histology (for H&E-stained slides, whole-slide images)
     - Radiology (for CT/MRI volumes, DICOM series)
     - Spatial Transcriptomics (for 10x Visium, MERFISH, Slide-seq data)
3. Click **Create** to save.

#### Viewing a Study

Click **View** on any study row to open the Study Detail page (`#/studies/<uuid>`). This page shows:

- Study header with description and a modality badge.
- Full study UUID and linked patient UUID.
- **Tabbed content:**
  - **Series tab** -- Table of all imaging series in this study with ID, Description, and Created date. Click View to drill into a series.
  - **Findings tab** -- List of all findings for this study, each with a severity color bar, finding label/type, severity badge, and confidence percentage.

#### Deleting a Study

Click **Del** on the study row and confirm the browser dialog. This cascades to all child series, images, and findings.

### 3.5 Browsing Images

**URL:** `#/images`
**Sidebar:** Data > Images

The Images page lists all image records across all series in a flat table: ID, Filename, Format, Series (truncated UUID), and Created date. Click **View** to open the Image Detail page.

#### Image Detail View

**URL:** `#/images/<uuid>`

The Image Detail page shows:

- The image filename as the card title.
- An embedded image viewer that attempts to render the image from the storage path. If the file is unavailable on disk, a "Image not available on disk" placeholder appears.
- Metadata: full UUID, file format, parent series UUID, and file size in KB.

#### Series Detail View

**URL:** `#/series/<uuid>`

Accessible from the Study Detail page, the Series Detail view shows:

- Series description, full UUID, and parent study UUID.
- A table of all images belonging to this series, with links to their individual detail views.

### 3.6 Uploading Images

**URL:** `#/upload`
**Sidebar:** Data > Upload

The Upload page provides a drag-and-drop file upload interface for adding medical images to the repository.

**Supported file formats:** PNG, JPG/JPEG, DICOM (.dcm), NIfTI (.nii, .nii.gz), Aperio SVS (.svs), TIFF/TIF, H5AD (.h5ad).

#### Upload Workflow

1. Navigate to the Upload page from the sidebar.
2. You will see a large drop zone with the text "Drop files here or click to browse".
3. **To add files:**
   - **Drag and drop:** Drag one or more files from your file manager onto the drop zone. The zone highlights with a visual border when files are dragged over it.
   - **Click to browse:** Click anywhere on the drop zone to open a file picker dialog. Select one or more files.
4. Selected files appear in a list below the drop zone, showing filename and size in KB. Each file has a **Remove** button to exclude it before upload.
5. Click the **Upload Files** button to begin the upload.
6. The button text changes to "Uploading..." and is disabled during transfer.
7. Files are streamed to the server in 8 KB chunks (memory-efficient for large files).
8. On success, a green toast shows "Uploaded N file(s)" and the file list clears.
9. On failure, a red toast shows the error message.

**Note:** Uploaded files are stored in the `uploads/` subdirectory of the configured repository path. Each file is saved with a UUID-based filename to prevent collisions, while the original filename is preserved in the database record.

### 3.7 Running Analysis

**URL:** `#/analysis`
**Sidebar:** Analysis > Run Analysis

The Analysis page is the launch pad for automated image analysis pipelines.

#### Launching an Analysis

1. Navigate to the Run Analysis page.
2. Fill in the analysis form:
   - **Study** (required) -- Select the study to analyse from the dropdown. The dropdown lists all studies with their description and modality.
   - **Pipeline / Modality** -- Select the analysis pipeline:
     - **Retinal (Fundus / OCT)** -- For retinal fundus photography, OCT scans, and slit-lamp images.
     - **Histology (H&E / WSI)** -- For histopathology whole-slide images.
     - **Radiology (CT / MRI)** -- For CT and MRI volumes in DICOM or NIfTI format.
     - **Spatial Transcriptomics** -- For spatial gene expression data in H5AD format.
   - **Parameters (optional JSON)** -- Advanced users can provide a JSON object with pipeline-specific parameters. Examples:
     ```json
     {"tile_size": 512, "tissue_threshold": 0.3}
     ```
     ```json
     {"window_center": 40, "window_width": 80}
     ```
     ```json
     {"min_genes_per_spot": 300, "n_top_genes": 3000}
     ```
3. Click **Run Analysis**.
4. The button changes to "Launching..." while the request is sent.
5. On success, a toast shows "Analysis queued! Process: <short-id>" and you are automatically redirected to the Processes page to monitor progress.
6. If the JSON parameters are malformed, a red toast shows "Invalid JSON in parameters."

**What happens internally:**

1. The frontend sends `POST /analysis/<modality>` with `{"id_study": "<uuid>", "parameters": {...}}`.
2. The API validates the study exists and creates a `PENDING` process record.
3. The `ProcessManagement` module spawns a daemon thread that dynamically imports the appropriate pipeline class.
4. The pipeline iterates over all selected images in the study, running a 5-stage processing chain on each:
   - **Load** -- Read the image from disk (format auto-detected).
   - **Preprocess** -- Normalize, enhance contrast, filter noise.
   - **Segment** -- Identify regions of interest.
   - **Statistics** -- Compute quantitative descriptors per region.
   - **Detect Anomalies** -- Compare against baselines and classify severity.
5. Findings are persisted to the database as they are detected (survives partial failures).
6. Progress is tracked as a percentage (0-100%) in the process record.

### 3.8 Monitoring Processes

**URL:** `#/processes`
**Sidebar:** Analysis > Processes

The Processes page shows all analysis jobs in a table with columns: ID, Type, Status, Study, Started, and User.

**Process statuses:**

| Status      | Badge Color | Description                                              |
|-------------|-------------|----------------------------------------------------------|
| `PENDING`   | Grey        | Job created but not yet started.                         |
| `RUNNING`   | Blue        | Pipeline is actively processing images. Progress (0-100%) is tracked. |
| `COMPLETED` | Green       | All images processed successfully. Findings are available. |
| `FAILED`    | Red         | Pipeline encountered an error. Error message is stored in the process record. |

**Refreshing:** Click the **Refresh** button above the table to reload process statuses. This is useful for monitoring a running analysis -- the page does not auto-refresh.

### 3.9 Reviewing Findings

**URL:** `#/findings`
**Sidebar:** Analysis > Findings

The Findings page displays all diagnostic findings across all studies in a table: ID, Type/Label, Severity (color-coded badge), Confidence (percentage), Study, and Date.

#### Severity Levels and Color Coding

| Severity   | Badge Color     | Meaning                                                 |
|------------|-----------------|----------------------------------------------------------|
| `CRITICAL` | Red             | Requires immediate clinical attention. High-confidence anomaly with significant deviation from baseline. |
| `HIGH`     | Orange          | Notable abnormality. Should be reviewed promptly.        |
| `MEDIUM`   | Yellow          | Moderate deviation. May warrant follow-up or additional imaging. |
| `LOW`      | Green/Grey      | Minor or borderline finding. Often within normal variation. |

#### Finding Detail View

**URL:** `#/findings/<uuid>`

Click **Detail** on any finding row to open the Finding Detail page, which displays:

- **Header:** Finding label/type and severity badge.
- **Metadata:**
  - Full finding UUID.
  - Confidence score (0-100%).
  - Parent study UUID.
  - Source image UUID.
- **Description:** Human-readable description of the finding (if available).
- **Details:** JSON-formatted detailed data from the analysis pipeline (e.g., statistical measures, z-scores, region coordinates). Displayed in a formatted code block.
- **Location:** Spatial coordinates or bounding box of the finding within the image (if applicable).

#### Reviewing Findings via API

Findings can be marked as reviewed by a clinician through the API endpoint `POST /findings/<uuid>` with a JSON body containing review data:

```json
{
    "reviewed": true,
    "review_notes": "Confirmed diabetic retinopathy. Recommend laser photocoagulation."
}
```

### 3.10 Managing Users (Admin Only)

**URL:** `#/users`
**Sidebar:** Admin > Users

The Users page lists all platform users in a table: ID, Username, Name, Role (blue badge), and Created date.

#### Adding a New User

1. Click the **+ Add User** button.
2. A modal opens with fields:
   - **Username** (required) -- The login username. Must be unique.
   - **Full Name** (required) -- The user's display name.
   - **Password** (required) -- Initial password.
   - **Role** -- A dropdown: User, Admin, or Viewer.
3. Click **Create** to save.

The password is automatically hashed with bcrypt before storage. The `password_hash` field is never returned in API responses.

#### Deleting a User

Click **Del** on the user row and confirm the browser dialog.

---

## 4. CLI Reference

The VIDIO CLI tool (`VidioTool.py`) provides administrative operations that can be performed without running the web server. It reads `cfg.json` from its own directory and initializes the database connection automatically.

**General syntax:**

```bash
python VidioTool.py -t <task> [options]
```

### 4.1 Available Tasks

#### `add_user` -- Create a New User

Create a new platform user with a bcrypt-hashed password.

| Parameter      | Required | Description                                    |
|----------------|----------|------------------------------------------------|
| `--username`   | Yes      | Login username (must be unique).               |
| `--password`   | Yes      | Plaintext password (will be hashed).           |
| `--name`       | Yes      | User's first/display name.                     |
| `--surname`    | No       | User's surname. Defaults to empty string.      |
| `--role`       | No       | One of: `admin`, `analyst`, `clinician`, `viewer`. Defaults to `analyst`. |

**Example:**

```bash
python VidioTool.py -t add_user \
    --username dr_garcia \
    --password s3cureP@ss \
    --name "Maria" \
    --surname "Garcia" \
    --role clinician
```

#### `add_patient` -- Register a Patient

Register a new patient record in the database.

| Parameter  | Required | Description                              |
|------------|----------|------------------------------------------|
| `--name`   | Yes      | Patient's full name.                     |
| `--mrn`    | No       | Medical record number (unique if given). |
| `--sex`    | No       | Sex: `M`, `F`, or `O`.                  |
| `--dob`    | No       | Date of birth in `YYYY-MM-DD` format.   |

**Example:**

```bash
python VidioTool.py -t add_patient \
    --name "Jane Doe" \
    --mrn MRN-2026-001 \
    --sex F \
    --dob 1965-03-15
```

#### `add_study` -- Create an Imaging Study

Create a new imaging study for an existing patient.

| Parameter       | Required | Description                                         |
|-----------------|----------|-----------------------------------------------------|
| `--patient-id`  | Yes      | UUID of the patient this study belongs to.          |
| `--modality`    | Yes      | One of: `RETINAL`, `HISTOLOGY`, `RADIOLOGY`, `SPATIAL`. |
| `--description` | No       | Free-text study description.                        |
| `--institution` | No       | Name of the institution where the study was performed. |

**Example:**

```bash
python VidioTool.py -t add_study \
    --patient-id a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
    --modality RETINAL \
    --description "Fundus screening exam 2026" \
    --institution "City Eye Hospital"
```

#### `run_analysis` -- Trigger an Analysis Pipeline

Launch a processing pipeline on a study. This command is **synchronous** -- it blocks until the pipeline completes.

| Parameter      | Required | Description                                |
|----------------|----------|--------------------------------------------|
| `--study-id`   | Yes      | UUID of the study to analyse.              |
| `--modality`   | Yes      | Pipeline: `retinal`, `histology`, `radiology`, or `spatial`. |
| `--parameters` | No       | JSON string with pipeline-specific parameters. |

**Examples:**

```bash
# Run retinal analysis with default parameters
python VidioTool.py -t run_analysis \
    --study-id b2c3d4e5-f6a7-8901-bcde-f12345678901 \
    --modality retinal

# Run histology analysis with custom tile size
python VidioTool.py -t run_analysis \
    --study-id b2c3d4e5-f6a7-8901-bcde-f12345678901 \
    --modality histology \
    --parameters '{"tile_size": 512, "tissue_threshold": 0.3}'

# Run radiology analysis with CT windowing
python VidioTool.py -t run_analysis \
    --study-id b2c3d4e5-f6a7-8901-bcde-f12345678901 \
    --modality radiology \
    --parameters '{"window_center": 40, "window_width": 80}'

# Run spatial transcriptomics with custom gene filtering
python VidioTool.py -t run_analysis \
    --study-id b2c3d4e5-f6a7-8901-bcde-f12345678901 \
    --modality spatial \
    --parameters '{"min_genes_per_spot": 300, "n_top_genes": 3000}'
```

#### `list_findings` -- Display Findings for a Study

Print all findings for a study with severity, type, and confidence score.

| Parameter    | Required | Description                     |
|--------------|----------|---------------------------------|
| `--study-id` | Yes      | UUID of the study to query.     |

**Example:**

```bash
python VidioTool.py -t list_findings \
    --study-id b2c3d4e5-f6a7-8901-bcde-f12345678901
```

**Sample output:**

```
2026-02-20 14:30:00 [INFO] VidioTool: Found 5 findings for study b2c3d4e5-...
2026-02-20 14:30:00 [INFO] VidioTool:   [CRITICAL] LESION (confidence=0.92)
2026-02-20 14:30:00 [INFO] VidioTool:   [HIGH] VESSEL_ANOMALY (confidence=0.87)
2026-02-20 14:30:00 [INFO] VidioTool:   [MEDIUM] REGION_ANOMALY (confidence=0.73)
2026-02-20 14:30:00 [INFO] VidioTool:   [LOW] REGION_ANOMALY (confidence=0.55)
2026-02-20 14:30:00 [INFO] VidioTool:   [LOW] REGION_ANOMALY (confidence=0.41)
```

#### `import_dicom` -- Import DICOM Files

Scan a directory for DICOM files, create a new series under the specified study, and register each file as an image record. Non-DICOM files are silently skipped with a warning.

| Parameter     | Required | Description                                     |
|---------------|----------|-------------------------------------------------|
| `--study-id`  | Yes      | UUID of the study to import into.               |
| `--directory` | Yes      | Path to the directory containing DICOM files.   |

**Example:**

```bash
python VidioTool.py -t import_dicom \
    --study-id b2c3d4e5-f6a7-8901-bcde-f12345678901 \
    --directory /data/dicom/patient001/CT_HEAD/
```

**What it does:**

1. Validates the directory exists.
2. Creates a new series record under the specified study with description "Imported from \<dirname\>" and modality_subtype "DICOM".
3. Iterates over all files in the directory (sorted alphabetically).
4. For each file, attempts to read it as DICOM using `pydicom`.
5. Extracts pixel spacing, dimensions, and DICOM metadata.
6. Creates an image record in the database with the storage path, format, file size, and metadata.
7. Logs the count of successfully imported images.

---

## 5. API Reference

All API endpoints require JWT authentication via the `Authorization: Bearer <token>` header, except `POST /auth` and static file routes. The API accepts and returns JSON (`application/json`) unless otherwise noted.

### 5.1 Authentication

| Method | Endpoint   | Description               | Auth Required |
|--------|------------|---------------------------|---------------|
| POST   | `/auth`    | Login; returns JWT token  | No            |

**Request body:**
```json
{"username": "admin", "password": "s3cret"}
```

**Response (200):**
```json
{
    "token": "<jwt_string>",
    "user": {"id": "<uuid>", "username": "admin", "name": "Admin", "role": "admin"}
}
```

**Error responses:** 400 (missing fields), 401 (invalid credentials), 500 (server error).

### 5.2 Patients

| Method | Endpoint              | Description                  | Response |
|--------|-----------------------|------------------------------|----------|
| GET    | `/patients`           | List/filter all patients     | 200      |
| POST   | `/patients`           | Create or update a patient   | 200      |
| GET    | `/patients/{id}`      | Get a single patient         | 200/404  |
| POST   | `/patients/{id}`      | Update a specific patient    | 200      |
| DELETE | `/patients/{id}`      | Delete a patient             | 200      |

**Filtering:** `GET /patients?filter={"name":"John"}` -- Pass a JSON-encoded filter as a query parameter.

**Create (POST without `id`):**
```json
{"name": "Jane Doe", "date_of_birth": "1965-03-15", "sex": "F"}
```

**Update (POST with `id`):**
```json
{"id": "<uuid>", "name": "Jane Smith"}
```

### 5.3 Studies

| Method | Endpoint                      | Description                  | Response |
|--------|-------------------------------|------------------------------|----------|
| GET    | `/studies`                    | List/filter all studies      | 200      |
| POST   | `/studies`                    | Create or update a study     | 200      |
| GET    | `/studies/{id}`               | Get a single study           | 200/404  |
| POST   | `/studies/{id}`               | Update a specific study      | 200      |
| DELETE | `/studies/{id}`               | Delete a study               | 200      |
| GET    | `/studies/{id}/series`        | List series for a study      | 200      |
| GET    | `/studies/{id}/findings`      | List findings for a study    | 200      |

**Create:**
```json
{"id_patient": "<uuid>", "modality": "RETINAL", "description": "Annual screening"}
```

### 5.4 Series

| Method | Endpoint                    | Description                  | Response |
|--------|-----------------------------|------------------------------|----------|
| GET    | `/series`                   | List/filter all series       | 200      |
| POST   | `/series`                   | Create or update a series    | 200      |
| GET    | `/series/{id}`              | Get a single series          | 200/404  |
| POST   | `/series/{id}`              | Update a specific series     | 200      |
| DELETE | `/series/{id}`              | Delete a series              | 200      |
| GET    | `/series/{id}/images`       | List images in a series      | 200      |

### 5.5 Images

| Method | Endpoint            | Description                    | Response |
|--------|---------------------|--------------------------------|----------|
| GET    | `/images`           | List/filter all images         | 200      |
| POST   | `/images`           | Create or update image record  | 200      |
| GET    | `/images/{id}`      | Get a single image record      | 200/404  |
| DELETE | `/images/{id}`      | Delete an image record         | 200      |

### 5.6 Annotations

| Method | Endpoint                | Description                     | Response |
|--------|-------------------------|---------------------------------|----------|
| GET    | `/annotations`          | List/filter annotations         | 200      |
| POST   | `/annotations`          | Create or update an annotation  | 200      |
| GET    | `/annotations/{id}`     | Get a single annotation         | 200/404  |
| POST   | `/annotations/{id}`     | Update a specific annotation    | 200      |
| DELETE | `/annotations/{id}`     | Delete an annotation            | 200      |

### 5.7 Findings

| Method | Endpoint             | Description                    | Response |
|--------|----------------------|--------------------------------|----------|
| GET    | `/findings`          | List/filter all findings       | 200      |
| GET    | `/findings/{id}`     | Get a single finding           | 200/404  |
| POST   | `/findings/{id}`     | Review a finding (approve/add notes) | 200 |

**Review body:**
```json
{"reviewed": true, "review_notes": "Confirmed by Dr. Smith."}
```

**Note:** Findings are created exclusively by analysis pipelines, not through direct POST requests.

### 5.8 Processes

| Method | Endpoint             | Description                   | Response |
|--------|----------------------|-------------------------------|----------|
| GET    | `/processes`         | List/filter all processes     | 200      |
| GET    | `/processes/{id}`    | Get a single process          | 200/404  |

Processes are read-only through the API. They are created internally when analysis is launched.

### 5.9 Users

| Method | Endpoint          | Description                   | Response |
|--------|-------------------|-------------------------------|----------|
| GET    | `/users`          | List/filter all users         | 200      |
| POST   | `/users`          | Create or update a user       | 200      |
| GET    | `/users/{id}`     | Get a single user             | 200/404  |
| POST   | `/users/{id}`     | Update a specific user        | 200      |
| DELETE | `/users/{id}`     | Delete a user                 | 200      |

**Note:** The `password_hash` field is stripped from all responses. When creating a user, supply `password` (plaintext); it is hashed server-side.

### 5.10 Uploads

| Method | Endpoint    | Description                         | Response |
|--------|-------------|-------------------------------------|----------|
| POST   | `/uploads`  | Upload files (multipart/form-data)  | 200      |

**Request:** `multipart/form-data` with one or more file parts named `file`.

**Response:**
```json
{"uploaded": [{"id": "<uuid>", "name": "scan001.dcm", "storage_path": "...", "file_size_bytes": 1234567}]}
```

### 5.11 ML Models

| Method | Endpoint          | Description                      | Response |
|--------|-------------------|----------------------------------|----------|
| GET    | `/models`         | List registered ML models        | 200      |
| POST   | `/models`         | Register or update a model       | 200      |
| GET    | `/models/{id}`    | Get a single model record        | 200/404  |

### 5.12 Analysis

| Method | Endpoint               | Description                            | Response |
|--------|------------------------|----------------------------------------|----------|
| POST   | `/analysis/retinal`    | Launch retinal analysis pipeline       | 202      |
| POST   | `/analysis/histology`  | Launch histology analysis pipeline     | 202      |
| POST   | `/analysis/radiology`  | Launch radiology analysis pipeline     | 202      |
| POST   | `/analysis/spatial`    | Launch spatial transcriptomics pipeline| 202      |

**Request body (all endpoints):**
```json
{"id_study": "<uuid>", "parameters": {}}
```

**Response (202 Accepted):**
```json
{"process_id": "<uuid>", "status": "PENDING", "message": "retinal analysis queued"}
```

---

## 6. Use Cases

### UC1: Diabetic Retinopathy Screening Program

**Scenario:** A regional ophthalmology clinic runs a diabetic retinopathy screening program. Trained technicians capture fundus photographs from diabetic patients, and VIDIO provides automated pre-screening to prioritize patients who need ophthalmologist review.

**User Workflow:**

1. **Register patients.** The clinic administrator navigates to **Patients** and clicks **+ Add Patient** for each new screening participant. They enter the patient name, date of birth, gender, and a note indicating "Diabetes Type 2, enrolled in DR screening."

2. **Create studies.** For each patient session, the administrator goes to **Studies** and clicks **+ Add Study**. They select the patient from the dropdown, enter a description such as "DR Screening Q1 2026", and select **Retinal** as the modality.

3. **Upload fundus images.** The technician navigates to **Upload** and drags the fundus photographs (typically 2-4 JPEG or PNG images per eye per patient: macula-centered, disc-centered, and peripheral fields) into the drop zone. They click **Upload Files** to transfer all images to the server.

4. **Associate images with the study.** The uploaded images are linked to the appropriate series within the study via the API or by organizing uploads per study.

5. **Run retinal analysis.** The technician navigates to **Run Analysis**, selects the patient's study from the dropdown, confirms the pipeline is set to **Retinal (Fundus / OCT)**, and clicks **Run Analysis**.

6. **Monitor progress.** The system redirects to the **Processes** page. The technician clicks **Refresh** periodically. The status progresses from PENDING to RUNNING (with percentage updates as each image completes) to COMPLETED.

7. **Review findings.** The technician navigates to **Findings** and filters or sorts by severity. Critical and High findings are flagged for immediate ophthalmologist review.

**What the Pipeline Does Internally:**

- **Load:** Reads each fundus image (JPEG/PNG) into a NumPy array.
- **Preprocess:** Extracts the green channel (maximizes vessel contrast), applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) to correct for uneven illumination across the fundus, then applies bilateral filtering to smooth noise while preserving vessel and lesion edges.
- **Segment:** Uses Laplacian edge detection to find boundaries, followed by morphological opening/closing to clean up noise and remove small artifacts. Finds contours of candidate regions.
- **Statistics:** Computes per-region intensity descriptors (mean, standard deviation, gradient magnitude, skewness, kurtosis) and geometric features (area, solidity, elongation, convexity) using ROI-masked measurements.
- **Detect Anomalies:** Compares each region's statistics against the whole-image baseline using z-scores. Classifies regions by geometric shape into finding types: LESION (compact, high-solidity regions -- potential exudates or haemorrhages), VESSEL_ANOMALY (elongated, low-solidity regions -- potential tortuous or dilated vessels), or REGION_ANOMALY (other deviations). Assigns severity based on z-score magnitude and confidence.

**Expected Findings:**

- **CRITICAL:** Large haemorrhages (high intensity deviation, large area, high confidence >0.90).
- **HIGH:** Hard exudates (bright lesions with high contrast against the retinal background), microaneurysm clusters.
- **MEDIUM:** Soft exudates (cotton-wool spots), moderate vessel tortuosity.
- **LOW:** Borderline intensity variations, small isolated candidate regions with low confidence.

---

### UC2: Digital Pathology Tumor Classification

**Scenario:** A university hospital pathology department digitizes H&E-stained biopsy slides using a whole-slide scanner. VIDIO analyses the gigapixel whole-slide images (WSI) to identify regions of abnormal tissue morphology, assisting pathologists in locating areas that warrant closer examination under virtual microscopy.

**User Workflow:**

1. **Register the patient.** Navigate to **Patients** and add the biopsy patient with their MRN and demographics.

2. **Create a histology study.** In **Studies**, click **+ Add Study**, select the patient, enter a description like "Breast Biopsy - Right Lobe", and choose **Histology** as the modality.

3. **Upload WSI files.** Navigate to **Upload** and drag the SVS or TIFF files from the slide scanner output. These files can be very large (500 MB to 5 GB each). The streaming upload mechanism handles large files without excessive memory consumption.

4. **Run histology analysis.** Go to **Run Analysis**, select the study, choose **Histology (H&E / WSI)**, and optionally specify custom parameters:
   ```json
   {"tile_size": 512, "tissue_threshold": 0.4}
   ```
   Click **Run Analysis**.

5. **Monitor processing.** On the **Processes** page, the analysis may take considerable time for large WSIs. The progress percentage updates as tiles are processed.

6. **Review findings.** Navigate to **Findings** to see flagged tissue regions. Each finding indicates the tile location and the nature of the anomaly (e.g., unusually dense staining, atypical nuclear morphology indicators).

**What the Pipeline Does Internally:**

- **Load:** Reads the WSI file. For SVS/TIFF files, uses OpenSlide to read at the appropriate magnification level. For in-memory images, loads the full array.
- **Preprocess:** Applies Macenko stain normalization to standardize H&E color appearance across slides from different scanners and staining batches. Converts to grayscale for intensity analysis.
- **Segment (Tiling):** Divides the image into non-overlapping square tiles (default 256x256 pixels). For each tile, applies a tissue detection heuristic: tiles where fewer than `tissue_threshold` fraction of pixels have grayscale intensity below 220 are discarded as background/glass. Only tissue-containing tiles proceed.
- **Statistics:** Computes per-tile intensity statistics (mean, standard deviation, gradient magnitude, skewness, kurtosis) over the tissue pixels within each tile.
- **Detect Anomalies:** Calculates z-scores for each tile's statistics relative to the global population of all tiles in the image. Tiles with extreme z-scores (indicating unusual staining intensity, texture, or morphological patterns) are flagged as findings. Severity is classified based on the magnitude of deviation.

**Expected Findings:**

- **CRITICAL:** Tiles with markedly elevated nuclear density or highly atypical staining patterns (potential high-grade tumor regions).
- **HIGH:** Tiles with significant deviation in texture metrics (potential tumor-stroma boundary, necrotic regions).
- **MEDIUM:** Moderately abnormal staining patterns (potential low-grade dysplasia, inflammation).
- **LOW:** Borderline tiles with minor statistical deviations from normal tissue.

---

### UC3: Brain MRI Atrophy Analysis

**Scenario:** A neurology research group studies brain atrophy patterns in Alzheimer's disease patients. They collect longitudinal T1-weighted MRI scans and use VIDIO to automatically detect and quantify regions of abnormal tissue intensity that may indicate atrophic changes, cortical thinning, or white matter lesions.

**User Workflow:**

1. **Register the patient.** Create a patient record with the research subject ID as the MRN.

2. **Create a radiology study.** Add a study with modality **Radiology** and description "Brain MRI - Baseline T1w".

3. **Import DICOM data.** Use the CLI tool to import the DICOM directory directly:
   ```bash
   python VidioTool.py -t import_dicom \
       --study-id <study-uuid> \
       --directory /data/dicom/subject042/T1w/
   ```
   This creates a series and registers all DICOM slices as image records with extracted metadata (pixel spacing, slice thickness, patient position).

4. **Run radiology analysis.** Either via the GUI (navigate to **Run Analysis**, select the study, choose **Radiology (CT / MRI)**) or via CLI:
   ```bash
   python VidioTool.py -t run_analysis \
       --study-id <study-uuid> \
       --modality radiology
   ```

5. **Review findings.** Check the Findings page for identified regions of abnormal intensity. Each finding includes spatial coordinates within the volume and statistical measures.

**What the Pipeline Does Internally:**

- **Load:** Detects the file format (DICOM, NIfTI, or raster). For DICOM, uses pydicom to read pixel data and extract metadata (modality, window center/width, pixel spacing). For NIfTI, uses nibabel. For 3D volumes, the full volumetric array is loaded.
- **Preprocess:** Branches by imaging modality:
  - **CT:** Applies Hounsfield Unit windowing using the window center and width from the DICOM header (or from parameters), then normalizes to [0, 1] float32.
  - **MRI:** Applies min-max normalization to float32, as MRI intensities are arbitrary and scanner-dependent.
- **Segment:** For 2D images, applies Laplacian edge detection with morphological cleanup. For 3D volumes, performs slice-by-slice 2D segmentation, detecting region boundaries on each axial slice independently. Finds contours on each processed slice.
- **Statistics:** Computes per-region intensity statistics (mean, standard deviation, gradient magnitude) for each detected ROI.
- **Detect Anomalies:** Evaluates each region's gradient and intensity deviation against the global image baseline. Assigns confidence and severity scores. Regions with high gradient combined with significant intensity deviation are flagged as potential lesions.

**Expected Findings:**

- **CRITICAL:** Large regions with very high intensity deviation and sharp borders (potential space-occupying lesions, large white matter hyperintensities).
- **HIGH:** Moderate-sized regions with significant contrast differences (potential periventricular white matter lesions, cortical abnormalities).
- **MEDIUM:** Smaller regions with moderate deviation (potential early-stage atrophic changes, small lacunar infarcts).
- **LOW:** Minor intensity variations within normal anatomical range.

---

### UC4: Spatial Transcriptomics Tissue Mapping

**Scenario:** A cancer biology lab performs 10x Visium spatial transcriptomics on tumor tissue sections to map gene expression heterogeneity. VIDIO processes the H5AD data files to identify spatially distinct expression clusters and flag regions with anomalous gene expression patterns that may correspond to tumor subclones, immune infiltrates, or treatment-resistant niches.

**User Workflow:**

1. **Register the sample as a patient.** Create a patient record using the tissue sample identifier as the name and the lab sample ID as the MRN.

2. **Create a spatial study.** Add a study with modality **Spatial Transcriptomics** and description "10x Visium - Tumor Section A".

3. **Upload H5AD file.** Navigate to **Upload** and drag the `.h5ad` file (AnnData format containing the spatial gene expression matrix, spot coordinates, and tissue image) into the upload zone.

4. **Run spatial analysis.** Go to **Run Analysis**, select the study, choose **Spatial Transcriptomics**, and optionally adjust parameters:
   ```json
   {"min_genes_per_spot": 250, "min_spots_per_gene": 15, "n_top_genes": 2500}
   ```
   Click **Run Analysis**.

5. **Review findings.** The Findings page shows clusters with anomalous expression profiles. Each finding includes the cluster identifier, the genes driving the anomaly, and spatial location information.

**What the Pipeline Does Internally:**

- **Load:** Reads the H5AD file using the `anndata` library, loading the sparse gene expression matrix, spot-level metadata, and spatial coordinates.
- **Preprocess (QC + Normalization):**
  - Filters out low-quality spots with fewer than `min_genes_per_spot` detected genes (likely empty droplets).
  - Filters out genes detected in fewer than `min_spots_per_gene` spots (too sparse for analysis).
  - Normalizes total counts per spot to a common target sum.
  - Log-transforms the data (`log1p`).
  - Identifies the top `n_top_genes` highly variable genes (HVGs) for dimensionality reduction.
  - Scales gene expression values.
- **Segment (Spatial Clustering):**
  - Runs PCA on the HVG matrix.
  - Computes a spatial neighbor graph incorporating both expression similarity and physical proximity of spots using squidpy.
  - Performs Leiden community detection to identify spatially coherent expression clusters.
- **Statistics:** Computes per-cluster mean expression levels for each gene. Calculates cluster-level summary statistics.
- **Detect Anomalies:** Compares each cluster's mean expression profile against the global (all-spots) mean. Genes whose cluster-level expression deviates significantly (by z-score) are identified as marker genes. Clusters with extreme overall deviation scores are flagged as findings, with severity based on the magnitude of expression anomaly and the number of significantly deregulated genes.

**Expected Findings:**

- **CRITICAL:** Clusters with extreme expression deviation across many genes (potential tumor subclones with distinct transcriptional programs, immune-excluded zones).
- **HIGH:** Clusters with strong deviation in specific gene sets (potential immune-infiltrate hotspots, hypoxic regions).
- **MEDIUM:** Clusters with moderate expression differences (transition zones between tumor and stroma, leading-edge invasion fronts).
- **LOW:** Clusters with minor expression variations (normal tissue heterogeneity, technical variation between capture areas).

---

### UC5: Multi-Modal Clinical Trial Image Analysis

**Scenario:** A pharmaceutical company is conducting a Phase II clinical trial for a new anti-VEGF drug targeting diabetic macular edema. The study protocol requires standardized analysis of retinal fundus images, macular OCT scans, and (in a subset of patients) fluorescein angiography images at baseline, 3 months, and 6 months. VIDIO serves as the central image management and analysis platform for the trial.

**User Workflow:**

1. **Bulk patient registration.** The clinical data manager uses the CLI tool in a batch script to register all enrolled subjects:
   ```bash
   for line in subjects.csv; do
       python VidioTool.py -t add_patient \
           --name "$NAME" --mrn "$SUBJECT_ID" --sex "$SEX" --dob "$DOB"
   done
   ```

2. **Create timepoint studies.** For each patient at each visit, a study is created with a descriptive label:
   ```bash
   python VidioTool.py -t add_study \
       --patient-id <uuid> \
       --modality RETINAL \
       --description "Visit 1 - Baseline" \
       --institution "Clinical Trial Site 001"
   ```

3. **Image upload by site coordinators.** Site coordinators log into the VIDIO web interface, navigate to **Upload**, and drag-drop the fundus and OCT images acquired during the visit. Each upload batch is associated with the correct study through the data management workflow.

4. **Batch analysis via CLI.** The central analysis team runs retinal analysis on all new studies:
   ```bash
   python VidioTool.py -t run_analysis \
       --study-id <study-uuid> \
       --modality retinal
   ```
   This is often scripted across all pending studies.

5. **Finding review by the reading center.** Certified graders at the reading center log into VIDIO, navigate to **Findings**, and review each finding. They use the Finding Detail view to examine severity, confidence, spatial location, and the underlying statistical data. Findings are marked as reviewed with clinical notes:
   ```json
   {
       "reviewed": true,
       "review_notes": "Grade 2 hard exudates confirmed, consistent with moderate NPDR. No change from baseline."
   }
   ```

6. **Longitudinal comparison.** Graders navigate to individual patients' studies across timepoints (Baseline, Month 3, Month 6) and compare the number, severity, and location of findings to assess treatment response.

7. **Results export.** The findings data, accessible via the API (`GET /studies/<id>/findings`), can be extracted programmatically for statistical analysis in the trial database.

**What the Pipeline Does Internally:**

The retinal pipeline processes each study's images through the same 5-stage workflow described in UC1. For a clinical trial context, the standardization provided by the pipeline is particularly valuable:

- **Preprocessing consistency:** CLAHE and bilateral filtering parameters are fixed, ensuring that images from different sites and cameras are normalized to a consistent baseline before analysis.
- **Automated lesion detection:** The segmentation and anomaly detection stages apply identical criteria to every image, eliminating inter-grader variability in the initial screening pass.
- **Quantitative metrics:** Per-finding confidence scores and per-region statistics provide continuous outcome variables that supplement categorical grading scales.

**Expected Findings:**

Across the trial timepoints, the platform generates a longitudinal profile:

- **Baseline findings** establish the starting severity distribution (mix of CRITICAL/HIGH/MEDIUM/LOW findings depending on disease stage).
- **Follow-up findings** should show changes in the severity distribution -- in treatment responders, fewer CRITICAL/HIGH findings and lower mean confidence scores indicate regression of lesions.
- **Non-responders** may show stable or increasing numbers of high-severity findings, flagging patients for protocol-specified rescue therapy.

---

## 7. Troubleshooting

### 7.1 Installation and Startup Issues

**Problem:** `ModuleNotFoundError: No module named 'falcon'`

**Cause:** Python dependencies are not installed or the virtual environment is not activated.

**Fix:**
```bash
# Activate the virtual environment first
source venv/bin/activate  # Linux
venv\Scripts\activate      # Windows

# Then install dependencies
pip install -r requirements.txt
```

---

**Problem:** `psycopg2.OperationalError: could not connect to server`

**Cause:** PostgreSQL is not running, or the connection parameters in `cfg.json` are incorrect.

**Fix:**
1. Verify PostgreSQL is running: `pg_isready -h localhost -p 5432`
2. Check `cfg.json` database settings: host, port, database name, user, and password.
3. Ensure the database exists: `psql -U postgres -c "\l"` and look for `vidio`.
4. Ensure the user has permissions: `psql -U postgres -c "\du"` and verify `admin_vidio` exists.

---

**Problem:** `relation "user" does not exist` when starting the server

**Cause:** The database schema has not been created.

**Fix:**
```bash
psql -U admin_vidio -d vidio -f SQL/create_db_vidio.sql
```

---

**Problem:** `Address already in use` when starting the server on port 7070

**Cause:** Another process is using port 7070.

**Fix:**
- Change the port in `cfg.json` under `server.port`, or
- Find and stop the conflicting process:
  ```bash
  # Linux
  lsof -i :7070
  kill <PID>

  # Windows
  netstat -ano | findstr :7070
  taskkill /PID <PID> /F
  ```

### 7.2 Authentication Issues

**Problem:** Login fails with "Invalid credentials"

**Fix:**
1. Verify the username exists. Check the database directly:
   ```sql
   SELECT username, role FROM "user" WHERE deleted = FALSE;
   ```
2. If no users exist, create one via CLI (see Section 2.1).
3. If the password was forgotten, reset it via CLI:
   ```bash
   python VidioTool.py -t add_user --username admin --password newpassword --name Admin --role admin
   ```
   Note: This creates a new user if the username does not exist. To reset an existing user's password, use the API or update the database directly.

---

**Problem:** "Session expired" or frequent logouts

**Cause:** JWT token has expired.

**Fix:** Increase `auth.expiration_days` in `cfg.json` (default is 7 days). After changing, restart the server. Users must log in again to receive a new token.

---

**Problem:** "Missing authorization token" on API requests

**Cause:** The `Authorization` header is missing or malformed.

**Fix:** Ensure the header format is `Authorization: Bearer <token>` or `Authorization: JWT <token>`. The scheme keyword is case-insensitive.

### 7.3 Upload Issues

**Problem:** File upload fails with a server error

**Cause:** The repository directory does not exist or has insufficient permissions.

**Fix:**
1. Check that the directory in `cfg.json` under `repository.location` (or `location_windows`) exists.
2. Create the `uploads` subdirectory:
   ```bash
   mkdir -p /data/repo-vidio/uploads
   ```
3. Ensure the application user has write permissions to this directory.

---

**Problem:** Uploaded files are not visible in the Images page

**Cause:** The Upload endpoint creates `file` records (raw uploads), not `image` records linked to a series. Images must be associated with a series to appear in the Images page.

**Fix:** After uploading, create image records that reference the uploaded file paths and link them to the appropriate series, either through the API (`POST /images`) or by importing via the CLI.

### 7.4 Analysis Issues

**Problem:** Analysis status shows `FAILED`

**Fix:** Check the process record for the error message:
```bash
curl -H "Authorization: Bearer <token>" http://localhost:7070/processes/<process-id>
```
Look at the `error_message` field. Common causes:
- **"Pipeline not implemented"**: The pipeline module could not be imported, usually because required dependencies (e.g., `torch`, `scanpy`, `openslide`) are not installed.
- **"No images found for study"**: The study has no series with images. Upload or import images first.
- **File not found errors**: Image `storage_path` values point to non-existent files.

---

**Problem:** Analysis runs but produces no findings

**Cause:** This can occur if:
- All images have `selected = False` (they are skipped by the pipeline).
- The preprocessing or segmentation stages find no regions of interest (e.g., blank images, tissue threshold too high for histology).
- The anomaly detection z-score thresholds are not exceeded (all regions are within normal range).

**Fix:**
- Verify images exist and are valid: check the file paths in the database.
- For histology, try lowering `tissue_threshold` (e.g., `0.3` instead of `0.5`).
- Check the `vidio.log` file for per-image processing messages.

---

**Problem:** `ImportError` for pipeline dependencies (e.g., `torch`, `scanpy`, `openslide`)

**Cause:** Pipeline dependencies are imported lazily (only when a specific modality is requested). If the required packages are not installed, the import fails at analysis time.

**Fix:** Install the required dependencies for the modality you need:
- **Retinal/Histology/Radiology:** `pip install torch torchvision monai`
- **Histology (WSI):** Install OpenSlide system library and `pip install openslide-python`
- **Spatial:** `pip install scanpy squidpy anndata`
- **Radiology (DICOM/NIfTI):** `pip install pydicom SimpleITK nibabel`

### 7.5 Frontend Issues

**Problem:** Blank page or JavaScript errors in the browser console

**Cause:** Static files are not being served correctly.

**Fix:**
1. Ensure the `frontend/` directory exists in the application root with `index.html`, `css/`, and `js/` subdirectories.
2. Check the browser console (F12 > Console) for specific error messages.
3. Verify the server is running and accessible at the expected URL.
4. Clear the browser cache and reload (Ctrl+Shift+R).

---

**Problem:** The sidebar shows "U" instead of the user's name

**Cause:** The user profile was not stored in `localStorage` during login.

**Fix:** Log out (click the power button in the sidebar) and log in again. The user profile is stored in `localStorage` under `vidio_user` during the login flow.

---

**Problem:** Toast notifications appear but the page data does not update

**Fix:** Some pages require manual refresh. For the Processes page, click the **Refresh** button. For other pages, navigate away and back using the sidebar links.

### 7.6 Database Issues

**Problem:** `UndefinedColumn` or `UndefinedTable` errors

**Cause:** The database schema is outdated or was not fully created.

**Fix:** Re-run the schema creation script. It uses `CREATE TABLE` (not `CREATE TABLE IF NOT EXISTS`), so drop existing tables first if needed, or run against a fresh database.

---

**Problem:** Slow queries on large datasets

**Fix:** The schema includes indexes on commonly queried columns. If you experience slow performance:
1. Run `ANALYZE` in PostgreSQL to update statistics: `psql -U admin_vidio -d vidio -c "ANALYZE;"`
2. Check that indexes exist: `psql -U admin_vidio -d vidio -c "\di"`
3. For very large datasets, consider adding pagination to API queries using filters.

### 7.7 Log Files

VIDIO writes log output to both the console and a `vidio.log` file in the application directory. The log format is:

```
2026-02-20 10:00:00,000 [INFO] module_name: Log message here
```

For pipeline-specific debugging, look for log entries from pipeline class names (e.g., `RetinalPipeline`, `HistologyPipeline`). Each image processing step is logged individually, making it possible to identify exactly where a failure occurred.

---

*End of VIDIO User Manual*
