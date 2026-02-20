#!/usr/bin/env python3
"""
VIDIO CLI Tool — Administrative Command-Line Interface
=======================================================

A command-line utility for performing administrative operations on the
VIDIO platform without running the web server.  Useful for:

- **Initial setup** : Creating the first admin user.
- **Data ingestion** : Importing DICOM directories into the database.
- **Batch analysis** : Triggering pipeline analysis from scripts or cron.
- **Quick inspection** : Listing findings for a study.

Usage
-----
::

    python VidioTool.py -t <task> [options]

Available Tasks
---------------
``add_user``
    Create a new platform user with bcrypt-hashed password.
    Required: ``--username``, ``--password``, ``--name``.
    Optional: ``--surname``, ``--role`` (default: ``analyst``).

``add_patient``
    Register a new patient record.
    Required: ``--name``.
    Optional: ``--mrn``, ``--sex``, ``--dob``.

``add_study``
    Create a new imaging study for a patient.
    Required: ``--patient-id``, ``--modality``.
    Optional: ``--description``, ``--institution``.

``run_analysis``
    Trigger a processing pipeline on a study (synchronous — waits for
    completion).
    Required: ``--study-id``, ``--modality``.
    Optional: ``--parameters`` (JSON string).

``list_findings``
    Display all findings for a study with severity and confidence.
    Required: ``--study-id``.

``import_dicom``
    Scan a directory for DICOM files, create a series, and register
    each file as an image in the database.
    Required: ``--study-id``, ``--directory``.

Examples
--------
::

    # Create admin user
    python VidioTool.py -t add_user --username admin --password s3cret --name Admin --role admin

    # Register a patient
    python VidioTool.py -t add_patient --name "Jane Doe" --mrn MRN001 --sex F --dob 1965-03-15

    # Import DICOM directory
    python VidioTool.py -t import_dicom --study-id <UUID> --directory /data/dicom/patient001/

    # Run retinal analysis
    python VidioTool.py -t run_analysis --study-id <UUID> --modality retinal

    # Check results
    python VidioTool.py -t list_findings --study-id <UUID>

Configuration
-------------
The tool reads ``cfg.json`` from its own directory and initialises the
database connection automatically.  On Windows, the repository path is
switched to ``location_windows`` from the config.

See Also
--------
:mod:`app` : WSGI web server entry point.
:mod:`ProcessManagement` : Async pipeline launcher.
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
log = logging.getLogger('VidioTool')


def load_config():
    """Load ``cfg.json`` and initialise the database connection.

    Reads the configuration file from the same directory as this script,
    sets the global config singleton (:data:`api.Cfg.gCfg`), and calls
    :func:`~api.db.DB.InitDatabase` to create the SQLAlchemy engine and
    session factory.

    On Windows, the repository storage path is swapped to the
    ``location_windows`` value from the config.

    Returns
    -------
    dict
        The parsed configuration dictionary.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(cur_dir, 'cfg.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    import api.Cfg
    api.Cfg.gCfg = cfg

    if sys.platform == 'win32':
        cfg['repository']['location'] = cfg['repository'].get('location_windows',
                                                               cfg['repository']['location'])

    from api.db.DB import InitDatabase
    InitDatabase(cfg)
    return cfg


def add_user(args):
    """Create a new platform user with bcrypt-hashed password.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: ``username``, ``password``, ``name``.
        Optional: ``surname``, ``role``.
    """
    from api.db.DB import DB
    from api.Authentication import hash_password

    db = DB()
    data = {
        'username': args.username,
        'password_hash': hash_password(args.password),
        'name': args.name,
        'surname': args.surname or '',
        'role': args.role or 'analyst',
    }
    result = db.AddUser(None, data)
    log.info(f'User created: {result["id"]} ({result["username"]})')
    return result


def add_patient(args):
    """Register a new patient record.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: ``name``.
        Optional: ``mrn``, ``sex``, ``dob``.
    """
    from api.db.DB import DB

    db = DB()
    data = {
        'name': args.name,
        'medical_record_number': args.mrn,
        'sex': args.sex,
    }
    if args.dob:
        data['date_of_birth'] = args.dob

    result = db.AddPatient(None, data)
    log.info(f'Patient created: {result["id"]} ({result["name"]})')
    return result


def add_study(args):
    """Create a new imaging study for a patient.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: ``patient_id``, ``modality``.
        Optional: ``description``, ``institution``.
    """
    from api.db.DB import DB

    db = DB()
    data = {
        'id_patient': args.patient_id,
        'modality': args.modality.upper(),
        'description': args.description or '',
        'institution': args.institution or '',
    }
    result = db.AddStudy(None, data)
    log.info(f'Study created: {result["id"]} (modality={result["modality"]})')
    return result


def run_analysis(args):
    """Trigger an analysis pipeline on a study (synchronous).

    Creates a ``process`` record, launches the appropriate modality
    pipeline, and **blocks** until the pipeline thread completes.
    This is the synchronous counterpart to the ``POST /analysis/``
    REST endpoint (which is non-blocking).

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: ``study_id``, ``modality``.
        Optional: ``parameters`` (JSON string).
    """
    from api.db.DB import DB
    from ProcessManagement import launch_analysis

    db = DB()
    process_data = {
        'id_study': args.study_id,
        'type': f'{args.modality.upper()}_ANALYSIS',
        'status': 'PENDING',
        'parameters': json.loads(args.parameters) if args.parameters else {},
    }
    process = db.AddProcess(None, process_data)
    log.info(f'Process created: {process["id"]}')

    thread = launch_analysis(
        process['id'],
        args.modality.lower(),
        args.study_id,
        json.loads(args.parameters) if args.parameters else {},
    )
    log.info('Analysis launched in background. Waiting for completion...')
    thread.join()
    log.info('Analysis complete.')


def list_findings(args):
    """List all findings for a study with severity and confidence.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: ``study_id``.
    """
    from api.db.DB import DB

    db = DB()
    findings = db.GetFindingsForStudy(args.study_id)
    log.info(f'Found {len(findings)} findings for study {args.study_id}')
    for f in findings:
        severity = f.get('severity', 'N/A')
        ftype = f.get('finding_type', 'N/A')
        confidence = f.get('confidence', 0)
        log.info(f'  [{severity}] {ftype} (confidence={confidence:.2f})')
    return findings


def import_dicom(args):
    """Import a directory of DICOM files into the database.

    Scans the given directory for files, reads each as DICOM (extracting
    pixel data and metadata), creates a new ``series`` record under the
    specified study, and registers each DICOM file as an ``image`` row.
    Non-DICOM files are silently skipped with a warning.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: ``study_id``, ``directory``.
    """
    from api.db.DB import DB
    from core.image_io import read_dicom

    db = DB()
    dicom_dir = args.directory

    if not os.path.isdir(dicom_dir):
        log.error(f'Directory not found: {dicom_dir}')
        return

    series_data = {
        'id_study': args.study_id,
        'description': f'Imported from {os.path.basename(dicom_dir)}',
        'modality_subtype': 'DICOM',
    }
    series = db.AddSeries(None, series_data)
    log.info(f'Series created: {series["id"]}')

    count = 0
    for fname in sorted(os.listdir(dicom_dir)):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            _, metadata = read_dicom(fpath)
            image_data = {
                'id_series': series['id'],
                'name': fname,
                'storage_path': fpath,
                'file_format': 'DICOM',
                'file_size_bytes': os.path.getsize(fpath),
                'info': metadata,
            }
            if 'pixel_spacing' in metadata:
                image_data['pixel_spacing'] = metadata['pixel_spacing']
            if 'rows' in metadata and 'columns' in metadata:
                image_data['dimensions'] = {
                    'width': metadata['columns'],
                    'height': metadata['rows'],
                }

            db.AddImage(None, image_data)
            count += 1
        except Exception as ex:
            log.warning(f'Skipping {fname}: {ex}')

    log.info(f'Imported {count} DICOM images into series {series["id"]}')


def main():
    parser = argparse.ArgumentParser(description='VIDIO CLI Tool')
    parser.add_argument('-t', '--task', required=True,
                        choices=['add_user', 'add_patient', 'add_study',
                                 'run_analysis', 'list_findings', 'import_dicom'],
                        help='Task to execute')

    # User args
    parser.add_argument('--username', help='Username for add_user')
    parser.add_argument('--password', help='Password for add_user')
    parser.add_argument('--role', help='Role for add_user (admin/analyst/clinician/viewer)')

    # Patient args
    parser.add_argument('--name', help='Name')
    parser.add_argument('--surname', help='Surname')
    parser.add_argument('--mrn', help='Medical record number')
    parser.add_argument('--sex', help='Sex (M/F/O)')
    parser.add_argument('--dob', help='Date of birth (YYYY-MM-DD)')

    # Study args
    parser.add_argument('--patient-id', dest='patient_id', help='Patient UUID')
    parser.add_argument('--modality', help='Modality (RETINAL/HISTOLOGY/RADIOLOGY/SPATIAL)')
    parser.add_argument('--description', help='Description')
    parser.add_argument('--institution', help='Institution name')

    # Analysis args
    parser.add_argument('--study-id', dest='study_id', help='Study UUID')
    parser.add_argument('--parameters', help='JSON parameters string')

    # Import args
    parser.add_argument('--directory', help='DICOM directory path')

    args = parser.parse_args()

    load_config()

    task_map = {
        'add_user': add_user,
        'add_patient': add_patient,
        'add_study': add_study,
        'run_analysis': run_analysis,
        'list_findings': list_findings,
        'import_dicom': import_dicom,
    }

    task_fn = task_map[args.task]
    task_fn(args)


if __name__ == '__main__':
    main()
