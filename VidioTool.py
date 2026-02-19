#!/usr/bin/env python3
"""
VIDIO CLI Tool - Administrative operations for the VIDIO platform.

Usage:
    python VidioTool.py -t <task> [options]

Tasks:
    add_user        Add a new user
    add_patient     Add a new patient
    add_study       Add a new study
    run_analysis    Trigger analysis on a study
    list_findings   List findings for a study
    import_dicom    Import a DICOM directory
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
