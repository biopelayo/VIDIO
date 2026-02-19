import os
import uuid


def generate_storage_path(base_dir, study_id, filename):
    ext = os.path.splitext(filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    study_dir = os.path.join(base_dir, str(study_id))
    os.makedirs(study_dir, exist_ok=True)
    return os.path.join(study_dir, unique_name)


def get_file_size(filepath):
    try:
        return os.path.getsize(filepath)
    except OSError:
        return None


def detect_format(filename):
    ext = os.path.splitext(filename)[1].lower()
    format_map = {
        '.dcm': 'DICOM',
        '.dicom': 'DICOM',
        '.svs': 'SVS',
        '.tif': 'TIFF',
        '.tiff': 'TIFF',
        '.png': 'PNG',
        '.jpg': 'JPG',
        '.jpeg': 'JPG',
        '.nii': 'NIfTI',
        '.nii.gz': 'NIfTI',
        '.h5ad': 'H5AD',
    }
    return format_map.get(ext, 'UNKNOWN')
