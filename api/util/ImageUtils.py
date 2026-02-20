"""
Image Utility Functions
=======================

Helper functions for managing medical image files on disk. Provides
path generation, file-size retrieval, and format detection for the
image formats supported by the VIDIO platform (DICOM, SVS, TIFF, PNG,
JPG, NIfTI, H5AD, etc.).
"""

import os
import uuid


def generate_storage_path(base_dir, study_id, filename):
    """Build a unique file-system path for storing an uploaded image.

    The generated path is::

        <base_dir>/<study_id>/<uuid_hex><original_extension>

    The study sub-directory is created automatically if it does not exist.

    Parameters
    ----------
    base_dir : str
        Root directory for the image repository.
    study_id : str or int
        Identifier of the study; used as the sub-directory name.
    filename : str
        Original filename of the uploaded image (used only to extract the
        file extension).

    Returns
    -------
    str
        Absolute path where the image should be written.
    """
    ext = os.path.splitext(filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    study_dir = os.path.join(base_dir, str(study_id))
    os.makedirs(study_dir, exist_ok=True)
    return os.path.join(study_dir, unique_name)


def get_file_size(filepath):
    """Return the size of a file in bytes.

    Parameters
    ----------
    filepath : str
        Path to the file on disk.

    Returns
    -------
    int or None
        File size in bytes, or ``None`` if the file does not exist or
        cannot be accessed (``OSError``).
    """
    try:
        return os.path.getsize(filepath)
    except OSError:
        return None


def detect_format(filename):
    """Detect the medical-image format from a filename's extension.

    The mapping is purely extension-based (no magic-byte inspection).

    Parameters
    ----------
    filename : str
        The filename (or full path) whose extension will be inspected.

    Returns
    -------
    str
        One of the recognised format labels: ``'DICOM'``, ``'SVS'``,
        ``'TIFF'``, ``'PNG'``, ``'JPG'``, ``'NIfTI'``, ``'H5AD'``, or
        ``'UNKNOWN'`` if the extension is not recognised.

    Notes
    -----
    The check is case-insensitive. Compound extensions like ``.nii.gz``
    are *not* fully handled by ``os.path.splitext`` (which would return
    ``'.gz'``), so ``.nii.gz`` files may be reported as ``'UNKNOWN'``
    unless the filename ends with the bare ``.nii`` extension.
    """
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
