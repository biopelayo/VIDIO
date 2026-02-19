"""
VIDIO Core - Universal Image I/O
=================================
Supports reading standard images (PNG, JPG, TIFF), DICOM, DICOM series,
NIfTI, whole-slide images (OpenSlide), and h5ad (scanpy/anndata).
Auto-detection dispatches to the correct reader based on file extension.
"""

import os
import logging
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------
try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False
    logger.debug("pydicom not available; DICOM reading disabled.")

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False
    logger.debug("SimpleITK not available; DICOM series reading disabled.")

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False
    logger.debug("nibabel not available; NIfTI reading disabled.")

try:
    import openslide
    _HAS_OPENSLIDE = True
except ImportError:
    _HAS_OPENSLIDE = False
    logger.debug("openslide not available; WSI reading disabled.")

try:
    import scanpy as sc
    _HAS_SCANPY = True
except ImportError:
    _HAS_SCANPY = False
    logger.debug("scanpy not available; h5ad reading disabled.")

# ---------------------------------------------------------------------------
# Standard image formats
# ---------------------------------------------------------------------------
STANDARD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
WSI_EXTENSIONS = {".svs", ".ndpi", ".tif", ".mrxs", ".vms", ".vmu", ".scn", ".bif"}
H5AD_EXTENSIONS = {".h5ad"}


def read_standard(path: str) -> np.ndarray:
    """Read a standard image file (PNG, JPG, TIFF, etc.) via OpenCV.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image as a NumPy array in BGR colour space (or grayscale).
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback to PIL for formats cv2 cannot handle
            from PIL import Image
            pil_img = Image.open(path)
            img = np.array(pil_img)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        logger.info("Read standard image %s  shape=%s dtype=%s", path, img.shape, img.dtype)
        return img
    except Exception as exc:
        logger.error("Failed to read standard image %s: %s", path, exc)
        raise


def read_dicom(path: str) -> tuple:
    """Read a single DICOM file.

    Parameters
    ----------
    path : str
        Path to the .dcm file.

    Returns
    -------
    tuple[np.ndarray, dict]
        (pixel_array, metadata_dict)
    """
    if not _HAS_PYDICOM:
        raise ImportError("pydicom is required for DICOM reading. Install with: pip install pydicom")
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DICOM file not found: {path}")
    try:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array.astype(np.float64)

        # Apply rescale slope / intercept if present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * float(slope) + float(intercept)

        metadata = {
            "PatientID": getattr(ds, "PatientID", ""),
            "StudyDescription": getattr(ds, "StudyDescription", ""),
            "SeriesDescription": getattr(ds, "SeriesDescription", ""),
            "Modality": getattr(ds, "Modality", ""),
            "Rows": getattr(ds, "Rows", None),
            "Columns": getattr(ds, "Columns", None),
            "PixelSpacing": [float(x) for x in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else None,
            "SliceThickness": float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else None,
            "RescaleSlope": float(slope),
            "RescaleIntercept": float(intercept),
        }
        logger.info("Read DICOM %s  shape=%s  modality=%s", path, pixel_array.shape, metadata["Modality"])
        return pixel_array, metadata
    except Exception as exc:
        logger.error("Failed to read DICOM %s: %s", path, exc)
        raise


def read_dicom_series(directory: str) -> tuple:
    """Read a DICOM series from a directory into a 3-D volume.

    Parameters
    ----------
    directory : str
        Path to directory containing DICOM slices.

    Returns
    -------
    tuple[np.ndarray, dict]
        (3D_numpy_array, metadata_dict)
    """
    if not _HAS_SITK:
        raise ImportError("SimpleITK is required for DICOM series reading. Install with: pip install SimpleITK")
    directory = str(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"DICOM series directory not found: {directory}")
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {directory}")
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

        volume = sitk.GetArrayFromImage(image)  # (slices, rows, cols)

        metadata = {
            "Origin": image.GetOrigin(),
            "Spacing": image.GetSpacing(),
            "Direction": image.GetDirection(),
            "Size": image.GetSize(),
            "NumSlices": volume.shape[0],
        }
        # Try to extract patient-level metadata from the first slice
        if reader.GetMetaDataKeys(0):
            for key in ("0010|0020", "0008|1030", "0008|103e", "0008|0060"):
                if reader.HasMetaDataKey(0, key):
                    metadata[key] = reader.GetMetaData(0, key)

        logger.info("Read DICOM series from %s  volume_shape=%s", directory, volume.shape)
        return volume, metadata
    except Exception as exc:
        logger.error("Failed to read DICOM series from %s: %s", directory, exc)
        raise


def read_nifti(path: str) -> tuple:
    """Read a NIfTI image file.

    Parameters
    ----------
    path : str
        Path to the .nii or .nii.gz file.

    Returns
    -------
    tuple[np.ndarray, dict]
        (numpy_array, header_dict)
    """
    if not _HAS_NIBABEL:
        raise ImportError("nibabel is required for NIfTI reading. Install with: pip install nibabel")
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    try:
        nii = nib.load(path)
        data = np.asarray(nii.dataobj)
        hdr = dict(nii.header)
        # Convert numpy scalar types to Python natives for JSON serialization
        header_dict = {}
        for k, v in hdr.items():
            if isinstance(v, np.ndarray):
                header_dict[k] = v.tolist()
            elif isinstance(v, (np.integer,)):
                header_dict[k] = int(v)
            elif isinstance(v, (np.floating,)):
                header_dict[k] = float(v)
            else:
                header_dict[k] = v
        header_dict["affine"] = nii.affine.tolist()
        logger.info("Read NIfTI %s  shape=%s", path, data.shape)
        return data, header_dict
    except Exception as exc:
        logger.error("Failed to read NIfTI %s: %s", path, exc)
        raise


def read_wsi(path: str, level: int = 0, region: tuple = None) -> np.ndarray:
    """Read a whole-slide image (or a region of it).

    Parameters
    ----------
    path : str
        Path to the WSI file (.svs, .ndpi, etc.).
    level : int
        Pyramid level (0 = highest resolution).
    region : tuple, optional
        (x, y, width, height) to read a specific region. If None, reads the
        full image at the given level.

    Returns
    -------
    np.ndarray
        Image as RGBA NumPy array.
    """
    if not _HAS_OPENSLIDE:
        raise ImportError("openslide-python is required for WSI reading. Install with: pip install openslide-python")
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WSI file not found: {path}")
    try:
        slide = openslide.OpenSlide(path)
        if region is not None:
            x, y, w, h = region
            pil_img = slide.read_region((x, y), level, (w, h))
        else:
            dims = slide.level_dimensions[level]
            pil_img = slide.read_region((0, 0), level, dims)
        img = np.array(pil_img)
        slide.close()
        logger.info("Read WSI %s  level=%d  shape=%s", path, level, img.shape)
        return img
    except Exception as exc:
        logger.error("Failed to read WSI %s: %s", path, exc)
        raise


def read_h5ad(path: str):
    """Read an h5ad file (single-cell data) via scanpy.

    Parameters
    ----------
    path : str
        Path to the .h5ad file.

    Returns
    -------
    anndata.AnnData
        The loaded AnnData object.
    """
    if not _HAS_SCANPY:
        raise ImportError("scanpy is required for h5ad reading. Install with: pip install scanpy")
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"h5ad file not found: {path}")
    try:
        adata = sc.read_h5ad(path)
        logger.info("Read h5ad %s  n_obs=%d  n_vars=%d", path, adata.n_obs, adata.n_vars)
        return adata
    except Exception as exc:
        logger.error("Failed to read h5ad %s: %s", path, exc)
        raise


def read_image(path: str):
    """Auto-detect the image format and dispatch to the appropriate reader.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray or tuple or anndata.AnnData
        The result from the appropriate reader function.
    """
    path_obj = Path(path)
    suffixes = "".join(path_obj.suffixes).lower()  # handles .nii.gz
    ext = path_obj.suffix.lower()

    # NIfTI (.nii or .nii.gz)
    if suffixes.endswith(".nii.gz") or ext == ".nii":
        return read_nifti(path)

    # DICOM
    if ext in DICOM_EXTENSIONS:
        return read_dicom(path)

    # h5ad
    if ext in H5AD_EXTENSIONS:
        return read_h5ad(path)

    # WSI — only if openslide is available and extension matches
    if _HAS_OPENSLIDE and ext in WSI_EXTENSIONS:
        try:
            return read_wsi(path)
        except Exception:
            logger.debug("OpenSlide failed for %s, falling back to standard reader.", path)

    # Standard image
    if ext in STANDARD_EXTENSIONS:
        return read_standard(path)

    # Directory → try DICOM series
    if os.path.isdir(path):
        return read_dicom_series(path)

    # Last resort — try standard reader
    logger.warning("Unknown extension '%s' for %s; attempting standard read.", ext, path)
    return read_standard(path)
