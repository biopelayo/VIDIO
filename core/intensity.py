"""
VIDIO Core - Intensity Normalization & Enhancement
====================================================
Functions for intensity normalization, histogram enhancement, CT windowing,
and H&E stain normalization (Macenko method).
"""

import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def normalize_minmax(img: np.ndarray, new_min: float = 0.0, new_max: float = 1.0) -> np.ndarray:
    """Min-max normalize an image to [new_min, new_max].

    Parameters
    ----------
    img : np.ndarray
        Input image (any dtype).
    new_min : float
        Desired minimum value.
    new_max : float
        Desired maximum value.

    Returns
    -------
    np.ndarray
        Normalized image as float64.
    """
    img = img.astype(np.float64)
    imin = img.min()
    imax = img.max()
    if imax - imin < 1e-12:
        logger.warning("normalize_minmax: constant image, returning new_min fill.")
        return np.full_like(img, new_min, dtype=np.float64)
    normalized = (img - imin) / (imax - imin) * (new_max - new_min) + new_min
    return normalized


def normalize_zscore(img: np.ndarray) -> np.ndarray:
    """Z-score normalize an image to zero mean and unit variance.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized image as float64.
    """
    img = img.astype(np.float64)
    mean = img.mean()
    std = img.std()
    if std < 1e-12:
        logger.warning("normalize_zscore: near-zero std, returning zeros.")
        return np.zeros_like(img, dtype=np.float64)
    return (img - mean) / std


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0,
                  grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale uint8, or colour — converted to LAB).
    clip_limit : float
        CLAHE clip limit.
    grid_size : tuple
        Tile grid size for CLAHE.

    Returns
    -------
    np.ndarray
        Enhanced image (same dtype as input after conversion).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    if img.ndim == 2:
        # Grayscale
        if img.dtype != np.uint8:
            img8 = normalize_minmax(img, 0, 255).astype(np.uint8)
        else:
            img8 = img
        return clahe.apply(img8)

    if img.ndim == 3 and img.shape[2] == 3:
        # Colour: apply CLAHE on the L channel of LAB
        if img.dtype != np.uint8:
            img8 = normalize_minmax(img, 0, 255).astype(np.uint8)
        else:
            img8 = img.copy()
        lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    logger.warning("clahe_enhance: unsupported image shape %s, returning copy.", img.shape)
    return img.copy()


def hu_windowing(img: np.ndarray, window_center: float,
                 window_width: float) -> np.ndarray:
    """Apply Hounsfield Unit windowing to a CT image.

    Parameters
    ----------
    img : np.ndarray
        Input CT image in HU (float or int).
    window_center : float
        Centre of the window (e.g. 40 for soft tissue).
    window_width : float
        Width of the window (e.g. 400 for soft tissue).

    Returns
    -------
    np.ndarray
        Windowed image scaled to [0, 255] as uint8.
    """
    img = img.astype(np.float64)
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    windowed = np.clip(img, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return windowed


def stain_normalize(img: np.ndarray, method: str = "macenko") -> np.ndarray:
    """Normalise H&E stained histopathology image colours.

    Parameters
    ----------
    img : np.ndarray
        Input H&E image in BGR (uint8).
    method : str
        Normalisation method. Currently only ``"macenko"`` is supported.

    Returns
    -------
    np.ndarray
        Colour-normalised image (uint8 BGR).
    """
    if method.lower() != "macenko":
        raise ValueError(f"Unsupported stain normalisation method: {method}")

    return _macenko_normalize(img)


def _macenko_normalize(img: np.ndarray) -> np.ndarray:
    """Macenko stain normalisation using SVD on optical density.

    Reference
    ---------
    Macenko et al., "A method for normalizing histology slides for
    quantitative analysis", ISBI 2009.
    """
    # Reference stain vectors (H&E from Macenko paper)
    ref_stain_matrix = np.array([
        [0.5626, 0.2159],   # R: Hematoxylin, Eosin
        [0.7201, 0.8012],   # G
        [0.4062, 0.5581],   # B
    ])

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    img_rgb = img_rgb.reshape(-1, 3)

    # Optical density
    img_rgb[img_rgb < 1.0] = 1.0  # avoid log(0)
    od = -np.log(img_rgb / 255.0)

    # Threshold: keep only significant OD pixels
    od_thresh = 0.15
    mask = np.all(od > od_thresh, axis=1)
    if mask.sum() < 100:
        logger.warning("stain_normalize: insufficient tissue pixels, returning original.")
        return img.copy()
    od_hat = od[mask]

    # SVD to find principal stain directions
    _, _, Vt = np.linalg.svd(od_hat, full_matrices=False)
    V = Vt[:2, :]  # top 2 singular vectors (2 x 3)

    # Project OD into 2-D plane
    proj = od_hat @ V.T  # (N, 2)

    # Find angle of each pixel in the 2-D plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    # Robust angle extremes (1st and 99th percentile)
    min_phi = np.percentile(phi, 1)
    max_phi = np.percentile(phi, 99)

    v1 = V.T @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = V.T @ np.array([np.cos(max_phi), np.sin(max_phi)])

    # Ensure Hematoxylin is the first vector (darker in the blue channel)
    if v1[0] > v2[0]:
        stain_matrix = np.stack([v1, v2], axis=1)
    else:
        stain_matrix = np.stack([v2, v1], axis=1)

    # Normalise stain vectors
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)

    # Concentrations via least-squares
    concentrations = np.linalg.lstsq(stain_matrix, od.T, rcond=None)[0]  # (2, N)

    # Robust max concentration (99th percentile)
    max_conc = np.percentile(concentrations, 99, axis=1, keepdims=True)
    max_conc[max_conc < 1e-6] = 1.0

    # Reference concentrations
    ref_max_conc = np.array([[1.9705], [1.0308]])

    concentrations = concentrations / max_conc * ref_max_conc

    # Reconstruct with reference stain matrix
    ref_stain_norm = ref_stain_matrix / np.linalg.norm(ref_stain_matrix, axis=0, keepdims=True)
    od_normalized = ref_stain_norm @ concentrations  # (3, N)
    img_normalized = 255.0 * np.exp(-od_normalized)
    img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
    img_normalized = img_normalized.T.reshape(img.shape[0], img.shape[1], 3)

    return cv2.cvtColor(img_normalized, cv2.COLOR_RGB2BGR)


def auto_brightness_contrast(img: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
    """Automatically adjust brightness and contrast by clipping histogram tails.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 grayscale or BGR colour).
    clip_percent : float
        Percentage of pixels to clip on each tail of the histogram.

    Returns
    -------
    np.ndarray
        Adjusted image (uint8).
    """
    if img.dtype != np.uint8:
        img = normalize_minmax(img, 0, 255).astype(np.uint8)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = gray.size
    clip_count = total_pixels * clip_percent / 100.0

    # Find low threshold
    cumsum = np.cumsum(hist)
    low = np.searchsorted(cumsum, clip_count)

    # Find high threshold
    high = 255 - np.searchsorted(cumsum[::-1], clip_count)

    if high <= low:
        logger.warning("auto_brightness_contrast: computed range is degenerate, returning copy.")
        return img.copy()

    # Compute alpha (contrast) and beta (brightness)
    alpha = 255.0 / (high - low)
    beta = -low * alpha

    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted
