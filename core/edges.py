"""
VIDIO Core - Edge Detection
=============================
Laplacian, Canny, and Sobel edge detectors with statistics.
"""

import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale uint8 if necessary."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def laplacian_edges(img: np.ndarray, ksize: int = 1) -> tuple:
    """Compute Laplacian edge map and associated statistics.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or colour).
    ksize : int
        Aperture size for the Laplacian operator (1, 3, 5, or 7).

    Returns
    -------
    tuple[np.ndarray, dict]
        (edge_map, stats_dict) where stats_dict contains avg, std, median,
        min, and max of the absolute Laplacian values.
    """
    gray = _ensure_grayscale(img)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    abs_laplacian = np.abs(laplacian)

    stats = {
        "avg": float(abs_laplacian.mean()),
        "std": float(abs_laplacian.std()),
        "median": float(np.median(abs_laplacian)),
        "min": float(abs_laplacian.min()),
        "max": float(abs_laplacian.max()),
    }
    logger.debug("laplacian_edges: ksize=%d  avg=%.2f  max=%.2f", ksize, stats["avg"], stats["max"])
    return abs_laplacian, stats


def canny_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Compute Canny edge map.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    low : int
        Lower hysteresis threshold.
    high : int
        Upper hysteresis threshold.

    Returns
    -------
    np.ndarray
        Binary edge map (uint8, values 0 or 255).
    """
    gray = _ensure_grayscale(img)
    edges = cv2.Canny(gray, low, high)
    logger.debug("canny_edges: low=%d high=%d  edge_pixels=%d", low, high, np.count_nonzero(edges))
    return edges


def sobel_edges(img: np.ndarray, ksize: int = 3) -> tuple:
    """Compute Sobel edge magnitude and direction.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    ksize : int
        Sobel kernel size (1, 3, 5, or 7).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (magnitude, direction) where magnitude is the gradient magnitude and
        direction is the gradient angle in radians.
    """
    gray = _ensure_grayscale(img)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(sobel_y, sobel_x)

    logger.debug("sobel_edges: ksize=%d  mag_range=[%.2f, %.2f]", ksize, magnitude.min(), magnitude.max())
    return magnitude, direction
