"""
VIDIO Core - Morphological Operations
=======================================
Opening, closing, dilation, erosion, hole filling, and small object removal.
"""

import logging

import numpy as np
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)


def _get_kernel(kernel_size: int) -> np.ndarray:
    """Create a square structuring element."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))


def morph_open(img: np.ndarray, kernel_size: int = 3,
               iterations: int = 1) -> np.ndarray:
    """Morphological opening (erosion followed by dilation).

    Useful for removing small bright spots / noise on a dark background.

    Parameters
    ----------
    img : np.ndarray
        Input binary or grayscale image (uint8).
    kernel_size : int
        Size of the square structuring element.
    iterations : int
        Number of times to apply the operation.

    Returns
    -------
    np.ndarray
        Opened image.
    """
    kernel = _get_kernel(kernel_size)
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    logger.debug("morph_open: kernel=%d iterations=%d", kernel_size, iterations)
    return result


def morph_close(img: np.ndarray, kernel_size: int = 3,
                iterations: int = 1) -> np.ndarray:
    """Morphological closing (dilation followed by erosion).

    Useful for closing small holes in foreground objects.

    Parameters
    ----------
    img : np.ndarray
        Input binary or grayscale image (uint8).
    kernel_size : int
        Size of the square structuring element.
    iterations : int
        Number of times to apply the operation.

    Returns
    -------
    np.ndarray
        Closed image.
    """
    kernel = _get_kernel(kernel_size)
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    logger.debug("morph_close: kernel=%d iterations=%d", kernel_size, iterations)
    return result


def morph_dilate(img: np.ndarray, kernel_size: int = 3,
                 iterations: int = 1) -> np.ndarray:
    """Dilate an image (expand bright/foreground regions).

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8).
    kernel_size : int
        Size of the square structuring element.
    iterations : int
        Number of times to apply.

    Returns
    -------
    np.ndarray
        Dilated image.
    """
    kernel = _get_kernel(kernel_size)
    result = cv2.dilate(img, kernel, iterations=iterations)
    logger.debug("morph_dilate: kernel=%d iterations=%d", kernel_size, iterations)
    return result


def morph_erode(img: np.ndarray, kernel_size: int = 3,
                iterations: int = 1) -> np.ndarray:
    """Erode an image (shrink bright/foreground regions).

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8).
    kernel_size : int
        Size of the square structuring element.
    iterations : int
        Number of times to apply.

    Returns
    -------
    np.ndarray
        Eroded image.
    """
    kernel = _get_kernel(kernel_size)
    result = cv2.erode(img, kernel, iterations=iterations)
    logger.debug("morph_erode: kernel=%d iterations=%d", kernel_size, iterations)
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using scipy.ndimage.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, values 0/255 or bool).

    Returns
    -------
    np.ndarray
        Binary mask with holes filled (uint8, 0/255).
    """
    # Normalise to boolean
    if mask.dtype == np.uint8:
        binary = mask > 127
    else:
        binary = mask.astype(bool)

    filled = ndimage.binary_fill_holes(binary)
    result = (filled.astype(np.uint8)) * 255
    logger.debug("fill_holes: holes_filled=%d pixels", int(result.sum() / 255 - binary.sum()))
    return result


def remove_small_objects(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove connected components smaller than a minimum area.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, 0/255).
    min_area : int
        Minimum area in pixels for a component to be kept.

    Returns
    -------
    np.ndarray
        Cleaned binary mask (uint8, 0/255).
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    removed_count = 0
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
        else:
            removed_count += 1

    logger.debug("remove_small_objects: min_area=%d  removed=%d components", min_area, removed_count)
    return cleaned
