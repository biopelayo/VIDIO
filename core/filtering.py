"""
VIDIO Core - Image Filtering
==============================
Bilateral, Gaussian, Median, Unsharp Mask, and Non-Local Means denoising.
"""

import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def bilateral_filter(img: np.ndarray, d: int = 0, sigma_color: float = 1.0,
                     sigma_space: float = 3.0) -> np.ndarray:
    """Apply bilateral filtering (edge-preserving smoothing).

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32).
    d : int
        Diameter of pixel neighbourhood. 0 = auto-compute from sigma_space.
    sigma_color : float
        Filter sigma in the colour space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    if d == 0:
        d = max(1, int(np.ceil(sigma_space * 2)) * 2 + 1)
    filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    logger.debug("bilateral_filter: d=%d sigma_color=%.2f sigma_space=%.2f", d, sigma_color, sigma_space)
    return filtered


def gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    ksize : int
        Kernel size (must be odd and positive).
    sigma : float
        Gaussian sigma. 0 = auto-compute from ksize.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1
        logger.warning("gaussian_filter: ksize must be odd, adjusted to %d.", ksize)
    filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    logger.debug("gaussian_filter: ksize=%d sigma=%.2f", ksize, sigma)
    return filtered


def median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply median blur (good for salt-and-pepper noise).

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32).
    ksize : int
        Kernel size (must be odd and positive).

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    if ksize % 2 == 0:
        ksize += 1
        logger.warning("median_filter: ksize must be odd, adjusted to %d.", ksize)
    filtered = cv2.medianBlur(img, ksize)
    logger.debug("median_filter: ksize=%d", ksize)
    return filtered


def unsharp_mask(img: np.ndarray, sigma: float = 1.0,
                 strength: float = 1.5) -> np.ndarray:
    """Sharpen an image using the unsharp mask technique.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    sigma : float
        Gaussian sigma for the blur pass.
    strength : float
        Sharpening strength (weight of the high-frequency component).

    Returns
    -------
    np.ndarray
        Sharpened image.
    """
    ksize = max(3, int(np.ceil(sigma * 6)) | 1)  # ensure odd
    blurred = cv2.GaussianBlur(img.astype(np.float64), (ksize, ksize), sigma)
    sharpened = img.astype(np.float64) + strength * (img.astype(np.float64) - blurred)
    if img.dtype == np.uint8:
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        sharpened = sharpened.astype(img.dtype)
    logger.debug("unsharp_mask: sigma=%.2f strength=%.2f", sigma, strength)
    return sharpened


def denoise_nlm(img: np.ndarray, h: float = 10.0, template_size: int = 7,
                search_size: int = 21) -> np.ndarray:
    """Denoise an image using Non-Local Means.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8).
    h : float
        Filter strength. Higher removes more noise but also detail.
    template_size : int
        Size of the template patch (must be odd).
    search_size : int
        Size of the search window (must be odd).

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    if template_size % 2 == 0:
        template_size += 1
    if search_size % 2 == 0:
        search_size += 1

    if img.dtype != np.uint8:
        from .intensity import normalize_minmax
        img = normalize_minmax(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        denoised = cv2.fastNlMeansDenoising(img, None, h, template_size, search_size)
    elif img.ndim == 3:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h,
                                                    template_size, search_size)
    else:
        logger.warning("denoise_nlm: unsupported ndim=%d, returning copy.", img.ndim)
        return img.copy()

    logger.debug("denoise_nlm: h=%.1f template=%d search=%d", h, template_size, search_size)
    return denoised
