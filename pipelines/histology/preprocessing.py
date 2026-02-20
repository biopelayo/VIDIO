"""
Histology preprocessing utilities.

This module provides helper functions for the earliest stages of
whole-slide image analysis: separating tissue from background glass and
normalising stain colour variation between laboratories / scanners.

Functions here are *stateless* and operate on plain NumPy arrays so they
can be used both inside the :class:`~pipelines.histology.pipeline.HistologyPipeline`
and as standalone utilities in notebooks or scripts.

Design Notes
------------
* **Tissue detection** relies on a simple grayscale intensity threshold
  (default 220).  In H&E-stained sections the glass background is nearly
  white (intensity ~240--255) while any stained tissue absorbs light and
  appears darker.  A threshold of 220 provides a comfortable margin above
  the darkest background pixels while remaining well below typical tissue
  intensities (30--200 depending on stain uptake).

* **Stain normalisation** delegates to :func:`core.intensity.stain_normalize`
  which implements the Macenko method.  The wrapper here exists so that
  callers in the histology package do not need to know about the ``core``
  internal API.
"""

import numpy as np
import cv2

from core.intensity import stain_normalize, normalize_minmax


def detect_tissue(img, threshold=220):
    """
    Create a binary mask separating tissue from glass background.

    Pixels with a grayscale intensity **below** *threshold* are classified
    as tissue (1); pixels at or above the threshold are background (0).

    Parameters
    ----------
    img : numpy.ndarray
        Input image, either RGB ``(H, W, 3)`` or grayscale ``(H, W)``.
        If RGB, it is converted to single-channel grayscale via OpenCV's
        ``COLOR_RGB2GRAY`` (weighted luminance formula).
    threshold : int, optional
        Intensity cut-off.  Pixels with ``gray < threshold`` are tissue.
        Default is ``220``, which works well for standard H&E staining on
        white glass slides.

    Returns
    -------
    numpy.ndarray
        Binary mask of shape ``(H, W)`` with dtype ``uint8``.  Values are
        ``1`` for tissue and ``0`` for background.

    Examples
    --------
    >>> mask = detect_tissue(slide_thumbnail, threshold=200)
    >>> tissue_area_px = np.sum(mask)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    tissue_mask = (gray < threshold).astype(np.uint8)
    return tissue_mask


def get_tissue_ratio(img, threshold=220):
    """
    Compute the fraction of an image occupied by tissue.

    This is a convenience wrapper around :func:`detect_tissue` that
    returns a single scalar instead of the full mask.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (RGB or grayscale).
    threshold : int, optional
        Intensity threshold passed to :func:`detect_tissue`.
        Default ``220``.

    Returns
    -------
    float
        Ratio in ``[0.0, 1.0]``.  A value of 0.0 means the tile is pure
        background; 1.0 means every pixel is tissue.  The pipeline's
        default retention threshold is 0.5 (50 % tissue).

    See Also
    --------
    detect_tissue : Full binary mask generation.
    """
    mask = detect_tissue(img, threshold)
    return np.sum(mask) / mask.size


def normalize_stain(img, method='macenko'):
    """
    Normalise stain colours to reduce inter-laboratory variation.

    Thin wrapper around :func:`core.intensity.stain_normalize` provided
    so that histology-package callers have a single, locally-scoped entry
    point without importing from ``core`` directly.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image ``(H, W, 3)`` uint8.
    method : str, optional
        Normalisation algorithm.  Currently only ``'macenko'`` is
        supported by the core module.  Default ``'macenko'``.

    Returns
    -------
    numpy.ndarray
        Stain-normalised image with the same shape and dtype as *img*.

    Raises
    ------
    ValueError
        If the image does not contain enough tissue pixels for reliable
        stain vector estimation (propagated from the core implementation).

    Notes
    -----
    The Macenko method works by:

    1. Converting the image to optical density (OD) space via
       ``OD = -log(I / I_0)`` where ``I_0`` is the illumination intensity
       (typically 255 for uint8).
    2. Running SVD on the OD values of tissue pixels to find the two
       principal stain directions (haematoxylin and eosin).
    3. Projecting those directions onto a fixed reference stain matrix,
       effectively mapping the source colour space to the target.
    """
    return stain_normalize(img, method=method)
