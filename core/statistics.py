"""
VIDIO Core - Image & ROI Statistics
=====================================
Compute descriptive statistics over regions of interest, compare regions,
and generate z-score-based analyses.
"""

import logging
from typing import List, Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def roi_stats(img: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Compute descriptive statistics for an image region.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale float/int or single-channel).
    mask : np.ndarray, optional
        Binary mask selecting the ROI. If None, the entire image is used.

    Returns
    -------
    dict
        Dictionary with keys: min, max, mean, std, median, mode,
        gradient (max - mean), skewness, kurtosis.
    """
    if img.ndim == 3:
        # Convert to grayscale intensity for statistics
        import cv2
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = img[:, :, 0].astype(np.float64)
    else:
        gray = img.astype(np.float64)

    if mask is not None:
        if mask.dtype == np.uint8:
            mask_bool = mask > 127
        else:
            mask_bool = mask.astype(bool)
        values = gray[mask_bool]
    else:
        values = gray.ravel()

    if values.size == 0:
        logger.warning("roi_stats: empty ROI (no pixels selected).")
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            "median": 0.0, "mode": 0.0, "gradient": 0.0,
            "skewness": 0.0, "kurtosis": 0.0,
        }

    v_min = float(values.min())
    v_max = float(values.max())
    v_mean = float(values.mean())
    v_std = float(values.std())
    v_median = float(np.median(values))

    # Mode: use scipy, take the smallest mode if multimodal
    mode_result = sp_stats.mode(values.astype(np.int64) if values.dtype == np.float64
                                else values, keepdims=True)
    v_mode = float(mode_result.mode[0]) if mode_result.mode.size > 0 else v_median

    # Gradient: difference between max and mean intensity
    gradient = v_max - v_mean

    # Skewness and kurtosis
    v_skew = float(sp_stats.skew(values))
    v_kurt = float(sp_stats.kurtosis(values))

    result = {
        "min": v_min,
        "max": v_max,
        "mean": v_mean,
        "std": v_std,
        "median": v_median,
        "mode": v_mode,
        "gradient": gradient,
        "skewness": v_skew,
        "kurtosis": v_kurt,
    }
    logger.debug("roi_stats: mean=%.2f std=%.2f gradient=%.2f", v_mean, v_std, gradient)
    return result


def region_stats(img: np.ndarray, regions: list) -> List[dict]:
    """Compute ROI statistics for each region (mask) in a list.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    regions : list[np.ndarray]
        List of binary masks, one per region.

    Returns
    -------
    list[dict]
        List of statistics dictionaries, one per region.
    """
    all_stats = []
    for i, mask in enumerate(regions):
        s = roi_stats(img, mask)
        s["region_index"] = i
        all_stats.append(s)
    logger.info("region_stats: computed stats for %d regions", len(all_stats))
    return all_stats


def compare_regions(stats_list: List[dict]) -> List[dict]:
    """Compare each region's statistics against the global (pooled) statistics
    using z-scores.

    Parameters
    ----------
    stats_list : list[dict]
        List of ROI statistics dicts as returned by :func:`roi_stats` or
        :func:`region_stats`.

    Returns
    -------
    list[dict]
        Each dict contains z-scores for the numeric keys in the stats,
        comparing that region to the global mean/std across all regions.
    """
    numeric_keys = ["min", "max", "mean", "std", "median", "mode",
                    "gradient", "skewness", "kurtosis"]

    # Collect values per key
    key_values = {k: [] for k in numeric_keys}
    for s in stats_list:
        for k in numeric_keys:
            if k in s:
                key_values[k].append(s[k])

    # Compute global mean and std for each key
    global_stats = {}
    for k in numeric_keys:
        vals = np.array(key_values[k])
        if vals.size > 0:
            global_stats[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        else:
            global_stats[k] = {"mean": 0.0, "std": 1.0}

    # Compute z-scores for each region
    z_scores_list = []
    for i, s in enumerate(stats_list):
        z = {"region_index": s.get("region_index", i)}
        for k in numeric_keys:
            g = global_stats[k]
            val = s.get(k, 0.0)
            if g["std"] > 1e-12:
                z[k] = float((val - g["mean"]) / g["std"])
            else:
                z[k] = 0.0
        z_scores_list.append(z)

    logger.info("compare_regions: compared %d regions to global stats", len(z_scores_list))
    return z_scores_list
