"""
VIDIO Core - Contour Detection & Analysis
==========================================
Find contours in binary images, compute geometric statistics, and draw
annotated overlays.
"""

import logging
from typing import List, Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def contour_stats(contour: np.ndarray) -> dict:
    """Compute geometric statistics for a single contour.

    Parameters
    ----------
    contour : np.ndarray
        Contour points as returned by cv2.findContours.

    Returns
    -------
    dict
        Dictionary with keys: area, perimeter, solidity, elongation,
        convexity, circularity, centroid (x, y), angle.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, closed=True)

    # Solidity: ratio of contour area to convex hull area
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Convexity: ratio of convex hull perimeter to contour perimeter
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0

    # Circularity: 4 * pi * area / perimeter^2
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Fit ellipse for elongation and angle (requires >= 5 points)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        center = ellipse[0]
        axes = ellipse[1]  # (minor, major) or (width, height)
        angle = ellipse[2]
        minor_axis = min(axes)
        major_axis = max(axes)
        elongation = minor_axis / major_axis if major_axis > 0 else 1.0
    else:
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        w, h = rect[1]
        angle = rect[2]
        minor_axis = min(w, h)
        major_axis = max(w, h)
        elongation = minor_axis / major_axis if major_axis > 0 else 1.0

    # Centroid via moments
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = center

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "solidity": float(solidity),
        "elongation": float(elongation),
        "convexity": float(convexity),
        "circularity": float(circularity),
        "centroid": (float(cx), float(cy)),
        "angle": float(angle),
    }


def find_contours(binary_img: np.ndarray, min_area: int = 20,
                  max_area: Optional[int] = None) -> List[dict]:
    """Find and filter contours in a binary image.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image (uint8, 0/255).
    min_area : int
        Minimum contour area to keep.
    max_area : int, optional
        Maximum contour area. None = no upper limit.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``contour``: np.ndarray of contour points
        - ``bounding_rect``: (x, y, w, h)
        - ``hull``: convex hull points
        - ``stats``: dict from :func:`contour_stats`
    """
    if binary_img.dtype != np.uint8:
        binary_img = (binary_img > 0).astype(np.uint8) * 255

    contours_raw, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        stats = contour_stats(cnt)
        bounding_rect = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)

        results.append({
            "contour": cnt,
            "bounding_rect": bounding_rect,
            "hull": hull,
            "stats": stats,
        })

    # Sort by area descending
    results.sort(key=lambda c: c["stats"]["area"], reverse=True)
    logger.info("find_contours: found %d contours (min_area=%d, max_area=%s)",
                len(results), min_area, max_area)
    return results


def draw_contours(img: np.ndarray, contours: list,
                  color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw contours on an image copy.

    Parameters
    ----------
    img : np.ndarray
        Background image (BGR or grayscale).
    contours : list
        List of contour dicts as returned by :func:`find_contours`,
        or a list of raw np.ndarray contours.
    color : tuple
        BGR colour for the contour lines.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Annotated image copy.
    """
    canvas = img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    raw_contours = []
    for item in contours:
        if isinstance(item, dict):
            raw_contours.append(item["contour"])
        elif isinstance(item, np.ndarray):
            raw_contours.append(item)
        else:
            logger.warning("draw_contours: skipping unrecognised contour type %s", type(item))

    cv2.drawContours(canvas, raw_contours, -1, color, thickness)
    logger.debug("draw_contours: drew %d contours", len(raw_contours))
    return canvas
