"""
VIDIO Core - Severity Classification & Finding Builder
========================================================
Classify analysis findings by severity and build structured finding
dictionaries ready for database storage or reporting.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default severity thresholds
# ---------------------------------------------------------------------------
# The gradient (max - mean intensity) and confidence are used together
# to determine severity. Higher gradient AND higher confidence push
# towards more severe classifications.
#
# Thresholds format: list of (min_gradient, min_confidence, severity_label)
# evaluated top-to-bottom; the FIRST match wins.
DEFAULT_SEVERITY_THRESHOLDS = [
    {"min_gradient": 80.0, "min_confidence": 0.85, "severity": "CRITICAL"},
    {"min_gradient": 50.0, "min_confidence": 0.70, "severity": "HIGH"},
    {"min_gradient": 25.0, "min_confidence": 0.50, "severity": "MEDIUM"},
    {"min_gradient": 0.0,  "min_confidence": 0.0,  "severity": "LOW"},
]


def classify_severity(gradient: float, confidence: float,
                      thresholds: list = None) -> str:
    """Classify the severity of a finding based on gradient and confidence.

    The gradient represents the difference between max and mean intensity in
    the region of interest (higher values indicate stronger anomalies).
    Confidence is the model or algorithm confidence in [0, 1].

    Parameters
    ----------
    gradient : float
        Intensity gradient value (max - mean).
    confidence : float
        Confidence score in [0, 1].
    thresholds : list[dict], optional
        Custom thresholds. Each dict must have ``min_gradient``,
        ``min_confidence``, and ``severity`` keys. Evaluated top-to-bottom;
        the first match wins. Defaults to :data:`DEFAULT_SEVERITY_THRESHOLDS`.

    Returns
    -------
    str
        Severity label: ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    """
    if thresholds is None:
        thresholds = DEFAULT_SEVERITY_THRESHOLDS

    for t in thresholds:
        if gradient >= t["min_gradient"] and confidence >= t["min_confidence"]:
            severity = t["severity"]
            logger.debug("classify_severity: gradient=%.2f confidence=%.2f -> %s",
                         gradient, confidence, severity)
            return severity

    # Fallback (should not happen with default thresholds)
    logger.warning("classify_severity: no threshold matched, defaulting to LOW.")
    return "LOW"


def build_finding(finding_type: str,
                  severity: str,
                  confidence: float,
                  statistics: dict,
                  geometric_properties: dict,
                  location: Optional[dict] = None,
                  classification: Optional[str] = None) -> dict:
    """Build a structured finding dictionary ready for database storage.

    Parameters
    ----------
    finding_type : str
        Type of finding (e.g. ``"hotspot"``, ``"lesion"``, ``"anomaly"``).
    severity : str
        Severity label (``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``).
    confidence : float
        Confidence score in [0, 1].
    statistics : dict
        ROI statistics dict (from :mod:`vidio.core.statistics`).
    geometric_properties : dict
        Geometric properties dict (area, perimeter, circularity, etc.).
    location : dict, optional
        Spatial location information. Typically contains keys like
        ``bounding_rect``, ``centroid``, ``slice_index``, ``region_id``.
    classification : str, optional
        Additional classification label (e.g. tissue type, pathology class).

    Returns
    -------
    dict
        Structured finding with the following top-level keys:
        ``finding_type``, ``severity``, ``confidence``, ``statistics``,
        ``geometric_properties``, ``location``, ``classification``,
        ``created_at``, ``version``.
    """
    finding = {
        "finding_type": str(finding_type),
        "severity": str(severity).upper(),
        "confidence": float(confidence),
        "statistics": _sanitize_for_json(statistics),
        "geometric_properties": _sanitize_for_json(geometric_properties),
        "location": _sanitize_for_json(location) if location else None,
        "classification": str(classification) if classification else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
    }

    logger.info("build_finding: type=%s severity=%s confidence=%.2f",
                finding_type, severity, confidence)
    return finding


def _sanitize_for_json(obj):
    """Recursively convert numpy types to Python natives for JSON serialisation."""
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
