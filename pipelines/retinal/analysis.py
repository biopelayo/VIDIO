"""
Retinal Analysis Functions
===========================

Higher-level analytical routines that transform segmented regions into
clinical findings and lesion-type masks.

This module provides three functions:

* :func:`analyze_retinal_regions` -- The primary anomaly-detection
  routine that computes z-scores for each segmented region against the
  whole-image baseline and produces structured clinical findings.
* :func:`detect_bright_lesions` -- Thresholding-based extraction of
  bright lesion candidates (exudates, drusen, cotton-wool spots).
* :func:`detect_dark_lesions` -- Thresholding-based extraction of dark
  lesion candidates (haemorrhages, microaneurysms).

The bright/dark lesion functions are utility masks that can be used for
rapid screening or as complementary evidence alongside the full
region-based analysis.

Clinical Mapping
----------------
+------------------+---------------------+-----------------------------+
| Lesion type      | Retinal appearance  | Associated pathologies      |
+==================+=====================+=============================+
| Bright lesions   | High reflectance,   | Exudates (DR), drusen       |
|                  | yellow/white in     | (AMD), cotton-wool spots    |
|                  | colour fundus       | (hypertensive retinopathy)  |
+------------------+---------------------+-----------------------------+
| Dark lesions     | Low reflectance,    | Dot/blot haemorrhages (DR), |
|                  | red/dark in colour  | microaneurysms (DR),        |
|                  | fundus              | choroidal naevi              |
+------------------+---------------------+-----------------------------+
"""
import numpy as np

from core.clustering import kmeans_segment
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


def analyze_retinal_regions(img, regions, global_stats=None):
    """
    Perform z-score-based anomaly detection on segmented retinal regions.

    For each region, the function extracts the bounding-box ROI,
    computes its intensity statistics, and compares the region's mean
    intensity against the whole-image mean using a z-score.  Regions
    whose z-score exceeds **1.5** are flagged as ``RETINAL_ANOMALY``
    findings with severity and confidence scores.

    This function is the standalone equivalent of the anomaly-detection
    logic in :meth:`RetinalPipeline.detect_anomalies`, but without the
    geometric subclassification (LESION / VESSEL_ANOMALY).  It is
    useful for quick screening or when geometric contour data is not
    available.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 image.
    regions : list of dict
        Segmented regions from :func:`segmentation.segment_lesions` or
        similar.  Each dict must contain a ``'stats'`` sub-dict with
        keys ``x``, ``y``, ``w``, ``h``.
    global_stats : dict or None, optional
        Pre-computed whole-image statistics (from
        :func:`core.statistics.roi_stats`).  If ``None``, they are
        computed from *img*.  Passing pre-computed stats avoids
        redundant computation when analysing multiple region sets from
        the same image.

    Returns
    -------
    list of dict
        Clinical findings.  Each finding dict (built by
        :func:`core.classification.build_finding`) contains:

        * ``'finding_type'`` : ``'RETINAL_ANOMALY'``
        * ``'severity'`` : ``'CRITICAL'`` / ``'HIGH'`` / ``'MEDIUM'`` /
          ``'LOW'``
        * ``'confidence'`` : float in [0, 1]
        * ``'statistics'`` : full ROI stats for the region
        * ``'geometric_properties'`` : bounding-box and shape descriptors
        * ``'location'`` : ``{x, y, w, h}``

    Notes
    -----
    **Z-score threshold of 1.5** :  Identical rationale to
    :meth:`RetinalPipeline.detect_anomalies` -- this is a recall-
    favouring first-pass filter.  In a normal distribution, ~13% of
    values fall beyond +/- 1.5 sigma.  For retinal images where most
    of the field is healthy tissue with a fairly tight intensity
    distribution, 1.5 sigma captures early-stage pathology (faint
    drusen, early exudates) while still rejecting the bulk of normal
    background variation.

    **Confidence = min(1.0, z_score / 5.0)** :  Linear mapping from
    z-score to a [0, 1] confidence value.  A z-score of 5 (extreme
    outlier) saturates at confidence 1.0.  Combined with the gradient
    magnitude by :func:`classify_severity`, this determines the final
    severity label.

    Algorithm
    ---------
    1. Compute (or reuse) global image statistics.
    2. For each region:
       a. Extract bounding-box ROI.
       b. Compute ROI intensity statistics.
       c. Calculate z-score = |ROI_mean - global_mean| / global_std.
       d. Skip regions with z_score < 1.5 (within normal range).
       e. Map z-score to confidence, classify severity.
       f. Build and append finding dict.
    3. Return findings list.
    """
    # Compute global baseline if not provided.
    if global_stats is None:
        global_stats = roi_stats(img)

    findings = []
    global_mean = global_stats.get('mean', 0)
    # Default std = 1 prevents division by zero on uniform images.
    global_std = global_stats.get('std', 1)

    for region in regions:
        stats_data = region.get('stats', {})
        x = stats_data.get('x', 0)
        y = stats_data.get('y', 0)
        w = stats_data.get('w', 0)
        h = stats_data.get('h', 0)

        # Extract bounding-box ROI, clamping to image boundaries.
        roi = img[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            continue

        # Compute per-region intensity descriptors.
        region_stat = roi_stats(roi)
        mean_val = region_stat.get('mean', 0)
        gradient = region_stat.get('gradient', 0)

        # Z-score: how many std devs does this region deviate from
        # the whole-image mean?
        z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

        # Skip regions within the normal range (z < 1.5).
        if z_score < 1.5:
            continue

        # Map z-score to confidence [0, 1], saturating at z = 5.
        confidence = min(1.0, z_score / 5.0)
        severity = classify_severity(gradient, confidence)

        finding = build_finding(
            finding_type='RETINAL_ANOMALY',
            severity=severity,
            confidence=confidence,
            statistics=region_stat,
            geometric_properties=stats_data,
            location={'x': x, 'y': y, 'w': w, 'h': h},
        )
        findings.append(finding)

    return findings


def detect_bright_lesions(img, threshold_factor=2.0):
    """
    Create a binary mask of bright lesion candidates.

    Bright lesions in retinal images correspond to areas of high
    reflectance / high pixel intensity in the green channel.  Clinically,
    these include:

    * **Hard exudates** -- waxy, yellow deposits of lipid and protein
      leaking from damaged retinal capillaries; a hallmark of diabetic
      retinopathy.
    * **Drusen** -- extracellular deposits beneath the retinal pigment
      epithelium; the earliest clinical sign of age-related macular
      degeneration (AMD).
    * **Cotton-wool spots** -- fluffy white patches caused by retinal
      nerve fibre layer infarction; seen in hypertensive retinopathy
      and diabetic retinopathy.

    The detection is a simple global thresholding at ``mean + k * std``
    where *k* is the ``threshold_factor``.  This is a fast screening
    mask, not a precise segmentation.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 image.
    threshold_factor : float, optional
        Number of standard deviations above the mean to use as the
        brightness threshold.  Default **2.0** (selects pixels brighter
        than ~97.7% of the image under a normal distribution).

    Returns
    -------
    numpy.ndarray
        Binary mask of shape ``(H, W)`` with dtype ``uint8``.
        Bright lesion pixels = 1, background = 0.

    Notes
    -----
    **threshold_factor = 2.0** :  Two standard deviations above the mean
    captures approximately the top 2.3% of pixel intensities.  This is
    a good first-pass filter for exudates and drusen, which are
    significantly brighter than the surrounding retinal tissue.  A
    higher factor (e.g. 3.0) would miss faint drusen; a lower factor
    (e.g. 1.0) would include the optic disc and peripapillary atrophy.

    This mask should be post-processed (morphological cleanup, area
    filtering, optic disc exclusion) before clinical use.

    See Also
    --------
    detect_dark_lesions : Complementary function for haemorrhages and
        microaneurysms.
    """
    stats = roi_stats(img)
    # Threshold at mean + k * std to isolate abnormally bright pixels.
    threshold = stats['mean'] + threshold_factor * stats['std']
    bright_mask = (img > threshold).astype(np.uint8)
    return bright_mask


def detect_dark_lesions(img, threshold_factor=2.0):
    """
    Create a binary mask of dark lesion candidates.

    Dark lesions in retinal images correspond to areas of low
    reflectance / low pixel intensity in the green channel.  Clinically,
    these include:

    * **Dot and blot haemorrhages** -- small intraretinal bleeds caused
      by capillary damage in diabetic retinopathy.  Dot haemorrhages
      are < 125 microns (similar in size to microaneurysms); blot
      haemorrhages are larger, irregular.
    * **Microaneurysms** -- the earliest clinically visible sign of
      diabetic retinopathy.  These are saccular outpouchings of
      weakened capillary walls that appear as small, round, dark spots
      (< 125 microns).
    * **Choroidal naevi** -- benign pigmented lesions beneath the
      retina.  These are typically larger and more diffuse than
      haemorrhages.

    The detection uses a simple global threshold at
    ``mean - k * std``, selecting pixels darker than the majority of
    the image.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 image.
    threshold_factor : float, optional
        Number of standard deviations below the mean to use as the
        darkness threshold.  Default **2.0** (selects pixels darker
        than ~97.7% of the image under a normal distribution).

    Returns
    -------
    numpy.ndarray
        Binary mask of shape ``(H, W)`` with dtype ``uint8``.
        Dark lesion pixels = 1, background = 0.

    Notes
    -----
    **threshold_factor = 2.0** :  Two standard deviations below the mean
    captures approximately the bottom 2.3% of pixel intensities.  This
    isolates haemorrhages and microaneurysms, which are significantly
    darker than the surrounding retinal tissue in the green channel.

    **Caution** :  This mask will also include the foveal pit (naturally
    dark) and vessel segments.  Post-processing should exclude the
    macular region (see :func:`segmentation.detect_macula`) and
    subtract the vessel mask (see :func:`segmentation.segment_vessels`)
    to isolate true dark lesions.

    See Also
    --------
    detect_bright_lesions : Complementary function for exudates and
        drusen.
    segment_vessels : Vessel mask for excluding vessel false positives.
    detect_macula : Macula location for excluding the foveal pit.
    """
    stats = roi_stats(img)
    # Threshold at mean - k * std to isolate abnormally dark pixels.
    threshold = stats['mean'] - threshold_factor * stats['std']
    dark_mask = (img < threshold).astype(np.uint8)
    return dark_mask
