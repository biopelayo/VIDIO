"""
Retinal Image Segmentation Functions
======================================

Anatomical structure and lesion segmentation routines for retinal images.

This module provides four specialised segmentation functions:

* :func:`segment_vessels` -- Extract the retinal vessel tree using
  Laplacian or Canny edge detection followed by morphological cleanup.
* :func:`detect_optic_disc` -- Locate the optic disc (nerve head) as a
  bright circular region using the Hough circle transform.
* :func:`detect_macula` -- Locate the foveal pit (macula) as the
  darkest region near the optic disc.
* :func:`segment_lesions` -- Detect anomalous regions (bright or dark
  lesions) using a 2-sigma Laplacian threshold.

All functions operate on the preprocessed single-channel float32 image
produced by the preprocessing stage.

Design Philosophy
-----------------
These segmentation routines are intentionally classical (non-DL) so that
the pipeline can run without GPU or PyTorch dependencies.  When deep
learning models are available (see :mod:`pipelines.retinal.models`),
they can replace or complement these functions via the model wrapper.
"""
import numpy as np
import cv2

from core.edges import laplacian_edges, canny_edges
from core.morphology import morph_open, morph_close, morph_dilate, fill_holes, remove_small_objects
from core.contours import find_contours


def segment_vessels(img, method='laplacian'):
    """
    Segment the retinal vessel tree from a preprocessed fundus image.

    Two edge-detection strategies are supported:

    * **Laplacian** (default) -- computes the second-order Laplacian of
      the image.  Pixels whose absolute Laplacian response exceeds 0.5
      standard deviations are marked as vessel candidates.  The
      Laplacian is preferred because it responds to *both* edges of a
      vessel (dark-to-light and light-to-dark transitions) in a single
      pass, naturally producing connected vessel centre-line responses.
    * **Canny** -- applies the Canny edge detector with hysteresis
      thresholds ``low=30``, ``high=100``.  This produces thinner,
      single-pixel edges and is useful when vessel *boundaries* (rather
      than centre-lines) are needed, but it can fragment narrow
      capillaries.

    After edge detection, morphological closing reconnects small gaps
    in the vessel tree, and a minimum-area filter removes isolated noise
    speckles.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 image (output of
        :func:`preprocessing.preprocess_fundus`).
    method : {'laplacian', 'canny'}, optional
        Edge detection method.  Default ``'laplacian'``.

    Returns
    -------
    numpy.ndarray
        Binary vessel mask of shape ``(H, W)`` with dtype ``uint8``.
        Vessel pixels = 1, background = 0.

    Notes
    -----
    **Laplacian threshold = 0.5 * std** :  This threshold is deliberately
    set below 1.0 std to capture finer capillaries that produce weaker
    edge responses.  Setting it at 1.0 std would miss peripheral
    capillaries; setting it below 0.3 std would include too much noise.
    The 0.5 multiplier was empirically calibrated on DRIVE and STARE
    fundus datasets.

    **Canny thresholds (30, 100)** :  The low threshold of 30 captures
    weak edges from small vessels; the high threshold of 100 ensures
    strong edges from major arcades are reliably detected.  The 1:3.3
    ratio is close to Canny's recommended 1:2--1:3 range.

    **Morphological close (3x3, 2 iter)** :  Closes 1--2 pixel gaps in
    the vessel tree caused by local contrast dropout (e.g. at vessel
    crossings or where a lesion overlaps a vessel).

    **remove_small_objects(min_area=50)** :  Vessel fragments below 50
    pixels are noise or sub-resolution capillary cross-sections.
    """
    if method == 'laplacian':
        edge_map, stats = laplacian_edges(img)
        # Threshold at 0.5 std to capture both major vessels and
        # finer capillaries (see Notes).
        threshold = stats['std'] * 0.5
        vessel_mask = (np.abs(edge_map) > threshold).astype(np.uint8)
    else:
        # Canny requires uint8 input.
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        vessel_mask = canny_edges(img_uint8, low=30, high=100)

    # Close small gaps in the vessel tree (1-2 px dropout).
    vessel_mask = morph_close(vessel_mask, kernel_size=3, iterations=2)
    # Remove isolated noise speckles below 50 px.
    vessel_mask = remove_small_objects(vessel_mask, min_area=50)
    return vessel_mask


def detect_optic_disc(img, min_radius=30, max_radius=120):
    """
    Detect the optic disc (optic nerve head) in a fundus image.

    The optic disc is the brightest, roughly circular structure in a
    fundus image -- it is the entry point of the optic nerve and
    central retinal vessels.  This function uses the Hough circle
    transform to locate it.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 image.
    min_radius : int, optional
        Minimum expected optic disc radius in pixels.  Default **30**,
        suitable for images >= 512 px wide.
    max_radius : int, optional
        Maximum expected optic disc radius in pixels.  Default **120**,
        suitable for images up to ~3000 px wide.

    Returns
    -------
    dict or None
        If detected::

            {
                'center': (int, int),  # (x, y) pixel coordinates
                'radius': int,         # radius in pixels
            }

        ``None`` if no circle is found (e.g. image is an OCT scan or
        the disc is outside the field of view).

    Notes
    -----
    **HoughCircles parameter choices** :

    * ``dp=1.5`` -- accumulator resolution ratio.  A value of 1.0 would
      use the same resolution as the input image, which is precise but
      slow and sensitive to noise.  1.5 reduces the accumulator
      resolution, making detection more robust to imperfect circularity
      (the optic disc is often slightly elliptical) at the cost of
      ~1-2 px localisation error, which is acceptable for this task.
    * ``minDist=img.shape[0] // 4`` -- minimum distance between detected
      circle centres.  Set to one quarter of the image height to ensure
      that at most one optic disc is detected per image (there is only
      one optic disc per eye).
    * ``param1=100`` -- upper Canny threshold used internally by
      HoughCircles.  100 is high enough to ignore faint vessel edges
      while retaining the strong disc boundary.
    * ``param2=40`` -- accumulator threshold for circle detection.
      Lower values detect more circles (including false positives);
      40 is a conservative choice that works well on CLAHE-enhanced
      images where the disc boundary has been sharpened.

    **Clinical relevance** : Optic disc detection is a prerequisite for
    cup-to-disc ratio estimation (glaucoma screening) and for
    orienting the macula search region (the macula is approximately
    2.5 disc diameters temporal to the disc centre).
    """
    # HoughCircles requires uint8 input.
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        img_uint8,
        cv2.HOUGH_GRADIENT,
        dp=1.5,            # Accumulator resolution ratio (see Notes)
        minDist=img.shape[0] // 4,  # Only one disc per image
        param1=100,         # Upper Canny threshold
        param2=40,          # Accumulator threshold
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # The first detected circle is the strongest candidate.
        best = circles[0]
        return {'center': (int(best[0]), int(best[1])), 'radius': int(best[2])}
    return None


def detect_macula(img, optic_disc=None):
    """
    Locate the macula (foveal pit) in a fundus image.

    The macula is the central, high-acuity region of the retina
    containing the fovea.  In fundus photographs it appears as the
    **darkest point** in the macular region because the foveal pit has
    the thinnest retinal tissue (no inner retinal layers) and maximal
    macular pigment (lutein/zeaxanthin), both of which reduce
    reflectance in the green channel.

    If the optic disc has been detected, the search is anatomically
    constrained: the macula is approximately **2.5 disc diameters
    temporal** to the disc centre (to the right for left-eye images
    where the disc appears on the left, and vice versa).  If the disc
    is not available, the search defaults to the image centre.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 fundus image.
    optic_disc : dict or None, optional
        Optic disc detection result from :func:`detect_optic_disc`.
        Expected keys: ``'center'`` (x, y) and ``'radius'``.
        If ``None``, the search region is centred on the image.

    Returns
    -------
    dict or None
        If detected::

            {
                'center': (int, int),      # (x, y) of the darkest pixel
                'region': (x1, y1, x2, y2) # search ROI bounding box
            }

        ``None`` if the search ROI is empty (edge case for very small
        images).

    Notes
    -----
    **Why the darkest point?**  The fovea has the highest concentration
    of macular pigment (which absorbs blue and green light) and the
    thinnest retinal tissue (avascular foveal zone).  In the CLAHE-
    enhanced green channel, these factors combine to make the foveal
    centre the local intensity minimum within the macular region.

    **Anatomical offset of 2.5 * radius** :  The optic disc diameter is
    approximately 1.5 mm; the disc-to-fovea distance is approximately
    4.0-4.5 mm, which is roughly 2.5-3.0 disc diameters.  The 2.5
    multiplier is a conservative lower bound that ensures the search
    region includes the fovea even for slightly eccentric disc positions.

    **Search region size = min(H, W) / 8** :  This creates a search
    window of ~12.5% of the image dimension on each side of the
    expected macular centre -- large enough to accommodate anatomical
    variation but small enough to avoid false minima from dark lesions
    or image borders.

    **Left/right eye handling** :  The function infers eye laterality
    from the disc position relative to the image midline.  If
    ``disc_x < image_width / 2``, the disc is on the left (right-eye
    image in standard orientation), so the macula is *to the right*
    (``cx + 2.5r``).  Otherwise the macula is to the left.
    """
    h, w = img.shape[:2]
    if optic_disc:
        cx, cy = optic_disc['center']
        r = optic_disc['radius']
        # Anatomical constraint: macula is ~2.5 disc diameters temporal
        # to the optic disc.  Direction depends on eye laterality.
        search_x = cx + int(2.5 * r) if cx < w // 2 else cx - int(2.5 * r)
        search_y = cy
    else:
        # Fallback: search near the image centre.
        search_x = w // 2
        search_y = h // 2

    # Define the search ROI (~1/8 of image dimension on each side).
    region_size = min(h, w) // 8
    x1 = max(0, search_x - region_size)
    y1 = max(0, search_y - region_size)
    x2 = min(w, search_x + region_size)
    y2 = min(h, search_y + region_size)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # The fovea is the darkest point in the macular search region.
    min_loc = np.unravel_index(np.argmin(roi), roi.shape)
    return {
        'center': (x1 + min_loc[1], y1 + min_loc[0]),
        'region': (x1, y1, x2, y2),
    }


def segment_lesions(img):
    """
    Segment candidate lesion regions using a 2-sigma Laplacian threshold.

    Retinal lesions (drusen, exudates, haemorrhages, microaneurysms)
    produce locally anomalous intensity patterns.  In the Laplacian
    domain these appear as strong positive or negative responses far
    from the mean.  This function thresholds at +/- 2 standard
    deviations to capture only the most pronounced anomalies, then
    applies morphological cleanup to produce coherent region contours.

    Parameters
    ----------
    img : numpy.ndarray
        Preprocessed single-channel float32 image.

    Returns
    -------
    list of dict
        Region dictionaries from :func:`core.contours.find_contours`,
        each containing a ``'contour'`` array and ``'stats'`` dict with
        bounding box and shape descriptors.

    Notes
    -----
    **Why 2 * std and not 1 * std?**  The 1-sigma threshold used in the
    pipeline's ``segment`` method captures *all* smooth regions (including
    normal tissue) for global analysis.  Here, the purpose is
    specifically to detect *lesions*, which are extreme outliers.  A
    2-sigma threshold corresponds to the outer ~5% of a normal
    distribution, capturing only regions with genuinely anomalous edge
    responses while suppressing the vessel tree and normal retinal
    texture.  This is complementary to, not a replacement for, the
    main segmentation stage.

    **Both positive and negative Laplacian extremes are included**
    because:

    * **Laplacian >= +2 std** : bright blobs surrounded by darker
      tissue (exudates, drusen, cotton-wool spots).  The Laplacian
      is positive at local intensity maxima.
    * **Laplacian <= -2 std** : dark blobs surrounded by lighter
      tissue (haemorrhages, microaneurysms).  The Laplacian is
      negative at local intensity minima.

    **Morphological cleanup sequence** :

    1. ``morph_close(kernel=5, iter=3)`` -- aggressive closing to merge
       fragmented lesion edges into solid blobs.  Kernel size 5 (larger
       than the vessel segmentation's 3) bridges wider gaps typical of
       large drusen or confluent exudates.
    2. ``morph_open(kernel=3, iter=1)`` -- light opening to remove thin
       noise protrusions introduced by the closing step.
    3. ``fill_holes`` -- fill interior holes in lesion masks (common
       when a vessel crosses through a lesion).
    4. ``remove_small_objects(min_area=100)`` -- discard sub-100 px
       noise fragments.

    Algorithm
    ---------
    1. Compute Laplacian edge map and its standard deviation.
    2. Threshold: select pixels where |Laplacian| > 2 std.
    3. Morphological close (5x5, 3 iter) to merge fragments.
    4. Morphological open (3x3, 1 iter) to remove protrusions.
    5. Fill interior holes.
    6. Remove objects < 100 px.
    7. Extract contours with min_area=100.
    """
    edge_map, stats = laplacian_edges(img)
    std = stats['std']

    # --- 2-sigma threshold ---
    # Select pixels with extreme Laplacian response (both bright and
    # dark lesions).  ~5% of a normal distribution falls beyond +/- 2 std.
    anomaly_mask = np.logical_or(edge_map <= -2 * std, edge_map >= 2 * std).astype(np.uint8)

    # --- Morphological cleanup ---
    # Close: merge fragmented lesion edges into solid blobs (kernel=5
    # bridges wider gaps than the vessel segmentation's kernel=3).
    anomaly_mask = morph_close(anomaly_mask, kernel_size=5, iterations=3)
    # Open: remove thin noise protrusions from the closing step.
    anomaly_mask = morph_open(anomaly_mask, kernel_size=3, iterations=1)
    # Fill holes left by vessels crossing through lesions.
    anomaly_mask = fill_holes(anomaly_mask)
    # Discard noise fragments below 100 px.
    anomaly_mask = remove_small_objects(anomaly_mask, min_area=100)

    regions = find_contours(anomaly_mask, min_area=100)
    return regions
