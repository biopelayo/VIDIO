"""
Tumour region segmentation for histopathology images.

This module implements an unsupervised segmentation strategy based on
K-Means colour clustering followed by morphological refinement.  The
key assumption is the **"darkest cluster = tumour" hypothesis**: in
H&E-stained tissue, tumour nuclei absorb haematoxylin heavily and
therefore appear as the *darkest* intensity cluster, while stroma and
background occupy lighter clusters.

Pipeline
--------
1. Convert image to single-channel float32 grayscale.
2. Run K-Means (k=3) to partition pixel intensities into three clusters:
   dark (tumour-suspicious), medium (stroma / cytoplasm), light
   (background / glass).
3. Select the cluster with the lowest centroid value as the tumour mask.
4. Apply a morphological **closing** (5x5 kernel, 3 iterations) to fill
   small intra-tumoural gaps (e.g. lumina, fixation artefacts).
5. Apply a morphological **opening** (3x3 kernel, 1 iteration) to detach
   thin bridges between separate tumour foci.
6. Remove connected components smaller than 500 px to suppress noise.
7. Extract contours from the cleaned binary mask, keeping only those
   with area >= 200 px.

Limitations
-----------
* The darkest-cluster heuristic can fail on images dominated by
  melanin-heavy tissue or ink annotations.
* K=3 is a fixed choice; some slides may have more than three distinct
  tissue compartments.
"""

import numpy as np

from core.edges import laplacian_edges
from core.morphology import morph_close, morph_open, remove_small_objects
from core.contours import find_contours
from core.clustering import kmeans_segment


def segment_tumor_regions(img):
    """
    Segment tumour regions from a histopathology image using K-Means
    clustering and morphological cleanup.

    Parameters
    ----------
    img : numpy.ndarray
        Input image, shape ``(H, W, 3)`` (RGB) or ``(H, W)`` (grayscale).
        Expected to be a tissue region (not the full WSI background).

    Returns
    -------
    list of dict
        Contour regions returned by :func:`core.contours.find_contours`.
        Each dict typically contains keys such as ``'contour'``,
        ``'area'``, ``'bbox'``, and ``'centroid'``.  Only contours with
        ``area >= 200`` px are included.

    Notes
    -----
    **K-Means details** -- :func:`core.clustering.kmeans_segment` returns
    three objects:

    * ``segmented`` : image where each pixel is replaced by its cluster
      centroid value.
    * ``labels`` : per-pixel cluster index (0, 1, or 2).
    * ``centers`` : array of centroid values, shape ``(k, 1)``.

    We sort the centroids and pick the smallest (darkest) value to build
    the binary tumour mask.

    **Morphological cleanup chain** (in order):

    1. ``morph_close(kernel=5, iter=3)`` -- aggressive closing merges
       nearby tumour fragments and fills holes up to ~15 px across.
    2. ``morph_open(kernel=3, iter=1)`` -- gentle opening removes thin
       spurs and disconnects loosely connected regions.
    3. ``remove_small_objects(min_area=500)`` -- eliminates isolated
       specks that survived the opening step.

    The ``min_area`` parameter in ``find_contours`` (200 px) is
    intentionally lower than the small-object removal threshold (500 px)
    because contour area and binary-mask connected-component area can
    differ slightly due to border effects.

    Examples
    --------
    >>> from pipelines.histology.segmentation import segment_tumor_regions
    >>> regions = segment_tumor_regions(tile_rgb)
    >>> print(f'Found {len(regions)} tumour foci')
    Found 3 tumour foci
    """
    # Convert to single-channel float for K-Means.
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    # K-Means with k=3: dark (tumour), medium (stroma), light (background).
    segmented, labels, centers = kmeans_segment(gray, k=3)

    # Identify the darkest cluster centroid -- the tumour hypothesis.
    sorted_centers = np.sort(centers.flatten())
    darkest_cluster = sorted_centers[0]

    # Build binary mask: 1 where pixel belongs to the darkest cluster.
    tumor_mask = (segmented.astype(np.float32) == darkest_cluster).astype(np.uint8)

    # --- Morphological cleanup chain ---
    # Close: fill small holes and merge nearby fragments.
    tumor_mask = morph_close(tumor_mask, kernel_size=5, iterations=3)
    # Open: remove thin bridges and small protrusions.
    tumor_mask = morph_open(tumor_mask, kernel_size=3, iterations=1)
    # Remove tiny connected components (likely noise).
    tumor_mask = remove_small_objects(tumor_mask, min_area=500)

    # Extract contours from the cleaned mask.
    regions = find_contours(tumor_mask, min_area=200)
    return regions
