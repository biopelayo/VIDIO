"""
Radiology Segmentation Utilities
==================================

Higher-level segmentation functions for brain tissue classification,
3D lesion extraction, and voxel-based volumetry.

These functions complement the 2D edge-based segmentation built into
:class:`~pipelines.radiology.pipeline.RadiologyPipeline` by providing:

* **K-Means tissue segmentation** -- unsupervised clustering into
  tissue classes (CSF, grey matter, white matter, background).
* **Threshold-based 3D lesion segmentation** -- global intensity
  thresholding with per-slice morphological refinement.
* **Volumetric measurement** -- conversion from voxel counts to
  physical volume (mm\ :sup:`3` and cm\ :sup:`3`).

Functions
---------
segment_brain_regions
    K-Means tissue classification on a 2D slice.
segment_lesions_3d
    Threshold-based volumetric lesion mask with morphological cleanup.
measure_volume
    Voxel-count to physical-volume conversion.
"""

import numpy as np

from core.clustering import kmeans_segment
from core.morphology import morph_close, morph_open, fill_holes, remove_small_objects
from core.contours import find_contours


def segment_brain_regions(slice_2d, k=4):
    """
    Segment a 2D brain slice into tissue classes using K-Means
    clustering.

    Partitions voxel intensities into *k* clusters.  With the default
    ``k=4``, the resulting clusters typically correspond to:

    ======= ===========================
    Cluster Tissue (approximate)
    ======= ===========================
    0       Background / air
    1       CSF (cerebrospinal fluid)
    2       Grey matter (GM)
    3       White matter (WM)
    ======= ===========================

    The exact cluster-to-tissue mapping depends on the intensity
    distribution of the input slice and is **not** explicitly labelled
    -- downstream consumers should sort clusters by centre intensity
    to infer tissue identity.

    Parameters
    ----------
    slice_2d : numpy.ndarray
        2D image of shape ``(H, W)``, any numeric dtype (will be cast
        to ``float32`` internally).
    k : int, optional
        Number of clusters (default ``4``).

    Returns
    -------
    segmented : numpy.ndarray
        Quantised image where each pixel takes the intensity value of
        its cluster centre.  Shape ``(H, W)``, dtype ``float32``.
    labels : numpy.ndarray
        Integer label map of shape ``(H, W)``.  Each pixel's value is
        its cluster index in ``[0, k)``.
    centers : numpy.ndarray
        1D array of cluster centre intensities, length *k*, sorted by
        the K-Means algorithm (not necessarily by magnitude).

    See Also
    --------
    core.clustering.kmeans_segment : Low-level K-Means implementation.
    """
    segmented, labels, centers = kmeans_segment(slice_2d.astype(np.float32), k=k)
    return segmented, labels, centers


def segment_lesions_3d(volume, threshold_factor=2.0):
    """
    Extract a 3D binary lesion mask using global thresholding and
    per-slice morphological refinement.

    The threshold is computed as::

        threshold = mean(volume) + threshold_factor * std(volume)

    Voxels above this threshold are considered potential lesion tissue.
    This approach is effective for **hyperintense** lesions (e.g. FLAIR
    white-matter lesions in MS, DWI bright spots in acute stroke,
    contrast-enhancing tumours) where lesion voxels are significantly
    brighter than the surrounding parenchyma.

    After initial thresholding, each axial slice undergoes:

    1. **Morphological closing** (3x3 kernel, 2 iterations) -- bridges
       small gaps within a lesion caused by noise or partial-volume
       effects.
    2. **Hole filling** -- ensures solid lesion masks without internal
       voids.
    3. **Small-object removal** (min 50 px) -- eliminates spurious
       bright spots (e.g. vessels, noise).

    Parameters
    ----------
    volume : numpy.ndarray
        3D image volume, shape ``(D, H, W)``, dtype ``float32`` or
        similar.  Should be intensity-normalised before calling.
    threshold_factor : float, optional
        Number of standard deviations above the mean to set the
        threshold (default ``2.0``).  Increase for higher specificity
        (fewer false positives); decrease for higher sensitivity.

    Returns
    -------
    numpy.ndarray
        Binary mask, shape ``(D, H, W)``, dtype ``uint8`` (0 or 1).

    Notes
    -----
    * The morphological cleanup is applied **per-slice** (2D) rather
      than in 3D, which is faster and avoids blurring across slices
      with different lesion distributions.
    * For very heterogeneous volumes (e.g. mixed tissue + air), the
      global mean/std threshold may be suboptimal.  Consider
      brain-masking the volume first.

    See Also
    --------
    measure_volume : Convert the output mask to physical volume.
    pipelines.radiology.analysis.analyze_lesion : Combined volumetrics
        and intensity statistics.
    """
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    threshold = mean_val + threshold_factor * std_val

    # Initial binary mask: voxels brighter than threshold
    lesion_mask = (volume > threshold).astype(np.uint8)

    # Per-slice morphological refinement
    for z in range(lesion_mask.shape[0]):
        slc = lesion_mask[z]
        slc = morph_close(slc, kernel_size=3, iterations=2)
        slc = fill_holes(slc)
        slc = remove_small_objects(slc, min_area=50)
        lesion_mask[z] = slc

    return lesion_mask


def measure_volume(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute the physical volume of a binary mask.

    Counts the number of non-zero voxels, multiplies by the per-voxel
    physical volume, and returns results in both mm\ :sup:`3` and
    cm\ :sup:`3`.

    The conversion follows::

        volume_mm3 = n_voxels * (spacing_x * spacing_y * spacing_z)
        volume_cm3 = volume_mm3 / 1000.0

    Parameters
    ----------
    mask : numpy.ndarray
        Binary (or label) mask where non-zero voxels are counted.
        Any shape is accepted.
    voxel_spacing : tuple of float, optional
        Physical size of a single voxel along each axis, in mm
        (default ``(1.0, 1.0, 1.0)``).  The order must match the
        array's dimension order.

    Returns
    -------
    dict
        ``'n_voxels'`` : int
            Total count of non-zero voxels.
        ``'volume_mm3'`` : float
            Total volume in cubic millimetres.
        ``'volume_cm3'`` : float
            Total volume in cubic centimetres (= mm\ :sup:`3` / 1000).

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.ones((10, 10, 10), dtype=np.uint8)
    >>> measure_volume(mask, voxel_spacing=(2.0, 0.5, 0.5))
    {'n_voxels': 1000, 'volume_mm3': 500.0, 'volume_cm3': 0.5}

    See Also
    --------
    segment_lesions_3d : Produces binary masks suitable for this
        function.
    """
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    n_voxels = np.sum(mask > 0)
    volume_mm3 = n_voxels * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0
    return {
        'n_voxels': int(n_voxels),
        'volume_mm3': float(volume_mm3),
        'volume_cm3': float(volume_cm3),
    }
