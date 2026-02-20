"""
Radiology Analysis Utilities
==============================

High-level clinical analysis functions that combine segmentation masks
with volumetric and intensity measurements to produce clinically
meaningful metrics.

Functions
---------
analyze_brain_atrophy
    Compute brain parenchymal fraction (BPF) as a global atrophy
    indicator.
analyze_lesion
    Combine volumetric and intensity statistics for a lesion mask.
"""

import numpy as np

from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


def analyze_brain_atrophy(volume, brain_mask=None):
    """
    Quantify global brain atrophy via the Brain Parenchymal Fraction.

    The **Brain Parenchymal Fraction (BPF)** is defined as::

        BPF = brain_voxels / total_voxels

    where *brain_voxels* is the count of voxels classified as brain
    parenchyma and *total_voxels* is the total number of voxels in the
    volume (or mask).

    Two modes of operation:

    1. **With brain mask** (``brain_mask is not None``):
       Non-zero voxels in the mask are counted as brain parenchyma.
       The mask should be a binary output from a skull-stripping or
       tissue-segmentation step.

    2. **Without brain mask** (heuristic fallback):
       A simple intensity threshold at ``0.3 * mean(volume)`` is
       applied.  This is a rough approximation that assumes the
       volume is already preprocessed (normalised) and that
       background / CSF voxels have low intensities.

    Parameters
    ----------
    volume : numpy.ndarray
        3D brain volume, shape ``(D, H, W)``.  Used for threshold
        computation when no mask is supplied.
    brain_mask : numpy.ndarray or None, optional
        Binary mask where non-zero voxels represent brain parenchyma.
        Must have the same shape as *volume* if provided.

    Returns
    -------
    dict
        ``'brain_volume_voxels'`` : int
            Number of brain parenchyma voxels.
        ``'total_volume_voxels'`` : int
            Total number of voxels in the volume.
        ``'brain_parenchymal_fraction'`` : float
            BPF ratio in [0.0, 1.0].  Lower values indicate greater
            atrophy.

    Notes
    -----
    * Clinically, BPF < 0.80 is often considered suggestive of
      significant atrophy, though thresholds vary by age and study.
    * The heuristic threshold (``0.3 * mean``) is intentionally
      conservative (low) to avoid under-counting brain tissue.  It is
      **not** suitable for publication-quality atrophy quantification;
      use a proper skull-stripping mask for clinical workflows.
    * To convert voxel counts to physical volume (cm\ :sup:`3`), use
      :func:`pipelines.radiology.segmentation.measure_volume`.

    See Also
    --------
    analyze_lesion : Lesion-specific volumetrics and intensity stats.
    """
    if brain_mask is not None:
        brain_volume = np.sum(brain_mask > 0)
        total_volume = brain_mask.size
        brain_ratio = brain_volume / total_volume
    else:
        # Heuristic: 30% of global mean as brain/background boundary
        threshold = np.mean(volume) * 0.3
        brain_volume = np.sum(volume > threshold)
        total_volume = volume.size
        brain_ratio = brain_volume / total_volume

    return {
        'brain_volume_voxels': int(brain_volume),
        'total_volume_voxels': int(total_volume),
        'brain_parenchymal_fraction': float(brain_ratio),
    }


def analyze_lesion(volume, lesion_mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute combined volumetric and intensity statistics for a lesion.

    Merges two complementary analyses:

    1. **Volumetrics** via
       :func:`~pipelines.radiology.segmentation.measure_volume` --
       voxel count, mm\ :sup:`3`, cm\ :sup:`3`.
    2. **Intensity statistics** -- mean, standard deviation, min, and
       max of the original volume intensities within the lesion mask.

    If the lesion mask is empty (no non-zero voxels), only the
    volumetric dictionary is returned (all volumes will be zero) and
    the intensity fields are omitted.

    Parameters
    ----------
    volume : numpy.ndarray
        3D image volume from which intensity values are extracted.
        Should be in its original (or normalised) intensity range --
        *not* a binary mask.
    lesion_mask : numpy.ndarray
        Binary mask of the same shape as *volume*, where non-zero
        voxels mark lesion tissue.  Typically produced by
        :func:`~pipelines.radiology.segmentation.segment_lesions_3d`.
    voxel_spacing : tuple of float, optional
        Physical voxel dimensions in mm (default ``(1.0, 1.0, 1.0)``).

    Returns
    -------
    dict
        Always contains:

        - ``'n_voxels'`` (int) -- Number of lesion voxels.
        - ``'volume_mm3'`` (float) -- Lesion volume in mm\ :sup:`3`.
        - ``'volume_cm3'`` (float) -- Lesion volume in cm\ :sup:`3`.

        If the mask is non-empty, additionally contains:

        - ``'mean_intensity'`` (float) -- Mean voxel intensity.
        - ``'std_intensity'`` (float) -- Standard deviation of voxel
          intensities.
        - ``'min_intensity'`` (float) -- Minimum voxel intensity.
        - ``'max_intensity'`` (float) -- Maximum voxel intensity.

    Notes
    -----
    The intensity statistics are computed over the **flattened** set of
    lesion voxels, without spatial weighting.  For lesions spanning
    multiple tissue types (e.g. enhancing rim + necrotic core),
    consider sub-masking before calling this function.

    See Also
    --------
    pipelines.radiology.segmentation.segment_lesions_3d :
        Produces lesion masks suitable for this function.
    pipelines.radiology.segmentation.measure_volume :
        Standalone volumetric measurement.
    analyze_brain_atrophy : Global atrophy quantification.
    """
    from pipelines.radiology.segmentation import measure_volume

    vol_info = measure_volume(lesion_mask, voxel_spacing)

    # Extract intensity values within the lesion
    lesion_values = volume[lesion_mask > 0]
    if lesion_values.size == 0:
        return vol_info

    stats = {
        'mean_intensity': float(np.mean(lesion_values)),
        'std_intensity': float(np.std(lesion_values)),
        'min_intensity': float(np.min(lesion_values)),
        'max_intensity': float(np.max(lesion_values)),
    }
    vol_info.update(stats)
    return vol_info
