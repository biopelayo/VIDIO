"""
Radiology Preprocessing Utilities
==================================

Standalone helper functions for CT and MRI volume preprocessing.
These are thin, composable wrappers around :mod:`core.intensity` that
encode clinically meaningful defaults (e.g. soft-tissue windowing for
CT).

They can be used directly when more granular control is needed than the
monolithic :meth:`RadiologyPipeline.preprocess` method provides -- for
instance when a research workflow needs to resample before windowing.

Functions
---------
preprocess_ct
    Apply Hounsfield-Unit windowing to a CT volume.
preprocess_mri
    Min-max normalise an MRI volume to [0, 255].
resample_isotropic
    Resample a volume to isotropic voxel spacing.
"""

import numpy as np

from core.intensity import hu_windowing, normalize_minmax


def preprocess_ct(volume, window_center=40, window_width=400):
    """
    Apply Hounsfield-Unit (HU) windowing to a CT volume.

    HU windowing maps a narrow intensity range to the displayable /
    processable [0, 1] range, suppressing irrelevant tissue densities.

    The **default window** (centre=40, width=400) corresponds to the
    standard **soft-tissue** window, which spans HU [--160, +240].
    Common alternatives:

    ======== ======= =======
    Tissue   Centre  Width
    ======== ======= =======
    Brain      40      80
    Subdural   75     215
    Bone      400    1800
    Lung     -600    1500
    ======== ======= =======

    Parameters
    ----------
    volume : numpy.ndarray
        CT volume in Hounsfield Units (after ``RescaleSlope`` and
        ``RescaleIntercept`` have been applied).  Any shape is
        accepted; windowing operates element-wise.
    window_center : float, optional
        Centre of the HU window (default ``40``).
    window_width : float, optional
        Width of the HU window (default ``400``).

    Returns
    -------
    numpy.ndarray
        Windowed volume with values in [0, 1].  Same shape and dtype
        as input (typically ``float32`` or ``float64`` depending on the
        input).

    See Also
    --------
    core.intensity.hu_windowing : Low-level implementation.
    """
    return hu_windowing(volume, window_center, window_width)


def preprocess_mri(volume):
    """
    Normalise an MRI volume to the [0, 255] intensity range.

    Unlike CT, MRI voxel intensities are **arbitrary** -- they depend
    on scanner gain, coil sensitivity, and sequence parameters.
    Min-max normalisation maps the observed dynamic range to [0, 255],
    making downstream processing (edge detection, thresholding)
    independent of the original intensity scale.

    The volume is first cast to ``float32`` to prevent integer overflow
    during the normalisation arithmetic.

    Parameters
    ----------
    volume : numpy.ndarray
        MRI volume of any shape and numeric dtype.

    Returns
    -------
    numpy.ndarray
        Normalised volume with values in [0.0, 255.0], dtype
        ``float32``.

    Notes
    -----
    The 0-255 range (rather than 0-1) is chosen for compatibility with
    OpenCV-based morphological and contour utilities in ``core.*``
    that expect 8-bit-equivalent ranges.

    See Also
    --------
    core.intensity.normalize_minmax : Low-level implementation.
    """
    return normalize_minmax(volume.astype(np.float32), 0, 255)


def resample_isotropic(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """
    Resample a volume to isotropic (or arbitrary target) voxel spacing.

    Many volumetric analyses (registration, atlas comparison, volume
    measurement) require isotropic voxels.  Clinical CT and MRI scans
    often have anisotropic spacing -- e.g. 0.5 x 0.5 x 2.0 mm for a
    typical brain MRI.

    Two back-ends are attempted in order of preference:

    1. **SimpleITK** (preferred) -- Uses ``ResampleImageFilter`` with
       **BSpline** interpolation (``sitkBSpline``).  BSpline is chosen
       over linear because it preserves fine anatomical detail and
       produces smoother results at the cost of modest additional
       computation.  The output inherits the original image's
       direction cosines and origin.
    2. **scipy.ndimage** (fallback) -- If SimpleITK is not installed,
       ``scipy.ndimage.zoom`` with ``order=3`` (cubic) is used.  This
       path does not preserve DICOM spatial metadata but produces
       numerically similar voxel values.

    Parameters
    ----------
    volume : numpy.ndarray
        3D array to resample.  Shape ``(D, H, W)`` or ``(H, W, D)``
        -- the function is agnostic to axis ordering as long as
        *current_spacing* matches the array axes.
    current_spacing : tuple of float
        Voxel spacing of the input volume in mm, ordered to match the
        array dimensions -- e.g. ``(slice_thickness, row_spacing,
        col_spacing)``.
    target_spacing : tuple of float, optional
        Desired output voxel spacing in mm (default ``(1.0, 1.0,
        1.0)`` for 1 mm isotropic).

    Returns
    -------
    numpy.ndarray
        Resampled volume.  Shape will differ from the input if the
        spacing ratios are not unity.

    Raises
    ------
    ImportError
        Only if **both** SimpleITK and scipy are unavailable (highly
        unlikely in practice).

    Notes
    -----
    * New output dimensions are computed as
      ``round(original_size * current_spacing / target_spacing)``.
    * The BSpline interpolator can produce slight ringing near sharp
      edges; for binary masks, consider using nearest-neighbour
      interpolation instead (not currently exposed by this wrapper).

    See Also
    --------
    SimpleITK.ResampleImageFilter : Underlying SimpleITK resampler.
    scipy.ndimage.zoom : Fallback resampler.
    """
    try:
        import SimpleITK as sitk

        # -- SimpleITK preferred path -----------------------------------------
        sitk_img = sitk.GetImageFromArray(volume)
        sitk_img.SetSpacing(current_spacing)

        # Compute new grid size to honour the target spacing.
        original_size = sitk_img.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, current_spacing, target_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkBSpline)       # cubic B-spline
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())

        resampled = resampler.Execute(sitk_img)
        return sitk.GetArrayFromImage(resampled)
    except ImportError:
        # -- scipy fallback (no spatial metadata) -----------------------------
        from scipy.ndimage import zoom

        zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
        return zoom(volume, zoom_factors, order=3)         # cubic interpolation
