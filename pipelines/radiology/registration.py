"""
Affine Registration to Standard Template
==========================================

Provides :func:`register_to_template`, which aligns a subject volume
to a reference template (typically MNI152) using an intensity-based
affine registration implemented in SimpleITK.

Registration is a critical prerequisite for group-level analyses,
atlas-based ROI extraction, and any comparison across subjects or
time-points.

Algorithm Summary
-----------------
1. **Metric** -- Mattes Mutual Information with 50 histogram bins.
   MI is the standard choice for inter-modal and intra-modal
   registration because it does not assume a linear relationship
   between intensities.
2. **Optimiser** -- Gradient Descent (learning rate 1.0, max 100
   iterations, convergence threshold 1e-6 over a 10-sample window).
3. **Transform model** -- 3D Affine (12 DOF: rotation, translation,
   scaling, shearing).  Initialised with a geometry-based centre
   alignment (``CenteredTransformInitializer.GEOMETRY``).
4. **Multi-resolution pyramid** -- Shrink factors ``[4, 2, 1]`` with
   Gaussian smoothing sigmas ``[2, 1, 0]`` (mm).  Coarse-to-fine
   registration avoids local minima and speeds convergence.
5. **Interpolation** -- Linear (``sitkLinear``) for the final
   resampling step.

Graceful Fallback
-----------------
* If SimpleITK is not installed, the function logs a warning and
  returns the **unregistered** input volume.
* If the registration optimiser diverges or any SimpleITK error
  occurs, the original volume is returned and the exception is logged.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


def register_to_template(volume, template_path, current_spacing=(1.0, 1.0, 1.0)):
    """
    Register a subject volume to a reference template via affine
    transformation.

    Uses SimpleITK's ``ImageRegistrationMethod`` with Mattes Mutual
    Information and a multi-resolution gradient-descent optimiser to
    find the best 12-DOF affine alignment between the subject
    (*moving*) image and a template (*fixed*) image stored on disk.

    Parameters
    ----------
    volume : numpy.ndarray
        Subject volume to register, shape ``(D, H, W)``.  Should
        already be preprocessed (e.g. intensity-normalised) for best
        registration accuracy.
    template_path : str
        File-system path to the fixed reference image.  Any format
        readable by ``SimpleITK.ReadImage`` is accepted (NIfTI, NRRD,
        MHA, DICOM series directory, etc.).  For brain imaging, this
        is typically the **MNI152 1 mm** template.
    current_spacing : tuple of float, optional
        Voxel spacing ``(sz, sy, sx)`` of the input volume in mm
        (default ``(1.0, 1.0, 1.0)``).  Must match the actual
        physical spacing of *volume*; incorrect values will cause
        misregistration.

    Returns
    -------
    numpy.ndarray
        Registered volume resampled into the template's physical
        space.  Shape matches the template grid.  If registration
        fails or SimpleITK is unavailable, the **original volume** is
        returned unchanged.

    Warns
    -----
    Logs a ``WARNING`` if SimpleITK is not installed (registration
    is silently skipped).

    Notes
    -----
    * The number of histogram bins (50) is a standard default that
      works well for brain images.  Very low-resolution images may
      benefit from fewer bins.
    * ``inPlace=False`` on ``SetInitialTransform`` ensures the
      initial transform is copied, avoiding unintended side-effects.
    * The smoothing sigmas are specified in **physical units** (mm)
      rather than voxel units, ensuring consistent behaviour across
      different voxel spacings.
    * The fill value for out-of-field-of-view voxels is ``0.0``.

    See Also
    --------
    SimpleITK.ImageRegistrationMethod : Full registration API.
    pipelines.radiology.preprocessing.resample_isotropic :
        Often used before registration to standardise voxel spacing.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        log.warning('SimpleITK not available. Skipping registration.')
        return volume

    # -- Build SimpleITK images ------------------------------------------------
    moving = sitk.GetImageFromArray(volume)
    moving.SetSpacing(current_spacing)

    fixed = sitk.ReadImage(template_path)

    # -- Configure registration method ----------------------------------------
    registration = sitk.ImageRegistrationMethod()

    # Mattes Mutual Information: robust to non-linear intensity relationships
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Gradient descent with convergence monitoring
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    registration.SetInterpolator(sitk.sitkLinear)

    # -- Initial transform: geometry-based centre alignment -------------------
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # -- Multi-resolution pyramid: coarse (4x) -> medium (2x) -> fine (1x) ---
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # -- Execute and resample -------------------------------------------------
    try:
        final_transform = registration.Execute(fixed, moving)
        resampled = sitk.Resample(
            moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID(),
        )
        return sitk.GetArrayFromImage(resampled)
    except Exception as ex:
        log.error(f'Registration failed: {ex}')
        return volume
