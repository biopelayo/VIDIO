import logging
import numpy as np

log = logging.getLogger(__name__)


def register_to_template(volume, template_path, current_spacing=(1.0, 1.0, 1.0)):
    try:
        import SimpleITK as sitk
    except ImportError:
        log.warning('SimpleITK not available. Skipping registration.')
        return volume

    moving = sitk.GetImageFromArray(volume)
    moving.SetSpacing(current_spacing)

    fixed = sitk.ReadImage(template_path)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    registration.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        final_transform = registration.Execute(fixed, moving)
        resampled = sitk.Resample(
            moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID(),
        )
        return sitk.GetArrayFromImage(resampled)
    except Exception as ex:
        log.error(f'Registration failed: {ex}')
        return volume
