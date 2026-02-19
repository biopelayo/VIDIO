import numpy as np

from core.intensity import hu_windowing, normalize_minmax


def preprocess_ct(volume, window_center=40, window_width=400):
    return hu_windowing(volume, window_center, window_width)


def preprocess_mri(volume):
    return normalize_minmax(volume.astype(np.float32), 0, 255)


def resample_isotropic(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    try:
        import SimpleITK as sitk

        sitk_img = sitk.GetImageFromArray(volume)
        sitk_img.SetSpacing(current_spacing)

        original_size = sitk_img.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, current_spacing, target_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())

        resampled = resampler.Execute(sitk_img)
        return sitk.GetArrayFromImage(resampled)
    except ImportError:
        from scipy.ndimage import zoom

        zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
        return zoom(volume, zoom_factors, order=3)
