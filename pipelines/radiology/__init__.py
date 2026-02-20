"""
VIDIO Radiology Pipeline
=========================

CT and MRI volume analysis pipeline for brain imaging, tumour
segmentation, atrophy measurement, and lesion characterisation.

Modalities
----------
- **CT (Computed Tomography)** : Hounsfield Unit-based tissue
  characterisation with windowing for soft tissue, bone, lung, or brain.
- **MRI (Magnetic Resonance Imaging)** : Multi-sequence brain imaging
  (T1, T2, FLAIR, DWI) for structural and functional analysis.
- **NIfTI volumes** : Pre-processed neuroimaging volumes for research
  workflows.

Processing Stages
-----------------
1. **Loading** : Format-aware reader dispatching (DICOM single-file,
   DICOM series via SimpleITK, or NIfTI via nibabel). Automatic
   RescaleSlope/Intercept application for DICOM.
2. **Preprocessing** : HU windowing (CT), min-max normalisation (MRI),
   isotropic voxel resampling via SimpleITK or scipy.ndimage fallback.
3. **Registration** : Affine registration to MNI152 standard space using
   SimpleITK (Mattes Mutual Information metric, gradient descent
   optimiser, multi-resolution pyramid).
4. **Segmentation** : 2D slice-by-slice Laplacian edge detection with
   morphological cleanup, or 3D threshold-based lesion extraction with
   per-slice morphological refinement. Optional 3D U-Net (MONAI).
5. **Analysis** : Brain parenchymal fraction (atrophy metric), lesion
   volumetrics (mm\ :sup:`3`, cm\ :sup:`3`), per-lesion intensity
   statistics.
6. **Severity Scoring** : Gradient x confidence -> CRITICAL / HIGH /
   MEDIUM / LOW with anatomical location context (slice index).

Clinical Context
----------------
- Alzheimer's disease brain atrophy quantification
- Glioblastoma tumour volumetrics
- Multiple sclerosis lesion load measurement
- Stroke lesion detection (DWI bright spots)
- CT trauma assessment

Key Dependencies
----------------
- pydicom : DICOM file reading
- SimpleITK : DICOM series, resampling, registration
- nibabel : NIfTI file reading
- MONAI : 3D U-Net segmentation models
- scipy.ndimage : Fallback resampling

See Also
--------
:mod:`pipelines.base` : Base pipeline class.
:mod:`core.intensity` : HU windowing, normalisation.
"""

from pipelines.radiology.pipeline import RadiologyPipeline

__all__ = ['RadiologyPipeline']
