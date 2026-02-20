"""
VIDIO Retinal Pipeline
======================

Fundus photography, OCT, and slit-lamp image analysis pipeline for
detecting retinal pathologies and crystalline lens abnormalities.

Modalities
----------
- **Fundus photography** : Colour retinal images for vessel analysis,
  optic disc/macula detection, and lesion identification (drusen,
  hemorrhages, exudates, microaneurysms).
- **Optical Coherence Tomography (OCT)** : Cross-sectional retinal
  layer imaging for thickness measurement and fluid detection.
- **Slit-lamp photography** : Anterior segment imaging for crystalline
  lens opacity and amyloid plaque candidate detection relevant to
  Alzheimer's disease biomarker research.

Processing Stages
-----------------
1. **Preprocessing** : Green channel extraction (fundus), CLAHE contrast
   enhancement, bilateral edge-preserving filtering.
2. **Segmentation** : Vessel tree extraction (Laplacian + morphology or
   U-Net), optic disc detection (Hough circles), macula localisation,
   lesion region extraction.
3. **Statistics** : Per-region intensity descriptors (mean, std, gradient,
   skewness, kurtosis) and geometric features (area, solidity,
   elongation, circularity, convexity).
4. **Anomaly Detection** : Z-score deviation from global statistics,
   adaptive K-means clustering, morphology-based classification of
   lesion type (bright lesions -> exudates/drusen, dark lesions ->
   hemorrhages/microaneurysms).
5. **Severity Scoring** : Gradient x confidence -> CRITICAL / HIGH /
   MEDIUM / LOW severity labels with 0-1 confidence scores.

Clinical Context
----------------
- Diabetic retinopathy (DR) staging
- Age-related macular degeneration (AMD) drusen quantification
- Glaucoma cup-to-disc ratio estimation
- Alzheimer's lens biomarker screening (supranuclear cataracts,
  amyloid-beta deposits in the crystalline lens)

Key Dependencies
----------------
- OpenCV (``cv2``) : Core image operations
- NumPy : Array manipulations
- PyTorch + torchvision : EfficientNet B4 classification model
- MONAI : U-Net vessel segmentation model (optional fallback to basic U-Net)

See Also
--------
:mod:`pipelines.base` : Base pipeline class with the 5-stage pattern.
:mod:`core` : Shared image processing building blocks.
"""
from pipelines.retinal.pipeline import RetinalPipeline

__all__ = ['RetinalPipeline']
