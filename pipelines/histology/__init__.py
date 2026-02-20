"""
VIDIO Histology Pipeline
========================

Whole-slide image (WSI) analysis pipeline for cancer detection, tissue
classification, and tumour grading in histopathology specimens.

Modalities
----------
- **H&E-stained WSI** : Standard Hematoxylin & Eosin stained tissue
  sections viewed at 20x or 40x magnification.
- **IHC-stained WSI** : Immunohistochemistry stained sections for
  specific protein marker analysis.
- **TCGA slides** : Publicly available cancer slide images from The
  Cancer Genome Atlas via the GDC Data Portal API.

Processing Stages
-----------------
1. **Preprocessing** : Tissue detection (Otsu thresholding on grayscale),
   Macenko stain normalisation to reduce inter-laboratory colour variance.
2. **Tiling** : Extract non-overlapping or overlapping tiles (default
   256x256 px) from gigapixel WSI. Tiles are filtered by tissue content
   ratio (default >=50 %% tissue pixels) to discard background.
3. **Segmentation** : K-Means colour clustering to identify darkly
   staining (tumour-suspicious) regions, morphological cleanup, contour
   extraction.
4. **Classification** : Per-tile analysis based on intensity statistics
   and z-score deviation from global tile population. Labels: tumour,
   normal, stroma, background.
5. **Grading** : Aggregate tile-level classifications into slide-level
   findings with severity scoring.

Clinical Context
----------------
- Breast cancer (TCGA-BRCA) tumour detection and grading
- Lung adenocarcinoma (TCGA-LUAD) subtype classification
- Glioblastoma (TCGA-GBM) histological analysis
- General tissue architecture assessment

Key Dependencies
----------------
- OpenSlide (``openslide-python``) : WSI file reading (SVS, NDPI, MRXS)
- OpenCV (``cv2``) : Image processing and K-Means
- PyTorch + torchvision : ResNet50 patch classifier (optional DL path)
- requests : GDC API interaction for TCGA slide retrieval

See Also
--------
:mod:`pipelines.base` : Base pipeline class.
:mod:`core.intensity` : Macenko stain normalisation.
:mod:`core.clustering` : K-Means segmentation.
"""
from pipelines.histology.pipeline import HistologyPipeline

__all__ = ['HistologyPipeline']
