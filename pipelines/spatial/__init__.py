"""
VIDIO Spatial Transcriptomics Pipeline
========================================

Single-cell spatial transcriptomics analysis pipeline for gene
expression mapping, spatial clustering, and tissue characterisation.

Platforms
---------
- **10x Visium** : Barcoded spot-based spatial transcriptomics with
  H&E tissue image registration.
- **MERFISH** : Multiplexed error-robust FISH for single-molecule
  resolution.
- **Slide-seq** : Bead-based spatial transcriptomics.
- **10x Xenium** : In-situ single-cell spatial profiling.

Processing Stages
-----------------
1. **Loading** : Read H5AD files (AnnData format) via scanpy. H5AD
   contains the expression matrix (spots x genes), spatial coordinates,
   and optional tissue image.
2. **Quality Control** : Filter spots by minimum gene count (default
   >=200 genes/spot) and filter genes by minimum spot coverage (default
   >=10 spots/gene).
3. **Normalisation** : Library size normalisation to 10,000 counts per
   spot, log1p transformation, highly variable gene (HVG) selection
   (default top 2,000 genes via Seurat v3 method), PCA dimensionality
   reduction (50 components).
4. **Spatial Clustering** : k-NN graph construction on PCA space
   (30 PCs), Leiden community detection (resolution 0.8), spatial
   neighbour graph via squidpy for tissue-aware analysis.
5. **Statistical Analysis** : Moran's I spatial autocorrelation per
   gene, neighbourhood enrichment analysis, differential expression
   (Wilcoxon rank-sum) between spatial clusters.
6. **Anomaly Detection** : Z-score deviation of cluster-level gene
   expression from global population identifies spatially distinct
   transcriptional programmes.

Clinical Context
----------------
- Tumour microenvironment characterisation (immune infiltration zones)
- Spatially resolved gene expression in neurodegenerative diseases
- Tissue architecture mapping in developmental biology
- Ligand-receptor interaction spatial mapping

Key Dependencies
----------------
- scanpy : Single-cell analysis framework (QC, normalisation, clustering)
- squidpy : Spatial transcriptomics analysis (spatial graphs, Moran's I,
  neighbourhood enrichment)
- anndata : H5AD data structures
- matplotlib : Gene expression overlay visualisation

See Also
--------
:mod:`pipelines.base` : Base pipeline class.
:mod:`core.image_io` : H5AD file reading.
"""
from pipelines.spatial.pipeline import SpatialPipeline

__all__ = ['SpatialPipeline']
