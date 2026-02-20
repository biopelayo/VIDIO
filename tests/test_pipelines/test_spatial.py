"""
Tests for :mod:`pipelines.spatial`
====================================

Integration tests for the spatial transcriptomics pipeline.
Tests use synthetic data structures that mimic scanpy/anndata
objects when possible.  Full integration tests require scanpy.
"""

import numpy as np
import pytest


# ------------------------------------------------------------------ #
#  Check optional dependencies                                         #
# ------------------------------------------------------------------ #

try:
    import scanpy as sc
    import anndata as ad
    _HAS_SCANPY = True
except ImportError:
    _HAS_SCANPY = False

try:
    import squidpy as sq
    _HAS_SQUIDPY = True
except ImportError:
    _HAS_SQUIDPY = False


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def synthetic_adata():
    """Create a synthetic AnnData object mimicking Visium data.

    Returns None if scanpy/anndata not installed.
    """
    if not _HAS_SCANPY:
        pytest.skip("scanpy not available")

    rng = np.random.default_rng(42)

    # 200 spots, 500 genes
    n_spots = 200
    n_genes = 500
    X = rng.negative_binomial(n=5, p=0.3, size=(n_spots, n_genes)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Spot_{i}" for i in range(n_spots)]

    # Add spatial coordinates
    coords = rng.uniform(0, 1000, size=(n_spots, 2))
    adata.obsm['spatial'] = coords

    return adata


# ------------------------------------------------------------------ #
#  Preprocessing Tests                                                 #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy not available")
class TestSpatialPreprocessing:
    def test_load_and_qc(self, synthetic_adata):
        from pipelines.spatial.preprocessing import normalize, select_hvg, run_pca

        # Normalize
        adata = normalize(synthetic_adata.copy())
        assert adata.X.shape == synthetic_adata.X.shape

    def test_hvg_selection(self, synthetic_adata):
        from pipelines.spatial.preprocessing import normalize, select_hvg

        adata = normalize(synthetic_adata.copy())
        adata = select_hvg(adata, n_top_genes=100)
        assert 'highly_variable' in adata.var.columns

    def test_pca(self, synthetic_adata):
        from pipelines.spatial.preprocessing import normalize, select_hvg, run_pca

        adata = normalize(synthetic_adata.copy())
        adata = select_hvg(adata, n_top_genes=100)
        adata = run_pca(adata, n_comps=20)
        assert 'X_pca' in adata.obsm
        assert adata.obsm['X_pca'].shape[1] == 20


# ------------------------------------------------------------------ #
#  Segmentation Tests                                                  #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy not available")
class TestSpatialSegmentation:
    def test_spatial_clustering(self, synthetic_adata):
        from pipelines.spatial.preprocessing import normalize, select_hvg, run_pca
        from pipelines.spatial.segmentation import spatial_clustering

        adata = normalize(synthetic_adata.copy())
        adata = select_hvg(adata, n_top_genes=100)
        adata = run_pca(adata, n_comps=20)
        adata = spatial_clustering(adata, resolution=0.5)
        assert 'spatial_cluster' in adata.obs.columns
        n_clusters = adata.obs['spatial_cluster'].nunique()
        assert n_clusters >= 2

    def test_tissue_segmentation(self):
        """Test tissue region segmentation from H&E image."""
        from pipelines.spatial.segmentation import segment_tissue_image

        # Create a simple tissue-like image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = [180, 160, 200]  # tissue-like region
        regions = segment_tissue_image(img)
        assert isinstance(regions, list)


# ------------------------------------------------------------------ #
#  Analysis Tests                                                      #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy not available")
class TestSpatialAnalysis:
    def test_differential_expression(self, synthetic_adata):
        from pipelines.spatial.preprocessing import normalize, select_hvg, run_pca
        from pipelines.spatial.segmentation import spatial_clustering
        from pipelines.spatial.analysis import differential_expression

        adata = normalize(synthetic_adata.copy())
        adata = select_hvg(adata, n_top_genes=100)
        adata = run_pca(adata, n_comps=20)
        adata = spatial_clustering(adata, resolution=0.5)
        result = differential_expression(adata)
        assert result is not None


# ------------------------------------------------------------------ #
#  Visualisation Tests                                                 #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy not available")
class TestSpatialVisualization:
    def test_overlay_expression(self, synthetic_adata):
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend for testing
        except ImportError:
            pytest.skip("matplotlib not available")

        from pipelines.spatial.visualization import overlay_expression

        fig = overlay_expression(None, synthetic_adata, 'Gene_0')
        if fig is not None:
            assert hasattr(fig, 'savefig')
