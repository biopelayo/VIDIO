"""
Spatial Transcriptomics Segmentation
======================================

Functions for clustering spots into transcriptionally and spatially
coherent regions, and for segmenting tissue morphology from H&E images.

This module provides three complementary segmentation strategies:

1. **Transcriptomic clustering** (``spatial_clustering``) -- Groups
   spots by gene expression similarity using Leiden community detection
   on a k-NN graph in PCA space.
2. **Spatial graph construction** (``build_spatial_graph``) -- Builds a
   physical-proximity graph via Delaunay triangulation for downstream
   spatial statistics (Moran's I, neighbourhood enrichment).
3. **Tissue image segmentation** (``segment_tissue_image``) -- Detects
   tissue regions from the H&E histology image using Laplacian edge
   detection and morphological operations (independent of expression
   data).
"""
import logging
import numpy as np

log = logging.getLogger(__name__)


def spatial_clustering(adata, resolution=0.8):
    """
    Cluster spots by transcriptomic similarity using Leiden on a k-NN graph.

    Workflow
    ~~~~~~~~
    1. **k-NN graph** -- ``sc.pp.neighbors`` builds a k-nearest-neighbour
       graph in PCA space using the top 30 principal components
       (``n_pcs=30``). The default number of neighbours (``n_neighbors=15``,
       scanpy default) is used. Each spot is connected to its 15 most
       transcriptionally similar spots, regardless of their physical
       location in the tissue.
    2. **Leiden clustering** -- ``sc.tl.leiden`` partitions the k-NN
       graph into communities using the Leiden algorithm (an improved
       version of Louvain). The *resolution* parameter controls the
       granularity: higher values produce more clusters, lower values
       produce fewer.

    Parameters
    ----------
    adata : anndata.AnnData
        Preprocessed AnnData with PCA embedding in
        ``adata.obsm['X_pca']`` (i.e., after ``run_pca()``).
    resolution : float, optional
        Leiden resolution parameter. Default is **0.8**, which is the
        scanpy default and typically yields 5--15 clusters for Visium
        datasets with ~3,000--5,000 spots.

        Guideline for tuning:

        - ``0.2 -- 0.5`` : coarse clusters (3--7), broad tissue zones.
        - ``0.5 -- 1.0`` : standard (5--15), typical cell-type groups.
        - ``1.0 -- 2.0`` : fine-grained (15--30+), sub-populations.

    Returns
    -------
    anndata.AnnData
        The AnnData with clustering results added (modified in-place):

        - ``adata.obs['spatial_cluster']`` -- Categorical series with
          cluster labels (string integers: ``'0'``, ``'1'``, ...).
        - ``adata.obsp['connectivities']`` -- k-NN graph sparse matrix.
        - ``adata.obsp['distances']`` -- k-NN distance matrix.

    Notes
    -----
    - Only 30 of the 50 available PCs are used for neighbour
      construction. The top 30 PCs capture the dominant biological
      signal; higher components increasingly reflect technical noise.
    - Cluster labels are **not** spatially aware -- two distant tissue
      regions may share the same label if they are transcriptionally
      similar. Use ``build_spatial_graph`` and neighbourhood enrichment
      for spatial co-localisation analysis.

    See Also
    --------
    build_spatial_graph : Adds a spatial proximity graph.
    pipelines.spatial.analysis.neighborhood_enrichment :
        Tests spatial co-localisation of clusters.
    """
    import scanpy as sc

    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.leiden(adata, resolution=resolution, key_added='spatial_cluster')
    log.info(f'Found {adata.obs["spatial_cluster"].nunique()} clusters')
    return adata


def build_spatial_graph(adata):
    """
    Build a spatial proximity graph from tissue coordinates.

    Uses squidpy's ``spatial_neighbors`` to construct a Delaunay
    triangulation-based graph from the spatial coordinates stored in
    ``adata.obsm['spatial']``. In this graph, spots are connected to
    their physically adjacent neighbours in tissue space (as opposed
    to the k-NN graph in PCA space built by ``spatial_clustering``).

    The spatial graph is stored in ``adata.obsp`` and consumed by
    squidpy's spatial statistics functions:

    - :func:`pipelines.spatial.analysis.spatial_autocorrelation` --
      Moran's I requires a spatial connectivity matrix to define
      "neighbours" for autocorrelation computation.
    - :func:`pipelines.spatial.analysis.neighborhood_enrichment` --
      Tests whether cluster pairs are spatially co-localised using
      the spatial neighbour graph.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with spatial coordinates in
        ``adata.obsm['spatial']`` (a 2D array of shape
        ``(n_spots, 2)``). This is automatically populated when
        reading 10x Visium H5AD files.

    Returns
    -------
    anndata.AnnData
        The AnnData with spatial graph added (modified in-place):

        - ``adata.obsp['spatial_connectivities']`` -- Binary adjacency
          matrix from Delaunay triangulation.
        - ``adata.obsp['spatial_distances']`` -- Euclidean distances
          between connected spots.

    Warns
    -----
    Logs a warning if squidpy is not installed. The function still
    returns the unmodified ``adata`` so that the pipeline can continue
    without spatial statistics.

    Notes
    -----
    - Delaunay triangulation is the default method in
      ``sq.gr.spatial_neighbors``. It connects each spot to its natural
      neighbours (those sharing a Voronoi boundary), which for the
      regular hexagonal Visium grid produces a 6-connected graph.
    - For irregularly spaced platforms (MERFISH, Slide-seq), the
      Delaunay graph adapts to the local point density.
    """
    try:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata)
        log.info('Spatial neighbor graph constructed')
    except ImportError:
        log.warning('squidpy not available for spatial graph')
    return adata


def segment_tissue_image(tissue_img):
    """
    Segment tissue regions from an H&E histology image.

    This function operates on the **tissue image** (not expression data)
    and identifies distinct tissue regions using classical image
    processing: Laplacian edge detection followed by morphological
    cleanup and contour extraction.

    Algorithm
    ~~~~~~~~~
    1. **Grayscale conversion** -- If the input is a colour image
       (H x W x C), convert to grayscale by averaging channels. This
       simple approach works well for H&E stains where tissue is
       uniformly pink/purple against a white background.
    2. **Laplacian edge detection** -- ``laplacian_edges`` computes the
       second spatial derivative (Laplacian) of the grayscale image.
       Tissue boundaries produce strong edge responses.
    3. **Thresholding** -- Pixels where ``|edge_response| < std(edges)``
       are classified as tissue (i.e., smooth, non-edge regions).
       Pixels above the threshold are background or boundary.
    4. **Morphological closing** -- A 5x5 kernel with 3 iterations fills
       small holes and gaps in the tissue mask (e.g., from nuclei or
       staining artefacts).
    5. **Small object removal** -- Connected components smaller than
       1,000 pixels are removed to eliminate noise and debris.
    6. **Contour extraction** -- ``find_contours`` traces the boundary
       of each remaining connected component, returning only regions
       with area >= 500 pixels.

    Parameters
    ----------
    tissue_img : numpy.ndarray
        H&E tissue image as a NumPy array. May be:

        - **Colour**: shape ``(H, W, C)`` where C is 3 (RGB) or 4
          (RGBA).
        - **Grayscale**: shape ``(H, W)``.

        Expected dtype: uint8 or float32.

    Returns
    -------
    list of dict
        Tissue region contours as returned by
        :func:`core.contours.find_contours`. Each dict typically
        contains:

        - ``contour`` -- (N, 2) array of boundary pixel coordinates.
        - ``area`` -- Area of the region in pixels.
        - ``bbox`` -- Bounding box ``(x, y, w, h)``.

    Notes
    -----
    - This function is independent of expression data and can be used
      for purely morphological analysis of tissue architecture.
    - The Laplacian threshold (1 standard deviation) is a heuristic
      that works well for typical 10x Visium H&E images at 20x
      magnification.
    - For heavily stained or artefact-rich images, the morphological
      parameters (``kernel_size``, ``iterations``, ``min_area``) may
      need adjustment.

    See Also
    --------
    core.edges.laplacian_edges : Laplacian edge detection.
    core.morphology.morph_close : Morphological closing operation.
    core.contours.find_contours : Contour extraction from binary masks.
    """
    from core.edges import laplacian_edges
    from core.morphology import morph_close, remove_small_objects
    from core.contours import find_contours

    # --- Convert colour image to grayscale if needed ---
    if len(tissue_img.shape) == 3:
        gray = np.mean(tissue_img, axis=2).astype(np.float32)
    else:
        gray = tissue_img.astype(np.float32)

    # --- Edge detection and tissue mask construction ---
    edge_map, stats = laplacian_edges(gray)
    tissue_mask = (np.abs(edge_map) < stats['std']).astype(np.uint8)

    # --- Morphological cleanup ---
    tissue_mask = morph_close(tissue_mask, kernel_size=5, iterations=3)
    tissue_mask = remove_small_objects(tissue_mask, min_area=1000)

    # --- Extract contours for each tissue region ---
    regions = find_contours(tissue_mask, min_area=500)
    return regions
