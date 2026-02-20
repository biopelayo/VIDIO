"""
Spatial Transcriptomics Preprocessing
======================================

Standalone preprocessing functions for spatial transcriptomics data.

These functions mirror the preprocessing steps performed inside
:meth:`SpatialPipeline.preprocess` but are exposed as independent,
composable functions so that they can be used in notebooks, custom
scripts, or alternative pipeline configurations without instantiating
the full ``SpatialPipeline`` object.

Typical call order::

    adata = load_and_qc('sample.h5ad', min_genes=200, min_spots=10)
    adata = normalize(adata, target_sum=1e4)
    adata = select_hvg(adata, n_top_genes=2000)
    adata = run_pca(adata, n_comps=50)

All functions accept and return an :class:`anndata.AnnData` object,
modifying it **in-place** (scanpy convention) and also returning it
for method-chaining convenience.

Notes
-----
scanpy is imported **inside each function** rather than at module
level so that the module can be imported in environments where scanpy
is not installed (e.g., lightweight test harnesses).
"""
import logging

log = logging.getLogger(__name__)


def load_and_qc(h5ad_path, min_genes=200, min_spots=10):
    """
    Load an H5AD file and apply quality-control filtering.

    Reads the file into an :class:`anndata.AnnData` object, then
    removes low-quality spots and uninformative genes.

    QC Rationale
    ~~~~~~~~~~~~~
    - **Spot filtering** (``min_genes``): Spots with very few detected
      genes are likely empty droplets, debris, or tissue regions with
      degraded RNA. The default threshold of 200 genes follows the 10x
      Genomics Visium QC guidelines.
    - **Gene filtering** (``min_spots``): Genes detected in very few
      spots provide insufficient statistical power for downstream
      variance estimation and clustering. Removing them reduces the
      feature space and speeds up computation.

    Parameters
    ----------
    h5ad_path : str or pathlib.Path
        Absolute path to the ``.h5ad`` file. The file must conform to
        the AnnData on-disk format (HDF5-backed sparse or dense
        expression matrix).
    min_genes : int, optional
        Minimum number of genes a spot must express to be retained.
        Default is **200**.
    min_spots : int, optional
        Minimum number of spots in which a gene must be detected for
        the gene to be retained. Default is **10**.

    Returns
    -------
    anndata.AnnData
        The loaded and QC-filtered AnnData object. Filtered spots and
        genes are **removed** (not merely masked).

    Raises
    ------
    RuntimeError
        If scanpy is not installed.
    FileNotFoundError
        If *h5ad_path* does not exist (raised by ``sc.read_h5ad``).

    See Also
    --------
    normalize : Next step in the preprocessing chain.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise RuntimeError('scanpy required for spatial transcriptomics')

    adata = sc.read_h5ad(h5ad_path)
    log.info(f'Loaded: {adata.n_obs} spots, {adata.n_vars} genes')

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_spots)
    log.info(f'After QC: {adata.n_obs} spots, {adata.n_vars} genes')

    return adata


def normalize(adata, target_sum=1e4):
    """
    Library-size normalisation followed by log1p transformation.

    Two operations are applied sequentially:

    1. **normalize_total** -- Scales each spot so that its total count
       equals *target_sum*. This removes the confounding effect of
       variable sequencing depth (i.e., spots with more reads are not
       artificially "brighter"). The default ``target_sum=10,000``
       (counts-per-ten-thousand, or CP10K) is the scanpy/Seurat
       community standard.
    2. **log1p** -- Applies the natural-log transformation
       ``X = log(X + 1)``. Gene expression counts follow an
       approximately negative-binomial distribution whose variance
       scales with the mean. The log transform variance-stabilises the
       data, making it more suitable for linear methods (PCA) and
       distance-based clustering (Leiden on k-NN graph).

    Parameters
    ----------
    adata : anndata.AnnData
        QC-filtered AnnData object. ``adata.X`` should contain raw
        integer UMI counts (pre-normalisation).
    target_sum : float, optional
        Target library size for each spot. Default is **1e4** (10,000).

    Returns
    -------
    anndata.AnnData
        The normalised AnnData (modified in-place and returned).

    Notes
    -----
    - After this step, ``adata.X`` contains log-normalised expression
      values, **not** raw counts. If raw counts are needed later (e.g.,
      for differential expression with negative-binomial models), they
      should be saved to ``adata.raw`` beforehand.
    - The ``+1`` in log1p prevents ``log(0)`` for genes with zero
      counts in a given spot.

    See Also
    --------
    load_and_qc : Previous step in the preprocessing chain.
    select_hvg : Next step in the preprocessing chain.
    """
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_hvg(adata, n_top_genes=2000):
    """
    Select highly variable genes (HVGs) using the Seurat v3 method.

    Identifies genes that exhibit the highest biological variance across
    spots, marking them in ``adata.var['highly_variable']``. Downstream
    PCA (via ``run_pca``) automatically uses only HVGs when this
    annotation is present.

    Method Details (Seurat v3 / ``flavor='seurat_v3'``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The Seurat v3 variance-stabilising transformation (VST) models the
    mean-variance relationship of each gene across spots. Genes whose
    observed variance significantly exceeds the expected variance given
    their mean expression are selected. This is preferred over the
    simpler ``'seurat'`` or ``'cell_ranger'`` flavours because it is
    more robust to differences in mean expression level.

    Parameters
    ----------
    adata : anndata.AnnData
        Normalised AnnData object (post ``normalize()``). Must contain
        log-normalised expression in ``adata.X``.
    n_top_genes : int, optional
        Number of highly variable genes to select. Default is **2000**,
        the widely adopted community standard that balances biological
        signal against computational cost.

    Returns
    -------
    anndata.AnnData
        The AnnData with HVG annotations added (modified in-place):

        - ``adata.var['highly_variable']`` -- boolean mask
        - ``adata.var['highly_variable_rank']`` -- rank (1 = most
          variable)
        - ``adata.var['means']``, ``adata.var['variances']``,
          ``adata.var['variances_norm']`` -- VST statistics

    Notes
    -----
    ``subset=False`` is used deliberately: the full gene set is retained
    in ``adata.X`` so that per-gene analyses (e.g., spatial
    autocorrelation, differential expression) can still access all genes.
    PCA will automatically subset to HVGs via the boolean mask.

    See Also
    --------
    normalize : Previous step in the preprocessing chain.
    run_pca : Next step in the preprocessing chain.
    """
    import scanpy as sc

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', subset=False)
    return adata


def run_pca(adata, n_comps=50):
    """
    Perform PCA dimensionality reduction.

    Computes principal components on the HVG-subset of the expression
    matrix. The resulting low-dimensional embedding is stored in
    ``adata.obsm['X_pca']`` and is used as the input feature space for
    k-NN graph construction and Leiden clustering.

    Parameters
    ----------
    adata : anndata.AnnData
        Normalised AnnData with HVG annotations (i.e., after
        ``select_hvg()``). If ``adata.var['highly_variable']`` is
        present, scanpy's PCA will automatically restrict computation
        to those genes.
    n_comps : int, optional
        Number of principal components to compute. Default is **50**,
        the scanpy default. This typically captures >95 %% of
        biologically meaningful variance in Visium datasets while
        discarding technical noise in higher components.

    Returns
    -------
    anndata.AnnData
        The AnnData with PCA results added (modified in-place):

        - ``adata.obsm['X_pca']`` -- (n_spots, n_comps) embedding
          matrix.
        - ``adata.uns['pca']['variance']`` -- Explained variance per
          component.
        - ``adata.uns['pca']['variance_ratio']`` -- Fraction of total
          variance explained.
        - ``adata.varm['PCs']`` -- Gene loadings (n_genes, n_comps).

    Notes
    -----
    - The number of components used for downstream k-NN graph
      construction (``n_pcs=30`` in ``spatial_clustering``) is
      intentionally **less** than the 50 computed here. Computing more
      components than needed allows users to experiment with different
      ``n_pcs`` values without re-running PCA.
    - PCA is computed using the ARPACK solver by default (scanpy
      internals), which is efficient for sparse matrices.

    See Also
    --------
    select_hvg : Previous step in the preprocessing chain.
    pipelines.spatial.segmentation.spatial_clustering :
        Consumes ``adata.obsm['X_pca']`` for graph construction.
    """
    import scanpy as sc

    sc.pp.pca(adata, n_comps=n_comps)
    return adata
