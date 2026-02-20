"""
Spatial Transcriptomics Statistical Analysis
==============================================

Functions for quantifying spatial patterns in gene expression data.

This module provides three core spatial analyses:

1. **Spatial autocorrelation** (``spatial_autocorrelation``) -- Moran's
   I statistic to identify genes whose expression is spatially
   clustered or spatially dispersed.
2. **Neighbourhood enrichment** (``neighborhood_enrichment``) --
   Permutation test to determine whether pairs of spatial clusters
   co-localise more (or less) than expected by chance.
3. **Differential expression** (``differential_expression``) --
   Wilcoxon rank-sum test to find marker genes that distinguish each
   spatial cluster from all others.

All functions require a spatial neighbour graph in
``adata.obsp['spatial_connectivities']`` (built by
:func:`pipelines.spatial.segmentation.build_spatial_graph`) for the
spatial analyses, and cluster labels in ``adata.obs`` for the cluster-
based analyses.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)


def spatial_autocorrelation(adata, genes=None):
    """
    Compute Moran's I spatial autocorrelation for gene expression.

    Moran's I quantifies the degree to which a gene's expression is
    spatially structured (i.e., whether nearby spots have similar
    expression levels).

    Interpretation of Moran's I
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    +----------------+-------------------------------------------------+
    | Moran's I      | Meaning                                         |
    +================+=================================================+
    | Close to +1    | **Positive spatial autocorrelation**: high-      |
    |                | expression spots cluster together. The gene is   |
    |                | expressed in a spatially coherent domain (e.g.,  |
    |                | a tumour region or a specific tissue layer).     |
    +----------------+-------------------------------------------------+
    | Close to 0     | **No spatial pattern**: expression is randomly   |
    |                | distributed across the tissue.                   |
    +----------------+-------------------------------------------------+
    | Close to -1    | **Negative spatial autocorrelation (dispersed)**:|
    |                | high-expression spots are surrounded by low-     |
    |                | expression spots (checkerboard pattern). Rare in |
    |                | practice for transcriptomics data.               |
    +----------------+-------------------------------------------------+

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with a spatial neighbour graph in
        ``adata.obsp['spatial_connectivities']`` (see
        :func:`~pipelines.spatial.segmentation.build_spatial_graph`).
    genes : list of str, optional
        Gene names to test. If ``None``, the function automatically
        selects:

        - The first 100 highly variable genes (if HVG annotation
          exists in ``adata.var['highly_variable']``), **or**
        - The first 100 genes in ``adata.var_names`` as a fallback.

        The cap of 100 genes balances statistical coverage against
        computation time (Moran's I is O(n_spots^2) per gene).

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame indexed by gene name with columns:

        - ``I`` -- Moran's I statistic.
        - ``pval_norm`` -- p-value under normality assumption.
        - ``var_norm`` -- Variance under normality assumption.
        - ``pval_z_sim`` -- p-value from permutation simulation.

        Returns ``None`` if squidpy is not installed.

    Notes
    -----
    - The result is also stored in ``adata.uns['moranI']`` by squidpy,
      so it persists on the AnnData object for downstream use.
    - For large datasets (>10,000 spots), consider subsetting the
      spatial graph or reducing the gene list to avoid long runtimes.

    See Also
    --------
    neighborhood_enrichment : Cluster-level spatial co-localisation.
    pipelines.spatial.segmentation.build_spatial_graph :
        Required prerequisite for building the spatial connectivity
        matrix.
    """
    try:
        import squidpy as sq
    except ImportError:
        log.error('squidpy required for spatial autocorrelation')
        return None

    # Default to top 100 HVGs or first 100 genes
    if genes is None:
        if 'highly_variable' in adata.var.columns:
            genes = adata.var_names[adata.var['highly_variable']].tolist()[:100]
        else:
            genes = adata.var_names[:100].tolist()

    sq.gr.spatial_autocorr(adata, mode='moran', genes=genes)
    log.info(f'Computed Moran I for {len(genes)} genes')
    return adata.uns.get('moranI', None)


def neighborhood_enrichment(adata, cluster_key='spatial_cluster'):
    """
    Test whether spatial clusters co-localise more than expected by chance.

    Performs a permutation-based neighbourhood enrichment analysis
    (squidpy). For each pair of clusters (A, B), it counts how many
    times a spot from cluster A is spatially adjacent to a spot from
    cluster B in the real data, then compares this count to a null
    distribution generated by randomly permuting cluster labels across
    spots (default 1,000 permutations).

    Interpretation
    ~~~~~~~~~~~~~~
    - **Enrichment z-score >> 0** : Clusters A and B are spatially
      co-localised (adjacent more often than expected). Biologically,
      this may indicate cell-cell interactions, paracrine signalling,
      or shared tissue microenvironments.
    - **Enrichment z-score << 0** : Clusters A and B are spatially
      segregated (adjacent less often than expected). May indicate
      mutually exclusive tissue domains.
    - **z-score ~ 0** : No spatial preference; clusters are randomly
      interleaved.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with:

        - Cluster labels in ``adata.obs[cluster_key]``.
        - Spatial neighbour graph in
          ``adata.obsp['spatial_connectivities']`` (from
          :func:`~pipelines.spatial.segmentation.build_spatial_graph`).
    cluster_key : str, optional
        Column in ``adata.obs`` containing cluster labels. Default is
        ``'spatial_cluster'`` (as set by ``spatial_clustering``).

    Returns
    -------
    dict or None
        A dictionary with keys:

        - ``zscore`` -- (n_clusters, n_clusters) numpy array of
          enrichment z-scores for each cluster pair.
        - ``count`` -- (n_clusters, n_clusters) numpy array of observed
          neighbour counts.

        Returns ``None`` if squidpy is not installed.

    Notes
    -----
    - The result is also stored in
      ``adata.uns['{cluster_key}_nhood_enrichment']`` by squidpy.
    - The permutation test is non-parametric and makes no distributional
      assumptions, but can be slow for very large datasets.
    - The z-score matrix is symmetric for undirected spatial graphs.

    See Also
    --------
    spatial_autocorrelation : Gene-level spatial pattern detection.
    differential_expression : Identify marker genes per cluster.
    """
    try:
        import squidpy as sq
    except ImportError:
        log.error('squidpy required for neighborhood enrichment')
        return None

    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
    result = adata.uns.get(f'{cluster_key}_nhood_enrichment', None)
    log.info('Neighborhood enrichment computed')
    return result


def differential_expression(adata, cluster_key='spatial_cluster'):
    """
    Identify marker genes for each spatial cluster via Wilcoxon rank-sum.

    For each cluster, performs a one-vs-rest Wilcoxon rank-sum test
    (Mann-Whitney U) to find genes that are significantly up- or
    down-regulated in that cluster compared to all other spots.

    Why Wilcoxon Rank-Sum?
    ~~~~~~~~~~~~~~~~~~~~~~
    The Wilcoxon rank-sum test is a non-parametric test that does **not**
    assume normality of gene expression distributions. This is important
    for single-cell / spatial data because:

    - Many genes have zero-inflated distributions (excess zeros from
      dropouts).
    - Log-normalised expression is only *approximately* normal.
    - The test is robust to outliers and skewed distributions.

    Parameters
    ----------
    adata : anndata.AnnData
        Preprocessed AnnData with cluster labels in
        ``adata.obs[cluster_key]`` and expression values in ``adata.X``.
    cluster_key : str, optional
        Column in ``adata.obs`` containing cluster labels for grouping.
        Default is ``'spatial_cluster'``.

    Returns
    -------
    dict or None
        The scanpy ``rank_genes_groups`` result dictionary, containing:

        - ``names`` -- Structured array of gene names ranked by
          significance, one column per cluster.
        - ``scores`` -- Test statistic (z-score approximation) for each
          gene in each cluster.
        - ``pvals`` -- Raw p-values.
        - ``pvals_adj`` -- Benjamini-Hochberg adjusted p-values.
        - ``logfoldchanges`` -- Log2 fold change of the cluster mean
          versus the rest.

        Returns ``None`` if the result key is not found (should not
        happen in normal operation).

    Notes
    -----
    - The result is stored in ``adata.uns['rank_genes_groups']`` by
      scanpy and persists on the AnnData for downstream use (e.g.,
      ``sc.pl.rank_genes_groups`` for volcano plots).
    - By default, scanpy tests **all** genes in ``adata.X``, not only
      HVGs. This is intentional because marker genes may include
      non-HVG genes that are cluster-specific but not highly variable
      across all spots.
    - Multiple testing correction uses Benjamini-Hochberg (FDR) at
      alpha=0.05 by default.

    See Also
    --------
    spatial_autocorrelation : Spatial pattern analysis per gene.
    neighborhood_enrichment : Cluster co-localisation analysis.
    """
    import scanpy as sc

    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    result = adata.uns.get('rank_genes_groups', None)
    log.info('Differential expression analysis completed')
    return result
