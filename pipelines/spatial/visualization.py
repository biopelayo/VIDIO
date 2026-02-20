"""
Spatial Transcriptomics Visualisation
======================================

Functions for creating publication-quality spatial plots of gene
expression and cluster assignments overlaid on tissue histology images.

Both functions produce matplotlib ``Figure`` objects that can be saved
to disk (``fig.savefig('plot.png')``) or displayed in Jupyter notebooks
(``plt.show()``).

Design Notes
------------
- All functions guard against missing matplotlib with a try/except and
  return ``None`` on import failure, so they can be safely called in
  headless server environments.
- Spatial coordinates are expected in ``adata.obsm['spatial']`` (the
  standard location for 10x Visium and squidpy).
- The tissue image is optional; if ``None``, expression or clusters are
  plotted on a blank canvas.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)


def overlay_expression(tissue_img, adata, gene, cmap='viridis', alpha=0.5):
    """
    Overlay a single gene's expression as a heatmap on the tissue image.

    Creates a scatter plot of spot positions coloured by expression
    intensity, optionally superimposed on the H&E tissue image. This
    is the standard way to visualise spatially resolved gene expression
    in Visium and similar platforms.

    Colour Normalisation
    ~~~~~~~~~~~~~~~~~~~~~
    Expression values are normalised using **percentile-based clipping**:

    - ``vmin = 5th percentile`` of expression values
    - ``vmax = 95th percentile`` of expression values

    This clips the top and bottom 5 %% of the expression distribution,
    which prevents a few extreme outlier spots from compressing the
    colour range and washing out biologically meaningful variation.
    Compared to min/max normalisation, percentile-based normalisation
    is robust to dropouts (zeros) at the low end and ultra-high
    expression artefacts at the high end.

    Parameters
    ----------
    tissue_img : numpy.ndarray or None
        H&E tissue image as a NumPy array (H x W x C). If ``None``,
        the scatter is plotted on a blank white canvas.
    adata : anndata.AnnData
        AnnData with spatial coordinates in ``adata.obsm['spatial']``
        (shape ``(n_spots, 2)``) and expression data in ``adata.X``.
    gene : str
        Gene symbol to visualise (must exist in ``adata.var_names``).
    cmap : str, optional
        Matplotlib colourmap name. Default is ``'viridis'`` (perceptually
        uniform, colourblind-friendly). Other useful options:

        - ``'magma'`` -- Dark background emphasis.
        - ``'RdBu_r'`` -- Diverging (good for fold-change plots).
        - ``'Reds'`` -- Single-hue (good for expression intensity).
    alpha : float, optional
        Transparency of the scatter points. Default is **0.5**, which
        allows the underlying tissue image to remain visible while
        still showing expression intensity. Range: 0 (invisible) to
        1 (fully opaque).

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object containing the plot. Returns ``None`` if:

        - matplotlib is not installed.
        - ``adata.obsm['spatial']`` is missing.
        - The requested gene is not found in ``adata.var_names``.

    Notes
    -----
    - The expression matrix ``adata.X`` may be sparse (scipy CSR/CSC)
      or dense (numpy). The function handles both via a
      ``hasattr(adata.X, 'toarray')`` check.
    - The figure is 10 x 10 inches (``figsize=(10, 10)``) with axes
      turned off for a clean presentation.
    - The caller is responsible for saving (``fig.savefig()``) or
      displaying (``plt.show()``) the returned figure and closing it
      afterwards to free memory.

    Examples
    --------
    >>> fig = overlay_expression(tissue_img, adata, 'EGFR')
    >>> if fig is not None:
    ...     fig.savefig('EGFR_expression.png', dpi=150, bbox_inches='tight')
    ...     plt.close(fig)

    See Also
    --------
    plot_spatial_clusters : Visualise cluster assignments on tissue.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        log.error('matplotlib required for visualization')
        return None

    if 'spatial' not in adata.obsm:
        log.error('Spatial coordinates not found in adata.obsm')
        return None

    coords = adata.obsm['spatial']

    # --- Look up gene index in the var_names index ---
    gene_idx = list(adata.var_names).index(gene) if gene in adata.var_names else None
    if gene_idx is None:
        log.error(f'Gene {gene} not found')
        return None

    # --- Extract expression vector (handle sparse and dense matrices) ---
    if hasattr(adata.X, 'toarray'):
        expression = adata.X[:, gene_idx].toarray().flatten()
    else:
        expression = adata.X[:, gene_idx].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if tissue_img is not None:
        ax.imshow(tissue_img)

    # --- Percentile-based colour normalisation (5th--95th) ---
    norm = Normalize(vmin=np.percentile(expression, 5), vmax=np.percentile(expression, 95))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=expression, cmap=cmap, norm=norm,
        s=10, alpha=alpha,
    )
    plt.colorbar(scatter, ax=ax, label=gene)
    ax.set_title(f'{gene} expression')
    ax.axis('off')

    return fig


def plot_spatial_clusters(adata, cluster_key='spatial_cluster', tissue_img=None):
    """
    Plot spatial cluster assignments as a colour-coded scatter on tissue.

    Each cluster is assigned a distinct colour from the matplotlib
    ``tab20`` colourmap (supports up to 20 visually distinct clusters).
    Spots are plotted at their tissue coordinates with a legend
    identifying each cluster.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with spatial coordinates in ``adata.obsm['spatial']``
        and cluster labels in ``adata.obs[cluster_key]``.
    cluster_key : str, optional
        Column in ``adata.obs`` containing cluster assignments.
        Default is ``'spatial_cluster'`` (as set by
        :func:`~pipelines.spatial.segmentation.spatial_clustering`).
    tissue_img : numpy.ndarray or None, optional
        H&E tissue image (H x W x C) to use as background. If ``None``,
        clusters are plotted on a blank canvas.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object. Returns ``None`` if matplotlib is missing or
        spatial coordinates are absent.

    Notes
    -----
    - Clusters are **sorted** before colour assignment so that the
      colour mapping is deterministic across runs.
    - The ``tab20`` colourmap provides 20 distinguishable colours. For
      datasets with more than 20 clusters, colours will wrap around and
      some clusters will share colours. In practice, Visium datasets
      rarely exceed 15 clusters at the default Leiden resolution.
    - ``alpha=0.7`` provides slight transparency so that overlapping
      spots at cluster boundaries remain visible.
    - The legend is placed outside the axes area
      (``bbox_to_anchor=(1.05, 1)``) to avoid obscuring the tissue.
      ``plt.tight_layout()`` ensures the legend is not clipped when
      saving to file.
    - The caller is responsible for saving or displaying the returned
      figure and closing it to free memory.

    Examples
    --------
    >>> fig = plot_spatial_clusters(adata, tissue_img=tissue_img)
    >>> if fig is not None:
    ...     fig.savefig('clusters.png', dpi=150, bbox_inches='tight')
    ...     plt.close(fig)

    See Also
    --------
    overlay_expression : Visualise per-gene expression on tissue.
    pipelines.spatial.segmentation.spatial_clustering :
        Produces the cluster labels consumed by this function.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.error('matplotlib required')
        return None

    if 'spatial' not in adata.obsm:
        log.error('Spatial coordinates not found')
        return None

    coords = adata.obsm['spatial']
    clusters = adata.obs[cluster_key]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if tissue_img is not None:
        ax.imshow(tissue_img)

    # --- Assign a distinct colour to each cluster (sorted for consistency) ---
    unique_clusters = sorted(clusters.unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=f'Cluster {cluster_id}',
            s=10, alpha=0.7,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Spatial Clusters ({cluster_key})')
    ax.axis('off')
    plt.tight_layout()

    return fig
