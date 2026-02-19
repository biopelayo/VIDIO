import logging
import numpy as np

log = logging.getLogger(__name__)


def overlay_expression(tissue_img, adata, gene, cmap='viridis', alpha=0.5):
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

    gene_idx = list(adata.var_names).index(gene) if gene in adata.var_names else None
    if gene_idx is None:
        log.error(f'Gene {gene} not found')
        return None

    if hasattr(adata.X, 'toarray'):
        expression = adata.X[:, gene_idx].toarray().flatten()
    else:
        expression = adata.X[:, gene_idx].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if tissue_img is not None:
        ax.imshow(tissue_img)

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
