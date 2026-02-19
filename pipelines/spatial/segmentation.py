import logging
import numpy as np

log = logging.getLogger(__name__)


def spatial_clustering(adata, resolution=0.8):
    import scanpy as sc

    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.leiden(adata, resolution=resolution, key_added='spatial_cluster')
    log.info(f'Found {adata.obs["spatial_cluster"].nunique()} clusters')
    return adata


def build_spatial_graph(adata):
    try:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata)
        log.info('Spatial neighbor graph constructed')
    except ImportError:
        log.warning('squidpy not available for spatial graph')
    return adata


def segment_tissue_image(tissue_img):
    from core.edges import laplacian_edges
    from core.morphology import morph_close, remove_small_objects
    from core.contours import find_contours

    if len(tissue_img.shape) == 3:
        gray = np.mean(tissue_img, axis=2).astype(np.float32)
    else:
        gray = tissue_img.astype(np.float32)

    edge_map, stats = laplacian_edges(gray)
    tissue_mask = (np.abs(edge_map) < stats['std']).astype(np.uint8)
    tissue_mask = morph_close(tissue_mask, kernel_size=5, iterations=3)
    tissue_mask = remove_small_objects(tissue_mask, min_area=1000)

    regions = find_contours(tissue_mask, min_area=500)
    return regions
