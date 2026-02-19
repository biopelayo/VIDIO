import logging
import numpy as np

log = logging.getLogger(__name__)


def spatial_autocorrelation(adata, genes=None):
    try:
        import squidpy as sq
    except ImportError:
        log.error('squidpy required for spatial autocorrelation')
        return None

    if genes is None:
        if 'highly_variable' in adata.var.columns:
            genes = adata.var_names[adata.var['highly_variable']].tolist()[:100]
        else:
            genes = adata.var_names[:100].tolist()

    sq.gr.spatial_autocorr(adata, mode='moran', genes=genes)
    log.info(f'Computed Moran I for {len(genes)} genes')
    return adata.uns.get('moranI', None)


def neighborhood_enrichment(adata, cluster_key='spatial_cluster'):
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
    import scanpy as sc

    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    result = adata.uns.get('rank_genes_groups', None)
    log.info('Differential expression analysis completed')
    return result
