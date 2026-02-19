import logging

log = logging.getLogger(__name__)


def load_and_qc(h5ad_path, min_genes=200, min_spots=10):
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
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_hvg(adata, n_top_genes=2000):
    import scanpy as sc

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', subset=False)
    return adata


def run_pca(adata, n_comps=50):
    import scanpy as sc

    sc.pp.pca(adata, n_comps=n_comps)
    return adata
