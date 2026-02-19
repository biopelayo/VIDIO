import logging
import numpy as np

from pipelines.base import BasePipeline
from core.classification import build_finding


class SpatialPipeline(BasePipeline):
    """
    Spatial transcriptomics analysis pipeline.

    Processes 10x Visium, MERFISH, Slide-seq data for:
    - Gene expression spatial mapping
    - Spatial clustering and neighborhood analysis
    - Spatially variable gene detection
    - Tissue region characterization
    """

    def __init__(self, process_id, study_id, parameters=None):
        super().__init__(process_id, study_id, parameters)
        self.min_genes = self.parameters.get('min_genes_per_spot', 200)
        self.min_spots = self.parameters.get('min_spots_per_gene', 10)
        self.n_top_genes = self.parameters.get('n_top_genes', 2000)

    def load_image(self, image_record):
        fmt = image_record.get('file_format', '').upper()
        path = image_record['storage_path']

        if fmt == 'H5AD' or path.endswith('.h5ad'):
            from core.image_io import read_h5ad
            return read_h5ad(path)
        else:
            from core.image_io import read_image
            return read_image(path)

    def preprocess(self, data, image_record):
        try:
            import scanpy as sc
        except ImportError:
            self.log.error('scanpy required for spatial transcriptomics')
            return data

        if not hasattr(data, 'obs'):
            return data

        adata = data
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.min_spots)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes, flavor='seurat_v3',
                                     subset=False)
        sc.pp.pca(adata, n_comps=50)

        self.log.info(f'Preprocessed: {adata.n_obs} spots, {adata.n_vars} genes')
        return adata

    def segment(self, data, image_record):
        try:
            import scanpy as sc
            import squidpy as sq
        except ImportError:
            self.log.error('scanpy and squidpy required')
            return []

        if not hasattr(data, 'obs'):
            return []

        adata = data
        sc.pp.neighbors(adata, n_pcs=30)
        sc.tl.leiden(adata, resolution=0.8, key_added='spatial_cluster')

        try:
            sq.gr.spatial_neighbors(adata)
        except Exception as ex:
            self.log.warning(f'Spatial neighbors failed: {ex}')

        clusters = adata.obs['spatial_cluster'].unique().tolist()
        self.log.info(f'Found {len(clusters)} spatial clusters')

        regions = []
        for cluster_id in clusters:
            mask = adata.obs['spatial_cluster'] == cluster_id
            n_spots = int(mask.sum())
            regions.append({
                'cluster_id': str(cluster_id),
                'n_spots': n_spots,
                'spot_indices': np.where(mask.values)[0].tolist(),
            })

        return regions

    def calculate_stats(self, data, regions):
        if not hasattr(data, 'obs') or not regions:
            return []

        adata = data
        stats_list = []

        for region in regions:
            indices = region['spot_indices']
            if not indices:
                stats_list.append({})
                continue

            subset = adata[indices]
            n_genes_per_spot = np.array(subset.X.sum(axis=1)).flatten()

            stats = {
                'n_spots': region['n_spots'],
                'cluster_id': region['cluster_id'],
                'mean_genes_per_spot': float(np.mean(n_genes_per_spot)),
                'std_genes_per_spot': float(np.std(n_genes_per_spot)),
                'median_genes_per_spot': float(np.median(n_genes_per_spot)),
            }
            stats_list.append(stats)

        return stats_list

    def detect_anomalies(self, data, regions, stats, image_record):
        findings = []
        if not stats or not regions:
            return findings

        all_means = [s.get('mean_genes_per_spot', 0) for s in stats if s]
        if not all_means:
            return findings
        global_mean = np.mean(all_means)
        global_std = np.std(all_means) if len(all_means) > 1 else 1.0

        for region, region_stats in zip(regions, stats):
            if not region_stats:
                continue

            mean_val = region_stats.get('mean_genes_per_spot', 0)
            z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

            if z_score < 1.5:
                continue

            confidence = min(1.0, z_score / 4.0)
            severity = 'HIGH' if z_score > 3 else 'MEDIUM' if z_score > 2 else 'LOW'

            finding = build_finding(
                finding_type='SPATIAL_CLUSTER_ANOMALY',
                severity=severity,
                confidence=confidence,
                statistics=region_stats,
                geometric_properties={'n_spots': region['n_spots']},
                location={'cluster_id': region['cluster_id']},
            )
            finding['disease_category'] = 'SPATIAL_TRANSCRIPTOMICS'
            findings.append(finding)

        return findings
