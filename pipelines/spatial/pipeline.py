"""
Spatial Transcriptomics Pipeline
=================================

Core pipeline class for spatial transcriptomics analysis. Orchestrates the
end-to-end workflow from H5AD loading through quality control,
normalisation, spatial clustering, per-cluster statistics, and anomaly
detection.

Design Decision
---------------
This module extends :class:`pipelines.base.BasePipeline` so that the
spatial pipeline integrates cleanly with VIDIO's job scheduler and
process management infrastructure. Each public method corresponds to a
well-defined pipeline stage that the orchestrator calls in sequence:
``load_image`` -> ``preprocess`` -> ``segment`` -> ``calculate_stats``
-> ``detect_anomalies``.

Notes
-----
- scanpy and squidpy are **imported lazily** inside methods rather than
  at module level. This allows the rest of the VIDIO platform to import
  this module without requiring these heavy bioinformatics dependencies
  when they are not needed (e.g., during unit testing of other
  pipelines).
"""
import logging
import numpy as np

from pipelines.base import BasePipeline
from core.classification import build_finding


class SpatialPipeline(BasePipeline):
    """
    Spatial transcriptomics analysis pipeline.

    Processes 10x Visium, MERFISH, Slide-seq, and 10x Xenium data through
    a standardised workflow: quality control, normalisation, spatial
    clustering, per-cluster gene expression statistics, and anomaly
    detection via z-score deviation from global expression levels.

    The pipeline inherits the job lifecycle from
    :class:`pipelines.base.BasePipeline` and overrides the five stage
    methods that the orchestrator calls in order.

    Parameters
    ----------
    process_id : str
        Unique identifier for this pipeline run, used for logging and
        artifact tracking.
    study_id : str
        Identifier of the study (project) this run belongs to.
    parameters : dict, optional
        Runtime configuration. Recognised keys:

        - ``min_genes_per_spot`` (int, default 200) -- Minimum number of
          detected genes for a spot to pass QC. Spots below this
          threshold are likely empty droplets or low-quality captures.
        - ``min_spots_per_gene`` (int, default 10) -- Minimum number of
          spots in which a gene must be detected. Removes genes that are
          too sparse for meaningful analysis.
        - ``n_top_genes`` (int, default 2000) -- Number of highly
          variable genes to retain after HVG selection. Balances
          information content against computational cost of downstream
          PCA and clustering.

    Attributes
    ----------
    min_genes : int
        Minimum genes per spot threshold (from parameters or default).
    min_spots : int
        Minimum spots per gene threshold (from parameters or default).
    n_top_genes : int
        Number of highly variable genes to select.
    log : logging.Logger
        Logger instance inherited from ``BasePipeline``.

    Notes
    -----
    - **QC thresholds**: The default ``min_genes=200`` follows the 10x
      Genomics recommended lower bound for Visium. For MERFISH panels
      with fewer total genes, callers should lower this value.
    - **HVG count**: 2,000 HVGs is the Seurat/scanpy community standard
      for capturing biologically meaningful variance without
      overfitting.
    - All heavy dependencies (scanpy, squidpy) are imported lazily
      inside methods to avoid import-time overhead.

    Examples
    --------
    >>> pipeline = SpatialPipeline('proc_001', 'study_42',
    ...                            {'min_genes_per_spot': 300})
    >>> adata = pipeline.load_image({'storage_path': 'data.h5ad',
    ...                              'file_format': 'H5AD'})
    >>> adata = pipeline.preprocess(adata, {})
    >>> regions = pipeline.segment(adata, {})
    >>> stats = pipeline.calculate_stats(adata, regions)
    >>> findings = pipeline.detect_anomalies(adata, regions, stats, {})

    See Also
    --------
    pipelines.base.BasePipeline : Abstract base class.
    pipelines.spatial.preprocessing : Standalone preprocessing functions.
    pipelines.spatial.segmentation : Standalone clustering functions.
    pipelines.spatial.analysis : Spatial statistics functions.
    """

    def __init__(self, process_id, study_id, parameters=None):
        """
        Initialise the spatial transcriptomics pipeline.

        Calls the parent ``BasePipeline.__init__`` to set up logging,
        process tracking, and the merged parameter dictionary, then
        extracts spatial-specific thresholds.

        Parameters
        ----------
        process_id : str
            Unique run identifier (passed to ``BasePipeline``).
        study_id : str
            Study/project identifier (passed to ``BasePipeline``).
        parameters : dict, optional
            Runtime overrides.  Unrecognised keys are silently ignored.
            Spatial-specific keys:

            - ``min_genes_per_spot`` (int) -- default **200**.
            - ``min_spots_per_gene`` (int) -- default **10**.
            - ``n_top_genes`` (int) -- default **2000**.

        Notes
        -----
        Defaults are chosen to match the 10x Genomics Visium
        recommended QC thresholds and the scanpy community convention
        for HVG selection.
        """
        super().__init__(process_id, study_id, parameters)
        self.min_genes = self.parameters.get('min_genes_per_spot', 200)
        self.min_spots = self.parameters.get('min_spots_per_gene', 10)
        self.n_top_genes = self.parameters.get('n_top_genes', 2000)

    def load_image(self, image_record):
        """
        Load spatial transcriptomics data or a tissue image from disk.

        Dispatches to the appropriate reader based on the file format:

        - **H5AD** files are read via :func:`core.image_io.read_h5ad`,
          which returns an :class:`anndata.AnnData` object containing
          the expression matrix, spatial coordinates, and (optionally)
          the tissue image.
        - All other formats (TIFF, PNG, JPEG, ...) are read via
          :func:`core.image_io.read_image`, returning a NumPy array.
          This path is used when only the tissue H&E image is provided
          (e.g., for morphology-only segmentation).

        Parameters
        ----------
        image_record : dict
            Database record describing the image/file. Required key:

            - ``storage_path`` (str) -- Absolute path to the file on
              disk.

            Optional key:

            - ``file_format`` (str) -- File format hint (e.g.,
              ``'H5AD'``). If absent, the method falls back to checking
              the file extension.

        Returns
        -------
        anndata.AnnData or numpy.ndarray
            ``AnnData`` if the input is H5AD; otherwise a NumPy image
            array (H x W x C).

        Notes
        -----
        The format check is case-insensitive (``'h5ad'``, ``'H5AD'``,
        ``'H5ad'`` all match). Extension-based fallback uses
        ``str.endswith`` on the storage path.
        """
        fmt = image_record.get('file_format', '').upper()
        path = image_record['storage_path']

        if fmt == 'H5AD' or path.endswith('.h5ad'):
            from core.image_io import read_h5ad
            return read_h5ad(path)
        else:
            from core.image_io import read_image
            return read_image(path)

    def preprocess(self, data, image_record):
        """
        Run the full scanpy preprocessing chain on spatial data.

        The method applies the following steps **in order** (the order
        matters because each step depends on the output of the
        previous):

        1. **filter_cells** -- Remove spots with fewer than
           ``self.min_genes`` detected genes. These are typically empty
           droplets, damaged tissue, or low-capture-efficiency spots.
        2. **filter_genes** -- Remove genes detected in fewer than
           ``self.min_spots`` spots. Ultra-rare genes add noise without
           contributing to clustering.
        3. **normalize_total** -- Scale each spot's total count to
           ``target_sum=10,000`` so that expression values are
           comparable across spots regardless of sequencing depth.
        4. **log1p** -- Apply ``log(x + 1)`` transformation. This
           stabilises variance (counts are Poisson-like) and makes the
           data approximately normally distributed, which improves PCA
           and Leiden performance.
        5. **highly_variable_genes** -- Identify the top
           ``self.n_top_genes`` most variable genes using the Seurat v3
           variance-stabilising transformation (``flavor='seurat_v3'``).
           ``subset=False`` keeps all genes in the matrix but flags HVGs
           in ``adata.var['highly_variable']``; downstream PCA
           automatically uses only HVGs.
        6. **PCA** -- Compute 50 principal components. 50 is the scanpy
           default and captures the vast majority of biological variance
           while discarding technical noise.

        Parameters
        ----------
        data : anndata.AnnData or object
            Input data. If this is not an ``AnnData`` object (checked
            via ``hasattr(data, 'obs')``), the method returns it
            unchanged -- this allows the pipeline to gracefully handle
            plain image inputs.
        image_record : dict
            Image metadata (unused in this method, but required by the
            ``BasePipeline`` interface).

        Returns
        -------
        anndata.AnnData or object
            The preprocessed ``AnnData`` (modified **in-place** and also
            returned), or the original ``data`` if scanpy is unavailable
            or the input is not ``AnnData``.

        Warns
        -----
        Logs an error and returns the unmodified data if scanpy is not
        installed.

        Notes
        -----
        The scanpy preprocessing functions modify ``adata`` in-place.
        The return value is the *same* object for API consistency with
        the ``BasePipeline`` interface, which expects each stage to
        return its output.
        """
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
        """
        Cluster spots into spatial regions using Leiden community detection.

        Two complementary graph structures are built:

        1. **k-NN graph in PCA space** (scanpy) -- connects spots that
           are transcriptionally similar. Uses the top 30 principal
           components (``n_pcs=30``) from the PCA computed during
           ``preprocess()``. This graph drives the Leiden clustering.
        2. **Spatial neighbour graph** (squidpy) -- connects spots that
           are physically adjacent in tissue space using Delaunay
           triangulation of ``adata.obsm['spatial']`` coordinates. This
           graph is stored in ``adata.obsp`` for downstream spatial
           statistics (Moran's I, neighbourhood enrichment) but does
           **not** influence the clustering itself.

        Leiden clustering is run at ``resolution=0.8``, which is the
        scanpy default and typically yields 5--15 clusters for Visium
        datasets.  Higher resolution produces more, smaller clusters.

        Parameters
        ----------
        data : anndata.AnnData or object
            Preprocessed AnnData (must have ``adata.obsm['X_pca']``).
            If the input is not AnnData, returns an empty list.
        image_record : dict
            Image metadata (unused here, required by ``BasePipeline``).

        Returns
        -------
        list of dict
            One dictionary per cluster with keys:

            - ``cluster_id`` (str) -- Leiden cluster label.
            - ``n_spots`` (int) -- Number of spots assigned to this
              cluster.
            - ``spot_indices`` (list of int) -- Row indices into
              ``adata`` for spots belonging to this cluster.

        Notes
        -----
        - The spatial neighbour graph construction is wrapped in a
          try/except because it requires spatial coordinates in
          ``adata.obsm['spatial']``, which may be absent for some
          input formats. Failure is logged as a warning but does not
          abort the pipeline.
        - Cluster labels are stored in
          ``adata.obs['spatial_cluster']`` as categorical strings.
        """
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
        """
        Compute per-cluster gene expression summary statistics.

        For each cluster (region), the method subsets the expression
        matrix to the spots belonging to that cluster, sums gene counts
        per spot (row sums of ``adata.X``), and computes central
        tendency and dispersion measures. These statistics feed into
        ``detect_anomalies()`` to identify clusters with aberrant
        expression levels.

        Parameters
        ----------
        data : anndata.AnnData or object
            Preprocessed and clustered AnnData. The expression matrix
            ``adata.X`` may be dense (numpy) or sparse (scipy CSR/CSC);
            both are handled via ``np.array(...).flatten()``.
        regions : list of dict
            Cluster region dictionaries as returned by ``segment()``.
            Each dict must contain ``spot_indices``, ``n_spots``, and
            ``cluster_id``.

        Returns
        -------
        list of dict
            One dictionary per region (aligned by index with *regions*).
            Each dictionary contains:

            - ``n_spots`` (int) -- Spot count.
            - ``cluster_id`` (str) -- Cluster label.
            - ``mean_genes_per_spot`` (float) -- Mean total gene count
              across spots in the cluster.
            - ``std_genes_per_spot`` (float) -- Standard deviation of
              total gene count.
            - ``median_genes_per_spot`` (float) -- Median total gene
              count.

            Empty dict ``{}`` is returned for clusters with no spots
            (edge case guard).

        Notes
        -----
        - ``subset.X.sum(axis=1)`` gives the library size per spot
          **after** normalisation. Because ``normalize_total`` scales
          each spot to 10,000 and ``log1p`` compresses values, the
          resulting means reflect *log-normalised* expression, not raw
          UMI counts.
        - The statistics are intentionally simple (mean/std/median)
          because they serve as inputs to a z-score anomaly detector,
          not as final clinical metrics.
        """
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
        """
        Identify spatial clusters with anomalous gene expression levels.

        Computes a **z-score** for each cluster's mean total gene
        expression relative to the global distribution of cluster means.
        Clusters whose absolute z-score exceeds 1.5 are flagged as
        anomalies and packaged into standardised *finding* dictionaries
        via :func:`core.classification.build_finding`.

        Z-score Threshold and Severity Tiers
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        +-----------+---------------+-------------------------------+
        | z-score   | Severity      | Interpretation                |
        +===========+===============+===============================+
        | < 1.5     | (not flagged) | Within normal variation.      |
        +-----------+---------------+-------------------------------+
        | 1.5 -- 2  | LOW           | Mildly deviant cluster.       |
        +-----------+---------------+-------------------------------+
        | 2 -- 3    | MEDIUM        | Moderately deviant cluster.   |
        +-----------+---------------+-------------------------------+
        | > 3       | HIGH          | Strongly deviant cluster;     |
        |           |               | likely biologically distinct.  |
        +-----------+---------------+-------------------------------+

        Confidence is linearly scaled as ``min(1.0, z_score / 4.0)``,
        reaching maximum confidence (1.0) at z >= 4.

        Parameters
        ----------
        data : anndata.AnnData or object
            The AnnData object (not directly used in this method, but
            required by the ``BasePipeline`` signature for interface
            consistency).
        regions : list of dict
            Cluster regions from ``segment()``.
        stats : list of dict
            Per-cluster statistics from ``calculate_stats()``, aligned
            by index with *regions*.
        image_record : dict
            Image metadata (unused here, required by ``BasePipeline``).

        Returns
        -------
        list of dict
            A list of finding dictionaries, one per anomalous cluster.
            Each finding contains:

            - ``finding_type`` -- ``'SPATIAL_CLUSTER_ANOMALY'``
            - ``severity`` -- ``'LOW'``, ``'MEDIUM'``, or ``'HIGH'``
            - ``confidence`` -- float in [0, 1]
            - ``statistics`` -- the cluster's stats dict
            - ``geometric_properties`` -- ``{'n_spots': int}``
            - ``location`` -- ``{'cluster_id': str}``
            - ``disease_category`` -- ``'SPATIAL_TRANSCRIPTOMICS'``

        Notes
        -----
        - When there is only **one** cluster, ``global_std`` is set to
          1.0 to avoid division by zero. In practice, single-cluster
          datasets will never trigger an anomaly because z = 0.
        - The method uses **absolute** z-score, so both unusually high
          and unusually low expression clusters are flagged. In
          biological terms, high-expression anomalies may represent
          actively transcribing tumour regions, while low-expression
          anomalies may indicate necrotic zones.

        See Also
        --------
        core.classification.build_finding : Standardised finding
            constructor.
        """
        findings = []
        if not stats or not regions:
            return findings

        # Collect mean expression values across all non-empty clusters
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

            # --- Threshold gate: only flag clusters beyond 1.5 sigma ---
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
