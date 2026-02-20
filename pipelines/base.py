"""
VIDIO Pipelines - Base Pipeline
================================

Abstract base class implementing the canonical 5-stage biomedical image
analysis pattern used by all VIDIO modality pipelines.

Architecture
------------
Every VIDIO pipeline follows the same five sequential stages::

    Load → Preprocess → Segment → Statistics → Detect Anomalies

This pattern is adapted from established medical image analysis
workflows and mirrors the solar panel thermal inspection pipeline that
inspired the VIDIO architecture.  Each stage is a virtual method that
subclasses override with modality-specific logic.

Execution Model
---------------
- Pipelines are launched asynchronously from :mod:`ProcessManagement` in
  daemon threads, one per ``POST /analysis/<modality>`` request.
- A :class:`~api.db.DB.DB` instance is created per pipeline for
  database persistence (session-per-method pattern).
- Progress is tracked in the ``process`` table (0–100 %) and updated
  after each image is processed.
- Findings are persisted individually to the ``finding`` table as they
  are detected, so partial results survive a crash.

Error Handling
--------------
- Per-image errors are caught and logged without aborting the study.
- Pipeline-level failures set the process status to ``FAILED`` with
  the exception message stored in ``process.error_message``.

See Also
--------
:mod:`ProcessManagement` : Launches pipelines in background threads.
:mod:`api.resources.AnalysisResource` : REST endpoint that triggers analysis.
"""

import logging
import datetime as dt

from api.db.DB import DB


class BasePipeline:
    """Abstract base for all VIDIO modality analysis pipelines.

    Implements the 5-stage biomedical image analysis pattern:

    1. **Load** — Read the image file from disk using the universal
       reader in :mod:`core.image_io`.
    2. **Preprocess** — Normalise intensities, apply contrast
       enhancement and edge-preserving filters.
    3. **Segment** — Identify regions of interest (ROIs) via edge
       detection, morphological operations, or deep-learning models.
    4. **Statistics** — Compute quantitative descriptors per ROI:
       intensity statistics (mean, std, gradient, skewness, kurtosis)
       and geometric features (area, solidity, elongation, convexity).
    5. **Detect Anomalies** — Compare ROI descriptors against global
       baselines using z-scores and classify severity.

    Subclasses **must** override the stage methods for their specific
    imaging modality.  The base implementations are no-ops that return
    empty results.

    Parameters
    ----------
    process_id : str (UUID)
        Primary key of the ``process`` row tracking this analysis job.
        Used for progress updates and finding linkage.
    study_id : str (UUID)
        Primary key of the ``study`` being analysed.  All series and
        images belonging to this study will be processed.
    parameters : dict, optional
        Modality-specific runtime parameters (e.g. tile_size, window
        center, min_genes).  Passed through from the API request body.

    Attributes
    ----------
    log : logging.Logger
        Logger named after the concrete subclass (e.g. ``RetinalPipeline``).
    db : DB
        Data access layer instance (session-per-method, thread-safe).
    process_id : str
        UUID of the tracking process row.
    study_id : str
        UUID of the study being analysed.
    parameters : dict
        Runtime parameters for this pipeline execution.

    Notes
    -----
    The pipeline iterates over **all selected images** in the study
    (across all series).  Images with ``selected = False`` are skipped.
    """

    def __init__(self, process_id, study_id, parameters=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.db = DB()
        self.process_id = process_id
        self.study_id = study_id
        self.parameters = parameters or {}

    def run(self):
        """Execute the full pipeline for every image in the study.

        Iterates over all selected images in the study, running the
        5-stage processing chain on each.  Progress is reported to the
        ``process`` table after each image completes.

        Returns
        -------
        list[dict]
            Accumulated findings from all images.  Each finding is a
            dict ready for database storage (see
            :func:`core.classification.build_finding`).

        Raises
        ------
        Exception
            Re-raised after setting the process status to ``FAILED``.
            Per-image exceptions are caught and logged individually.
        """
        self.log.info(f'Pipeline started: process={self.process_id}, study={self.study_id}')
        self.update_status('RUNNING', progress=0)

        try:
            images = self.get_images()
            if not images:
                self.log.warning('No images found for study')
                self.update_status('COMPLETED', progress=100, result={'findings_count': 0})
                return []

            all_findings = []
            total = len(images)

            for idx, image_record in enumerate(images):
                self.log.info(f'Processing image {idx + 1}/{total}: {image_record["name"]}')
                try:
                    findings = self.process_image(image_record)
                    all_findings.extend(findings)
                except Exception as ex:
                    self.log.error(f'Error processing image {image_record["name"]}: {ex}')

                progress = int(((idx + 1) / total) * 100)
                self.update_status('RUNNING', progress=progress)

            self.update_status('COMPLETED', progress=100, result={
                'findings_count': len(all_findings),
                'images_processed': total,
            })

            self.log.info(f'Pipeline completed: {len(all_findings)} findings in {total} images')
            return all_findings

        except Exception as ex:
            self.log.error(f'Pipeline failed: {ex}')
            self.update_status('FAILED', error_message=str(ex))
            raise

    def process_image(self, image_record):
        """Run the 5-stage chain on a single image.

        Parameters
        ----------
        image_record : dict
            Row from the ``image`` table (as returned by
            :meth:`~api.db.DB.DB.GetImage`).  Must contain at least
            ``id``, ``name``, and ``storage_path``.

        Returns
        -------
        list[dict]
            Findings detected in this image.
        """
        img = self.load_image(image_record)
        img = self.preprocess(img, image_record)
        regions = self.segment(img, image_record)
        stats = self.calculate_stats(img, regions)
        findings = self.detect_anomalies(img, regions, stats, image_record)
        self.save_findings(findings, image_record)
        return findings

    def get_images(self):
        """Retrieve all selected images for the study.

        Collects images across all series belonging to
        ``self.study_id``, filtering out any with ``selected = False``.

        Returns
        -------
        list[dict]
            Image records from the database.
        """
        series_list = self.db.GetSeriesForStudy(self.study_id)
        images = []
        for s in series_list:
            imgs = self.db.GetImagesForSeries(s['id'])
            images.extend(imgs)
        return [img for img in images if img.get('selected', True)]

    # ------------------------------------------------------------------ #
    #  Stage methods — override in subclasses                              #
    # ------------------------------------------------------------------ #

    def load_image(self, image_record):
        """Stage 1: Load the image from disk.

        Default implementation uses the universal auto-detecting reader.
        Override for modality-specific loading (e.g. DICOM with
        metadata extraction, or H5AD for spatial data).

        Parameters
        ----------
        image_record : dict
            Image row from the database.

        Returns
        -------
        np.ndarray or tuple or anndata.AnnData
            Loaded image data.  Format depends on the file type.
        """
        from core.image_io import read_image
        return read_image(image_record['storage_path'])

    def preprocess(self, img, image_record):
        """Stage 2: Preprocess the image (normalise, filter, enhance).

        Default is a no-op.  Subclasses apply modality-specific
        preprocessing (green channel extraction, HU windowing, stain
        normalisation, etc.).

        Parameters
        ----------
        img : np.ndarray
            Raw image data.
        image_record : dict
            Image metadata for context-aware preprocessing.

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return img

    def segment(self, img, image_record):
        """Stage 3: Segment the image into regions of interest.

        Default returns an empty list.  Subclasses perform edge
        detection, morphological operations, clustering, or DL-based
        segmentation.

        Parameters
        ----------
        img : np.ndarray
            Preprocessed image.
        image_record : dict
            Image metadata.

        Returns
        -------
        list
            Detected regions.  Format varies by modality (contour dicts,
            tile dicts, cluster dicts, etc.).
        """
        return []

    def calculate_stats(self, img, regions):
        """Stage 4: Compute quantitative statistics per region.

        Default returns an empty list.  Subclasses extract intensity
        and geometric descriptors for each ROI.

        Parameters
        ----------
        img : np.ndarray
            Preprocessed image (for intensity measurements).
        regions : list
            Regions from :meth:`segment`.

        Returns
        -------
        list[dict]
            One statistics dict per region.
        """
        return []

    def detect_anomalies(self, img, regions, stats, image_record):
        """Stage 5: Detect anomalies and classify severity.

        Default returns an empty list.  Subclasses compare per-region
        statistics against global baselines and apply classification
        rules.

        Parameters
        ----------
        img : np.ndarray
            Preprocessed image.
        regions : list
            Regions from :meth:`segment`.
        stats : list[dict]
            Per-region statistics from :meth:`calculate_stats`.
        image_record : dict
            Image metadata.

        Returns
        -------
        list[dict]
            Detected findings (see :func:`core.classification.build_finding`).
        """
        return []

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_findings(self, findings, image_record):
        """Persist detected findings to the database.

        Each finding dict is augmented with ``id_image``, ``id_study``,
        and ``id_process`` before being inserted into the ``finding``
        table.  This is called automatically by :meth:`process_image`.

        Parameters
        ----------
        findings : list[dict]
            Findings to persist.
        image_record : dict
            Source image (provides ``id``).
        """
        for finding_data in findings:
            finding_data['id_image'] = image_record['id']
            finding_data['id_study'] = self.study_id
            finding_data['id_process'] = self.process_id
            self.db.AddFinding(None, finding_data)

    # ------------------------------------------------------------------ #
    #  Status updates                                                      #
    # ------------------------------------------------------------------ #

    def update_status(self, status, progress=None, result=None, error_message=None):
        """Update the process tracking row in the database.

        Parameters
        ----------
        status : str
            New status: ``"RUNNING"``, ``"COMPLETED"``, or ``"FAILED"``.
        progress : int, optional
            Completion percentage (0–100).
        result : dict, optional
            Summary result payload (stored as JSONB).
        error_message : str, optional
            Error description if ``status == "FAILED"``.

        Notes
        -----
        ``time_start`` is set when status first becomes ``RUNNING``
        (progress == 0).  ``time_end`` is set on ``COMPLETED`` or
        ``FAILED``.  Timestamps use UTC.
        """
        data = {
            'id': self.process_id,
            'status': status,
        }
        if progress is not None:
            data['progress'] = progress
        if result is not None:
            data['result'] = result
        if error_message is not None:
            data['error_message'] = error_message
        if status == 'RUNNING' and progress == 0:
            data['time_start'] = dt.datetime.utcnow().isoformat()
        if status in ('COMPLETED', 'FAILED'):
            data['time_end'] = dt.datetime.utcnow().isoformat()

        try:
            self.db.ModifyProcess(None, data)
        except Exception as ex:
            self.log.error(f'Failed to update process status: {ex}')
