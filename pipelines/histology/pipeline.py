"""
Histology pipeline implementation.

This module contains the main :class:`HistologyPipeline` class which
orchestrates the end-to-end analysis of whole-slide histopathology images.
It inherits from :class:`~pipelines.base.BasePipeline` and implements the
five canonical pipeline stages: load, preprocess, segment, statistics, and
anomaly detection.

Design Decision -- Tile-Based Architecture
-------------------------------------------
Gigapixel whole-slide images (WSI) cannot fit in GPU or even CPU memory in
their entirety (a typical 40x SVS file is 50 000 x 50 000 px or larger).
This pipeline therefore operates on a *tile-based* strategy:

1. The full image (or a down-sampled thumbnail for in-memory images) is
   divided into non-overlapping square tiles.
2. Each tile is independently assessed for tissue content so that blank
   glass / background tiles are discarded early.
3. Per-tile statistics are computed, and outlier tiles (by z-score) are
   flagged as anomalies.

This design keeps peak memory proportional to ``tile_size**2`` rather than
to the full WSI dimensions.
"""

import logging
import numpy as np

from pipelines.base import BasePipeline
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


class HistologyPipeline(BasePipeline):
    """
    Histopathology whole-slide image analysis pipeline.

    Processes SVS/TIFF whole-slide images through a tile-based workflow:
    tissue detection, Macenko stain normalisation, non-overlapping tiling,
    per-tile intensity statistics, and z-score anomaly detection.

    The pipeline inherits the orchestration loop from
    :class:`~pipelines.base.BasePipeline`.  Each ``run()`` invocation
    iterates over the image records for the given ``study_id`` and calls
    :meth:`load_image`, :meth:`preprocess`, :meth:`segment`,
    :meth:`calculate_stats`, and :meth:`detect_anomalies` in sequence.

    Parameters
    ----------
    process_id : str
        Unique identifier for this pipeline execution, used in logging and
        result persistence.
    study_id : str
        Identifier of the study (patient / case / experiment) whose images
        will be processed.
    parameters : dict, optional
        Runtime configuration dictionary.  Recognised keys:

        - ``tile_size`` (int) -- Edge length in pixels of the square tiles
          extracted from each image.  Default ``256``.
        - ``tissue_threshold`` (float) -- Minimum fraction of tissue pixels
          a tile must contain to be retained for analysis.  Range [0, 1].
          Default ``0.5`` (50 %).

    Attributes
    ----------
    tile_size : int
        Edge length of square tiles (pixels).  Set from ``parameters``.
    tissue_threshold : float
        Minimum tissue-pixel fraction for tile retention.
    log : logging.Logger
        Logger instance inherited from ``BasePipeline``.

    Notes
    -----
    * The tissue detection heuristic (grayscale intensity < 220) is tuned
      for H&E-stained sections where tissue appears darker than the white
      glass background.  IHC or special stains with very light chromogens
      may require a higher threshold.
    * Stain normalisation uses the Macenko method via
      :func:`core.intensity.stain_normalize`.  If the image is not a
      3-channel RGB image (e.g. grayscale fluorescence), the normalisation
      step is silently skipped.

    See Also
    --------
    pipelines.base.BasePipeline : Abstract base class defining the stage
        interface.
    pipelines.histology.tiling.extract_tiles : Standalone tiling utility
        with overlap support.
    pipelines.histology.segmentation.segment_tumor_regions : K-Means-based
        tumour segmentation alternative.

    Examples
    --------
    >>> pipe = HistologyPipeline('proc-001', 'study-42',
    ...                          parameters={'tile_size': 512})
    >>> pipe.run()  # processes all images for study-42
    """

    def __init__(self, process_id, study_id, parameters=None):
        """
        Initialise the histology pipeline.

        Calls the parent constructor to set up logging, result storage, and
        the ``self.parameters`` dict, then extracts histology-specific
        configuration values.

        Parameters
        ----------
        process_id : str
            Unique execution identifier (written into every finding record
            for traceability).
        study_id : str
            Study / case whose images will be analysed.
        parameters : dict, optional
            Configuration overrides.  If ``None``, empty-dict defaults are
            used by the parent class.

            ``tile_size`` : int, default 256
                Width and height (in pixels) of each square tile.  Common
                values are 224 (ImageNet-compatible) and 256.
            ``tissue_threshold`` : float, default 0.5
                Fraction of pixels that must be classified as *tissue*
                (grayscale < 220) for a tile to be kept.  Raising this
                value discards tiles with partial background.

        Notes
        -----
        The ``tile_size`` and ``tissue_threshold`` are cached as instance
        attributes for fast access during the hot tiling loop in
        :meth:`_extract_tiles`.
        """
        super().__init__(process_id, study_id, parameters)
        self.tile_size = self.parameters.get('tile_size', 256)
        self.tissue_threshold = self.parameters.get('tissue_threshold', 0.5)

    def load_image(self, image_record):
        """
        Load an image from disk given its database record.

        Delegates to :func:`core.image_io.read_image` which supports
        multiple formats (PNG, TIFF, SVS thumbnails, DICOM).  The import
        is deferred to avoid a hard dependency at module-import time.

        Parameters
        ----------
        image_record : dict
            Database row (or dict) for the image.  Must contain the key
            ``'storage_path'`` pointing to an on-disk file path.

        Returns
        -------
        numpy.ndarray
            Loaded image as an ``(H, W, C)`` uint8 RGB array (for colour
            images) or ``(H, W)`` for grayscale.

        Notes
        -----
        ``read_image`` returns a *tuple* ``(pixel_array, metadata)`` for
        DICOM files but a plain ``ndarray`` for standard image formats.
        This method performs tuple-unpacking when necessary so that the
        rest of the pipeline always receives a single array.
        """
        from core.image_io import read_image
        result = read_image(image_record['storage_path'])
        # DICOM reader returns (pixel_array, dicom_metadata) tuple;
        # standard image readers return a bare ndarray.
        if isinstance(result, tuple):
            return result[0]
        return result

    def preprocess(self, img, image_record):
        """
        Apply Macenko stain normalisation to the loaded image.

        Stain normalisation is critical in histopathology because colour
        variation between laboratories, scanners, and staining protocols
        can be dramatic.  The Macenko method decomposes the image into
        optical-density space, identifies the two principal stain vectors
        (haematoxylin and eosin), and projects them onto a reference
        stain matrix.  This removes batch effects while preserving
        morphological structure.

        Parameters
        ----------
        img : numpy.ndarray or None
            The loaded image (``H, W, 3`` uint8 RGB).  If ``None`` (e.g.
            the load stage failed), it is returned unchanged.
        image_record : dict
            Image metadata (unused in this stage but required by the
            ``BasePipeline`` interface signature).

        Returns
        -------
        numpy.ndarray or None
            Stain-normalised image with the same shape and dtype as the
            input, or the original image if normalisation was skipped or
            failed.

        Notes
        -----
        * Normalisation is only attempted on 3-channel images.  Grayscale
          or single-channel inputs pass through untouched.
        * If ``stain_normalize`` raises (e.g. the tissue region is too
          small to compute reliable stain vectors), the exception is
          caught, logged as a warning, and the *original* image is
          returned.  This graceful fallback ensures the pipeline does not
          abort due to edge-case preprocessing failures.
        """
        if img is None:
            return img

        from core.intensity import stain_normalize
        # Only normalise 3-channel (RGB) images; grayscale / RGBA are left
        # unchanged because the Macenko decomposition requires exactly the
        # haematoxylin + eosin two-stain model.
        if len(img.shape) == 3 and img.shape[2] == 3:
            try:
                img = stain_normalize(img)
            except Exception as ex:
                # Graceful fallback: log and continue with the original image
                # rather than aborting the entire pipeline run.
                self.log.warning(f'Stain normalization failed: {ex}')

        return img

    def segment(self, img, image_record):
        """
        Segment the image into tissue-containing tiles.

        In histopathology the "segmentation" stage is tile extraction:
        the gigapixel image is divided into a regular grid of square
        patches and background-only tiles are discarded.

        Parameters
        ----------
        img : numpy.ndarray or None
            Preprocessed image.  If ``None``, an empty list is returned
            immediately (no tiles to process).
        image_record : dict
            Image metadata (unused here but part of the stage contract).

        Returns
        -------
        list of dict
            Each dict represents one retained tile with keys ``'tile'``,
            ``'x'``, ``'y'``, ``'w'``, ``'h'``, and ``'tissue_ratio'``.
            See :meth:`_extract_tiles` for detailed field descriptions.
        """
        if img is None:
            return []

        tiles = self._extract_tiles(img)
        return tiles

    def _extract_tiles(self, img):
        """
        Divide the image into a non-overlapping grid of square tiles and
        keep only those tiles that contain sufficient tissue.

        The algorithm raster-scans the image in row-major order with a
        step equal to ``self.tile_size`` (i.e. no overlap).  For each
        candidate tile:

        1. Convert to grayscale by averaging across the colour channels
           (or use the image directly if already single-channel).
        2. Compute ``tissue_ratio`` as the fraction of pixels with
           grayscale intensity **< 220**.  The threshold of 220 separates
           stained tissue (darker) from the near-white glass background
           in H&E-stained slides.
        3. If ``tissue_ratio >= self.tissue_threshold``, append the tile
           and its metadata to the output list.

        Parameters
        ----------
        img : numpy.ndarray
            Input image, shape ``(H, W)`` or ``(H, W, 3)``.

        Returns
        -------
        list of dict
            Each dictionary contains:

            ``tile`` : numpy.ndarray
                The raw pixel data for this tile, shape
                ``(tile_size, tile_size)`` or
                ``(tile_size, tile_size, 3)``.
            ``x``, ``y`` : int
                Top-left corner coordinates (pixels) of the tile within
                the original image.
            ``w``, ``h`` : int
                Width and height of the tile (always equal to
                ``self.tile_size``).
            ``tissue_ratio`` : float
                Fraction of pixels classified as tissue, in [0, 1].

        Notes
        -----
        * Tiles that would extend beyond the image boundary on the right
          or bottom edge are *not* extracted (no zero-padding).  This
          means a thin strip of up to ``tile_size - 1`` pixels on each
          edge may be ignored.
        * The intensity threshold of 220 is a hard-coded heuristic.  For
          very lightly stained or fluorescence images, consider using the
          standalone :func:`~pipelines.histology.preprocessing.detect_tissue`
          function which allows an adjustable threshold.
        """
        h, w = img.shape[:2]
        tiles = []
        for y in range(0, h - self.tile_size + 1, self.tile_size):
            for x in range(0, w - self.tile_size + 1, self.tile_size):
                tile = img[y:y + self.tile_size, x:x + self.tile_size]

                # Convert to single-channel grayscale for tissue detection.
                if len(tile.shape) == 3:
                    gray = np.mean(tile, axis=2)
                else:
                    gray = tile

                # Tissue heuristic: pixels darker than 220 are tissue.
                tissue_ratio = np.sum(gray < 220) / gray.size
                if tissue_ratio >= self.tissue_threshold:
                    tiles.append({
                        'tile': tile,
                        'x': x, 'y': y,
                        'w': self.tile_size, 'h': self.tile_size,
                        'tissue_ratio': tissue_ratio,
                    })

        self.log.info(f'Extracted {len(tiles)} tiles from {h}x{w} image')
        return tiles

    def calculate_stats(self, img, regions):
        """
        Compute intensity statistics for every retained tile.

        Each tile is converted to a float32 grayscale representation and
        passed to :func:`core.statistics.roi_stats`, which returns a dict
        of summary statistics (typically ``mean``, ``std``, ``min``,
        ``max``, ``gradient``, etc.).  The tile's spatial location and
        tissue ratio are appended for downstream traceability.

        Parameters
        ----------
        img : numpy.ndarray
            The full preprocessed image (not used directly -- each tile's
            pixel data is obtained from *regions*).  Kept in the signature
            for compatibility with the ``BasePipeline`` interface.
        regions : list of dict
            Tile dicts produced by :meth:`segment` / :meth:`_extract_tiles`.

        Returns
        -------
        list of dict
            One statistics dict per tile.  In addition to the keys
            returned by ``roi_stats`` (``mean``, ``std``, ``gradient``,
            ...), each dict is augmented with:

            ``tile_x``, ``tile_y`` : int
                Tile origin in the full image.
            ``tissue_ratio`` : float
                Fraction of tissue pixels in the tile.

        Notes
        -----
        Conversion to ``float32`` before computing statistics avoids
        integer-overflow artefacts in variance / standard-deviation
        calculations (uint8 range is only 0--255).
        """
        stats_list = []
        for tile_info in regions:
            tile = tile_info['tile']
            # Convert to float32 grayscale for numerically stable stats.
            if len(tile.shape) == 3:
                gray = np.mean(tile, axis=2).astype(np.float32)
            else:
                gray = tile.astype(np.float32)
            stats = roi_stats(gray)
            # Carry tile location forward so findings can be geo-located.
            stats['tile_x'] = tile_info['x']
            stats['tile_y'] = tile_info['y']
            stats['tissue_ratio'] = tile_info['tissue_ratio']
            stats_list.append(stats)
        return stats_list

    def detect_anomalies(self, img, regions, stats, image_record):
        """
        Flag tiles whose mean intensity deviates significantly from the
        slide-level population as potential tissue anomalies.

        The detection logic is a simple **z-score outlier test**:

        1. Compute the global mean and standard deviation of all per-tile
           mean intensities across the entire image.
        2. For each tile, calculate
           ``z = |tile_mean - global_mean| / global_std``.
        3. Tiles with ``z >= 2.0`` are considered anomalous (roughly the
           top/bottom 2.3 % of a normal distribution).

        For every anomalous tile a *finding* record is created with:

        * **confidence** -- linearly scaled: ``min(1.0, z / 5.0)``.  A
          z-score of 5.0 or above maps to maximum confidence (1.0).
        * **severity** -- determined by
          :func:`core.classification.classify_severity` using the tile's
          gradient magnitude and the computed confidence.
        * **disease_category** -- hard-coded to ``'HISTOLOGY'``.

        Parameters
        ----------
        img : numpy.ndarray
            Full preprocessed image (not used directly; kept for
            interface compatibility).
        regions : list of dict
            Tile dicts from the segmentation stage.
        stats : list of dict
            Per-tile statistics from :meth:`calculate_stats`.
        image_record : dict
            Image metadata (not used directly; part of the stage
            contract).

        Returns
        -------
        list of dict
            A list of finding dictionaries.  Each finding contains:

            ``finding_type`` : str
                Always ``'TISSUE_ANOMALY'``.
            ``severity`` : str
                One of ``'LOW'``, ``'MEDIUM'``, ``'HIGH'`` (from
                ``classify_severity``).
            ``confidence`` : float
                In [0, 1].
            ``statistics`` : dict
                The full per-tile statistics dict.
            ``geometric_properties`` : dict
                ``area`` (px^2) and ``tissue_ratio``.
            ``location`` : dict
                Bounding box ``{x, y, w, h}`` in full-image coordinates.
            ``disease_category`` : str
                ``'HISTOLOGY'``.

        Notes
        -----
        * If ``stats`` is empty or contains no ``'mean'`` key, an empty
          list is returned immediately -- this handles the edge case of
          an image yielding zero tissue tiles.
        * When only a single tile is present, ``global_std`` is set to
          ``1.0`` to avoid division-by-zero; that tile would need a mean
          at least 2.0 intensity units away from itself (impossible), so
          no false finding is generated.
        """
        if not stats:
            return []

        # Collect per-tile means to build the global reference distribution.
        all_means = [s['mean'] for s in stats if 'mean' in s]
        if not all_means:
            return []
        global_mean = np.mean(all_means)
        # Use 1.0 as a safe fallback when only one tile exists to prevent
        # division-by-zero.
        global_std = np.std(all_means) if len(all_means) > 1 else 1.0

        findings = []
        for tile_info, tile_stats in zip(regions, stats):
            mean_val = tile_stats.get('mean', 0)
            z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

            # --- Anomaly gate: z >= 2.0 (approx. p < 0.023 two-tailed) ---
            if z_score < 2.0:
                continue

            gradient = tile_stats.get('gradient', 0)
            # Confidence scales linearly from 0 at z=0 to 1.0 at z=5.0,
            # clamped at 1.0 for very extreme outliers.
            confidence = min(1.0, z_score / 5.0)
            severity = classify_severity(gradient, confidence)

            finding = build_finding(
                finding_type='TISSUE_ANOMALY',
                severity=severity,
                confidence=confidence,
                statistics=tile_stats,
                geometric_properties={
                    'area': tile_info['w'] * tile_info['h'],
                    'tissue_ratio': tile_info['tissue_ratio'],
                },
                location={
                    'x': tile_info['x'],
                    'y': tile_info['y'],
                    'w': tile_info['w'],
                    'h': tile_info['h'],
                },
            )
            finding['disease_category'] = 'HISTOLOGY'
            findings.append(finding)

        return findings
