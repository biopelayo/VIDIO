"""
Radiology Pipeline -- Core Orchestrator
========================================

Implements :class:`RadiologyPipeline`, the central entry point for all
CT and MRI volume analysis.  The class inherits from
:class:`pipelines.base.BasePipeline` and overrides each abstract stage
(load, preprocess, segment, stats, anomaly detection) with
radiology-specific logic.

Design Decisions
----------------
* **Lazy imports** -- Heavy dependencies (``pydicom``, ``SimpleITK``,
  ``nibabel``, ``core.edges``, etc.) are imported inside methods so that
  the module loads quickly even when only a subset of functionality is
  needed.
* **Modality branching** -- Preprocessing diverges for CT (HU windowing)
  vs MRI (min-max normalisation) based on the ``modality`` key in
  ``image_record['info']``.
* **2D / 3D dispatch** -- Segmentation automatically selects a 2D
  single-image path or a 3D slice-by-slice path depending on the
  dimensionality of the input array.
* **Graceful degradation** -- ``None`` inputs are propagated rather than
  raising, so upstream read failures can be reported without crashing the
  pipeline.
"""

import logging
import numpy as np

from pipelines.base import BasePipeline
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


class RadiologyPipeline(BasePipeline):
    """
    Radiology image analysis pipeline for CT and MRI volumes.

    Orchestrates the full analysis chain -- loading, preprocessing,
    segmentation, statistical characterisation, and anomaly detection --
    for radiological images stored as DICOM, NIfTI, or common raster
    formats (PNG/JPEG used for single-slice review images).

    The pipeline is **stateless**: every public method receives its
    required context via arguments and returns results without mutating
    instance attributes beyond those inherited from
    :class:`~pipelines.base.BasePipeline`.

    Processing Stages
    -----------------
    1. ``load_image``   -- Format-aware I/O dispatch.
    2. ``preprocess``    -- Modality-dependent intensity normalisation.
    3. ``segment``       -- Edge-based region extraction (2D or 3D).
    4. ``calculate_stats`` -- Per-region ROI intensity statistics.
    5. ``detect_anomalies`` -- Gradient/confidence severity scoring.

    Attributes
    ----------
    Inherited from :class:`~pipelines.base.BasePipeline`.  No additional
    instance attributes are defined by this subclass.

    Notes
    -----
    * All image arrays are converted to ``np.float32`` during
      preprocessing to ensure consistent numerical behaviour across
      windowing, normalisation, and edge-detection steps.
    * 3D segmentation is performed **slice-by-slice** using 2D Laplacian
      edge detection; a volumetric U-Net alternative is available in
      :mod:`pipelines.radiology.models`.
    * The ``image_record`` dictionary expected by several methods must
      contain at least ``'storage_path'`` (str) and optionally
      ``'file_format'`` (str) and ``'info'`` (dict with ``'modality'``,
      ``'window_center'``, ``'window_width'``).

    See Also
    --------
    pipelines.base.BasePipeline : Abstract base class defining the
        stage interface.
    pipelines.radiology.preprocessing : Standalone preprocessing helpers.
    pipelines.radiology.segmentation : Advanced 3D segmentation and
        volumetry.
    pipelines.radiology.models : MONAI-based 3D U-Net wrapper.

    Examples
    --------
    >>> pipe = RadiologyPipeline()
    >>> record = {'storage_path': '/data/scan.dcm', 'file_format': 'DICOM',
    ...           'info': {'modality': 'CT', 'window_center': 40,
    ...                    'window_width': 400}}
    >>> img = pipe.load_image(record)
    >>> img = pipe.preprocess(img, record)
    >>> regions = pipe.segment(img, record)
    >>> stats = pipe.calculate_stats(img, regions)
    >>> findings = pipe.detect_anomalies(img, regions, stats, record)
    """

    def load_image(self, image_record):
        """
        Load a radiological image from disk.

        Dispatches to one of three readers based on file format:

        1. **NIfTI** -- ``file_format == 'NIFTI'`` or path ending in
           ``.nii`` / ``.nii.gz``.  Uses :func:`core.image_io.read_nifti`
           (backed by *nibabel*), which returns ``(np.ndarray, header)``.
        2. **DICOM** -- ``file_format == 'DICOM'``.  Uses
           :func:`core.image_io.read_dicom` (backed by *pydicom*),
           applying ``RescaleSlope`` and ``RescaleIntercept``
           automatically.
        3. **Generic** -- Anything else (PNG, JPEG, TIFF).  Falls back
           to :func:`core.image_io.read_image` which uses OpenCV /
           PIL.

        Parameters
        ----------
        image_record : dict
            Must contain:

            - ``'storage_path'`` (str) : Absolute path to the image
              file.
            - ``'file_format'`` (str, optional) : One of ``'NIFTI'``,
              ``'DICOM'``, or omitted for generic loading.  Comparison
              is case-insensitive.

        Returns
        -------
        numpy.ndarray or tuple(numpy.ndarray, dict)
            For NIfTI: a ``(data, metadata)`` tuple.  For DICOM and
            generic: a plain ``np.ndarray``.  Returns ``None`` if the
            underlying reader fails.

        Notes
        -----
        The format string is normalised to upper-case before
        comparison.  If ``file_format`` is absent but the path has a
        NIfTI extension, NIfTI loading is still triggered.
        """
        fmt = image_record.get('file_format', '').upper()
        path = image_record['storage_path']

        if fmt == 'NIFTI' or path.endswith(('.nii', '.nii.gz')):
            from core.image_io import read_nifti
            return read_nifti(path)
        elif fmt == 'DICOM':
            from core.image_io import read_dicom
            return read_dicom(path)
        else:
            from core.image_io import read_image
            return read_image(path)

    def preprocess(self, img, image_record):
        """
        Apply modality-specific intensity preprocessing.

        Two branches exist:

        * **CT** (``image_record['info']['modality'] == 'CT'``):
          Hounsfield-Unit windowing via :func:`core.intensity.hu_windowing`.
          Default window (centre=40, width=400) corresponds to **soft
          tissue** visualisation.  Custom values can be supplied via
          ``image_record['info']['window_center']`` and
          ``image_record['info']['window_width']``.
        * **MRI / Other**: Min-max normalisation to [0, 255] via
          :func:`core.intensity.normalize_minmax`, which is appropriate
          for T1/T2/FLAIR sequences where absolute intensity values are
          arbitrary.

        If the input is a ``(data, metadata)`` tuple (as returned by
        NIfTI loaders), only the *data* component is preprocessed; the
        metadata is discarded at this stage.

        Parameters
        ----------
        img : numpy.ndarray or tuple(numpy.ndarray, dict) or None
            Raw image data.  If ``None``, the method returns ``None``
            immediately (graceful propagation of upstream load
            failures).
        image_record : dict
            Image metadata.  Relevant keys:

            - ``'info'`` (dict) :
                - ``'modality'`` (str) : ``'CT'`` triggers HU windowing;
                  anything else triggers min-max normalisation.
                - ``'window_center'`` (float, optional) : HU window
                  centre (default ``40``).
                - ``'window_width'`` (float, optional) : HU window width
                  (default ``400``).

        Returns
        -------
        numpy.ndarray or None
            Preprocessed image as ``np.float32``.  Shape is unchanged.

        Notes
        -----
        The entire array is cast to ``np.float32`` *before* windowing
        or normalisation to avoid integer-overflow artefacts when the
        source DICOM stores pixel data as ``int16``.
        """
        if img is None:
            return img

        if isinstance(img, tuple):
            img_data, metadata = img
        else:
            img_data = img
            metadata = {}

        img_data = img_data.astype(np.float32)

        modality = image_record.get('info', {}).get('modality', '')
        if modality == 'CT':
            from core.intensity import hu_windowing
            window_center = image_record.get('info', {}).get('window_center', 40)
            window_width = image_record.get('info', {}).get('window_width', 400)
            img_data = hu_windowing(img_data, window_center, window_width).astype(np.float32)
        else:
            from core.intensity import normalize_minmax
            img_data = normalize_minmax(img_data, 0, 255)

        return img_data

    def segment(self, img, image_record):
        """
        Segment regions of interest from a preprocessed image.

        Dispatches to either :meth:`_segment_3d` (for volumetric data
        with ``ndim == 3``) or :meth:`_segment_2d` (for single 2D
        slices).  The dispatch is based solely on the *shape* of
        ``img``, not on any metadata flag.

        Parameters
        ----------
        img : numpy.ndarray or None
            Preprocessed image (``float32``).  2D shape ``(H, W)`` or
            3D shape ``(D, H, W)`` / ``(H, W, D)``.  If ``None``,
            returns an empty list.
        image_record : dict
            Image metadata (currently unused inside this method but
            passed for interface consistency with
            :class:`~pipelines.base.BasePipeline`).

        Returns
        -------
        list[dict]
            Each element is a region dictionary produced by
            :func:`core.contours.find_contours`, augmented with a
            ``'z'`` key in ``region['stats']`` for 3D inputs indicating
            the source slice index.

        See Also
        --------
        _segment_2d : Single-slice Laplacian edge segmentation.
        _segment_3d : Slice-by-slice volumetric segmentation.
        """
        if img is None:
            return []

        if len(img.shape) == 3:
            return self._segment_3d(img)
        else:
            return self._segment_2d(img)

    def _segment_2d(self, img):
        """
        Segment a single 2D slice using Laplacian edge detection.

        Processing steps:

        1. **Edge detection** -- Compute second-derivative (Laplacian)
           edge map via :func:`core.edges.laplacian_edges`.
        2. **Thresholding** -- Pixels with absolute edge response above
           ``1.5 * std(edge_map)`` are marked as anomalous.  The 1.5x
           factor is an empirically chosen trade-off between
           sensitivity (detecting faint lesion borders) and specificity
           (avoiding noise in homogeneous tissue).
        3. **Morphological cleanup** -- Close small gaps with a 5x5
           kernel (3 iterations) and remove connected components
           smaller than 200 pixels.
        4. **Contour extraction** -- Extract external contours with a
           minimum area of 100 pixels.

        Parameters
        ----------
        img : numpy.ndarray
            2D image slice of shape ``(H, W)``, dtype ``float32``.

        Returns
        -------
        list[dict]
            Region dictionaries from :func:`core.contours.find_contours`,
            each containing ``'contour'``, ``'stats'`` (with ``'x'``,
            ``'y'``, ``'w'``, ``'h'``, ``'area'``, ``'solidity'``,
            ``'elongation'``).

        Notes
        -----
        The threshold ``1.5 * std`` adapts to the overall contrast of
        each slice, making the method reasonably robust across
        different modalities and intensity ranges.  However, very
        low-contrast slices (e.g. near the vertex in brain MRI) may
        yield few or no regions.
        """
        from core.edges import laplacian_edges
        from core.morphology import morph_close, remove_small_objects
        from core.contours import find_contours

        edge_map, stats = laplacian_edges(img.astype(np.float32))
        threshold = stats['std'] * 1.5
        anomaly_mask = (np.abs(edge_map) > threshold).astype(np.uint8)
        anomaly_mask = morph_close(anomaly_mask, kernel_size=5, iterations=3)
        anomaly_mask = remove_small_objects(anomaly_mask, min_area=200)
        regions = find_contours(anomaly_mask, min_area=100)
        return regions

    def _segment_3d(self, volume):
        """
        Segment a 3D volume by iterating over 2D slices.

        Determines the slice axis using a heuristic: if the *last*
        dimension (``shape[2]``) is smaller than the *first*
        (``shape[0]``), the volume is assumed to be stored as
        ``(H, W, D)`` (common in some DICOM loaders), and slicing
        proceeds along axis 2.  Otherwise the volume is assumed to be
        ``(D, H, W)`` (NumPy / NIfTI convention) and slicing proceeds
        along axis 0.

        Each 2D slice is segmented via :meth:`_segment_2d`, and a
        ``'z'`` key is injected into every region's ``stats`` dict to
        record the source slice index.

        Parameters
        ----------
        volume : numpy.ndarray
            3D array of shape ``(D, H, W)`` or ``(H, W, D)``, dtype
            ``float32``.

        Returns
        -------
        list[dict]
            Flat list of region dictionaries aggregated across all
            slices.  Each region includes ``region['stats']['z']``
            indicating its axial slice index.

        Notes
        -----
        The dimension heuristic (``shape[2] < shape[0]``) works well
        for typical neuroimaging data where the number of axial slices
        (~100-200) is smaller than the in-plane resolution (~256-512).
        It may produce incorrect axis selection for non-standard
        geometries such as very thick-slice acquisitions.
        """
        regions = []
        n_slices = volume.shape[2] if volume.ndim == 3 else volume.shape[0]

        for z in range(n_slices):
            if volume.ndim == 3 and volume.shape[2] < volume.shape[0]:
                slc = volume[:, :, z]
            else:
                slc = volume[z]

            slice_regions = self._segment_2d(slc)
            for r in slice_regions:
                r['stats']['z'] = z
            regions.extend(slice_regions)

        return regions

    def calculate_stats(self, img, regions):
        """
        Compute intensity statistics for each segmented region.

        For 3D images, uses the ``'z'`` key stored by
        :meth:`_segment_3d` to extract the correct axial slice before
        cropping the bounding-box ROI.  The same dimension heuristic
        as :meth:`_segment_3d` is applied (``shape[2] < shape[0]``
        implies ``(H, W, D)`` layout).

        Parameters
        ----------
        img : numpy.ndarray
            The preprocessed image, either 2D ``(H, W)`` or 3D
            ``(D, H, W)`` / ``(H, W, D)``.
        regions : list[dict]
            Region dictionaries returned by :meth:`segment`.  Each
            must contain ``region['stats']`` with at least ``'x'``,
            ``'y'``, ``'w'``, ``'h'``, and optionally ``'z'`` (for
            3D volumes).

        Returns
        -------
        list[dict]
            One statistics dictionary per region, as returned by
            :func:`core.statistics.roi_stats`, augmented with a
            ``'geometric'`` key containing the original bounding-box
            and shape descriptors.  An empty dict ``{}`` is inserted
            for regions whose ROI has zero pixels (e.g. out-of-bounds
            bounding boxes).

        Notes
        -----
        Bounding-box coordinates are clamped to ``max(0, ...)`` to
        avoid negative indexing, but no upper-bound clamping is
        performed -- NumPy's slice semantics handle that gracefully.
        """
        stats_list = []
        for region in regions:
            s = region.get('stats', {})
            x, y, w, h = s.get('x', 0), s.get('y', 0), s.get('w', 0), s.get('h', 0)
            z = s.get('z', None)

            if z is not None and len(img.shape) == 3:
                if img.shape[2] < img.shape[0]:
                    slc = img[:, :, z]
                else:
                    slc = img[z]
            else:
                slc = img

            roi = slc[max(0, y):y + h, max(0, x):x + w]
            if roi.size == 0:
                stats_list.append({})
                continue

            stats = roi_stats(roi.astype(np.float32))
            stats['geometric'] = s
            stats_list.append(stats)

        return stats_list

    def detect_anomalies(self, img, regions, stats, image_record):
        """
        Score segmented regions and classify their clinical severity.

        For each region, the method applies two gate thresholds to
        filter out low-signal noise:

        * ``gradient < 10`` **and** ``std < 5`` -- region is discarded
          as background noise (both conditions must be true to skip).

        Regions that pass the gate are scored:

        * **Confidence** = ``min(1.0, gradient / 100.0)`` -- a linear
          mapping that saturates at gradient = 100.  This heuristic
          assumes that stronger intensity gradients indicate more
          clearly delineated anomalies.
        * **Severity** is determined by
          :func:`core.classification.classify_severity` based on the
          gradient magnitude and confidence score, producing one of
          ``CRITICAL``, ``HIGH``, ``MEDIUM``, or ``LOW``.

        Parameters
        ----------
        img : numpy.ndarray
            Preprocessed image (not directly used in this method, but
            part of the base class interface for pipelines that need
            raw pixel access during anomaly detection).
        regions : list[dict]
            Region dictionaries from :meth:`segment`.
        stats : list[dict]
            Per-region statistics from :meth:`calculate_stats`.  Must
            be aligned index-wise with *regions*.  Keys used:
            ``'gradient'``, ``'std'``, ``'geometric'``.
        image_record : dict
            Image metadata (currently unused but reserved for future
            modality-specific severity adjustments).

        Returns
        -------
        list[dict]
            Findings built by :func:`core.classification.build_finding`,
            each containing ``'finding_type'``, ``'severity'``,
            ``'confidence'``, ``'statistics'``,
            ``'geometric_properties'``, ``'location'`` (with ``x``,
            ``y``, ``z``, ``w``, ``h``), and
            ``'disease_category' == 'RADIOLOGY'``.

        Notes
        -----
        The ``z`` coordinate in ``location`` is ``None`` for 2D
        inputs, preserving downstream consumers' ability to
        distinguish 2D from 3D findings.
        """
        findings = []
        if not regions or not stats:
            return findings

        for region, region_stats in zip(regions, stats):
            if not region_stats:
                continue

            gradient = region_stats.get('gradient', 0)
            std_val = region_stats.get('std', 0)
            geometric = region_stats.get('geometric', {})

            if gradient < 10 and std_val < 5:
                continue

            confidence = min(1.0, gradient / 100.0)
            severity = classify_severity(gradient, confidence)

            finding = build_finding(
                finding_type='RADIOLOGICAL_ANOMALY',
                severity=severity,
                confidence=confidence,
                statistics=region_stats,
                geometric_properties={
                    'area': geometric.get('area', 0),
                    'solidity': geometric.get('solidity', 0),
                    'elongation': geometric.get('elongation', 0),
                },
                location={
                    'x': geometric.get('x', 0),
                    'y': geometric.get('y', 0),
                    'z': geometric.get('z'),
                    'w': geometric.get('w', 0),
                    'h': geometric.get('h', 0),
                },
            )
            finding['disease_category'] = 'RADIOLOGY'
            findings.append(finding)

        return findings
