"""
VIDIO Process Management
=========================

Asynchronous pipeline orchestration for the VIDIO platform.

This module provides the bridge between the REST API
(:class:`~api.resources.AnalysisResource.AnalysisResource`) and the
modality-specific processing pipelines.  When a user triggers an
analysis via ``POST /analysis/<modality>``, this module:

1. Dynamically imports the correct pipeline class based on the
   ``modality`` string (``retinal``, ``histology``, ``radiology``,
   ``spatial``).
2. Launches the pipeline in a **daemon thread** so the API can return
   immediately with an ``HTTP 202 Accepted`` and the process UUID.
3. The pipeline itself manages its own progress updates and finding
   persistence via the :class:`~pipelines.base.BasePipeline` base class.

Thread Safety
-------------
Each pipeline creates its own :class:`~api.db.DB.DB` instance, and the
DB layer uses SQLAlchemy's session-per-method pattern, so concurrent
pipelines do not share database sessions.

Concurrency Limit
-----------------
Currently there is no concurrency limiter (the ``max_concurrent_jobs``
config key is reserved for future use).  Each analysis request spawns
a new thread.

See Also
--------
:mod:`pipelines.base` : Base pipeline class with the 5-stage pattern.
:mod:`api.resources.AnalysisResource` : REST endpoint that calls
    :func:`launch_analysis`.
"""

import os
import sys
import logging
import threading

log = logging.getLogger(__name__)

# Registry of modality → fully-qualified pipeline class path.
# Used for dynamic import so that heavy dependencies (torch, scanpy,
# SimpleITK, etc.) are only loaded when a specific modality is requested.
_PIPELINE_MAP = {
    'retinal': 'pipelines.retinal.pipeline.RetinalPipeline',
    'histology': 'pipelines.histology.pipeline.HistologyPipeline',
    'radiology': 'pipelines.radiology.pipeline.RadiologyPipeline',
    'spatial': 'pipelines.spatial.pipeline.SpatialPipeline',
}


def _import_pipeline_class(modality):
    """Dynamically import and return the pipeline class for a modality.

    Parameters
    ----------
    modality : str
        One of ``"retinal"``, ``"histology"``, ``"radiology"``,
        ``"spatial"``.

    Returns
    -------
    type
        The pipeline class (a subclass of :class:`~pipelines.base.BasePipeline`).

    Raises
    ------
    KeyError
        If ``modality`` is not in :data:`_PIPELINE_MAP`.
    ImportError
        If the pipeline module or its dependencies are not installed.
    """
    module_path, class_name = _PIPELINE_MAP[modality].rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _run_pipeline(process_id, modality, study_id, parameters):
    """Thread target: instantiate and execute a pipeline.

    This function runs inside a daemon thread.  It catches all
    exceptions to prevent silent thread death and records failures
    in the ``process`` table.

    Parameters
    ----------
    process_id : str (UUID)
        Process tracking row ID.
    modality : str
        Pipeline modality key.
    study_id : str (UUID)
        Study to analyse.
    parameters : dict
        Runtime parameters for the pipeline.
    """
    try:
        PipelineClass = _import_pipeline_class(modality)
        pipeline = PipelineClass(process_id, study_id, parameters)
        pipeline.run()
    except ImportError as ex:
        log.error(f'Pipeline module not available for {modality}: {ex}')
        from api.db.DB import DB
        db = DB()
        db.ModifyProcess(None, {
            'id': process_id,
            'status': 'FAILED',
            'error_message': f'Pipeline not implemented: {modality}',
        })
    except Exception as ex:
        log.error(f'Pipeline failed for {modality}: {ex}')


def launch_analysis(process_id, modality, study_id, parameters=None):
    """Launch an analysis pipeline in a background thread.

    This is the main entry point called by
    :class:`~api.resources.AnalysisResource.AnalysisResource` and by
    :func:`VidioTool.run_analysis`.

    Parameters
    ----------
    process_id : str (UUID)
        The ``process.id`` row that tracks this analysis job.  Must
        already exist in the database (created by the caller).
    modality : str
        Pipeline type.  Must be one of:
        ``"retinal"``, ``"histology"``, ``"radiology"``, ``"spatial"``.
    study_id : str (UUID)
        The study to analyse.
    parameters : dict, optional
        Modality-specific runtime parameters (e.g. ``tile_size``,
        ``window_center``, ``min_genes_per_spot``).

    Returns
    -------
    threading.Thread
        The daemon thread running the pipeline.  Can be joined for
        synchronous execution (as done in :func:`VidioTool.run_analysis`).

    Raises
    ------
    ValueError
        If ``modality`` is not in :data:`_PIPELINE_MAP`.
    """
    if modality not in _PIPELINE_MAP:
        raise ValueError(f'Unknown modality: {modality}. Available: {list(_PIPELINE_MAP.keys())}')

    log.info(f'Launching {modality} analysis: process={process_id}, study={study_id}')

    thread = threading.Thread(
        target=_run_pipeline,
        args=(process_id, modality, study_id, parameters or {}),
        daemon=True,
        name=f'vidio-{modality}-{process_id[:8]}',
    )
    thread.start()
    return thread
