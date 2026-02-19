import os
import sys
import logging
import threading

log = logging.getLogger(__name__)

_PIPELINE_MAP = {
    'retinal': 'pipelines.retinal.pipeline.RetinalPipeline',
    'histology': 'pipelines.histology.pipeline.HistologyPipeline',
    'radiology': 'pipelines.radiology.pipeline.RadiologyPipeline',
    'spatial': 'pipelines.spatial.pipeline.SpatialPipeline',
}


def _import_pipeline_class(modality):
    module_path, class_name = _PIPELINE_MAP[modality].rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _run_pipeline(process_id, modality, study_id, parameters):
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
