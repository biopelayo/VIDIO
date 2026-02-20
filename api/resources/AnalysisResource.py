"""
Analysis Resource
=================

Falcon resource that exposes endpoints for launching asynchronous medical
image analysis pipelines. Four imaging modalities are supported, each
mapped to its own POST endpoint:

* ``POST /analysis/retinal``   -- retinal image analysis.
* ``POST /analysis/histology`` -- histology slide analysis.
* ``POST /analysis/radiology`` -- radiology (CT / MRI) analysis.
* ``POST /analysis/spatial``   -- spatial transcriptomics analysis.

All four endpoints share the same internal logic provided by the
:meth:`AnalysisResource._launch_analysis` helper, which validates the
study, creates a ``PENDING`` process record, and delegates actual
execution to :func:`ProcessManagement.launch_analysis`.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class AnalysisResource(BaseResource):
    """Falcon resource for the ``/analysis`` endpoint family.

    Provides four modality-specific POST endpoints that all funnel into
    the shared :meth:`_launch_analysis` helper.  The helper validates
    input, creates a process record, and dispatches the analysis job to
    the background pipeline runner.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def _launch_analysis(self, req, resp, modality):
        """Shared helper that validates, records, and dispatches an analysis job.

        This method is called by each modality-specific ``on_post_*``
        handler.  It performs the following steps:

        1. Reads ``id_study`` (required) and optional ``parameters`` from
           the request body.
        2. Validates that the referenced study exists.
        3. Creates a ``PENDING`` process record in the database.
        4. Calls :func:`ProcessManagement.launch_analysis` to start the
           background pipeline.

        Parameters
        ----------
        req : falcon.Request
            JSON body must contain at minimum::

                {"id_study": "<uuid>", "parameters": {...}}

        resp : falcon.Response
            On success (HTTP 202), ``resp.media`` contains::

                {
                    "process_id": "<uuid>",
                    "status": "PENDING",
                    "message": "<modality> analysis queued"
                }

        modality : str
            One of ``'retinal'``, ``'histology'``, ``'radiology'``, or
            ``'spatial'``.

        Response Codes
        --------------
        * **202** -- analysis job accepted and queued.
        * **400** -- ``id_study`` missing, or database validation error.
        * **404** -- referenced study not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            study_id = data.get('id_study')

            if not study_id:
                resp.status = falcon.HTTP_400
                resp.media = {'error': 'id_study is required'}
                return

            study = self.db.GetStudy(study_id)
            if not study:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Study not found'}
                return

            process_data = {
                'id_user': user_id,
                'id_study': study_id,
                'type': f'{modality.upper()}_ANALYSIS',
                'status': 'PENDING',
                'parameters': data.get('parameters', {}),
            }
            process = self.db.AddProcess(user_id, process_data)

            from ProcessManagement import launch_analysis
            launch_analysis(process['id'], modality, study_id, data.get('parameters', {}))

            resp.media = {
                'process_id': process['id'],
                'status': 'PENDING',
                'message': f'{modality} analysis queued',
            }
            resp.status = falcon.HTTP_202
        except DBException as ex:
            log.error(ex)
            resp.status = falcon.HTTP_400
            resp.media = {'error': str(ex)}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_retinal(self, req, resp):
        """Handle ``POST /analysis/retinal`` -- launch a retinal image analysis.

        Parameters
        ----------
        req : falcon.Request
            JSON body with ``id_study`` and optional ``parameters``.
        resp : falcon.Response
            Delegated to :meth:`_launch_analysis`.

        See Also
        --------
        _launch_analysis : Shared implementation for all modality endpoints.
        """
        self._launch_analysis(req, resp, 'retinal')

    def on_post_histology(self, req, resp):
        """Handle ``POST /analysis/histology`` -- launch a histology slide analysis.

        Parameters
        ----------
        req : falcon.Request
            JSON body with ``id_study`` and optional ``parameters``.
        resp : falcon.Response
            Delegated to :meth:`_launch_analysis`.

        See Also
        --------
        _launch_analysis : Shared implementation for all modality endpoints.
        """
        self._launch_analysis(req, resp, 'histology')

    def on_post_radiology(self, req, resp):
        """Handle ``POST /analysis/radiology`` -- launch a radiology (CT/MRI) analysis.

        Parameters
        ----------
        req : falcon.Request
            JSON body with ``id_study`` and optional ``parameters``.
        resp : falcon.Response
            Delegated to :meth:`_launch_analysis`.

        See Also
        --------
        _launch_analysis : Shared implementation for all modality endpoints.
        """
        self._launch_analysis(req, resp, 'radiology')

    def on_post_spatial(self, req, resp):
        """Handle ``POST /analysis/spatial`` -- launch a spatial transcriptomics analysis.

        Parameters
        ----------
        req : falcon.Request
            JSON body with ``id_study`` and optional ``parameters``.
        resp : falcon.Response
            Delegated to :meth:`_launch_analysis`.

        See Also
        --------
        _launch_analysis : Shared implementation for all modality endpoints.
        """
        self._launch_analysis(req, resp, 'spatial')
