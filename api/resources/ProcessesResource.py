"""
Processes Resource
==================

Falcon resource for querying analysis process records. Processes track
the lifecycle of asynchronous analysis jobs (status, parameters, results).

Routes
------
* ``GET /processes``              -- list / filter processes.
* ``GET /processes/{entity_id}``  -- retrieve a single process.

Notes
-----
Process records are created internally by the analysis launch flow (see
:class:`~api.resources.AnalysisResource.AnalysisResource`) and are
read-only through this endpoint.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource


class ProcessesResource(BaseResource):
    """Falcon resource for the ``/processes`` endpoint family.

    Provides read-only access to analysis process records.  Process
    creation is handled by the analysis endpoints, not directly through
    this resource.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /processes`` -- list or filter process records.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter
            (e.g. filter by status or study).
        resp : falcon.Response
            On success, ``resp.media`` is a list of process dicts.

        Response Codes
        --------------
        * **200** -- processes retrieved.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetProcesses(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_uuid(self, req, resp, entity_id):
        """Handle ``GET /processes/{entity_id}`` -- retrieve a single process.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the process dict (includes
            ``status``, ``parameters``, ``type``, etc.).
        entity_id : str
            UUID of the process.

        Response Codes
        --------------
        * **200** -- process found.
        * **404** -- process not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetProcess(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Process not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
