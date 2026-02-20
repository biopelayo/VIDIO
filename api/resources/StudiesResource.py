"""
Studies Resource
================

Falcon resource handling CRUD operations on medical study records, plus
sub-resource access to the series and findings that belong to a study.

Routes
------
* ``GET    /studies``                       -- list / filter studies.
* ``POST   /studies``                       -- create or update a study.
* ``GET    /studies/{entity_id}``           -- retrieve a single study.
* ``POST   /studies/{entity_id}``           -- update a specific study.
* ``DELETE /studies/{entity_id}``           -- delete a specific study.
* ``GET    /studies/{entity_id}/series``    -- list series for a study.
* ``GET    /studies/{entity_id}/findings``  -- list findings for a study.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class StudiesResource(BaseResource):
    """Falcon resource for the ``/studies`` endpoint family.

    In addition to standard CRUD methods this resource exposes two
    sub-resource GET endpoints for retrieving the series and findings
    associated with a particular study.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /studies`` -- list or filter study records.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of study dicts.

        Response Codes
        --------------
        * **200** -- studies retrieved successfully.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetStudies(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except DBException as ex:
            log.error(ex)
            resp.status = falcon.HTTP_400
            resp.media = {'error': str(ex)}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post(self, req, resp):
        """Handle ``POST /studies`` -- create or update a study.

        If the JSON body contains an ``id`` key the request is treated as
        an update; otherwise a new study is created.

        Parameters
        ----------
        req : falcon.Request
            JSON body with study fields.
        resp : falcon.Response
            On success, ``resp.media`` is the created / updated study dict.

        Response Codes
        --------------
        * **200** -- study created or updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifyStudy(user_id, data)
            else:
                result = self.db.AddStudy(user_id, data)
            resp.media = result
            resp.status = falcon.HTTP_200
        except DBException as ex:
            log.error(ex)
            resp.status = falcon.HTTP_400
            resp.media = {'error': str(ex)}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_uuid(self, req, resp, entity_id):
        """Handle ``GET /studies/{entity_id}`` -- retrieve a single study.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the study dict.
        entity_id : str
            UUID of the study (from the URL path).

        Response Codes
        --------------
        * **200** -- study found.
        * **404** -- study not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetStudy(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Study not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_uuid(self, req, resp, entity_id):
        """Handle ``POST /studies/{entity_id}`` -- update a specific study.

        Parameters
        ----------
        req : falcon.Request
            JSON body with the fields to update.
        resp : falcon.Response
            On success, ``resp.media`` is the updated study dict.
        entity_id : str
            UUID of the study to update.

        Response Codes
        --------------
        * **200** -- study updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            data['id'] = entity_id
            user_id = self.get_user_id(req)
            result = self.db.ModifyStudy(user_id, data)
            resp.media = result
            resp.status = falcon.HTTP_200
        except DBException as ex:
            log.error(ex)
            resp.status = falcon.HTTP_400
            resp.media = {'error': str(ex)}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_delete_uuid(self, req, resp, entity_id):
        """Handle ``DELETE /studies/{entity_id}`` -- delete a study.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is ``{'deleted': '<entity_id>'}``.
        entity_id : str
            UUID of the study to delete.

        Response Codes
        --------------
        * **200** -- study deleted.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeleteStudy(user_id, entity_id)
            resp.status = falcon.HTTP_200
            resp.media = {'deleted': entity_id}
        except DBException as ex:
            log.error(ex)
            resp.status = falcon.HTTP_400
            resp.media = {'error': str(ex)}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_series(self, req, resp, entity_id):
        """Handle ``GET /studies/{entity_id}/series`` -- list series for a study.

        Returns all imaging series that belong to the specified study.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is a list of series dicts.
        entity_id : str
            UUID of the parent study.

        Response Codes
        --------------
        * **200** -- series list returned.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetSeriesForStudy(entity_id)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_findings(self, req, resp, entity_id):
        """Handle ``GET /studies/{entity_id}/findings`` -- list findings for a study.

        Returns all analysis findings associated with the specified study.
        An optional ``?filter={...}`` query parameter narrows the results.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of finding dicts.
        entity_id : str
            UUID of the parent study.

        Response Codes
        --------------
        * **200** -- findings list returned.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetFindingsForStudy(entity_id, flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
