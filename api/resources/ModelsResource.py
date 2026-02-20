"""
Models Resource
===============

Falcon resource handling CRUD operations on machine-learning model
registry records.  Each model record stores metadata such as name,
version, modality, and the path to the serialised weights.

Routes
------
* ``GET  /models``              -- list / filter registered ML models.
* ``POST /models``              -- create or update a model record.
* ``GET  /models/{entity_id}``  -- retrieve a single model record.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class ModelsResource(BaseResource):
    """Falcon resource for the ``/models`` endpoint family.

    Provides listing, creation / update, and single-record retrieval for
    ML model registry entries used by the analysis pipelines.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /models`` -- list or filter ML model records.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of model dicts.

        Response Codes
        --------------
        * **200** -- models retrieved.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetMLModels(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post(self, req, resp):
        """Handle ``POST /models`` -- create or update a model record.

        If the JSON body contains an ``id`` key the request is treated as
        an update; otherwise a new model record is created.

        Parameters
        ----------
        req : falcon.Request
            JSON body with model metadata fields (name, version, path,
            modality, etc.).
        resp : falcon.Response
            On success, ``resp.media`` is the created / updated model dict.

        Response Codes
        --------------
        * **200** -- model created or updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifyMLModel(user_id, data)
            else:
                result = self.db.AddMLModel(user_id, data)
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
        """Handle ``GET /models/{entity_id}`` -- retrieve a single model.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the model dict.
        entity_id : str
            UUID of the model.

        Response Codes
        --------------
        * **200** -- model found.
        * **404** -- model not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetMLModel(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Model not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
