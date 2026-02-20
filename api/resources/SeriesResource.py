"""
Series Resource
===============

Falcon resource handling CRUD operations on imaging series records, plus
a sub-resource endpoint for listing the images within a series.

Routes
------
* ``GET    /series``                      -- list / filter series.
* ``POST   /series``                      -- create or update a series.
* ``GET    /series/{entity_id}``          -- retrieve a single series.
* ``POST   /series/{entity_id}``          -- update a specific series.
* ``DELETE /series/{entity_id}``          -- delete a specific series.
* ``GET    /series/{entity_id}/images``   -- list images for a series.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class SeriesResource(BaseResource):
    """Falcon resource for the ``/series`` endpoint family.

    Provides standard CRUD plus a sub-resource ``/series/{id}/images``
    endpoint for retrieving the images that belong to a series.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /series`` -- list or filter series records.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of series dicts.

        Response Codes
        --------------
        * **200** -- series retrieved.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetAllSeries(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post(self, req, resp):
        """Handle ``POST /series`` -- create or update a series.

        If the JSON body contains an ``id`` key the request is treated as
        an update; otherwise a new series record is created.

        Parameters
        ----------
        req : falcon.Request
            JSON body with series fields.
        resp : falcon.Response
            On success, ``resp.media`` is the created / updated series dict.

        Response Codes
        --------------
        * **200** -- series created or updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifySeries(user_id, data)
            else:
                result = self.db.AddSeries(user_id, data)
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
        """Handle ``GET /series/{entity_id}`` -- retrieve a single series.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the series dict.
        entity_id : str
            UUID of the series.

        Response Codes
        --------------
        * **200** -- series found.
        * **404** -- series not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetSeries(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Series not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_uuid(self, req, resp, entity_id):
        """Handle ``POST /series/{entity_id}`` -- update a specific series.

        Parameters
        ----------
        req : falcon.Request
            JSON body with the fields to update.
        resp : falcon.Response
            On success, ``resp.media`` is the updated series dict.
        entity_id : str
            UUID of the series to update.

        Response Codes
        --------------
        * **200** -- series updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            data['id'] = entity_id
            user_id = self.get_user_id(req)
            result = self.db.ModifySeries(user_id, data)
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
        """Handle ``DELETE /series/{entity_id}`` -- delete a series.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is ``{'deleted': '<entity_id>'}``.
        entity_id : str
            UUID of the series to delete.

        Response Codes
        --------------
        * **200** -- series deleted.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeleteSeries(user_id, entity_id)
            resp.status = falcon.HTTP_200
            resp.media = {'deleted': entity_id}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_images(self, req, resp, entity_id):
        """Handle ``GET /series/{entity_id}/images`` -- list images in a series.

        Returns all image records that belong to the specified series.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is a list of image dicts.
        entity_id : str
            UUID of the parent series.

        Response Codes
        --------------
        * **200** -- image list returned.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetImagesForSeries(entity_id)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
