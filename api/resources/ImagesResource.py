"""
Images Resource
===============

Falcon resource handling CRUD operations on individual image records
within the VIDIO platform.

Routes
------
* ``GET    /images``              -- list / filter images.
* ``POST   /images``              -- create or update an image record.
* ``GET    /images/{entity_id}``  -- retrieve a single image record.
* ``DELETE /images/{entity_id}``  -- delete an image record.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class ImagesResource(BaseResource):
    """Falcon resource for the ``/images`` endpoint family.

    Supports listing, creating, retrieving, and deleting image metadata
    records.  Note that actual file data is uploaded via
    :class:`~api.resources.UploadsResource.UploadsResource`.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /images`` -- list or filter image records.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of image dicts.

        Response Codes
        --------------
        * **200** -- images retrieved.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetImages(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post(self, req, resp):
        """Handle ``POST /images`` -- create or update an image record.

        If the JSON body contains an ``id`` key the request is treated as
        an update; otherwise a new image record is created.

        Parameters
        ----------
        req : falcon.Request
            JSON body with image metadata fields.
        resp : falcon.Response
            On success, ``resp.media`` is the created / updated image dict.

        Response Codes
        --------------
        * **200** -- image created or updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifyImage(user_id, data)
            else:
                result = self.db.AddImage(user_id, data)
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
        """Handle ``GET /images/{entity_id}`` -- retrieve a single image record.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the image dict.
        entity_id : str
            UUID of the image.

        Response Codes
        --------------
        * **200** -- image found.
        * **404** -- image not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetImage(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Image not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_delete_uuid(self, req, resp, entity_id):
        """Handle ``DELETE /images/{entity_id}`` -- delete an image record.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is ``{'deleted': '<entity_id>'}``.
        entity_id : str
            UUID of the image to delete.

        Response Codes
        --------------
        * **200** -- image deleted.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeleteImage(user_id, entity_id)
            resp.status = falcon.HTTP_200
            resp.media = {'deleted': entity_id}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
