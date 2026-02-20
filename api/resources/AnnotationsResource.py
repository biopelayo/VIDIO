"""
Annotations Resource
====================

Falcon resource handling CRUD operations on annotation records. Annotations
represent user-created markings, labels, or regions of interest drawn on
medical images.

Routes
------
* ``GET    /annotations``              -- list / filter annotations.
* ``POST   /annotations``              -- create or update an annotation.
* ``GET    /annotations/{entity_id}``  -- retrieve a single annotation.
* ``POST   /annotations/{entity_id}``  -- update a specific annotation.
* ``DELETE /annotations/{entity_id}``  -- delete a specific annotation.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class AnnotationsResource(BaseResource):
    """Falcon resource for the ``/annotations`` endpoint family.

    Supports full CRUD on annotation records.  Annotations are typically
    linked to an image and contain geometry (e.g. polygons, bounding
    boxes) together with classification labels.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /annotations`` -- list or filter annotations.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of annotation dicts.

        Response Codes
        --------------
        * **200** -- annotations retrieved.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetAnnotations(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post(self, req, resp):
        """Handle ``POST /annotations`` -- create or update an annotation.

        If the JSON body contains an ``id`` key the request is treated as
        an update; otherwise a new annotation is created.

        Parameters
        ----------
        req : falcon.Request
            JSON body with annotation fields (geometry, labels, etc.).
        resp : falcon.Response
            On success, ``resp.media`` is the created / updated annotation
            dict.

        Response Codes
        --------------
        * **200** -- annotation created or updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifyAnnotation(user_id, data)
            else:
                result = self.db.AddAnnotation(user_id, data)
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
        """Handle ``GET /annotations/{entity_id}`` -- retrieve a single annotation.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the annotation dict.
        entity_id : str
            UUID of the annotation.

        Response Codes
        --------------
        * **200** -- annotation found.
        * **404** -- annotation not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetAnnotation(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Annotation not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_uuid(self, req, resp, entity_id):
        """Handle ``POST /annotations/{entity_id}`` -- update a specific annotation.

        Parameters
        ----------
        req : falcon.Request
            JSON body with the fields to update.
        resp : falcon.Response
            On success, ``resp.media`` is the updated annotation dict.
        entity_id : str
            UUID of the annotation to update.

        Response Codes
        --------------
        * **200** -- annotation updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            data['id'] = entity_id
            user_id = self.get_user_id(req)
            result = self.db.ModifyAnnotation(user_id, data)
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
        """Handle ``DELETE /annotations/{entity_id}`` -- delete an annotation.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is ``{'deleted': '<entity_id>'}``.
        entity_id : str
            UUID of the annotation to delete.

        Response Codes
        --------------
        * **200** -- annotation deleted.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeleteAnnotation(user_id, entity_id)
            resp.status = falcon.HTTP_200
            resp.media = {'deleted': entity_id}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
