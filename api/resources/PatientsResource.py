"""
Patients Resource
=================

Falcon resource handling CRUD operations on patient records.

Routes
------
* ``GET  /patients``              -- list / filter patients.
* ``POST /patients``              -- create or update a patient.
* ``GET  /patients/{entity_id}``  -- retrieve a single patient.
* ``POST /patients/{entity_id}``  -- update a specific patient.
* ``DELETE /patients/{entity_id}``-- delete a specific patient.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class PatientsResource(BaseResource):
    """Falcon resource for the ``/patients`` endpoint family.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    All methods require JWT authentication (enforced by
    :class:`~api.Authentication.AuthMiddleware`).
    """

    def on_get(self, req, resp):
        """Handle ``GET /patients`` -- list or filter patient records.

        Parameters
        ----------
        req : falcon.Request
            May contain an optional ``?filter={...}`` query parameter
            with a JSON-encoded filter object.
        resp : falcon.Response
            On success, ``resp.media`` is set to a list of patient dicts.

        Response Codes
        --------------
        * **200** -- patients retrieved successfully.
        * **400** -- database-layer validation error (``DBException``).
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetPatients(flt)
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
        """Handle ``POST /patients`` -- create or update a patient.

        If the JSON body contains an ``id`` key the request is treated as
        an **update** (``ModifyPatient``); otherwise a new patient record
        is **created** (``AddPatient``).

        Parameters
        ----------
        req : falcon.Request
            JSON body with patient fields. Must include ``id`` for updates.
        resp : falcon.Response
            On success, ``resp.media`` contains the created / updated
            patient dict.

        Response Codes
        --------------
        * **200** -- patient created or updated.
        * **400** -- validation error from the database layer.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifyPatient(user_id, data)
            else:
                result = self.db.AddPatient(user_id, data)
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
        """Handle ``GET /patients/{entity_id}`` -- retrieve a single patient.

        Parameters
        ----------
        req : falcon.Request
            The incoming request (no special parameters needed).
        resp : falcon.Response
            On success, ``resp.media`` is the patient dict.
        entity_id : str
            UUID of the patient to retrieve (from the URL path).

        Response Codes
        --------------
        * **200** -- patient found and returned.
        * **404** -- no patient exists with the given ID.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetPatient(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Patient not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_uuid(self, req, resp, entity_id):
        """Handle ``POST /patients/{entity_id}`` -- update a specific patient.

        The ``entity_id`` from the URL is injected into the body data so
        the database layer knows which record to modify.

        Parameters
        ----------
        req : falcon.Request
            JSON body with the fields to update.
        resp : falcon.Response
            On success, ``resp.media`` contains the updated patient dict.
        entity_id : str
            UUID of the patient to update (from the URL path).

        Response Codes
        --------------
        * **200** -- patient updated successfully.
        * **400** -- validation error from the database layer.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            data['id'] = entity_id
            user_id = self.get_user_id(req)
            result = self.db.ModifyPatient(user_id, data)
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
        """Handle ``DELETE /patients/{entity_id}`` -- delete a patient.

        Parameters
        ----------
        req : falcon.Request
            The incoming request (authenticated user extracted via
            :meth:`get_user_id`).
        resp : falcon.Response
            On success, ``resp.media`` is ``{'deleted': '<entity_id>'}``.
        entity_id : str
            UUID of the patient to delete (from the URL path).

        Response Codes
        --------------
        * **200** -- patient deleted successfully.
        * **400** -- database-layer validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeletePatient(user_id, entity_id)
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
