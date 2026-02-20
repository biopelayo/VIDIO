"""
Users Resource
==============

Falcon resource handling CRUD operations on user accounts. Passwords are
hashed with bcrypt before storage and ``password_hash`` is stripped from
all responses to prevent accidental credential leakage.

Routes
------
* ``GET    /users``              -- list / filter users.
* ``POST   /users``              -- create or update a user.
* ``GET    /users/{entity_id}``  -- retrieve a single user.
* ``POST   /users/{entity_id}``  -- update a specific user.
* ``DELETE /users/{entity_id}``  -- delete a user.
"""

import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException
from api.Authentication import hash_password


class UsersResource(BaseResource):
    """Falcon resource for the ``/users`` endpoint family.

    Handles user account management.  Password fields supplied by the
    client are transparently hashed before being forwarded to the
    database layer, and ``password_hash`` is always removed from
    outgoing responses.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_get(self, req, resp):
        """Handle ``GET /users`` -- list or filter user accounts.

        The ``password_hash`` field is stripped from every user record
        before the response is sent.

        Parameters
        ----------
        req : falcon.Request
            May contain ``?filter={...}`` with a JSON-encoded filter.
        resp : falcon.Response
            On success, ``resp.media`` is a list of user dicts (without
            ``password_hash``).

        Response Codes
        --------------
        * **200** -- users retrieved.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetUsers(flt)
            for u in result:
                u.pop('password_hash', None)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post(self, req, resp):
        """Handle ``POST /users`` -- create or update a user account.

        If the body contains a ``password`` field it is hashed via
        :func:`~api.Authentication.hash_password` and replaced with
        ``password_hash`` before being forwarded to the database.

        For **new** users (no ``id`` in body), a ``password`` field is
        mandatory.  For **updates** (``id`` present), the password is
        optional.

        Parameters
        ----------
        req : falcon.Request
            JSON body with user fields. Must include ``password`` for
            creation.
        resp : falcon.Response
            On success, ``resp.media`` is the created / updated user dict
            (without ``password_hash``).

        Response Codes
        --------------
        * **200** -- user created or updated.
        * **400** -- missing password for new user, or database validation
          error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)

            if 'password' in data:
                data['password_hash'] = hash_password(data.pop('password'))

            if 'id' in data:
                result = self.db.ModifyUser(user_id, data)
            else:
                if 'password_hash' not in data:
                    resp.status = falcon.HTTP_400
                    resp.media = {'error': 'password is required for new users'}
                    return
                result = self.db.AddUser(user_id, data)

            result.pop('password_hash', None)
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
        """Handle ``GET /users/{entity_id}`` -- retrieve a single user.

        The ``password_hash`` field is stripped before returning.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is the user dict (without
            ``password_hash``).
        entity_id : str
            UUID of the user.

        Response Codes
        --------------
        * **200** -- user found.
        * **404** -- user not found.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetUser(entity_id)
            if result:
                result.pop('password_hash', None)
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'User not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_uuid(self, req, resp, entity_id):
        """Handle ``POST /users/{entity_id}`` -- update a specific user.

        If the body includes a ``password`` field it is hashed before
        storage.  The ``password_hash`` is stripped from the response.

        Parameters
        ----------
        req : falcon.Request
            JSON body with the fields to update.
        resp : falcon.Response
            On success, ``resp.media`` is the updated user dict (without
            ``password_hash``).
        entity_id : str
            UUID of the user to update.

        Response Codes
        --------------
        * **200** -- user updated.
        * **400** -- database validation error.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            data['id'] = entity_id
            user_id = self.get_user_id(req)
            if 'password' in data:
                data['password_hash'] = hash_password(data.pop('password'))
            result = self.db.ModifyUser(user_id, data)
            result.pop('password_hash', None)
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
        """Handle ``DELETE /users/{entity_id}`` -- delete a user account.

        Parameters
        ----------
        req : falcon.Request
            The incoming request.
        resp : falcon.Response
            On success, ``resp.media`` is ``{'deleted': '<entity_id>'}``.
        entity_id : str
            UUID of the user to delete.

        Response Codes
        --------------
        * **200** -- user deleted.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeleteUser(user_id, entity_id)
            resp.status = falcon.HTTP_200
            resp.media = {'deleted': entity_id}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
