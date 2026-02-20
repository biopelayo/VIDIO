"""
Authentication Module
=====================

Provides password hashing, JWT token management, and Falcon middleware /
resource classes that together implement the API's authentication layer.

Flow
----
1. A client sends ``POST /auth`` with ``{"username": ..., "password": ...}``.
2. :class:`AuthResource` validates the credentials against the database.
3. On success a signed JWT is returned to the client.
4. Subsequent requests include the token in the ``Authorization`` header.
5. :class:`AuthMiddleware` intercepts every request, verifies the token, and
   injects ``req.context['user_id']`` for downstream resource handlers.

Dependencies
------------
* **bcrypt** -- password hashing and verification.
* **PyJWT** -- JSON Web Token creation and decoding (HS256).
* Configuration values are read from :data:`api.Cfg.gCfg`.
"""

import logging
import datetime as dt

import jwt
import bcrypt
import falcon

from api import Cfg
from api.db.DB import DB


def hash_password(password):
    """Hash a plaintext password using bcrypt.

    A random salt is generated automatically by ``bcrypt.gensalt()``.

    Parameters
    ----------
    password : str
        The plaintext password to hash.

    Returns
    -------
    str
        The bcrypt hash as a UTF-8 string, suitable for storage in the
        database.
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password, password_hash):
    """Verify a plaintext password against a bcrypt hash.

    Parameters
    ----------
    password : str
        The plaintext password supplied by the user.
    password_hash : str
        The stored bcrypt hash to compare against.

    Returns
    -------
    bool
        ``True`` if the password matches the hash, ``False`` otherwise.
    """
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def create_token(user_id):
    """Create a signed JWT for the given user.

    The token payload contains:

    * ``id_user`` -- the user identifier (stringified).
    * ``exp`` -- expiration timestamp, ``now + expiration_days`` from config.
    * ``iat`` -- issued-at timestamp (UTC).

    The token is signed with the ``secret_key`` and ``algorithm`` (HS256)
    specified in ``Cfg.gCfg['auth']``.

    Parameters
    ----------
    user_id : str or int
        The unique identifier of the authenticated user.

    Returns
    -------
    str
        The encoded JWT string.
    """
    cfg = Cfg.gCfg
    now = dt.datetime.utcnow()
    payload = {
        'id_user': str(user_id),
        'exp': now + dt.timedelta(days=cfg['auth']['expiration_days']),
        'iat': now,
    }
    return jwt.encode(payload, cfg['auth']['secret_key'], algorithm=cfg['auth']['algorithm'])


def decode_token(token):
    """Decode and validate a JWT token.

    Parameters
    ----------
    token : str
        The raw JWT string (without the ``Bearer`` / ``JWT`` prefix).

    Returns
    -------
    dict or None
        The decoded payload dictionary on success (containing at least
        ``id_user``, ``exp``, ``iat``).  Returns ``None`` if the token
        is expired (``jwt.ExpiredSignatureError``) or otherwise invalid
        (``jwt.InvalidTokenError``).
    """
    cfg = Cfg.gCfg
    try:
        return jwt.decode(token, cfg['auth']['secret_key'], algorithms=[cfg['auth']['algorithm']])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


class AuthMiddleware:
    """Falcon middleware that enforces JWT authentication on every request.

    For each incoming request the middleware:

    1. Skips processing for CORS preflight (``OPTIONS``) requests.
    2. Skips processing for routes listed in :attr:`EXEMPT_ROUTES`.
    3. Extracts the bearer token from the ``Authorization`` header.
    4. Decodes and validates the JWT via :func:`decode_token`.
    5. On success, stores the user ID in ``req.context['user_id']``.
    6. On failure, raises ``falcon.HTTPUnauthorized``.

    Attributes
    ----------
    EXEMPT_ROUTES : set of str
        URL paths that do **not** require authentication (e.g. the login
        endpoint ``/auth``).
    """

    EXEMPT_ROUTES = {'/auth'}
    EXEMPT_PREFIXES = ('/static',)

    def process_request(self, req, resp):
        """Validate the JWT token present in the request.

        Parameters
        ----------
        req : falcon.Request
            The incoming request. The ``Authorization`` header is inspected.
        resp : falcon.Response
            The outgoing response (unused but required by Falcon).

        Raises
        ------
        falcon.HTTPUnauthorized
            If the token is missing, expired, or invalid.

        Side Effects
        ------------
        On success, ``req.context['user_id']`` is set to the authenticated
        user's identifier extracted from the JWT payload.
        """
        if req.method == 'OPTIONS':
            return

        if req.path in self.EXEMPT_ROUTES:
            return

        if any(req.path.startswith(p) for p in self.EXEMPT_PREFIXES):
            return

        token = self._extract_token(req)
        if not token:
            raise falcon.HTTPUnauthorized(description='Missing authorization token')

        payload = decode_token(token)
        if not payload:
            raise falcon.HTTPUnauthorized(description='Invalid or expired token')

        req.context['user_id'] = payload['id_user']

    def _extract_token(self, req):
        """Extract the raw JWT string from the ``Authorization`` header.

        Accepts headers in the form ``Bearer <token>`` or ``JWT <token>``
        (case-insensitive scheme).

        Parameters
        ----------
        req : falcon.Request
            The incoming Falcon request.

        Returns
        -------
        str or None
            The raw token string, or ``None`` if the header is absent or
            malformed.
        """
        auth_header = req.get_header('Authorization')
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) == 2 and parts[0].upper() in ('JWT', 'BEARER'):
            return parts[1]
        return None


class AuthResource:
    """Falcon resource for the ``POST /auth`` login endpoint.

    Authenticates a user by username / password and returns a signed JWT
    together with basic user profile data.

    Attributes
    ----------
    db : api.db.DB.DB
        Database access object used to look up users and persist tokens.
    """

    def __init__(self):
        """Initialise the resource with a database connection."""
        self.db = DB()

    def on_post(self, req, resp):
        """Handle ``POST /auth`` -- user login.

        Expects a JSON body with ``username`` and ``password`` fields.
        On successful authentication a JWT is created, persisted via
        ``db.SaveToken``, and returned to the client alongside the user
        profile.

        Parameters
        ----------
        req : falcon.Request
            The incoming request containing the JSON credentials.
        resp : falcon.Response
            The response object.

        Response (200)
        --------------
        .. code-block:: json

            {
                "token": "<jwt_string>",
                "user": {
                    "id": "<uuid>",
                    "username": "<str>",
                    "name": "<str>",
                    "role": "<str>"
                }
            }

        Error Responses
        ---------------
        * **400** -- ``username`` or ``password`` missing from body.
        * **401** -- invalid credentials.
        * **500** -- unexpected server error.
        """
        log = logging.getLogger(__name__)
        try:
            import json
            data = json.loads(req.bounded_stream.read(req.content_length or 0))
            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                resp.status = falcon.HTTP_400
                resp.media = {'error': 'username and password required'}
                return

            user = self.db.GetUserByUsername(username)
            if not user or not verify_password(password, user['password_hash']):
                resp.status = falcon.HTTP_401
                resp.media = {'error': 'Invalid credentials'}
                return

            token = create_token(user['id'])
            self.db.SaveToken(user['id'], token)

            resp.status = falcon.HTTP_200
            resp.media = {
                'token': token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'name': user['name'],
                    'role': user['role'],
                }
            }
        except Exception as ex:
            log.error(str(ex))
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
