"""
Base Resource
=============

Provides the :class:`BaseResource` superclass from which every concrete
Falcon resource in the VIDIO API inherits.  It centralises:

* Database access (one :class:`~api.db.DB.DB` instance per resource).
* Query-parameter filter parsing (``?filter={...}``).
* Authenticated-user extraction from the Falcon request context.
* Safe reading of JSON request bodies.
"""

import json
import logging

from api.db.DB import DB


class BaseResource:
    """Abstract base class for all VIDIO API resource handlers.

    Concrete resources (e.g. :class:`PatientsResource`,
    :class:`StudiesResource`) inherit from this class to gain shared
    helper methods and a pre-initialised database connection.

    Attributes
    ----------
    db : api.db.DB.DB
        Shared database access object.
    filter : dict or None
        Placeholder for the most recently loaded query filter. Subclasses
        typically call :meth:`load_filter` to populate this.
    """

    def __init__(self):
        """Initialise the resource with a database connection."""
        self.db = DB()
        self.filter = None

    def load_filter(self, req):
        """Parse the JSON ``filter`` query parameter from the request URL.

        Clients may pass a JSON-encoded filter object as a query string
        parameter, e.g.::

            GET /patients?filter={"name":"John"}

        Parameters
        ----------
        req : falcon.Request
            The incoming Falcon request.

        Returns
        -------
        dict or None
            The parsed filter dictionary, or ``None`` if the parameter is
            absent or cannot be decoded.
        """
        raw = req.get_param('filter')
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def get_user_id(self, req):
        """Retrieve the authenticated user's ID from the request context.

        The ID is placed into ``req.context['user_id']`` by
        :class:`~api.Authentication.AuthMiddleware` during request
        processing.

        Parameters
        ----------
        req : falcon.Request
            The incoming Falcon request.

        Returns
        -------
        str or None
            The user ID string, or ``None`` if not present (should only
            happen on exempt / unauthenticated routes).
        """
        return req.context.get('user_id')

    def read_body(self, req):
        """Read and JSON-decode the request body.

        Parameters
        ----------
        req : falcon.Request
            The incoming Falcon request whose body will be consumed.

        Returns
        -------
        dict
            The parsed JSON object. Returns an empty dictionary if the
            body is empty or cannot be parsed.

        Notes
        -----
        Any parse or I/O error is logged at ``ERROR`` level and silently
        swallowed, returning ``{}``.  This keeps resource handlers simple
        but means callers should validate expected keys after calling
        this method.
        """
        log = logging.getLogger(__name__)
        try:
            raw = req.bounded_stream.read(req.content_length or 0)
            if not raw:
                return {}
            return json.loads(raw)
        except Exception as ex:
            log.error(f'Error reading request body: {ex}')
            return {}
