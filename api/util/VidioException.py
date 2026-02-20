"""
Custom Exception Hierarchy
==========================

Defines the application-specific exception classes used throughout the
VIDIO API. Every exception carries a structured payload (code, operation,
message) that serialises to JSON, making it straightforward to convert
into an HTTP error response.

Class Hierarchy
---------------
::

    Exception
     +-- VidioException        (base for all VIDIO errors)
          +-- DBException      (database / persistence layer errors)
          +-- WSException      (web-service / HTTP layer errors)
"""

import json


class VidioException(Exception):
    """Base exception for all VIDIO application errors.

    Stores a machine-readable ``code``, the ``operation`` that failed,
    and a human-readable ``message``.  Its ``__str__`` method returns a
    JSON representation of the error, which resource handlers can embed
    directly in HTTP error responses.

    Parameters
    ----------
    code : int or str
        A machine-readable error code (e.g. HTTP status code or internal
        error identifier).
    operation : str
        Short label for the operation that triggered the exception
        (e.g. ``'GetPatient'``, ``'AddStudy'``).
    message : str
        Human-readable description of the error.

    Attributes
    ----------
    code : int or str
        The error code supplied at construction time.
    operation : str
        The originating operation label.
    message : str
        The human-readable error description.
    """

    def __init__(self, code, operation, message):
        self.code = code
        self.operation = operation
        self.message = message

    def __str__(self):
        """Return a JSON string representation of the error.

        Returns
        -------
        str
            JSON object with keys ``code``, ``operation``, and ``message``.
        """
        return json.dumps({
            'code': self.code,
            'operation': self.operation,
            'message': self.message,
        })


class DBException(VidioException):
    """Exception raised by the database / persistence layer.

    Inherits from :class:`VidioException` without modification. Using a
    distinct type allows resource handlers to differentiate DB errors
    (typically mapped to HTTP 400) from other failures.

    Parameters
    ----------
    code : int or str
        Error code.
    operation : str
        The database operation that failed (e.g. ``'InsertPatient'``).
    message : str
        Human-readable error description.
    """

    def __init__(self, code, operation, message):
        super().__init__(code, operation, message)


class WSException(VidioException):
    """Exception raised by the web-service / HTTP layer.

    Inherits from :class:`VidioException` without modification. Used to
    signal problems that originate in request handling, serialization,
    or external service calls rather than in the database.

    Parameters
    ----------
    code : int or str
        Error code (often an HTTP status code).
    operation : str
        The web-service operation that failed.
    message : str
        Human-readable error description.
    """

    def __init__(self, code, operation, message):
        super().__init__(code, operation, message)
