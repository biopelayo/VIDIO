"""
CORS Middleware Component
=========================

Provides Cross-Origin Resource Sharing (CORS) support for the Falcon API
application. This middleware ensures that browser-based clients hosted on
different origins can communicate with the API without being blocked by
the same-origin policy.

The component is registered as Falcon middleware and hooks into both the
request and response processing phases.
"""

import falcon


class CORSComponent:
    """Falcon middleware that injects CORS headers into every response.

    This component performs two duties:

    1. **Response phase** -- attaches permissive ``Access-Control-Allow-*``
       headers to every outgoing response so that any origin can consume
       the API.
    2. **Request phase** -- intercepts CORS preflight ``OPTIONS`` requests
       and returns an immediate ``200 OK`` before the request reaches any
       resource handler.

    Attributes
    ----------
    None

    Notes
    -----
    The current implementation uses a wildcard (``*``) for
    ``Access-Control-Allow-Origin``. In production deployments this should
    be restricted to the set of trusted front-end origins.

    The ``Access-Control-Max-Age`` header is set to 86 400 seconds (24 h),
    meaning browsers may cache the preflight response for up to one day.

    Examples
    --------
    Register the component when building the Falcon application::

        app = falcon.App(middleware=[CORSComponent(), ...])
    """

    def process_response(self, req, resp, resource, req_succeeded):
        """Append CORS headers to every HTTP response.

        Parameters
        ----------
        req : falcon.Request
            The incoming request object.
        resp : falcon.Response
            The outgoing response object; CORS headers are set here.
        resource : object or None
            The matched Falcon resource instance, or ``None`` if no route
            matched.
        req_succeeded : bool
            ``True`` if no unhandled exception occurred during request
            processing.

        Notes
        -----
        Headers set:

        * ``Access-Control-Allow-Origin: *``
        * ``Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS``
        * ``Access-Control-Allow-Headers: Authorization, Content-Type,
          Accept, Origin, X-Requested-With``
        * ``Access-Control-Max-Age: 86400``
        """
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        resp.set_header('Access-Control-Allow-Headers',
                        'Authorization, Content-Type, Accept, Origin, X-Requested-With')
        resp.set_header('Access-Control-Max-Age', '86400')

    def process_request(self, req, resp):
        """Handle CORS preflight ``OPTIONS`` requests.

        If the incoming request method is ``OPTIONS``, this method
        immediately responds with ``HTTP 200`` and raises
        ``falcon.HTTPStatus`` to short-circuit further processing.
        The actual CORS headers are added later by
        :meth:`process_response`.

        Parameters
        ----------
        req : falcon.Request
            The incoming request object.
        resp : falcon.Response
            The outgoing response object.

        Raises
        ------
        falcon.HTTPStatus
            Always raised for ``OPTIONS`` requests to halt the middleware
            chain and return immediately.
        """
        if req.method == 'OPTIONS':
            resp.status = falcon.HTTP_200
            raise falcon.HTTPStatus(falcon.HTTP_200)
