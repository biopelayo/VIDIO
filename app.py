"""
VIDIO Application Entry Point
===============================

WSGI application factory and development server for the VIDIO
biomedical image analysis platform.

Architecture
------------
The application uses the **Falcon** WSGI framework with the following
middleware stack (applied in order):

1. :class:`~api.Authentication.AuthMiddleware` — JWT token validation.
   Exempts ``/auth`` (login) and ``OPTIONS`` preflight requests.
2. :class:`~api.CORS.CORSComponent` — Cross-Origin Resource Sharing
   headers for browser-based API clients.

Route Map
---------
.. list-table::
   :header-rows: 1

   * - Endpoint
     - Resource
     - Methods
   * - ``/auth``
     - :class:`~api.Authentication.AuthResource`
     - POST (login)
   * - ``/patients[/{id}]``
     - :class:`~api.resources.PatientsResource.PatientsResource`
     - GET, POST, DELETE
   * - ``/studies[/{id}][/series|/findings]``
     - :class:`~api.resources.StudiesResource.StudiesResource`
     - GET, POST, DELETE
   * - ``/series[/{id}][/images]``
     - :class:`~api.resources.SeriesResource.SeriesResource`
     - GET, POST, DELETE
   * - ``/images[/{id}]``
     - :class:`~api.resources.ImagesResource.ImagesResource`
     - GET, POST, DELETE
   * - ``/annotations[/{id}]``
     - :class:`~api.resources.AnnotationsResource.AnnotationsResource`
     - GET, POST, DELETE
   * - ``/findings[/{id}]``
     - :class:`~api.resources.FindingsResource.FindingsResource`
     - GET, POST
   * - ``/processes[/{id}]``
     - :class:`~api.resources.ProcessesResource.ProcessesResource`
     - GET
   * - ``/users[/{id}]``
     - :class:`~api.resources.UsersResource.UsersResource`
     - GET, POST, DELETE
   * - ``/uploads``
     - :class:`~api.resources.UploadsResource.UploadsResource`
     - POST (multipart)
   * - ``/models[/{id}]``
     - :class:`~api.resources.ModelsResource.ModelsResource`
     - GET, POST
   * - ``/analysis/{modality}``
     - :class:`~api.resources.AnalysisResource.AnalysisResource`
     - POST (triggers async pipeline)

Running
-------
**Development** ::

    python app.py

**Production** (gunicorn) ::

    gunicorn app:application -w 4 -b 0.0.0.0:7070

See Also
--------
:mod:`VidioTool` : CLI administrative tool.
:mod:`ProcessManagement` : Async pipeline orchestration.
"""

import os
import sys
import json
import logging

import falcon

import api.Cfg
from api.CORS import CORSComponent
from api.Authentication import AuthMiddleware, AuthResource
from api.db.DB import InitDatabase

from api.resources.PatientsResource import PatientsResource
from api.resources.StudiesResource import StudiesResource
from api.resources.SeriesResource import SeriesResource
from api.resources.ImagesResource import ImagesResource
from api.resources.AnnotationsResource import AnnotationsResource
from api.resources.FindingsResource import FindingsResource
from api.resources.ProcessesResource import ProcessesResource
from api.resources.UsersResource import UsersResource
from api.resources.UploadsResource import UploadsResource
from api.resources.ModelsResource import ModelsResource
from api.resources.AnalysisResource import AnalysisResource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vidio.log'),
    ]
)
log = logging.getLogger(__name__)


def create_app():
    """Create and configure the Falcon WSGI application.

    Performs the following initialisation steps:

    1. Load ``cfg.json`` from the application directory.
    2. Set the global config singleton (:data:`api.Cfg.gCfg`).
    3. Adjust repository path for Windows if needed.
    4. Initialise the PostgreSQL database connection.
    5. Create the Falcon ``App`` with middleware.
    6. Register all route handlers.

    Returns
    -------
    falcon.App
        Configured WSGI application ready to serve requests.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, 'cfg.json')) as f:
        cfg = json.load(f)

    api.Cfg.gCfg = cfg

    if sys.platform == 'win32':
        cfg['repository']['location'] = cfg['repository'].get('location_windows',
                                                               cfg['repository']['location'])

    InitDatabase(cfg)

    app = falcon.App(middleware=[
        AuthMiddleware(),
        CORSComponent(),
    ])

    # --- Resources ---
    auth = AuthResource()
    patients = PatientsResource()
    studies = StudiesResource()
    series = SeriesResource()
    images = ImagesResource()
    annotations = AnnotationsResource()
    findings = FindingsResource()
    processes = ProcessesResource()
    users = UsersResource()
    uploads = UploadsResource()
    models = ModelsResource()
    analysis = AnalysisResource()

    # --- Routes ---

    # Auth
    app.add_route('/auth', auth)

    # Patients
    app.add_route('/patients', patients)
    app.add_route('/patients/{entity_id}', patients, suffix='uuid')

    # Studies
    app.add_route('/studies', studies)
    app.add_route('/studies/{entity_id}', studies, suffix='uuid')
    app.add_route('/studies/{entity_id}/series', studies, suffix='series')
    app.add_route('/studies/{entity_id}/findings', studies, suffix='findings')

    # Series
    app.add_route('/series', series)
    app.add_route('/series/{entity_id}', series, suffix='uuid')
    app.add_route('/series/{entity_id}/images', series, suffix='images')

    # Images
    app.add_route('/images', images)
    app.add_route('/images/{entity_id}', images, suffix='uuid')

    # Annotations
    app.add_route('/annotations', annotations)
    app.add_route('/annotations/{entity_id}', annotations, suffix='uuid')

    # Findings
    app.add_route('/findings', findings)
    app.add_route('/findings/{entity_id}', findings, suffix='uuid')

    # Processes
    app.add_route('/processes', processes)
    app.add_route('/processes/{entity_id}', processes, suffix='uuid')

    # Users
    app.add_route('/users', users)
    app.add_route('/users/{entity_id}', users, suffix='uuid')

    # Uploads
    app.add_route('/uploads', uploads)

    # ML Models
    app.add_route('/models', models)
    app.add_route('/models/{entity_id}', models, suffix='uuid')

    # Analysis triggers
    app.add_route('/analysis/retinal', analysis, suffix='retinal')
    app.add_route('/analysis/histology', analysis, suffix='histology')
    app.add_route('/analysis/radiology', analysis, suffix='radiology')
    app.add_route('/analysis/spatial', analysis, suffix='spatial')

    # --- Static Files (Frontend SPA) ---
    frontend_dir = os.path.join(cur_dir, 'frontend')
    app.add_static_route('/static', frontend_dir)

    # Catch-all: serve index.html for SPA routes
    app.add_sink(_spa_sink(frontend_dir), prefix='/')

    log.info('VIDIO application initialized')
    return app


def _spa_sink(frontend_dir):
    """Return a sink callable that serves index.html for unmatched routes."""
    index_path = os.path.join(frontend_dir, 'index.html')

    def sink(req, resp):
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html; charset=utf-8'
        with open(index_path, 'r', encoding='utf-8') as f:
            resp.text = f.read()

    return sink


application = create_app()

if __name__ == '__main__':
    from wsgiref.simple_server import make_server

    cfg = api.Cfg.gCfg
    host = cfg.get('server', {}).get('host', '127.0.0.1')
    port = cfg.get('server', {}).get('port', 7070)

    log.info(f'Starting VIDIO server on {host}:{port}')
    with make_server(host, port, application) as httpd:
        httpd.serve_forever()
