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

    log.info('VIDIO application initialized')
    return app


application = create_app()

if __name__ == '__main__':
    from wsgiref.simple_server import make_server

    cfg = api.Cfg.gCfg
    host = cfg.get('server', {}).get('host', '127.0.0.1')
    port = cfg.get('server', {}).get('port', 7070)

    log.info(f'Starting VIDIO server on {host}:{port}')
    with make_server(host, port, application) as httpd:
        httpd.serve_forever()
