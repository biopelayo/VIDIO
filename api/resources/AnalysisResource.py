import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class AnalysisResource(BaseResource):

    def _launch_analysis(self, req, resp, modality):
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            study_id = data.get('id_study')

            if not study_id:
                resp.status = falcon.HTTP_400
                resp.media = {'error': 'id_study is required'}
                return

            study = self.db.GetStudy(study_id)
            if not study:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Study not found'}
                return

            process_data = {
                'id_user': user_id,
                'id_study': study_id,
                'type': f'{modality.upper()}_ANALYSIS',
                'status': 'PENDING',
                'parameters': data.get('parameters', {}),
            }
            process = self.db.AddProcess(user_id, process_data)

            from ProcessManagement import launch_analysis
            launch_analysis(process['id'], modality, study_id, data.get('parameters', {}))

            resp.media = {
                'process_id': process['id'],
                'status': 'PENDING',
                'message': f'{modality} analysis queued',
            }
            resp.status = falcon.HTTP_202
        except DBException as ex:
            log.error(ex)
            resp.status = falcon.HTTP_400
            resp.media = {'error': str(ex)}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_retinal(self, req, resp):
        self._launch_analysis(req, resp, 'retinal')

    def on_post_histology(self, req, resp):
        self._launch_analysis(req, resp, 'histology')

    def on_post_radiology(self, req, resp):
        self._launch_analysis(req, resp, 'radiology')

    def on_post_spatial(self, req, resp):
        self._launch_analysis(req, resp, 'spatial')
