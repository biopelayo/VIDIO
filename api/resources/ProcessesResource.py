import logging

import falcon

from api.resources.BaseResource import BaseResource


class ProcessesResource(BaseResource):

    def on_get(self, req, resp):
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetProcesses(flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_uuid(self, req, resp, entity_id):
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetProcess(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Process not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
