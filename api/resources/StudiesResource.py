import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException


class StudiesResource(BaseResource):

    def on_get(self, req, resp):
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetStudies(flt)
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
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            user_id = self.get_user_id(req)
            if 'id' in data:
                result = self.db.ModifyStudy(user_id, data)
            else:
                result = self.db.AddStudy(user_id, data)
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
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetStudy(entity_id)
            if result:
                resp.media = result
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_404
                resp.media = {'error': 'Study not found'}
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_post_uuid(self, req, resp, entity_id):
        log = logging.getLogger(__name__)
        try:
            data = self.read_body(req)
            data['id'] = entity_id
            user_id = self.get_user_id(req)
            result = self.db.ModifyStudy(user_id, data)
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
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            self.db.DeleteStudy(user_id, entity_id)
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

    def on_get_series(self, req, resp, entity_id):
        log = logging.getLogger(__name__)
        try:
            result = self.db.GetSeriesForStudy(entity_id)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}

    def on_get_findings(self, req, resp, entity_id):
        log = logging.getLogger(__name__)
        try:
            flt = self.load_filter(req)
            result = self.db.GetFindingsForStudy(entity_id, flt)
            resp.media = result
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
