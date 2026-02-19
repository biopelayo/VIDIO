import logging

import falcon

from api.resources.BaseResource import BaseResource
from api.util.VidioException import DBException
from api.Authentication import hash_password


class UsersResource(BaseResource):

    def on_get(self, req, resp):
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
