import logging
import datetime as dt

import jwt
import bcrypt
import falcon

from api import Cfg
from api.db.DB import DB


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password, password_hash):
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def create_token(user_id):
    cfg = Cfg.gCfg
    now = dt.datetime.utcnow()
    payload = {
        'id_user': str(user_id),
        'exp': now + dt.timedelta(days=cfg['auth']['expiration_days']),
        'iat': now,
    }
    return jwt.encode(payload, cfg['auth']['secret_key'], algorithm=cfg['auth']['algorithm'])


def decode_token(token):
    cfg = Cfg.gCfg
    try:
        return jwt.decode(token, cfg['auth']['secret_key'], algorithms=[cfg['auth']['algorithm']])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


class AuthMiddleware:
    EXEMPT_ROUTES = {'/auth'}

    def process_request(self, req, resp):
        if req.method == 'OPTIONS':
            return

        if req.path in self.EXEMPT_ROUTES:
            return

        token = self._extract_token(req)
        if not token:
            raise falcon.HTTPUnauthorized(description='Missing authorization token')

        payload = decode_token(token)
        if not payload:
            raise falcon.HTTPUnauthorized(description='Invalid or expired token')

        req.context['user_id'] = payload['id_user']

    def _extract_token(self, req):
        auth_header = req.get_header('Authorization')
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) == 2 and parts[0].upper() in ('JWT', 'BEARER'):
            return parts[1]
        return None


class AuthResource:
    def __init__(self):
        self.db = DB()

    def on_post(self, req, resp):
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
