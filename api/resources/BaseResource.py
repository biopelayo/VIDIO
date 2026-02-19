import json
import logging

from api.db.DB import DB


class BaseResource:
    def __init__(self):
        self.db = DB()
        self.filter = None

    def load_filter(self, req):
        raw = req.get_param('filter')
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def get_user_id(self, req):
        return req.context.get('user_id')

    def read_body(self, req):
        log = logging.getLogger(__name__)
        try:
            raw = req.bounded_stream.read(req.content_length or 0)
            if not raw:
                return {}
            return json.loads(raw)
        except Exception as ex:
            log.error(f'Error reading request body: {ex}')
            return {}
