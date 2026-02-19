import os
import logging
import uuid

import falcon

from api import Cfg
from api.resources.BaseResource import BaseResource
from api.util.ImageUtils import detect_format, get_file_size


class UploadsResource(BaseResource):

    def on_post(self, req, resp):
        log = logging.getLogger(__name__)
        try:
            user_id = self.get_user_id(req)
            repo_dir = Cfg.gCfg['repository']['location']

            uploads = []
            for part in req.media:
                if part.name == 'file':
                    original_name = part.filename or f'upload_{uuid.uuid4().hex}'
                    ext = os.path.splitext(original_name)[1]
                    unique_name = f'{uuid.uuid4().hex}{ext}'

                    upload_dir = os.path.join(repo_dir, 'uploads')
                    os.makedirs(upload_dir, exist_ok=True)
                    filepath = os.path.join(upload_dir, unique_name)

                    with open(filepath, 'wb') as f:
                        while True:
                            chunk = part.stream.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)

                    file_record = self.db.AddFile(user_id, {
                        'name': original_name,
                        'storage_path': filepath,
                        'file_size_bytes': get_file_size(filepath),
                        'content_type': part.content_type,
                    })
                    uploads.append(file_record)

            resp.media = {'uploaded': uploads}
            resp.status = falcon.HTTP_200
        except Exception as ex:
            log.error(ex)
            resp.status = falcon.HTTP_500
            resp.media = {'error': str(ex)}
