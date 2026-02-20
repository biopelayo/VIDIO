"""
Uploads Resource
================

Falcon resource for handling multipart file uploads. Uploaded files are
persisted to the repository directory configured in ``Cfg.gCfg`` and a
corresponding database record is created for each file.

Routes
------
* ``POST /uploads`` -- upload one or more files via multipart form data.
"""

import os
import logging
import uuid

import falcon

from api import Cfg
from api.resources.BaseResource import BaseResource
from api.util.ImageUtils import detect_format, get_file_size


class UploadsResource(BaseResource):
    """Falcon resource for the ``POST /uploads`` endpoint.

    Accepts ``multipart/form-data`` requests containing one or more file
    parts named ``file``.  Each file is saved to disk under a UUID-based
    filename inside the configured repository's ``uploads/`` directory,
    and a file record is inserted into the database.

    Inherits shared helpers from :class:`~api.resources.BaseResource.BaseResource`.
    """

    def on_post(self, req, resp):
        """Handle ``POST /uploads`` -- upload files via multipart form data.

        Iterates over all multipart parts named ``file``, writes each to
        disk with a unique name, records metadata in the database, and
        returns a summary of all successfully uploaded files.

        Parameters
        ----------
        req : falcon.Request
            A ``multipart/form-data`` request.  Each file part must use
            the field name ``file``.
        resp : falcon.Response
            On success, ``resp.media`` is::

                {"uploaded": [<file_record>, ...]}

        Response Codes
        --------------
        * **200** -- all files uploaded and recorded successfully.
        * **500** -- unexpected server error (e.g. I/O failure).

        Notes
        -----
        Files are streamed to disk in 8 KiB chunks to keep memory usage
        constant regardless of file size.
        """
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
