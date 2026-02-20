"""
TCGA slide retrieval via the GDC (Genomic Data Commons) REST API.

This module wraps the NCI Genomic Data Commons API to search for cancer
cases and download whole-slide images (SVS format) from The Cancer Genome
Atlas (TCGA) programme.

API Reference
-------------
* **Cases endpoint** : ``GET /cases`` -- returns clinical metadata
  (case_id, disease type, demographics) filtered by project.
* **Files endpoint** : ``GET /files`` -- returns file metadata
  (file_id, file_name, size, format) filtered by case or project.
* **Data endpoint** : ``GET /data/<file_id>`` -- streams the raw file
  bytes for download.

All endpoints live under ``https://api.gdc.cancer.gov``.

Authentication
--------------
Public TCGA data does not require authentication.  Controlled-access
data (e.g. germline variants) requires a GDC token, which this module
does *not* currently support.

Typical Workflow
----------------
1. ``search_cases('TCGA-BRCA')`` -- find breast-cancer cases.
2. ``search_slides(case_id='<uuid>')`` -- find SVS files for a case.
3. ``download_slide('<file_id>', '/data/slides/example.svs')`` -- stream
   the SVS file to local disk.

Notes
-----
* The GDC filter syntax uses a nested JSON structure with ``op`` / ``content``
  nodes.  Python dicts are serialised to JSON-like strings by replacing
  single quotes with double quotes -- a pragmatic (if brittle) approach
  that works because none of the filter values contain quotes.
* Network timeouts are set conservatively: 30 s for metadata queries and
  300 s for file downloads (SVS files can be several GB).
"""

import logging
import requests

log = logging.getLogger(__name__)

# Base URL for the NCI Genomic Data Commons REST API (v1).
GDC_API_BASE = 'https://api.gdc.cancer.gov'


def search_cases(project, disease_type=None, limit=100):
    """
    Search for TCGA cases (patients) by project and optional disease type.

    Constructs a GDC filter expression and queries the ``/cases`` endpoint.

    Parameters
    ----------
    project : str
        TCGA project identifier, e.g. ``'TCGA-BRCA'`` (breast cancer),
        ``'TCGA-LUAD'`` (lung adenocarcinoma), ``'TCGA-GBM'``
        (glioblastoma).
    disease_type : str, optional
        Free-text disease type string to further narrow results, e.g.
        ``'Ductal and Lobular Neoplasms'``.  If ``None``, all disease
        types within the project are returned.
    limit : int, optional
        Maximum number of case records to return.  Default ``100``.
        The GDC API supports values up to 10 000.

    Returns
    -------
    list of dict
        Each dict is a GDC "hit" containing fields ``case_id``,
        ``submitter_id``, ``disease_type``, ``primary_site``, and
        ``demographic.gender`` (where available).

    Raises
    ------
    requests.HTTPError
        If the GDC API returns a non-2xx status code.

    Notes
    -----
    * The filter is built as an ``'and'`` composite.  When only *project*
      is specified the composite contains a single child; the GDC API
      accepts this degenerate case.
    * The ``str(filters).replace("'", '"')`` serialisation trick converts
      the Python dict to a JSON-compatible string.  This works because
      project IDs and disease types never contain quote characters.

    Examples
    --------
    >>> cases = search_cases('TCGA-BRCA', limit=5)
    >>> len(cases)
    5
    >>> cases[0]['case_id']
    'a1b2c3d4-...'
    """
    filters = {
        'op': 'and',
        'content': [
            {'op': '=', 'content': {'field': 'project.project_id', 'value': project}},
        ]
    }
    if disease_type:
        filters['content'].append(
            {'op': '=', 'content': {'field': 'disease_type', 'value': disease_type}}
        )

    params = {
        'filters': str(filters).replace("'", '"'),
        'fields': 'case_id,submitter_id,disease_type,primary_site,demographic.gender',
        'size': limit,
        'format': 'json',
    }

    resp = requests.get(f'{GDC_API_BASE}/cases', params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get('data', {}).get('hits', [])


def search_slides(case_id=None, project=None, limit=100):
    """
    Search for slide image files associated with a case or project.

    Queries the GDC ``/files`` endpoint with filters on ``cases.case_id``
    and/or ``cases.project.project_id``.

    Parameters
    ----------
    case_id : str, optional
        UUID of a specific case.  When provided, only files linked to
        this case are returned.
    project : str, optional
        TCGA project identifier (e.g. ``'TCGA-BRCA'``).  Can be used
        alone to retrieve all slides for a project, or combined with
        *case_id* for additional specificity.
    limit : int, optional
        Maximum number of file records.  Default ``100``.

    Returns
    -------
    list of dict
        GDC file hits with fields ``file_id``, ``file_name``,
        ``file_size`` (bytes), ``data_format`` (e.g. ``'SVS'``), and
        ``data_type`` (e.g. ``'Slide Image'``).

    Raises
    ------
    requests.HTTPError
        On non-2xx API response.

    Notes
    -----
    * At least one of *case_id* or *project* should be provided;
      otherwise the filter will be empty and the API may return
      unrelated files.
    * The returned ``file_id`` is the UUID needed for
      :func:`download_slide`.
    """
    filters_content = []
    if case_id:
        filters_content.append(
            {'op': '=', 'content': {'field': 'cases.case_id', 'value': case_id}}
        )
    if project:
        filters_content.append(
            {'op': '=', 'content': {'field': 'cases.project.project_id', 'value': project}}
        )

    filters = {'op': 'and', 'content': filters_content}
    params = {
        'filters': str(filters).replace("'", '"'),
        'fields': 'file_id,file_name,file_size,data_format,data_type',
        'size': limit,
        'format': 'json',
    }

    resp = requests.get(f'{GDC_API_BASE}/files', params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get('data', {}).get('hits', [])


def download_slide(file_id, output_path):
    """
    Download a slide image from the GDC data endpoint.

    Streams the file in 8 KiB chunks to keep memory usage constant
    regardless of file size (TCGA SVS files can exceed 2 GB).

    Parameters
    ----------
    file_id : str
        UUID of the file as returned by :func:`search_slides`.
    output_path : str or pathlib.Path
        Local filesystem path where the downloaded file will be written.
        Parent directories must already exist.

    Returns
    -------
    str
        The *output_path* argument, echoed back for convenience in
        pipeline chaining (e.g. ``path = download_slide(fid, dest)``).

    Raises
    ------
    requests.HTTPError
        If the GDC API returns a non-2xx status (e.g. 404 for an
        invalid file_id, 403 for controlled-access data).
    IOError
        If *output_path* cannot be opened for writing.

    Notes
    -----
    * The download timeout is set to 300 seconds.  For very large files
      on slow connections this may need to be increased.
    * ``stream=True`` is essential: without it, ``requests`` would buffer
      the entire response body in memory before returning, which is
      infeasible for multi-GB slide files.
    * The chunk size of 8192 bytes is a reasonable balance between system
      call overhead and memory footprint.
    """
    url = f'{GDC_API_BASE}/data/{file_id}'
    log.info(f'Downloading slide {file_id} to {output_path}')

    # stream=True ensures the response body is not loaded into memory at
    # once; we iterate over it in fixed-size chunks instead.
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    log.info(f'Downloaded {output_path}')
    return output_path
