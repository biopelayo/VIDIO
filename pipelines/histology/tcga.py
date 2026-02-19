import logging
import requests

log = logging.getLogger(__name__)

GDC_API_BASE = 'https://api.gdc.cancer.gov'


def search_cases(project, disease_type=None, limit=100):
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
    url = f'{GDC_API_BASE}/data/{file_id}'
    log.info(f'Downloading slide {file_id} to {output_path}')

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    log.info(f'Downloaded {output_path}')
    return output_path
