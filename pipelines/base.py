import logging
import datetime as dt

from api.db.DB import DB


class BasePipeline:
    """
    Base class for all VIDIO analysis pipelines.

    Implements the 5-stage biomedical image analysis pattern:
        1. Load    → Read image from disk
        2. Preprocess → Normalize, filter, enhance
        3. Segment → Identify regions of interest
        4. Statistics → Compute quantitative metrics per region
        5. Detect  → Identify anomalies and classify severity

    Subclasses override the stage methods for modality-specific logic.
    """

    def __init__(self, process_id, study_id, parameters=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.db = DB()
        self.process_id = process_id
        self.study_id = study_id
        self.parameters = parameters or {}

    def run(self):
        self.log.info(f'Pipeline started: process={self.process_id}, study={self.study_id}')
        self.update_status('RUNNING', progress=0)

        try:
            images = self.get_images()
            if not images:
                self.log.warning('No images found for study')
                self.update_status('COMPLETED', progress=100, result={'findings_count': 0})
                return []

            all_findings = []
            total = len(images)

            for idx, image_record in enumerate(images):
                self.log.info(f'Processing image {idx + 1}/{total}: {image_record["name"]}')
                try:
                    findings = self.process_image(image_record)
                    all_findings.extend(findings)
                except Exception as ex:
                    self.log.error(f'Error processing image {image_record["name"]}: {ex}')

                progress = int(((idx + 1) / total) * 100)
                self.update_status('RUNNING', progress=progress)

            self.update_status('COMPLETED', progress=100, result={
                'findings_count': len(all_findings),
                'images_processed': total,
            })

            self.log.info(f'Pipeline completed: {len(all_findings)} findings in {total} images')
            return all_findings

        except Exception as ex:
            self.log.error(f'Pipeline failed: {ex}')
            self.update_status('FAILED', error_message=str(ex))
            raise

    def process_image(self, image_record):
        img = self.load_image(image_record)
        img = self.preprocess(img, image_record)
        regions = self.segment(img, image_record)
        stats = self.calculate_stats(img, regions)
        findings = self.detect_anomalies(img, regions, stats, image_record)
        self.save_findings(findings, image_record)
        return findings

    def get_images(self):
        series_list = self.db.GetSeriesForStudy(self.study_id)
        images = []
        for s in series_list:
            imgs = self.db.GetImagesForSeries(s['id'])
            images.extend(imgs)
        return [img for img in images if img.get('selected', True)]

    # --- Stage methods (override in subclasses) ---

    def load_image(self, image_record):
        from core.image_io import read_image
        return read_image(image_record['storage_path'])

    def preprocess(self, img, image_record):
        return img

    def segment(self, img, image_record):
        return []

    def calculate_stats(self, img, regions):
        return []

    def detect_anomalies(self, img, regions, stats, image_record):
        return []

    # --- Persistence ---

    def save_findings(self, findings, image_record):
        for finding_data in findings:
            finding_data['id_image'] = image_record['id']
            finding_data['id_study'] = self.study_id
            finding_data['id_process'] = self.process_id
            self.db.AddFinding(None, finding_data)

    # --- Status updates ---

    def update_status(self, status, progress=None, result=None, error_message=None):
        data = {
            'id': self.process_id,
            'status': status,
        }
        if progress is not None:
            data['progress'] = progress
        if result is not None:
            data['result'] = result
        if error_message is not None:
            data['error_message'] = error_message
        if status == 'RUNNING' and progress == 0:
            data['time_start'] = dt.datetime.utcnow().isoformat()
        if status in ('COMPLETED', 'FAILED'):
            data['time_end'] = dt.datetime.utcnow().isoformat()

        try:
            self.db.ModifyProcess(None, data)
        except Exception as ex:
            self.log.error(f'Failed to update process status: {ex}')
