import logging
import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from api.util.VidioException import DBException
from api.db.model.Base import Base
from api.db.model.User import User, UserToken
from api.db.model.Patient import Patient
from api.db.model.Study import Study
from api.db.model.Series import Series as SeriesModel
from api.db.model.Image import Image
from api.db.model.Annotation import Annotation
from api.db.model.Finding import Finding
from api.db.model.TCGACase import TCGACase
from api.db.model.TCGASlide import TCGASlide
from api.db.model.SpatialExperiment import SpatialExperiment
from api.db.model.Process import Process
from api.db.model.MLModel import MLModel
from api.db.model.Log import Log
from api.db.model.File import File

gDbEngine = None
gDBSessionFactory = None


def InitDatabase(cfg):
    global gDbEngine, gDBSessionFactory
    db_cfg = cfg['db']
    conn_str = 'postgresql://%s:%s@%s:%d/%s' % (
        db_cfg['user'], db_cfg['password'],
        db_cfg['host'], db_cfg['port'], db_cfg['db'],
    )
    gDbEngine = create_engine(conn_str, pool_size=20, max_overflow=10)
    gDBSessionFactory = sessionmaker(bind=gDbEngine, autoflush=True)
    logging.getLogger(__name__).info(f'Database initialized: {db_cfg["host"]}:{db_cfg["port"]}/{db_cfg["db"]}')


def row2dict(row):
    d = {}
    for col in row.__table__.columns:
        val = getattr(row, col.name)
        if isinstance(val, dt.datetime):
            d[col.name] = val.isoformat()
        elif isinstance(val, dt.date):
            d[col.name] = val.isoformat()
        else:
            d[col.name] = val
    return d


class DB:

    def GetDBSession(self):
        if gDBSessionFactory is None:
            raise DBException('0.0', 'session', 'Database not initialized')
        return gDBSessionFactory()

    # ==================== LOGGING ====================

    def LogEvent(self, user_id, event_type, message, info=None):
        session = self.GetDBSession()
        try:
            entry = Log(id_user=user_id, type=event_type, message=message, info=info or {})
            session.add(entry)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    # ==================== USERS ====================

    def GetUser(self, user_id):
        session = self.GetDBSession()
        try:
            row = session.query(User).filter(User.id == user_id, User.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetUserByUsername(self, username):
        session = self.GetDBSession()
        try:
            row = session.query(User).filter(User.username == username, User.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetUsers(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(User).filter(User.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddUser(self, actor_id, data):
        session = self.GetDBSession()
        try:
            user = User(
                username=data['username'],
                password_hash=data['password_hash'],
                name=data.get('name', ''),
                surname=data.get('surname'),
                role=data.get('role', 'analyst'),
                info=data.get('info'),
            )
            session.add(user)
            session.commit()
            result = row2dict(user)
            self.LogEvent(actor_id, 'info', f'User created: {user.username}')
            return result
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.1.1', 'AddUser', str(ex))
        finally:
            session.close()

    def ModifyUser(self, actor_id, data):
        session = self.GetDBSession()
        try:
            user = session.query(User).filter(User.id == data['id']).first()
            if not user:
                raise DBException('1.1.0', 'ModifyUser', 'User not found')
            for key in ('username', 'password_hash', 'name', 'surname', 'role', 'info'):
                if key in data:
                    setattr(user, key, data[key])
            session.commit()
            return row2dict(user)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.1.2', 'ModifyUser', str(ex))
        finally:
            session.close()

    def DeleteUser(self, actor_id, user_id):
        session = self.GetDBSession()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.deleted = True
                session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.1.3', 'DeleteUser', str(ex))
        finally:
            session.close()

    def SaveToken(self, user_id, token):
        session = self.GetDBSession()
        try:
            from api import Cfg
            days = Cfg.gCfg.get('auth', {}).get('expiration_days', 7)
            now = dt.datetime.utcnow()
            existing = session.query(UserToken).filter(UserToken.id_user == user_id).first()
            if existing:
                existing.token = token
                existing.expires = now + dt.timedelta(days=days)
                existing.created = now
            else:
                ut = UserToken(id_user=user_id, token=token, expires=now + dt.timedelta(days=days), created=now)
                session.add(ut)
            session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.1.4', 'SaveToken', str(ex))
        finally:
            session.close()

    def GetUserForToken(self, user_id):
        return self.GetUser(user_id)

    # ==================== PATIENTS ====================

    def GetPatient(self, patient_id):
        session = self.GetDBSession()
        try:
            row = session.query(Patient).filter(Patient.id == patient_id, Patient.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetPatients(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Patient).filter(Patient.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddPatient(self, actor_id, data):
        session = self.GetDBSession()
        try:
            patient = Patient(
                medical_record_number=data.get('medical_record_number'),
                name=data.get('name'),
                date_of_birth=data.get('date_of_birth'),
                sex=data.get('sex'),
                info=data.get('info'),
            )
            session.add(patient)
            session.commit()
            result = row2dict(patient)
            self.LogEvent(actor_id, 'info', f'Patient created: {patient.id}')
            return result
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.2.1', 'AddPatient', str(ex))
        finally:
            session.close()

    def ModifyPatient(self, actor_id, data):
        session = self.GetDBSession()
        try:
            patient = session.query(Patient).filter(Patient.id == data['id']).first()
            if not patient:
                raise DBException('1.2.0', 'ModifyPatient', 'Patient not found')
            for key in ('medical_record_number', 'name', 'date_of_birth', 'sex', 'info'):
                if key in data:
                    setattr(patient, key, data[key])
            session.commit()
            return row2dict(patient)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.2.2', 'ModifyPatient', str(ex))
        finally:
            session.close()

    def DeletePatient(self, actor_id, patient_id):
        session = self.GetDBSession()
        try:
            patient = session.query(Patient).filter(Patient.id == patient_id).first()
            if patient:
                patient.deleted = True
                session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.2.3', 'DeletePatient', str(ex))
        finally:
            session.close()

    # ==================== STUDIES ====================

    def GetStudy(self, study_id):
        session = self.GetDBSession()
        try:
            row = session.query(Study).filter(Study.id == study_id, Study.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetStudies(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Study).filter(Study.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetStudiesForPatient(self, patient_id):
        session = self.GetDBSession()
        try:
            q = session.query(Study).filter(Study.id_patient == patient_id, Study.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddStudy(self, actor_id, data):
        session = self.GetDBSession()
        try:
            study = Study(
                id_patient=data.get('id_patient'),
                study_date=data.get('study_date'),
                modality=data['modality'],
                institution=data.get('institution'),
                protocol=data.get('protocol'),
                description=data.get('description'),
                info=data.get('info'),
            )
            session.add(study)
            session.commit()
            result = row2dict(study)
            self.LogEvent(actor_id, 'info', f'Study created: {study.id}')
            return result
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.3.1', 'AddStudy', str(ex))
        finally:
            session.close()

    def ModifyStudy(self, actor_id, data):
        session = self.GetDBSession()
        try:
            study = session.query(Study).filter(Study.id == data['id']).first()
            if not study:
                raise DBException('1.3.0', 'ModifyStudy', 'Study not found')
            for key in ('id_patient', 'study_date', 'modality', 'institution', 'protocol', 'description', 'info'):
                if key in data:
                    setattr(study, key, data[key])
            session.commit()
            return row2dict(study)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.3.2', 'ModifyStudy', str(ex))
        finally:
            session.close()

    def DeleteStudy(self, actor_id, study_id):
        session = self.GetDBSession()
        try:
            study = session.query(Study).filter(Study.id == study_id).first()
            if study:
                study.deleted = True
                session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.3.3', 'DeleteStudy', str(ex))
        finally:
            session.close()

    # ==================== SERIES ====================

    def GetSeries(self, series_id):
        session = self.GetDBSession()
        try:
            row = session.query(SeriesModel).filter(SeriesModel.id == series_id, SeriesModel.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetAllSeries(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(SeriesModel).filter(SeriesModel.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetSeriesForStudy(self, study_id):
        session = self.GetDBSession()
        try:
            q = session.query(SeriesModel).filter(SeriesModel.id_study == study_id, SeriesModel.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddSeries(self, actor_id, data):
        session = self.GetDBSession()
        try:
            s = SeriesModel(
                id_study=data.get('id_study'),
                series_number=data.get('series_number'),
                description=data.get('description'),
                body_part=data.get('body_part'),
                modality_subtype=data.get('modality_subtype'),
                manufacturer=data.get('manufacturer'),
                equipment_model=data.get('equipment_model'),
                info=data.get('info'),
            )
            session.add(s)
            session.commit()
            return row2dict(s)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.4.1', 'AddSeries', str(ex))
        finally:
            session.close()

    def ModifySeries(self, actor_id, data):
        session = self.GetDBSession()
        try:
            s = session.query(SeriesModel).filter(SeriesModel.id == data['id']).first()
            if not s:
                raise DBException('1.4.0', 'ModifySeries', 'Series not found')
            for key in ('id_study', 'series_number', 'description', 'body_part', 'modality_subtype',
                        'manufacturer', 'equipment_model', 'info'):
                if key in data:
                    setattr(s, key, data[key])
            session.commit()
            return row2dict(s)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.4.2', 'ModifySeries', str(ex))
        finally:
            session.close()

    def DeleteSeries(self, actor_id, series_id):
        session = self.GetDBSession()
        try:
            s = session.query(SeriesModel).filter(SeriesModel.id == series_id).first()
            if s:
                s.deleted = True
                session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.4.3', 'DeleteSeries', str(ex))
        finally:
            session.close()

    # ==================== IMAGES ====================

    def GetImage(self, image_id):
        session = self.GetDBSession()
        try:
            row = session.query(Image).filter(Image.id == image_id, Image.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetImages(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Image).filter(Image.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetImagesForSeries(self, series_id):
        session = self.GetDBSession()
        try:
            q = session.query(Image).filter(Image.id_series == series_id, Image.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddImage(self, actor_id, data):
        session = self.GetDBSession()
        try:
            img = Image(
                id_series=data.get('id_series'),
                name=data['name'],
                storage_path=data['storage_path'],
                file_format=data.get('file_format'),
                file_size_bytes=data.get('file_size_bytes'),
                pixel_spacing=data.get('pixel_spacing'),
                dimensions=data.get('dimensions'),
                sop_instance_uid=data.get('sop_instance_uid'),
                info=data.get('info'),
            )
            session.add(img)
            session.commit()
            return row2dict(img)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.5.1', 'AddImage', str(ex))
        finally:
            session.close()

    def ModifyImage(self, actor_id, data):
        session = self.GetDBSession()
        try:
            img = session.query(Image).filter(Image.id == data['id']).first()
            if not img:
                raise DBException('1.5.0', 'ModifyImage', 'Image not found')
            for key in ('id_series', 'name', 'storage_path', 'file_format', 'file_size_bytes',
                        'pixel_spacing', 'dimensions', 'sop_instance_uid', 'info', 'selected'):
                if key in data:
                    setattr(img, key, data[key])
            session.commit()
            return row2dict(img)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.5.2', 'ModifyImage', str(ex))
        finally:
            session.close()

    def DeleteImage(self, actor_id, image_id):
        session = self.GetDBSession()
        try:
            img = session.query(Image).filter(Image.id == image_id).first()
            if img:
                img.deleted = True
                session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.5.3', 'DeleteImage', str(ex))
        finally:
            session.close()

    # ==================== ANNOTATIONS ====================

    def GetAnnotation(self, annotation_id):
        session = self.GetDBSession()
        try:
            row = session.query(Annotation).filter(Annotation.id == annotation_id, Annotation.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetAnnotations(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Annotation).filter(Annotation.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddAnnotation(self, actor_id, data):
        session = self.GetDBSession()
        try:
            ann = Annotation(
                id_image=data.get('id_image'),
                id_user=data.get('id_user', actor_id),
                annotation_type=data['annotation_type'],
                label=data.get('label'),
                geometry=data['geometry'],
                confidence=data.get('confidence'),
                info=data.get('info'),
            )
            session.add(ann)
            session.commit()
            return row2dict(ann)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.6.1', 'AddAnnotation', str(ex))
        finally:
            session.close()

    def ModifyAnnotation(self, actor_id, data):
        session = self.GetDBSession()
        try:
            ann = session.query(Annotation).filter(Annotation.id == data['id']).first()
            if not ann:
                raise DBException('1.6.0', 'ModifyAnnotation', 'Annotation not found')
            for key in ('id_image', 'annotation_type', 'label', 'geometry', 'confidence', 'info'):
                if key in data:
                    setattr(ann, key, data[key])
            session.commit()
            return row2dict(ann)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.6.2', 'ModifyAnnotation', str(ex))
        finally:
            session.close()

    def DeleteAnnotation(self, actor_id, annotation_id):
        session = self.GetDBSession()
        try:
            ann = session.query(Annotation).filter(Annotation.id == annotation_id).first()
            if ann:
                ann.deleted = True
                session.commit()
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.6.3', 'DeleteAnnotation', str(ex))
        finally:
            session.close()

    # ==================== FINDINGS ====================

    def GetFinding(self, finding_id):
        session = self.GetDBSession()
        try:
            row = session.query(Finding).filter(Finding.id == finding_id, Finding.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetFindings(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Finding).filter(Finding.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetFindingsForStudy(self, study_id, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Finding).filter(Finding.id_study == study_id, Finding.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddFinding(self, actor_id, data):
        session = self.GetDBSession()
        try:
            finding = Finding(
                id_image=data.get('id_image'),
                id_study=data.get('id_study'),
                id_process=data.get('id_process'),
                finding_type=data['finding_type'],
                disease_category=data.get('disease_category'),
                severity=data.get('severity'),
                confidence=data.get('confidence'),
                ml_model_id=data.get('ml_model_id'),
                info=data.get('info', {}),
            )
            session.add(finding)
            session.commit()
            return row2dict(finding)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.7.1', 'AddFinding', str(ex))
        finally:
            session.close()

    def ReviewFinding(self, reviewer_id, data):
        session = self.GetDBSession()
        try:
            finding = session.query(Finding).filter(Finding.id == data['id']).first()
            if not finding:
                raise DBException('1.7.0', 'ReviewFinding', 'Finding not found')
            finding.reviewed = True
            finding.id_reviewer = reviewer_id
            if 'review_notes' in data:
                finding.review_notes = data['review_notes']
            if 'severity' in data:
                finding.severity = data['severity']
            session.commit()
            return row2dict(finding)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.7.2', 'ReviewFinding', str(ex))
        finally:
            session.close()

    # ==================== PROCESSES ====================

    def GetProcess(self, process_id):
        session = self.GetDBSession()
        try:
            row = session.query(Process).filter(Process.id == process_id).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetProcesses(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(Process)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddProcess(self, actor_id, data):
        session = self.GetDBSession()
        try:
            proc = Process(
                id_user=data.get('id_user', actor_id),
                id_study=data.get('id_study'),
                type=data['type'],
                status=data.get('status', 'PENDING'),
                parameters=data.get('parameters'),
            )
            session.add(proc)
            session.commit()
            return row2dict(proc)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.8.1', 'AddProcess', str(ex))
        finally:
            session.close()

    def ModifyProcess(self, actor_id, data):
        session = self.GetDBSession()
        try:
            proc = session.query(Process).filter(Process.id == data['id']).first()
            if not proc:
                raise DBException('1.8.0', 'ModifyProcess', 'Process not found')
            for key in ('status', 'progress', 'pid', 'result', 'time_start', 'time_end', 'error_message'):
                if key in data:
                    setattr(proc, key, data[key])
            session.commit()
            return row2dict(proc)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.8.2', 'ModifyProcess', str(ex))
        finally:
            session.close()

    # ==================== ML MODELS ====================

    def GetMLModel(self, model_id):
        session = self.GetDBSession()
        try:
            row = session.query(MLModel).filter(MLModel.id == model_id).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetMLModels(self, flt=None):
        session = self.GetDBSession()
        try:
            q = session.query(MLModel)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddMLModel(self, actor_id, data):
        session = self.GetDBSession()
        try:
            model = MLModel(
                name=data['name'],
                version=data['version'],
                modality=data['modality'],
                architecture=data.get('architecture'),
                task=data.get('task'),
                weights_path=data.get('weights_path'),
                info=data.get('info'),
            )
            session.add(model)
            session.commit()
            return row2dict(model)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.9.1', 'AddMLModel', str(ex))
        finally:
            session.close()

    def ModifyMLModel(self, actor_id, data):
        session = self.GetDBSession()
        try:
            model = session.query(MLModel).filter(MLModel.id == data['id']).first()
            if not model:
                raise DBException('1.9.0', 'ModifyMLModel', 'Model not found')
            for key in ('name', 'version', 'modality', 'architecture', 'task', 'weights_path', 'info'):
                if key in data:
                    setattr(model, key, data[key])
            session.commit()
            return row2dict(model)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.9.2', 'ModifyMLModel', str(ex))
        finally:
            session.close()

    # ==================== FILES ====================

    def AddFile(self, actor_id, data):
        session = self.GetDBSession()
        try:
            f = File(
                id_user=actor_id,
                name=data['name'],
                storage_path=data['storage_path'],
                file_size_bytes=data.get('file_size_bytes'),
                content_type=data.get('content_type'),
                info=data.get('info'),
            )
            session.add(f)
            session.commit()
            return row2dict(f)
        except SQLAlchemyError as ex:
            session.rollback()
            raise DBException('1.10.1', 'AddFile', str(ex))
        finally:
            session.close()
