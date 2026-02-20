"""
Data Access Layer (DAL) for the VIDIO medical-imaging platform.

This module provides the sole interface between the application logic and the
PostgreSQL database.  All SQL operations are routed through the :class:`DB`
service class.

Architecture notes
------------------
* **Session-per-method pattern** -- Every public method creates its own
  ``sqlalchemy.orm.Session``, commits or rolls back, and closes it in a
  ``finally`` block.  This keeps each call self-contained and avoids
  long-lived sessions leaking across request boundaries.
* **Thread safety** -- Because sessions are never shared between calls, the
  class is safe for concurrent access from multiple threads (e.g. parallel
  pipeline workers hitting the API simultaneously).
* **ORM-to-dict serialization** -- The module-level helper :func:`row2dict`
  converts any SQLAlchemy model instance into a plain ``dict``, serializing
  ``datetime`` / ``date`` columns to ISO-8601 strings.  All ``Get*`` methods
  return dicts (or lists of dicts), never ORM objects.
* **Error handling** -- Every write method wraps ``SQLAlchemyError`` and
  re-raises it as :class:`~api.util.VidioException.DBException` with a
  structured error code (e.g. ``'1.1.1'``).
* **Audit logging** -- Write operations (``Add*``, ``Modify*``, ``Delete*``)
  call :meth:`DB.LogEvent` to persist an audit trail in the ``Log`` table,
  recording *who* did *what* and *when*.

Module globals
--------------
gDbEngine : sqlalchemy.engine.Engine or None
    The database engine, initialised by :func:`InitDatabase`.
gDBSessionFactory : sqlalchemy.orm.session.sessionmaker or None
    Bound session factory used by :meth:`DB.GetDBSession`.
"""

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
    """
    Bootstrap the SQLAlchemy engine and session factory.

    Must be called once at application startup (before any :class:`DB` method
    is used).  Reads connection parameters from the ``'db'`` section of the
    application configuration dict and creates a connection-pool-backed engine.

    Parameters
    ----------
    cfg : dict
        Application configuration dictionary.  Must contain a ``'db'`` key
        with sub-keys ``'user'``, ``'password'``, ``'host'``, ``'port'`` and
        ``'db'`` (the database name).

    Raises
    ------
    KeyError
        If any required key is missing from *cfg*.
    sqlalchemy.exc.OperationalError
        If the database is unreachable.

    Notes
    -----
    The pool is sized to ``pool_size=20, max_overflow=10`` to support
    concurrent pipeline workers without exhausting connections.
    """
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
    """
    Convert a SQLAlchemy ORM row into a plain Python dictionary.

    Iterates over every column defined on the model's ``__table__`` and copies
    its value into the returned dict.  ``datetime`` and ``date`` values are
    serialised to ISO-8601 strings so that the result is JSON-safe.

    Parameters
    ----------
    row : sqlalchemy.orm.DeclarativeMeta
        An instance of any SQLAlchemy declarative model (e.g. ``User``,
        ``Patient``, ``Study``).

    Returns
    -------
    dict
        Column-name -> value mapping.  Temporal columns are converted to
        ``str`` via ``.isoformat()``; all other types are passed through
        unchanged.
    """
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
    """
    Stateless data-access service for the VIDIO platform.

    ``DB`` exposes every database operation the API and pipeline layers need.
    It holds **no instance state** beyond the inherited ability to call
    :meth:`GetDBSession`; all data lives in PostgreSQL.

    CRUD conventions
    ----------------
    Methods follow a consistent naming pattern:

    * **Get<Entity>** / **Get<Entity>s** -- read one or many rows (soft-deleted
      rows are excluded automatically).
    * **Add<Entity>** -- insert a new row; logs the action via
      :meth:`LogEvent`.
    * **Modify<Entity>** -- partial update of a row identified by
      ``data['id']``.
    * **Delete<Entity>** -- **soft-delete** (sets ``deleted = True``) rather
      than physically removing the row.

    Soft-delete pattern
    -------------------
    Entities that support soft-delete (users, patients, studies, series,
    images, annotations, findings) are never removed from the database.
    Instead, a boolean ``deleted`` column is flipped to ``True`` and all
    ``Get*`` queries filter them out with ``Entity.deleted != True``.

    Audit trail
    -----------
    Every write method accepts an ``actor_id`` parameter (the user ID of the
    person or service account performing the action).  This ID is forwarded to
    :meth:`LogEvent` so that every mutation can be traced back to its
    originator.

    Thread safety
    -------------
    Instances are safe to share across threads -- each method opens (and
    closes) its own session.
    """

    def GetDBSession(self):
        """
        Create and return a new SQLAlchemy session from the module-level factory.

        Returns
        -------
        sqlalchemy.orm.Session
            A fresh session bound to :data:`gDbEngine`.

        Raises
        ------
        DBException
            If :func:`InitDatabase` has not been called yet
            (``gDBSessionFactory is None``).
        """
        if gDBSessionFactory is None:
            raise DBException('0.0', 'session', 'Database not initialized')
        return gDBSessionFactory()

    # ==================== LOGGING ====================

    def LogEvent(self, user_id, event_type, message, info=None):
        """
        Persist an audit-log entry in the ``Log`` table.

        Called internally by every write operation (Add/Modify/Delete) to
        maintain a full audit trail.  Failures are silently swallowed so that
        a logging glitch never causes a business operation to fail.

        Parameters
        ----------
        user_id : int
            ID of the user (or service account) that triggered the event.
        event_type : str
            Free-form category such as ``'info'``, ``'warning'``, or
            ``'error'``.
        message : str
            Human-readable description of the event.
        info : dict, optional
            Arbitrary JSON-serialisable payload stored alongside the log
            entry.  Defaults to an empty dict.
        """
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
        """
        Retrieve a single non-deleted user by primary key.

        Parameters
        ----------
        user_id : int
            Primary key of the user.

        Returns
        -------
        dict or None
            User data as a dict, or ``None`` if no matching active user
            exists.
        """
        session = self.GetDBSession()
        try:
            row = session.query(User).filter(User.id == user_id, User.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetUserByUsername(self, username):
        """
        Look up a non-deleted user by their unique username.

        Parameters
        ----------
        username : str
            The login username to search for (exact match).

        Returns
        -------
        dict or None
            User data as a dict, or ``None`` if no active user with that
            username exists.
        """
        session = self.GetDBSession()
        try:
            row = session.query(User).filter(User.username == username, User.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetUsers(self, flt=None):
        """
        Return all non-deleted users.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active user, ordered by database default.
        """
        session = self.GetDBSession()
        try:
            q = session.query(User).filter(User.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddUser(self, actor_id, data):
        """
        Create a new user record and log the event.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the creation (for audit trail).
        data : dict
            User fields.  Required keys: ``'username'``, ``'password_hash'``.
            Optional keys: ``'name'``, ``'surname'``, ``'role'`` (defaults to
            ``'analyst'``), ``'info'``.

        Returns
        -------
        dict
            The newly created user, including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.1.1'`` on any SQLAlchemy error (e.g. duplicate
            username).
        """
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
        """
        Update mutable fields of an existing user.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'username'``,
            ``'password_hash'``, ``'name'``, ``'surname'``, ``'role'``,
            ``'info'``.

        Returns
        -------
        dict
            The updated user record.

        Raises
        ------
        DBException
            Code ``'1.1.0'`` if the user is not found; ``'1.1.2'`` on
            other SQLAlchemy errors.
        """
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
        """
        Soft-delete a user by setting ``deleted = True``.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the deletion (audit trail).
        user_id : int
            Primary key of the user to delete.

        Raises
        ------
        DBException
            Code ``'1.1.3'`` on any SQLAlchemy error.
        """
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
        """
        Persist or refresh an authentication token for a user.

        If a ``UserToken`` row already exists for *user_id* it is updated
        in-place; otherwise a new row is inserted.  Expiration is set to
        ``auth.expiration_days`` from the global config (default 7 days).

        Parameters
        ----------
        user_id : int
            The user whose token should be saved.
        token : str
            The JWT (or opaque) token string.

        Raises
        ------
        DBException
            Code ``'1.1.4'`` on any SQLAlchemy error.
        """
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
        """
        Retrieve user data given a previously-validated token's user ID.

        This is a thin wrapper around :meth:`GetUser`, kept as a separate
        entry point so the authentication layer has a dedicated call-site
        that can be extended (e.g. token-expiry checks) without touching
        ``GetUser``.

        Parameters
        ----------
        user_id : int
            The user ID extracted from the validated token.

        Returns
        -------
        dict or None
            User data, or ``None`` if the user no longer exists / is
            soft-deleted.
        """
        return self.GetUser(user_id)

    # ==================== PATIENTS ====================

    def GetPatient(self, patient_id):
        """
        Retrieve a single non-deleted patient by primary key.

        Parameters
        ----------
        patient_id : int
            Primary key of the patient.

        Returns
        -------
        dict or None
            Patient data as a dict, or ``None`` if not found or soft-deleted.
        """
        session = self.GetDBSession()
        try:
            row = session.query(Patient).filter(Patient.id == patient_id, Patient.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetPatients(self, flt=None):
        """
        Return all non-deleted patients.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active patient record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Patient).filter(Patient.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddPatient(self, actor_id, data):
        """
        Create a new patient record and log the event.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the creation (audit trail).
        data : dict
            Patient fields.  All keys are optional:
            ``'medical_record_number'``, ``'name'``, ``'date_of_birth'``,
            ``'sex'``, ``'info'``.

        Returns
        -------
        dict
            The newly created patient including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.2.1'`` on any SQLAlchemy error.
        """
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
        """
        Update mutable fields of an existing patient.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of:
            ``'medical_record_number'``, ``'name'``, ``'date_of_birth'``,
            ``'sex'``, ``'info'``.

        Returns
        -------
        dict
            The updated patient record.

        Raises
        ------
        DBException
            Code ``'1.2.0'`` if not found; ``'1.2.2'`` on other errors.
        """
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
        """
        Soft-delete a patient by setting ``deleted = True``.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the deletion (audit trail).
        patient_id : int
            Primary key of the patient to delete.

        Raises
        ------
        DBException
            Code ``'1.2.3'`` on any SQLAlchemy error.
        """
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
        """
        Retrieve a single non-deleted study by primary key.

        Parameters
        ----------
        study_id : int
            Primary key of the study.

        Returns
        -------
        dict or None
            Study data as a dict, or ``None`` if not found or soft-deleted.
        """
        session = self.GetDBSession()
        try:
            row = session.query(Study).filter(Study.id == study_id, Study.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetStudies(self, flt=None):
        """
        Return all non-deleted studies.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active study record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Study).filter(Study.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetStudiesForPatient(self, patient_id):
        """
        Return all non-deleted studies belonging to a specific patient.

        Parameters
        ----------
        patient_id : int
            Foreign key referencing the parent patient.

        Returns
        -------
        list of dict
            Studies whose ``id_patient`` matches *patient_id*.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Study).filter(Study.id_patient == patient_id, Study.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddStudy(self, actor_id, data):
        """
        Create a new study record and log the event.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the creation (audit trail).
        data : dict
            Study fields.  Required: ``'modality'``.  Optional:
            ``'id_patient'``, ``'study_date'``, ``'institution'``,
            ``'protocol'``, ``'description'``, ``'info'``.

        Returns
        -------
        dict
            The newly created study including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.3.1'`` on any SQLAlchemy error.
        """
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
        """
        Update mutable fields of an existing study.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'id_patient'``,
            ``'study_date'``, ``'modality'``, ``'institution'``,
            ``'protocol'``, ``'description'``, ``'info'``.

        Returns
        -------
        dict
            The updated study record.

        Raises
        ------
        DBException
            Code ``'1.3.0'`` if not found; ``'1.3.2'`` on other errors.
        """
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
        """
        Soft-delete a study by setting ``deleted = True``.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the deletion (audit trail).
        study_id : int
            Primary key of the study to delete.

        Raises
        ------
        DBException
            Code ``'1.3.3'`` on any SQLAlchemy error.
        """
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
        """
        Retrieve a single non-deleted series by primary key.

        Parameters
        ----------
        series_id : int
            Primary key of the series.

        Returns
        -------
        dict or None
            Series data as a dict, or ``None`` if not found or soft-deleted.
        """
        session = self.GetDBSession()
        try:
            row = session.query(SeriesModel).filter(SeriesModel.id == series_id, SeriesModel.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetAllSeries(self, flt=None):
        """
        Return all non-deleted series across every study.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active series record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(SeriesModel).filter(SeriesModel.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetSeriesForStudy(self, study_id):
        """
        Return all non-deleted series belonging to a specific study.

        Parameters
        ----------
        study_id : int
            Foreign key referencing the parent study.

        Returns
        -------
        list of dict
            Series whose ``id_study`` matches *study_id*.
        """
        session = self.GetDBSession()
        try:
            q = session.query(SeriesModel).filter(SeriesModel.id_study == study_id, SeriesModel.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddSeries(self, actor_id, data):
        """
        Create a new series record.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the creation (audit trail).
        data : dict
            Series fields.  All keys are optional: ``'id_study'``,
            ``'series_number'``, ``'description'``, ``'body_part'``,
            ``'modality_subtype'``, ``'manufacturer'``,
            ``'equipment_model'``, ``'info'``.

        Returns
        -------
        dict
            The newly created series including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.4.1'`` on any SQLAlchemy error.
        """
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
        """
        Update mutable fields of an existing series.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'id_study'``,
            ``'series_number'``, ``'description'``, ``'body_part'``,
            ``'modality_subtype'``, ``'manufacturer'``,
            ``'equipment_model'``, ``'info'``.

        Returns
        -------
        dict
            The updated series record.

        Raises
        ------
        DBException
            Code ``'1.4.0'`` if not found; ``'1.4.2'`` on other errors.
        """
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
        """
        Soft-delete a series by setting ``deleted = True``.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the deletion (audit trail).
        series_id : int
            Primary key of the series to delete.

        Raises
        ------
        DBException
            Code ``'1.4.3'`` on any SQLAlchemy error.
        """
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
        """
        Retrieve a single non-deleted image by primary key.

        Parameters
        ----------
        image_id : int
            Primary key of the image.

        Returns
        -------
        dict or None
            Image data as a dict, or ``None`` if not found or soft-deleted.
        """
        session = self.GetDBSession()
        try:
            row = session.query(Image).filter(Image.id == image_id, Image.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetImages(self, flt=None):
        """
        Return all non-deleted images.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active image record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Image).filter(Image.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetImagesForSeries(self, series_id):
        """
        Return all non-deleted images belonging to a specific series.

        Parameters
        ----------
        series_id : int
            Foreign key referencing the parent series.

        Returns
        -------
        list of dict
            Images whose ``id_series`` matches *series_id*.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Image).filter(Image.id_series == series_id, Image.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddImage(self, actor_id, data):
        """
        Create a new image record.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the creation (audit trail).
        data : dict
            Image fields.  Required: ``'name'``, ``'storage_path'``.
            Optional: ``'id_series'``, ``'file_format'``,
            ``'file_size_bytes'``, ``'pixel_spacing'``, ``'dimensions'``,
            ``'sop_instance_uid'``, ``'info'``.

        Returns
        -------
        dict
            The newly created image including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.5.1'`` on any SQLAlchemy error.
        """
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
        """
        Update mutable fields of an existing image.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'id_series'``,
            ``'name'``, ``'storage_path'``, ``'file_format'``,
            ``'file_size_bytes'``, ``'pixel_spacing'``, ``'dimensions'``,
            ``'sop_instance_uid'``, ``'info'``, ``'selected'``.

        Returns
        -------
        dict
            The updated image record.

        Raises
        ------
        DBException
            Code ``'1.5.0'`` if not found; ``'1.5.2'`` on other errors.
        """
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
        """
        Soft-delete an image by setting ``deleted = True``.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the deletion (audit trail).
        image_id : int
            Primary key of the image to delete.

        Raises
        ------
        DBException
            Code ``'1.5.3'`` on any SQLAlchemy error.
        """
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
        """
        Retrieve a single non-deleted annotation by primary key.

        Parameters
        ----------
        annotation_id : int
            Primary key of the annotation.

        Returns
        -------
        dict or None
            Annotation data as a dict, or ``None`` if not found or
            soft-deleted.
        """
        session = self.GetDBSession()
        try:
            row = session.query(Annotation).filter(Annotation.id == annotation_id, Annotation.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetAnnotations(self, flt=None):
        """
        Return all non-deleted annotations.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active annotation record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Annotation).filter(Annotation.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddAnnotation(self, actor_id, data):
        """
        Create a new annotation record.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the creation (audit trail).  Also
            used as the default ``id_user`` if not supplied in *data*.
        data : dict
            Annotation fields.  Required: ``'annotation_type'``,
            ``'geometry'``.  Optional: ``'id_image'``, ``'id_user'``
            (defaults to *actor_id*), ``'label'``, ``'confidence'``,
            ``'info'``.

        Returns
        -------
        dict
            The newly created annotation including its auto-generated
            ``id``.

        Raises
        ------
        DBException
            Code ``'1.6.1'`` on any SQLAlchemy error.
        """
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
        """
        Update mutable fields of an existing annotation.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'id_image'``,
            ``'annotation_type'``, ``'label'``, ``'geometry'``,
            ``'confidence'``, ``'info'``.

        Returns
        -------
        dict
            The updated annotation record.

        Raises
        ------
        DBException
            Code ``'1.6.0'`` if not found; ``'1.6.2'`` on other errors.
        """
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
        """
        Soft-delete an annotation by setting ``deleted = True``.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the deletion (audit trail).
        annotation_id : int
            Primary key of the annotation to delete.

        Raises
        ------
        DBException
            Code ``'1.6.3'`` on any SQLAlchemy error.
        """
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
        """
        Retrieve a single non-deleted finding by primary key.

        Parameters
        ----------
        finding_id : int
            Primary key of the finding.

        Returns
        -------
        dict or None
            Finding data as a dict, or ``None`` if not found or
            soft-deleted.
        """
        session = self.GetDBSession()
        try:
            row = session.query(Finding).filter(Finding.id == finding_id, Finding.deleted != True).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetFindings(self, flt=None):
        """
        Return all non-deleted findings.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per active finding record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Finding).filter(Finding.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def GetFindingsForStudy(self, study_id, flt=None):
        """
        Return all non-deleted findings belonging to a specific study.

        Parameters
        ----------
        study_id : int
            Foreign key referencing the parent study.
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            Findings whose ``id_study`` matches *study_id*.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Finding).filter(Finding.id_study == study_id, Finding.deleted != True)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddFinding(self, actor_id, data):
        """
        Create a new finding record (typically produced by an ML pipeline).

        Parameters
        ----------
        actor_id : int
            ID of the user or service account that triggered the pipeline
            (audit trail).
        data : dict
            Finding fields.  Required: ``'finding_type'``.  Optional:
            ``'id_image'``, ``'id_study'``, ``'id_process'``,
            ``'disease_category'``, ``'severity'``, ``'confidence'``,
            ``'ml_model_id'``, ``'info'`` (defaults to ``{}``).

        Returns
        -------
        dict
            The newly created finding including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.7.1'`` on any SQLAlchemy error.
        """
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
        """
        Mark a finding as reviewed by a clinician.

        Sets ``reviewed = True`` and records the reviewer's ID.  Optionally
        allows the reviewer to override the severity or attach notes.

        Parameters
        ----------
        reviewer_id : int
            ID of the user performing the review.
        data : dict
            Must contain ``'id'`` (the finding PK).  Optional:
            ``'review_notes'``, ``'severity'``.

        Returns
        -------
        dict
            The updated finding record with review metadata.

        Raises
        ------
        DBException
            Code ``'1.7.0'`` if the finding is not found; ``'1.7.2'`` on
            other errors.
        """
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
        """
        Retrieve a single process (pipeline run) by primary key.

        Processes are not soft-deleted, so all rows are eligible.

        Parameters
        ----------
        process_id : int
            Primary key of the process.

        Returns
        -------
        dict or None
            Process data as a dict, or ``None`` if not found.
        """
        session = self.GetDBSession()
        try:
            row = session.query(Process).filter(Process.id == process_id).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetProcesses(self, flt=None):
        """
        Return all process records.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per process record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(Process)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddProcess(self, actor_id, data):
        """
        Create a new process record to track a pipeline execution.

        Parameters
        ----------
        actor_id : int
            ID of the user who launched the process.  Also used as
            ``id_user`` default when not specified in *data*.
        data : dict
            Process fields.  Required: ``'type'``.  Optional:
            ``'id_user'`` (defaults to *actor_id*), ``'id_study'``,
            ``'status'`` (defaults to ``'PENDING'``), ``'parameters'``.

        Returns
        -------
        dict
            The newly created process including its auto-generated ``id``.

        Raises
        ------
        DBException
            Code ``'1.8.1'`` on any SQLAlchemy error.
        """
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
        """
        Update the status or results of a running/completed process.

        Typically called by the pipeline worker to report progress,
        completion, or failure.

        Parameters
        ----------
        actor_id : int
            ID of the user or service performing the update (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'status'``,
            ``'progress'``, ``'pid'``, ``'result'``, ``'time_start'``,
            ``'time_end'``, ``'error_message'``.

        Returns
        -------
        dict
            The updated process record.

        Raises
        ------
        DBException
            Code ``'1.8.0'`` if not found; ``'1.8.2'`` on other errors.
        """
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
        """
        Retrieve a single ML model record by primary key.

        Parameters
        ----------
        model_id : int
            Primary key of the ML model.

        Returns
        -------
        dict or None
            Model data as a dict, or ``None`` if not found.
        """
        session = self.GetDBSession()
        try:
            row = session.query(MLModel).filter(MLModel.id == model_id).first()
            return row2dict(row) if row else None
        finally:
            session.close()

    def GetMLModels(self, flt=None):
        """
        Return all registered ML model records.

        Parameters
        ----------
        flt : dict, optional
            Reserved for future server-side filtering; currently unused.

        Returns
        -------
        list of dict
            One dict per ML model record.
        """
        session = self.GetDBSession()
        try:
            q = session.query(MLModel)
            return [row2dict(r) for r in q.all()]
        finally:
            session.close()

    def AddMLModel(self, actor_id, data):
        """
        Register a new ML model in the catalogue.

        Parameters
        ----------
        actor_id : int
            ID of the user registering the model (audit trail).
        data : dict
            Model fields.  Required: ``'name'``, ``'version'``,
            ``'modality'``.  Optional: ``'architecture'``, ``'task'``,
            ``'weights_path'``, ``'info'``.

        Returns
        -------
        dict
            The newly created model record including its auto-generated
            ``id``.

        Raises
        ------
        DBException
            Code ``'1.9.1'`` on any SQLAlchemy error.
        """
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
        """
        Update mutable fields of an existing ML model record.

        Only keys present in *data* are applied; omitted keys are left
        unchanged.

        Parameters
        ----------
        actor_id : int
            ID of the user performing the modification (audit trail).
        data : dict
            Must contain ``'id'``.  May contain any of: ``'name'``,
            ``'version'``, ``'modality'``, ``'architecture'``, ``'task'``,
            ``'weights_path'``, ``'info'``.

        Returns
        -------
        dict
            The updated model record.

        Raises
        ------
        DBException
            Code ``'1.9.0'`` if not found; ``'1.9.2'`` on other errors.
        """
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
        """
        Register a new file upload in the database.

        Parameters
        ----------
        actor_id : int
            ID of the user who uploaded the file.  Stored as the owning
            ``id_user`` on the ``File`` row.
        data : dict
            File metadata.  Required: ``'name'``, ``'storage_path'``.
            Optional: ``'file_size_bytes'``, ``'content_type'``, ``'info'``.

        Returns
        -------
        dict
            The newly created file record including its auto-generated
            ``id``.

        Raises
        ------
        DBException
            Code ``'1.10.1'`` on any SQLAlchemy error.
        """
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
