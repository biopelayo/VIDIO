from sqlalchemy import Column, TEXT, BOOLEAN, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Study(Base):
    __tablename__ = 'study'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_patient = Column(UUID(as_uuid=False), ForeignKey('patient.id'))
    study_date = Column(TIMESTAMP(), nullable=True)
    modality = Column(TEXT(), nullable=False)
    institution = Column(TEXT(), nullable=True)
    protocol = Column(TEXT(), nullable=True)
    description = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=True)
