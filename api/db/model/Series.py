from sqlalchemy import Column, TEXT, BOOLEAN, INTEGER, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Series(Base):
    __tablename__ = 'series'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_study = Column(UUID(as_uuid=False), ForeignKey('study.id'))
    series_number = Column(INTEGER(), nullable=True)
    description = Column(TEXT(), nullable=True)
    body_part = Column(TEXT(), nullable=True)
    modality_subtype = Column(TEXT(), nullable=True)
    manufacturer = Column(TEXT(), nullable=True)
    equipment_model = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
