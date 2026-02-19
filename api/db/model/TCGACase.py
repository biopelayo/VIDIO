from sqlalchemy import Column, TEXT, BOOLEAN, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class TCGACase(Base):
    __tablename__ = 'tcga_case'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_patient = Column(UUID(as_uuid=False), ForeignKey('patient.id'))
    case_barcode = Column(TEXT(), nullable=True, unique=True)
    project = Column(TEXT(), nullable=True)
    disease_type = Column(TEXT(), nullable=True)
    primary_diagnosis = Column(TEXT(), nullable=True)
    tumor_stage = Column(TEXT(), nullable=True)
    gdc_case_id = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
