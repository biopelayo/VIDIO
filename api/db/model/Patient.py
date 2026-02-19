from sqlalchemy import Column, TEXT, BOOLEAN, DATE
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Patient(Base):
    __tablename__ = 'patient'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    medical_record_number = Column(TEXT(), nullable=True, unique=True)
    name = Column(TEXT(), nullable=True)
    date_of_birth = Column(DATE(), nullable=True)
    sex = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
