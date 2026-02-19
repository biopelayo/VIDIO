from sqlalchemy import Column, TEXT, INTEGER, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Process(Base):
    __tablename__ = 'process'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_user = Column(UUID(as_uuid=False), ForeignKey('user.id'))
    id_study = Column(UUID(as_uuid=False), ForeignKey('study.id'))
    type = Column(TEXT(), nullable=False)
    status = Column(TEXT(), default='PENDING')
    progress = Column(INTEGER(), default=0)
    pid = Column(INTEGER(), nullable=True)
    parameters = Column(MutableDict.as_mutable(JSONB), nullable=True)
    result = Column(MutableDict.as_mutable(JSONB), nullable=True)
    time_start = Column(TIMESTAMP(), nullable=True)
    time_end = Column(TIMESTAMP(), nullable=True)
    error_message = Column(TEXT(), nullable=True)
