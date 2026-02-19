from sqlalchemy import Column, TEXT, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Log(Base):
    __tablename__ = 'log'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_user = Column(UUID(as_uuid=False), nullable=True)
    type = Column(TEXT(), nullable=True)
    date = Column(TIMESTAMP(), default=func.now())
    message = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
