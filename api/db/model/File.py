from sqlalchemy import Column, TEXT, BIGINT, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class File(Base):
    __tablename__ = 'file'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_user = Column(UUID(as_uuid=False), ForeignKey('user.id'))
    name = Column(TEXT(), nullable=False)
    storage_path = Column(TEXT(), nullable=False)
    file_size_bytes = Column(BIGINT(), nullable=True)
    content_type = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    uploaded_at = Column(TIMESTAMP(), default=func.now())
