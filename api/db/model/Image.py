from sqlalchemy import Column, TEXT, BOOLEAN, BIGINT, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Image(Base):
    __tablename__ = 'image'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_series = Column(UUID(as_uuid=False), ForeignKey('series.id'))
    name = Column(TEXT(), nullable=False)
    storage_path = Column(TEXT(), nullable=False)
    file_format = Column(TEXT(), nullable=True)
    file_size_bytes = Column(BIGINT(), nullable=True)
    pixel_spacing = Column(MutableDict.as_mutable(JSONB), nullable=True)
    dimensions = Column(MutableDict.as_mutable(JSONB), nullable=True)
    sop_instance_uid = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    selected = Column(BOOLEAN(), default=True, nullable=False)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
