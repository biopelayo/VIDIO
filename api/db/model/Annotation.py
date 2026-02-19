from sqlalchemy import Column, TEXT, BOOLEAN, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Annotation(Base):
    __tablename__ = 'annotation'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_image = Column(UUID(as_uuid=False), ForeignKey('image.id'))
    id_user = Column(UUID(as_uuid=False), ForeignKey('user.id'))
    annotation_type = Column(TEXT(), nullable=False)
    label = Column(TEXT(), nullable=True)
    geometry = Column(MutableDict.as_mutable(JSONB), nullable=False)
    confidence = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
