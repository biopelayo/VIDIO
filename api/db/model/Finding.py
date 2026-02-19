from sqlalchemy import Column, TEXT, BOOLEAN, REAL, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class Finding(Base):
    __tablename__ = 'finding'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_image = Column(UUID(as_uuid=False), ForeignKey('image.id'))
    id_study = Column(UUID(as_uuid=False), ForeignKey('study.id'))
    id_process = Column(UUID(as_uuid=False), nullable=True)
    finding_type = Column(TEXT(), nullable=False)
    disease_category = Column(TEXT(), nullable=True)
    severity = Column(TEXT(), nullable=True)
    confidence = Column(REAL(), nullable=True)
    ml_model_id = Column(UUID(as_uuid=False), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=False)
    reviewed = Column(BOOLEAN(), default=False)
    id_reviewer = Column(UUID(as_uuid=False), ForeignKey('user.id'), nullable=True)
    review_notes = Column(TEXT(), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
