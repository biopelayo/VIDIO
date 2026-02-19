from sqlalchemy import Column, TEXT, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class MLModel(Base):
    __tablename__ = 'ml_model'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    name = Column(TEXT(), nullable=False)
    version = Column(TEXT(), nullable=False)
    modality = Column(TEXT(), nullable=False)
    architecture = Column(TEXT(), nullable=True)
    task = Column(TEXT(), nullable=True)
    weights_path = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)

    __table_args__ = (
        UniqueConstraint('name', 'version'),
    )
