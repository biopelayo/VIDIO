from sqlalchemy import Column, TEXT, BOOLEAN, INTEGER, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class SpatialExperiment(Base):
    __tablename__ = 'spatial_experiment'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_study = Column(UUID(as_uuid=False), ForeignKey('study.id'))
    id_image = Column(UUID(as_uuid=False), ForeignKey('image.id'))
    platform = Column(TEXT(), nullable=True)
    h5ad_path = Column(TEXT(), nullable=True)
    n_spots = Column(INTEGER(), nullable=True)
    n_genes = Column(INTEGER(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
