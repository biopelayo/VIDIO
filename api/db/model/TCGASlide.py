from sqlalchemy import Column, TEXT, BOOLEAN, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class TCGASlide(Base):
    __tablename__ = 'tcga_slide'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    id_tcga_case = Column(UUID(as_uuid=False), ForeignKey('tcga_case.id'))
    id_image = Column(UUID(as_uuid=False), ForeignKey('image.id'))
    slide_barcode = Column(TEXT(), nullable=True)
    gdc_file_id = Column(TEXT(), nullable=True)
    tissue_type = Column(TEXT(), nullable=True)
    stain_type = Column(TEXT(), nullable=True)
    magnification = Column(TEXT(), nullable=True)
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)
