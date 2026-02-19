from sqlalchemy import Column, TEXT, BOOLEAN, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func

from api.db.model.Base import Base


class User(Base):
    __tablename__ = 'user'

    id = Column(UUID(as_uuid=False), primary_key=True, default=func.uuid_generate_v1())
    username = Column(TEXT(), nullable=False, unique=True)
    password_hash = Column(TEXT(), nullable=False)
    name = Column(TEXT(), nullable=False)
    surname = Column(TEXT(), nullable=True)
    role = Column(TEXT(), default='analyst')
    info = Column(MutableDict.as_mutable(JSONB), nullable=True)
    deleted = Column(BOOLEAN(), default=False, nullable=False)


class UserToken(Base):
    __tablename__ = 'user_token'

    id_user = Column(UUID(as_uuid=False), ForeignKey('user.id'), primary_key=True)
    token = Column(TEXT(), nullable=False)
    expires = Column(TIMESTAMP(), nullable=False)
    created = Column(TIMESTAMP(), nullable=False)
