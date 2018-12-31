import os
from contextlib import contextmanager

from sqlalchemy import Column, Integer, String, FLOAT, BOOLEAN, create_engine, BLOB, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

_data_dir = 'data'
_data_file = os.path.abspath(f'{_data_dir}/urlclustering.db')
if not os.path.exists(_data_dir):
    os.mkdir(_data_dir)
_Base = declarative_base()
_engine = create_engine(f'sqlite:///{_data_file}')
_Session = sessionmaker(bind=_engine)


class Sample(_Base):
    __tablename__ = 'samples'

    id = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String, nullable=False, index=True)
    url = Column(String, nullable=False)
    frequency = Column(FLOAT, index=True, nullable=False)
    amount = Column(Integer, index=True, nullable=False)
    features = Column(BLOB)
    label = Column(BOOLEAN, index=True)

    @staticmethod
    def truncate_table(cls, session):
        session.execute(Table(cls.__tablename__, _Base.metadata, autoload=True, autoload_with=_engine).delete())


@contextmanager
def create_session(**kwargs):
    session = _Session(**kwargs)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


_Base.metadata.create_all(_engine)
