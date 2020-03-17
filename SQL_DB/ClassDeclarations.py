from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.types import DateTime, Boolean, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

# info necessary to understand this from below link
#   https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/
Base = declarative_base()
 
class Clip(Base):
    #table for a parsed clips meta data
    __tablename__ = 'clip'
    id = Column(Integer, primary_key=True)
    #num_frames = Column(Integer, nullable=False)
    #start_time = Column(DateTime, nullable=False)
    #end_time = Column(DateTime, nullable=False)
    #parent_clip_path = Column(String(250), nullable=False)
    #clip_path = Column(String(250), nullable=False)
    #processed = Column(Boolean, default=False)
    clip_name = Column(String(250), nullable=False)

class Frame(Base):
    __tablename__ = 'frame'
    clip_id = Column(Integer, ForeignKey('clip.id'), nullable=False)
    #how will we be handling the waiter activities
    #waiter_present = Column(Boolean, default=False)
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, nullable=False)
    clip = relationship(Clip)

class Pose(Base):
    __tablename__ = 'pose'
    id = Column(Integer, primary_key=True)
    confidence_score = Column(Float, nullable=False)
    # Expect to use pickle to store/retrieve numpy arrays into/from this column, respectively
    pose_data = Column(LargeBinary, nullable=False)
    frame_id = Column(Integer, ForeignKey('frame.id'), nullable=False)
    frame = relationship(Frame)

class Object(Base):
    __tablename__ = 'object'
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frame.id'), nullable=False)
    object_type = Column(String(250), nullable=False)
    confidence_score = Column(Float, nullable=False)
    object_data = Column(LargeBinary, nullable=False)
    frame = relationship(Frame)

class Activity(Base):
    __tablename__ = 'activity'
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey('frame.id'), nullable=False)
    activity = Column(Integer, nullable=False)
    person_ID = Column(Integer, nullable=False)
    frame = relationship(Frame)

def maketable():
    # Create an engine that stores data in the local directory's
    # parsed_clips.db file.
    engine = create_engine('sqlite:///../src/gastro.db')
 
    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)
maketable()
