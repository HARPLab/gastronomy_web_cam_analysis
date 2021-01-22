from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
 
from SQL_DB.ClassDeclarations import Base, Clip, Frame, Pose, Object

#also learned main logic for this wrapper [here](https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/)
class DBWrapper:
    DB_ABSOLUTE_PATH='home/mghuang/data/gastronomy_analysis/SQL_DB/parsed_clips.db'
    def __init__(self):
        self.engine = create_engine('sqlite:////{}'.format(DBWrapper.DB_ABSOLUTE_PATH))
        # Bind the engine to the metadata of the Base class so that the
        # declaratives can be accessed through a DBSession instance
        Base.metadata.bind = self.engine
        self.DBSession = sessionmaker(bind=self.engine)
        self.session = self.DBSession()

    def get_session(self):
        return self.session
    
