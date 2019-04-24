from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
 
from ClassDeclarations import Base, Clip, Frame, Pose, Object

#also learned main logic for this wrapper [here](https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/)
class DBWrapper:
    def __init__(self):
        self.lit = True
        engine = create_engine('sqlite:///sqlalchemy_example.db')
        # Bind the engine to the metadata of the Base class so that the
        # declaratives can be accessed through a DBSession instance
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        self.session = DBSession()

    def get_session(self):
        return self.session
    
