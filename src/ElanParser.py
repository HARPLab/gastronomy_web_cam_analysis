##### Activity Key #####
# 0 away-from-table
# 1 idle
# 2 eating
# 3 drinking
# 4 talking
# 5 ordering
# 6 standing
# 7 talking:waiter
# 8 looking:window
# 9 looking:waiter
# 10 reading:bill
# 11 reading:menu
# 12 paying:check
# 13 using:phone
# 14 using:napkin
# 15 using:purse
# 16 using:glasses
# 17 using:wallet
# 18 looking:PersonA
# 19 looking:PersonB
# 20 take-out-food
######################
##
## loop through each frame and check if there is an annotation available for that frame.
## record the frame id for which there are annotations and use the annotation key
## fram nump should start at 0, then convert frame number to time through 30 fps
import xml.etree.ElementTree as ET
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from SQL_DB.ClassDeclarations import Frame, Pose, Object, Activity, Clip, Base
import cv2
from OPwrapper import OP
from tensorflow_human_detection import DetectorAPI
def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

root = parseXML('../Annotations/8-21-18-michael.eaf')
timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20}

##cv2 create framedictionary
cap = cv2.VideoCapture("PATH_TO_CLIP")
frames = {}
frame_id = 0
while True:
    ret,frame =cap.read()
    if not ret:
        break
    frames[frame_id] = frame
    frame_id = frame_id + 1

## object detector initialization
object_detector = DetectorAPI()

## database initialization
engine = create_engine('sqlite:///gastro.db')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()
new_clip = Clip(clip_name='8-21-18-michael.eaf')
session.add(new_clip)
session.commit()
def maskTable(frame):
    ## TODO write this function to mask out the able portion of the frame?
    return frame
def classToString(class_id):
    ## TODO figure out the class dictionary for the tensorflow model
    return "plate"
## start looping through annotation labels
for child in root:
    if child.tag == 'TIME_ORDER':
        for times in child:
            timedict[times.attrib['TIME_SLOT_ID']] = times.attrib['TIME_VALUE']
    elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonA':
        for annotation in child:
            print("adding PersonA's annotations...")
            for temp in annotation: ## this should only loop once, couldnt figure out how to access a child xml tag without looping
                #print(temp.attrib['TIME_SLOT_REF1'])
                ## beginning frame
                beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
                activity = None
                print("beginning new annotation: \n\n")
                for anno in temp: ## another single iteration loop
                    activity=anno.text
                for f_id in range (beginning_frame, ending_frame):
                    print("adding another frame from annotation")
                    new_frame = Frame(frame_id=f_id, clip=new_clip)
                    session.add(new_frame)
                    session.commit()
                    new_activity = Activity(activity=activitydict[activity], person_ID=0, frame=new_frame)
                    session.add(new_activity)
                    session.commit()
                    currentframe = frames[f_id]
                    pose_datum = OP.getOpenposeDataFrom(currentframe)
                    ## TODO not sure if this typechecks, hard to see because I haven't installed cv2 on my laptop yet/havent
                    ## set it up in this environment. I am also unsure of how the LargeBinary column type the pose data
                    ## is stored as in the database- is this the standard way to store arrays in a sqlalchemy database?
                    new_pose = Pose(body_data=pose_datum.poseKeypoints, face_data=pose_datum.faceKeypoints,
                                    left_hand=pose_datum.handKeypoints[0],right_hand=pose_datum.handKeypoints[1],
                                    frame=new_frame)
                    session.add(new_pose)
                    session.commit()
                    ## TODO look into what classes are defined by the tensorflow model.  and just store the semantic
                    ## meanings of the class_labels(as a string describing th eobject)
                    currenttable = maskTable(currentframe)
                    boxes_list, conf_scores, class_labels, number_of_detections = object_detector.processFrame(currenttable)
                    for object_id in range(0, number_of_detections):
                        new_object = Object(condifence_score=conf_scores[object_id],object_type=classToString(class_labels[object_id]),
                                            frame=new_frame)
                        session.add(new_object)
                        session.commit()


    elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonB':
        print("Todo")
                # add frames within these bounds with correct activities to data base
                # for each frame, add (pose, objects, activity): For future persons, need to check if frame is already
                # added, in which case only need to add another activity entry related to the frame. This should make
                # each frame be only related to exactly one pose, one objectdata, and at most two(n) activities
                # using the frame id, get the correct frame run openpose and add coresponding pose to database
                # using frame id, get correct frame and run object detection
#activity = session.query(Activity).all()

#for act in activity:
    #print(act.__dict__['frame_id'])
    #parent = session.query(Frame).filter(Frame.frame_id == act.__dict__['frame_id']).all()
    #for par in parent:
    #    print(par)