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
engine = create_engine('sqlite:///gastro.db')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()
new_clip = Clip(clip_name='8-21-18-michael.eaf')
session.add(new_clip)
session.commit()
for child in root:
    if child.tag == 'TIME_ORDER':
        for times in child:
            timedict[times.attrib['TIME_SLOT_ID']] = times.attrib['TIME_VALUE']
    elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonA':
        for annotation in child:
            print("adding PersonA's annotations...")
            for temp in annotation:
                #print(temp.attrib['TIME_SLOT_REF1'])
                ## beginning frame
                beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
                activity = None
                print("beginning new annotation: \n\n")
                for anno in temp:
                    activity=anno.text
                for f_id in range (beginning_frame, ending_frame):
                    print("adding another frame from annotation")
                    new_frame = Frame(frame_id=f_id, clip=new_clip)
                    session.add(new_frame)
                    session.commit()
                    new_activity = Activity(activity=activitydict[activity], person_ID=0, frame=new_frame)
                    session.add(new_activity)
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