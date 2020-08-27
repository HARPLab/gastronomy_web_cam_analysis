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
import cv2
from OPwrapper import OP
import cvtest
def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

filename = "8-21-18"
root = parseXML('../Annotations/' + filename+ '-michael.eaf')
timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20}
#object detection model 
model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
detection_model = load_model(model_name)
PATH_TO_LABELS = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
objectdict = {'cup': 0, 'fork':1, 'knife':2, 'spoon': 3, 'bowl':4}
##cv2 create framedictionary
cap = cv2.VideoCapture("../videos/" + filename + "_cropped.mp4")
frames = {}
#frame_id = 0
#while True:
#    ret,frame =cap.read()
#    if not ret:
#        break
#    frames[frame_id] = frame
#    frame_id = frame_id + 1

OPobj = OP()
## database initialization
#engine = create_engine('sqlite:///gastro.db')
#Base.metadata.bind = engine
#DBSession = sessionmaker(bind=engine)
#session = DBSession()
#new_clip = Clip(clip_name=filename + '-michael.eaf')
#session.add(new_clip)
#session.commit()i
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)
def maskTable(frame):
    ## TODO write this function to mask out the able portion of the frame?
    return frame
def classToString(class_id):
    ## TODO figure out the class dictionary for the tensorflow model
    return "plate"
## start looping through annotation labels
frame_to_data_frame = {}
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
                    if f_id >= totalFrames:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES,f_id)
                    ret, currentframe = cap.read()
                    masked = currentframe[100:400, 100:370]
                    detectedclasses = retrieveclasses(detection_model, maskedimage) 
                    objectarray = np.zeros((1,5)) 
                    for label in detectedclasses.keys():
                         if label == "cup":
                             objectarray[0] = 1
                         elif label == "bowl":
                             objectarray[1] = 1
			 elif label == "fork":
                             objectarray[2] = 1
                         elif label == "knife":
                             objectarray[3] = 1
                         elif label == "spoon":
                             objectarray[4] = 1
                    pose_datum = OPobj.getOpenposeDataFrom(frame=currentframe)
                    writestring = "Body:" + str(pose_datum.poseKeypoints) + "LH:" + str(pose_datum.handKeypoints[0]) + "RH:" + str(pose_datum.handKeypoints[1]) + "Face:"+str(pose_datum.faceKeypoints)
                    print(writestring)
                    frame_to_data_frame[f_id] = (writestring, activity, "NONE", objectarray)

    elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonB':
        for annotation in child:
            print("adding PersonB's annotations...")
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
                    if f_id >= totalFrames:
                        continue
                    if f_id in frame_to_data_frame.keys():
                        (a,b,c, d) = frame_to_dataframe[f_id]
                        frame_to_data_frame[f_id] = (a,b,activity,d)	
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,currentframe)
                        ret, currentframe = cap.read()
                        masked = currentframe[100:400, 100:370]
                        detectedclasses = retrieveclasses(detection_model, maskedimage)
                        objectarray = np.zeros((1,5))
                        for label in detectedclasses.keys():
                            if label == "cup":
                                objectarray[0] = 1
                            elif label == "bowl":
                                objectarray[1] = 1
                            elif label == "fork":
                                objectarray[2] = 1
                            elif label == "knife":
                                objectarray[3] = 1
                            elif label == "spoon":
                                objectarray[4] = 1
                         pose_datum = OP.getOpenposeDataFrom(currentframe)
                         writestring = "Body:" + str(pose_datum.poseKeypoints) + "LH:" + str(pose_datum.handKeypoints[0]) + "RH:" + str(pose_datum.handKeypoints[1]) + "Face:"+str(pose_datum.faceKeypoints)
                         print(writestring)
                         frame_to_data_frame[fid] = (writestring, "NONE", activity, objectarray)
#activity = session.query(Activity).all()
f = open("hello.txt", "wb") 
for key in frame_to_data_frame:
    (a,b,c,d) = frame_to_data_frame[key]
    f.write(a + "PA:" + b + "PB:" + c + "OBJ:" + d.tostring() + "\n")
#for act in activity:
    #print(act.__dict__['frame_id'])
    #parent = session.query(Frame).filter(Frame.frame_id == act.__dict__['frame_id']).all()
    #for par in parent:
    #    print(par)
