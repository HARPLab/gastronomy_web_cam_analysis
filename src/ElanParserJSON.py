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
import xml.etree.ElementTree as ET
import cv2
from collections import defaultdict
from OPwrapper import OP
import json

def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

filename_root = "8-21-18"

root = parseXML('../Annotations/' + filename_root + '-michael.eaf')
timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20}

##cv2 create framedictionary
cap = cv2.VideoCapture("../videos/" + filename_root + "_cropped.mp4")
frames = {}
frame_id = 0
while True:
    ret,frame =cap.read()
    if not ret:
        break
    frames[frame_id] = frame
    frame_id = frame_id + 1
print(frame_id)

def getFeatureObj(currentframe, pose_datum):
    feature_data = {}
    feature_data['frame'] = currentframe
    feature_data['pose_body'] = pose_datum.poseKeypoints
    feature_data['pose_hand_left'] = pose_datum.handKeypoints[0]
    feature_data['pose_hand_right'] = pose_datum.handKeypoints[1]
    feature_data['pose_face'] = pose_datum.faceKeypoints
   
    return feature_data

def addActivityToFeatureObj(feature_obj, personLabel, activity):
    feature_obj[personLabel] = activity
    return feature_obj



## database initialization
## start looping through annotation labels
write_file = "restaurant_features_full-" + filename_root + ".json"
outfile = open(write_file, "w")
openpose_wrapper = OP()
frame_to_poseact ={} #f_id -> (posedata, personAactivity, personBactivity)
surf = cv2.SURF(400)
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
                #print("beginning new annotation: \n\n")
                for anno in temp: ## another single iteration loop
                    activity=anno.text
                for f_id in range (beginning_frame, ending_frame):
                    #print("adding another frame from annotation")
                    if f_id >= frame_id:
                        continue
                    currentframe = frames[f_id]
                    gray = cv2.cvtColor(currentframe, cv2.COLOR_BGR2GRAY)
                    kp, des = surf.detectAndCompute(gray,None)
                    print(len(kp))
                    pose_datum = openpose_wrapper.getOpenposeDataFrom(frame=currentframe)
                    feature_data = getFeatureObj(currentframe, pose_datum)
                    feature_data = addActivityToFeatureObj(feature_data, 'person-A', activity)
                    frame_to_poseact[f_id] = feature_data
                    print("added personA " + f_id)


    elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonB':
         for annotation in child:
            print("adding PersonB's annotations...")
            for temp in annotation: ## this should only loop once, couldnt figure out how to access a child xml tag without looping
                ## beginning frame
                beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
                activity = None
                for anno in temp: ## another single iteration loop
                    activity=anno.text
                for f_id in range (beginning_frame, ending_frame):
                    #print("adding another frame from annotation")
                    if f_id in frame_to_poseact.keys():
                        feature_data = frame_to_poseact[f_id]
                        frame_to_poseact[f_id] = addActivityToFeatureObj(feature_data, 'person-B', activity)
                        print("added personB " + f_id)
                        continue                    
                    #print(str(f_id))
                    if f_id >= frame_id:
                        continue
                    else:
                        currentframe = frames[f_id]
                        #print(currentframe)
                        #print(frames[30])
                        
                        pose_datum = openpose_wrapper.getOpenposeDataFrom(frame=currentframe)
                        feature_data = getFeatureObj(currentframe, pose_datum)
                        feature_data = addActivityToFeatureObj(feature_data, 'person-B', activity)
                        
                        print("updated personB " + f_id)

                        frame_to_poseact[f_id] = feature_data
                    #print(writestring + activity)
                    #outfile.write(writestring + "\n")
        
print(len(frame_to_poseact))

outfile.write(json.dumps(frame_to_poseact))

