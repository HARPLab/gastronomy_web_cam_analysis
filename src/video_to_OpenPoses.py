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


def getFeatureObj(currentframe, pose_datum):
    feature_data = {}
    feature_data['frame'] = currentframe
    feature_data['pose_body'] = pose_datum.poseKeypoints
    
    return feature_data

def addActivityToFeatureObj(feature_obj, personLabel, activity):
    feature_obj[personLabel] = activity
    return feature_obj



# 8-21-18
filenames_all = ['8-21-18', '8-13-18', '8-18-18', '8-19-18', '9-10-18']

for filename_root in filenames_all:
    print("parsing file " + filename_root)

    root = parseXML('../Annotations/' + filename_root + '-michael.eaf')
    timedict = {}
    activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20}

    # Open the corresponding video file for analysis
    cap = cv2.VideoCapture("../videos/" + filename_root + "_cropped.mp4")
    frames = {}
    frame_id = 0

    openpose_wrapper = OP()
    while True:
        ret,frame =cap.read()
        if not ret:
            break

        pose_datum = openpose_wrapper.getOpenposeDataFrom(frame=frame)
        frames[frame_id] = pose_datum
        frame_id = frame_id + 1
        if frame_id % 1000 == 0:
            print("processing through frame " + str(f_id))

    print("Total number of frames: " + str(frame_id))



    ## database initialization
    ## start looping through annotation labels
    write_file = "restaurant_features_full-" + filename_root + ".json"
    outfile = open(write_file, "w")
    
    outfile.write(json.dumps(frames))
    logfile.write("finished dumping\n") 
    logfile.close()
  
