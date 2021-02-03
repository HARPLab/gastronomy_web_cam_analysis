import time
import cv2
import random
import pickle
import pandas as pd

activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}

activity_from_key = {0:'away-from-table', 1:'idle', 2:'eating', 3: 'drinking', 4: 'talking', 5: 'ordering', 6: 'standing',
                        7: 'talking:waiter', 8: 'looking:window', 9: 'looking:waiter', 10: 'reading:bill', 11: 'reading:menu',
                        12: 'paying:check', 13: 'using:phone', 14: 'using:napkin', 15: 'using:purse', 16: 'using:glasses',
                        17: 'using:wallet', 18: 'looking:PersonA', 19: 'looking:PersonB', 20: 'takeoutfood', 21: 'leaving-table', 22: 'cleaning-up', 23: 'NONE'}


# Lookup table for OpenPose keypoint indices
keypoint_labels = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
                                                "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
                                                "LEar","LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background", '']


filenames_all = ['8-13-18', '8-18-18', '8-19-18', '8-17-18', '8-21-18']
prefix_output = './quality-checks/'
prefix_input = './output-vectors/'

INDEX_PA = 0
INDEX_PB = 1


def get_label_PA(input_row_Y):
    if input_row_Y[INDEX_PA] in activity_from_key:
        return activity_from_key[input_row_Y[INDEX_PA]]
    else:
        return -1

def get_label_PB(input_row_Y):
    if input_row_Y[INDEX_PB] in activity_from_key:
        return activity_from_key[input_row_Y[INDEX_PB]]
    else:
        return -1

def get_label_id_PA(input_row_Y):
    return input_row_Y[INDEX_PA]

def get_label_id_PB(input_row_Y):
    return input_row_Y[INDEX_PB]

def get_PA(input_row_X):
    if (input_row_X.shape[0] == 2):
        print("Wrong vector passed for pose extraction")

    return input_row_X[0:25]

def get_PB(input_row_X):
    if (input_row_X.shape[0] == 2):
        print("Wrong vector passed for pose extraction")

    return input_row_X[25:]


def add_pose_to_image(pose, img, color):
    # TODO verify 
    for p in pose:
        x1, y1, c1 = p[0], p[1], p[2]
        x1 = int(x1)
        y1 = int(y1)
        frame_img = cv2.circle(img, (x1,y1), 3, color, -1)

    return frame_img

def export_annotated_frame(f_id, row_X, row_Y, raw_X, label, cap, export_all_poses=False, frame_group=0):
    COLOR_BLACK = (0, 0, 0)
    COLOR_RED = (255, 0, 0)
    COLOR_BLUE = (0, 0, 255)

    print("exporting outlier")
    label_a = str(get_label_PA(row_Y))
    label_b = str(get_label_PB(row_Y))
    pose_a  = get_PA(row_X)
    pose_b  = get_PB(row_X)
    frame_num = int(f_id)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame_img = cap.read()
    
    height, width, channels = frame_img.shape

    # for pose in all_poses(raw_X):
    #     frame_img = add_pose_to_image(pose_a, frame_img, COLOR_BLACK)


    frame_img = add_pose_to_image(pose_a, frame_img, COLOR_RED)
    frame_img = add_pose_to_image(pose_b, frame_img, COLOR_BLUE)


    halfway = int(width / 2)

    org_a = (50, 50) 
    org_b = (50 + halfway, 50) 
      

    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = .25
    color = (255, 0, 0) 
    thickness = 2
   
    print(label_a)
    frame_imag = cv2.putText(frame_img, label_a, org_a, font,  
                   fontScale, COLOR_RED, thickness, cv2.LINE_AA) 

    frame_imag = cv2.putText(frame_img, label_b, org_b, font, fontScale, COLOR_RED, thickness, cv2.LINE_AA) 


    title = filename + "_shows_" + label + "_f" + str(frame_num) + ".jpg"
    cv2.imwrite(prefix_output + title, frame_img) 
    print("Exported outlier " + title)


EXPORT_DROPOUT_STATS                = True
EXPORT_TYPE_TOO_MANY_POSES          = False
EXPORT_TYPE_AWAY_BUT_POSE_DETECTED  = True
EXPORT_TYPE_RANDOM                  = True
EXPORT_WAITER_MOMENTS               = True

LABEL_TOO_MANY_POSES                = 'err-too-many-poses'
LABEL_TYPE_AWAY_BUT_POSE_DETECTED   = 'err-away-but-pose'
LABEL_TYPE_RANDOM                   = 'quality-check-random'
LABEL_WAITER_MOMENTS                = 'quality-check-waiter'


# run the experiment
def check_quality_and_export_trimmed(filename):
    print("Running quality checks")
    X_all = pickle.load(open(prefix_input + filename + '_roles_X.p',"rb"))
    X_raw = pickle.load(open(prefix_input + filename + '_raw_X.p',"rb"))
    Y_all = pickle.load(open(prefix_input + filename + '_Y.p',"rb"))

    print("loaded pickle datasets for " + filename)
    print(X_all.shape)
    print(Y_all.shape)

    # Verify that the two input vectors line up
    # Always guaranteed to start at 0, and be the length of the clip
    # TODO could add check if is the length of the clip
    if X_all.shape[0] != Y_all.shape[0]:
        print("Error processing due to different lengths of input vector")

    counter = 0

    prev_frame_num = -1
    frame_group = 0
    filename_root = filenames_all[frame_group]
    cap = cv2.VideoCapture("../../videos/" + filename + "_cropped.mp4")

    vector_length = X_all.shape[0]
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if vector_length != video_length:
        print("Warning: raw video is of length " + str(video_length) + " while vector is of length " + str(vector_length))
        print("This may lead to off-by-" + str(video_length - vector_length) + " errors.")
        print("Are you trying to re-trim a clip? Careful!")
        print("Exiting without analysis or trimming export")
        return

    # For each RestaurantFrame in the list
    for rid in range(vector_length):
        row_X = X_all[rid]
        raw_X = X_raw[rid]
        row_Y = Y_all[rid]
        
        label_pa = get_label_PA(row_Y)
        label_pb = get_label_PB(row_Y)
        
        # num_poses_raw   = get_num_poses_raw(row)
        # num_poses_clean = get_num_poses_clean(row)

        pose_pa = get_PA(row_X)
        pose_pb = get_PB(row_X)
        frame_num = rid

        chance = random.random()

        # if label_pa == 'leaving-table' and label_pb == 'leaving-table':
        #     print("meal finish lt noted at " + str(frame_num))

        # if label_pa == 'away-from-table' and label_pb == 'away-from-table':
        #     print("meal finish aa noted at " + str(frame_num))


        # if label_pa == 'away-from-table' and pose_pb is not None and chance < .01:
        #     export_annotated_frame(frame_num, row_X, row_Y, LABEL_TYPE_AWAY_BUT_POSE_DETECTED, cap, frame_group)
        #     counter += 1

        # if label_pb == 'away-from-table' and pose_pb is not None and chance < .01:
        #     export_annotated_frame(frame_num, row_X, row_Y, LABEL_TYPE_AWAY_BUT_POSE_DETECTED, cap, frame_group)
        #     counter += 1

        if chance < .0001:
            export_annotated_frame(frame_num, row_X, row_Y, raw_X, LABEL_TYPE_RANDOM, cap, filename)
            counter += 1



    print(frame_num)
    print(label_pa)
    print(label_pb)

    

    filehandler = open(prefix_output + "QC_" + filename_root + "_X.p", "wb")
    pickle.dump(X_all, filehandler)
    filehandler.close()

    filehandler = open(prefix_output + "QC_" + filename_root + "_X.p", "wb")
    pickle.dump(Y_all, open(prefix_output + "QC_" + filename_root + "_Y.p", "wb"))
    filehandler.close()


    # for filename_root in filenames_all:
    #     print("parsing file " + filename_root)
    #     root = parseXML('../Annotations/' + filename_root + '-michael.eaf')

    #     cap = cv2.VideoCapture("../videos/" + filename_root + "_cropped.mp4")
    #     frames = {}
    #     frame_id = 0
    #     while True:
    #         ret,frame =cap.read()
    #         if not ret:
    #             break
    #         frames[frame_id] = frame
    #         frame_id = frame_id + 1
    #         if frame_id % 1000 == 0:
    #             print("processing through frame " + str(f_id))

for filename in filenames_all:
    check_quality_and_export_trimmed(filename)
