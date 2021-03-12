import time
import cv2
import random
import pickle
import pandas as pd
import numpy as np
import json

import sys
sys.path.append("..")
import qchecks
import arconsts

# np.set_printoptions(suppress=True)

# activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
#                         'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
#                         'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
#                         'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}

# activity_from_key = {0:'away-from-table', 1:'idle', 2:'eating', 3: 'drinking', 4: 'talking', 5: 'ordering', 6: 'standing',
#                         7: 'talking:waiter', 8: 'looking:window', 9: 'looking:waiter', 10: 'reading:bill', 11: 'reading:menu',
#                         12: 'paying:check', 13: 'using:phone', 14: 'using:napkin', 15: 'using:purse', 16: 'using:glasses',
#                         17: 'using:wallet', 18: 'looking:PersonA', 19: 'looking:PersonB', 20: 'takeoutfood', 21: 'leaving-table', 22: 'cleaning-up', 23: 'NONE'}

activity_from_key = arconsts.activity_from_key_full

# Lookup table for OpenPose keypoint indices
keypoint_labels = arconsts.keypoint_labels
filenames_all = arconsts.filenames_all

prefix_qc = './quality-checks/'
prefix_vectors_in = './output-vectors/raws/'
prefix_vectors_out = './output-vectors/trimmed/'

INDEX_PA = 0
INDEX_PB = 1

EXPORT_DROPOUT_STATS                = True
EXPORT_TYPE_TOO_MANY_POSES          = False
EXPORT_TYPE_AWAY_BUT_POSE_DETECTED  = True
EXPORT_TYPE_RANDOM                  = True
EXPORT_WAITER_MOMENTS               = True

LABEL_TOO_MANY_POSES                = 'err-too-many-poses'
LABEL_TYPE_AWAY_BUT_POSE_DETECTED   = 'err-away-but-pose'
LABEL_TYPE_RANDOM                   = 'quality-check-random'
LABEL_WAITER_MOMENTS                = 'quality-check-waiter'
LABEL_TYPE_DELETED                  = 'deleted'
LABEL_TYPE_FIRST                    = 'first'
LABEL_TYPE_FLIPPED                  = 'flipped'


def get_label_PA(input_row_Y):
    if input_row_Y[INDEX_PA] in activity_from_key.keys():
        return activity_from_key[input_row_Y[INDEX_PA]]
    else:
        return -1

def get_label_PB(input_row_Y):
    if input_row_Y[INDEX_PB] in activity_from_key.keys():
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
    frame_img = img
    # TODO verify 
    for p in pose:
        x1, y1, c1 = p[0], p[1], p[2]
        x1 = int(x1)
        y1 = int(y1)
        frame_img = cv2.circle(img, (x1,y1), 3, color, -1)

    return frame_img

def get_mirror_A_to_B(pose_A):
    points_mirrored = []
    print(pose_A)

    for pt in pose_A:
        new_pt = (400 - pt[0], pt[1], pt[2])

    print(points_mirrored)
    return points_mirrored



def get_mirror_B_to_A(pose_B):
    points_mirrored = []
    print(pose_B)

    for pt in pose_B:
        new_pt = (400 - pt[0], pt[1], pt[2])

    print(points_mirrored)
    exit()
    return points_mirrored

pts_A = []
pts_B = []

def get_homography_A_to_B():
    

def get_homography_B_to_A():
    



def export_flipped_frame(can_annotate, f_id, row_X, row_Y, raw_X, label, cap, export_all_poses=False, frame_group=0):
    label_a = str(get_label_PA(row_Y))
    label_b = str(get_label_PB(row_Y))
    pose_a  = get_PA(row_X)
    pose_b  = get_PB(row_X)
    frame_num = int(f_id)

    color_a = (100, 200, 100)
    color_b = (0, 255, 0)

    pose_A_mirrored = get_mirror_A_to_B(pose_a)
    pose_B_mirrored = get_mirror_B_to_A(pose_b)

    export_frame(can_annotate, f_id, pose_a, pose_b, label_a, label_b, cap, export_reason=label, color_a=color_a, color_b=color_b)
    export_frame(can_annotate, f_id, pose_B_mirrored, pose_A_mirrored, label_b + "-flip", label_a + "-flip", cap, export_reason=label + "flop", color_a=color_b, color_b=color_a)
    print("Exported flipped frame")



def export_annotated_frame(can_annotate, f_id, row_X, row_Y, raw_X, label, cap, export_all_poses=False, frame_group=0):
    if not can_annotate:
        return

    print("exporting outlier: " + label)
    label_a = str(get_label_PA(row_Y))
    label_b = str(get_label_PB(row_Y))
    pose_a  = get_PA(row_X)
    pose_b  = get_PB(row_X)
    frame_num = int(f_id)
    export_reason = label
    export_frame(can_annotate, f_id, pose_a, pose_b, label_a, label_b, cap, export_reason, raw_poses=raw_X)

def export_frame(can_annotate, f_id, pose_a, pose_b, label_a, label_b, cap, export_reason, raw_poses=None, color_a=None, color_b=None):
    COLOR_NEUTRAL = (255, 255, 255)
    if color_a == None:
        COLOR_A = (255, 0, 0)
    else:
        COLOR_A = color_a

    if color_b == None:
        COLOR_B = (0, 0, 255)
    else:
        COLOR_B = color_b

    label = export_reason
    output_file = {}

    if isinstance(pose_a, (np.ndarray, np.generic)):
        output_file['pose-A'] = pose_a.tolist()
    if isinstance(pose_b, (np.ndarray, np.generic)):
        output_file['pose-B'] = pose_b.tolist()

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_id)
    ret, frame_img = cap.read()
    
    height, width, channels = frame_img.shape

    bd_box_A = arconsts.bd_box_A
    bd_box_B = arconsts.bd_box_B
    frame_img = cv2.rectangle(frame_img, bd_box_A[0], bd_box_A[1], COLOR_A, 1)
    frame_img = cv2.rectangle(frame_img, bd_box_B[0], bd_box_B[1], COLOR_B, 1)

    all_poses = []
    if raw_poses:
        for pose in raw_poses:
            all_poses.append(pose)
            # pslice = pose_size * pid
            # pose = raw_X[pslice : pslice + pose_size]
            pose = np.array(pose).reshape((25, 3))
            frame_img = add_pose_to_image(pose, frame_img, COLOR_NEUTRAL)
        
        output_file['all-poses'] = all_poses

    frame_img = add_pose_to_image(pose_a, frame_img, COLOR_A)
    frame_img = add_pose_to_image(pose_b, frame_img, COLOR_B)

    halfway = int(width / 2)

    org_a = (50, 50) 
    org_b = (45 + halfway, 50) 
      

    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = .6
    color = (255, 0, 0) 
    thickness = 2
   
    frame_imag = cv2.putText(frame_img, label_a, org_a, font,  
                   fontScale, COLOR_A, thickness, cv2.LINE_AA) 

    frame_imag = cv2.putText(frame_img, label_b, org_b, font, fontScale, COLOR_B, thickness, cv2.LINE_AA) 


    title = filename + "_shows_" + label + "_f" + str(f_id)
    cv2.imwrite(prefix_qc + title + ".jpg", frame_img) 
    print("Exported outlier " + title)

    with open(prefix_qc + title + ".json", 'w') as outfile:  
        json.dump(output_file, outfile)


# run the experiment
def check_quality_and_export_trimmed(filename, export_frames=False):
    print("Running quality checks")
    try:
        print(prefix_vectors_in + filename + '_raw_X.p')
        print(prefix_vectors_in + filename + '_roles_X.p')
        print("\t ~~~")
        X_all = pickle.load(open(prefix_vectors_in + filename + '_roles_X.p',"rb"))
        X_raw = pickle.load(open(prefix_vectors_in + filename + '_raw_X.p',"rb"))
    except FileNotFoundError:
        print("X pose input information not found- please run pose_to_vectors!")
        exit()

    try:
        Y_all = pickle.load(open(prefix_vectors_in + filename + '_raw_Y.p',"rb"))
    except FileNotFoundError:
        print("Y labels not found: please run elan to vectors!")
        exit()

    qchecks.verify_io_expanded(X_all, Y_all)

    print("loaded pickle datasets for " + filename)
    print("\tDimensions of input X: " + str(X_all.shape) + " (video length x 25 OpenPose Pts x (x,y,confidence))")
    print("\tDimensions of input Y: " + str(Y_all.shape) + " (video length x (labela, labelb))")

    # Verify that the two input vectors line up
    # Always guaranteed to start at 0, and be the length of the clip
    # TODO could add check if is the length of the clip
    if X_all.shape[0] != Y_all.shape[0]:
        print("Error processing due to different lengths of input vector")

    counter = 0

    prev_frame_num = -1
    cap = cv2.VideoCapture("../../videos/" + filename + "_cropped.mp4")

    vector_length = X_all.shape[0]
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    CAN_ANNOTATE = True

    if video_length == 0:
        print("Video file not found for " + filename)
        print("\tIf you're confident in your vectors this is fine, otherwise, might want to look into that!")
        # print("Exiting without analysis or trimming export, since slides can't be annotated")
        print("\t WARNING ON " + filename)
        CAN_ANNOTATE = False

    elif vector_length != video_length:
        print("Warning: raw video is of length " + str(video_length) + " while vector is of length " + str(vector_length))
        print("\tThis may lead to off-by-" + str(video_length - vector_length) + " errors.")
        print("\tAre you trying to re-trim a clip? Careful!")
        print("\t WARNING ON " + filename)
        CAN_ANNOTATE = False


    deletion_log = []
    export_annotated_frame(CAN_ANNOTATE, 0, X_all[0], Y_all[0], X_raw[0], LABEL_TYPE_FIRST, cap, filename)


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

        if label_pa == 'away-from-table' and label_pb == 'away-from-table':
            deletion_log.append(rid)
            if export_frames and chance < .001:
                export_annotated_frame(CAN_ANNOTATE, frame_num, row_X, row_Y, raw_X, LABEL_TYPE_DELETED, cap, filename)
                counter += 1


        # if export_frames and label_pa == 'away-from-table' and pose_pb is not None and chance < .01:
        #     export_annotated_frame(frame_num, row_X, row_Y, LABEL_TYPE_AWAY_BUT_POSE_DETECTED, cap, frame_group)
        #     counter += 1

        # if export_frames and label_pb == 'away-from-table' and pose_pb is not None and chance < .01:
        #     export_annotated_frame(frame_num, row_X, row_Y, LABEL_TYPE_AWAY_BUT_POSE_DETECTED, cap, frame_group)
        #     counter += 1

        if export_frames and chance < .00005:
            export_annotated_frame(CAN_ANNOTATE, frame_num, row_X, row_Y, raw_X, LABEL_TYPE_RANDOM, cap, filename)
            counter += 1

        if export_frames and chance < .0001:
            if get_label_PA(row_Y) != -1 or get_label_PB != -1:
                print("Exporting flipped frame")
                export_flipped_frame(CAN_ANNOTATE, frame_num, row_X, row_Y, raw_X, LABEL_TYPE_FLIPPED, cap, filename)
                export_annotated_frame(CAN_ANNOTATE, frame_num, row_X, row_Y, raw_X, LABEL_TYPE_FLIPPED + '0', cap, filename)
                counter += 1


    # print(frame_num)
    # Final removal of incorrect away-from-table-s

    print("\tDeletion log contains " + str(len(deletion_log)) + " items")
    X_final = np.delete(X_all, deletion_log, axis=0)
    Y_final = np.delete(Y_all, deletion_log, axis=0)

    print("\tPost-trim shape: " + str(Y_final.shape))

    qchecks.verify_io_expanded(X_final, Y_final)

    # filehandler = open(prefix_qc + "QC_" + filename_root + "_X.p", "wb")
    # json.dump(X_final, filehandler)
    # filehandler.close()

    filehandler = open(prefix_vectors_out + "trimmed_" + filename + "_X.p", "wb")
    pickle.dump(X_final, filehandler)
    filehandler.close()

    filehandler = open(prefix_vectors_out + "trimmed_" + filename + "_Y.p", "wb")
    pickle.dump(Y_final, filehandler)
    filehandler.close()

    print("Exported trimmed final clip for " + filename)
    print("\n")


for filename in filenames_all:
    check_quality_and_export_trimmed(filename, export_frames=True)
