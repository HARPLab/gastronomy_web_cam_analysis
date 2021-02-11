from feature_utils import *
import cv2
from RestaurantFrame import RestaurantFrame
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}

# Lookup table for OpenPose keypoint indices
keypoint_labels = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
                                                "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
                                                "LEar","LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background", '']

FLAG_EXPORT_OUTLIER_SAMPLES = False
FLAG_PER_ACTIVITY_ANALYSIS = False
FLAG_DISPLAY_HOTSPOTS = False
FLAG_COVERAGE_DATA = False
FLAG_NUMBER_PEOPLE_DATA = False
FLAG_SVM =False

ROLES_BY_BUCKET = 0
ROLES_BY_PATH = 1
FLAG_ROLE_ASSIGNMENT = ROLES_BY_BUCKET

FEATURES_SET_PA = 0
FEATURES_SET_PB = 1
FEATURES_SET_BOTH = 2


FEATURES_TYPE_POSES_RAW = 10
FEATURES_TYPE_POSES_ROLES = 11
FEATURES_TYPE_ALSO_VEL = 12


FLAG_FEATURE_TYPE = FEATURES_TYPE_POSES_RAW
FLAG_FEATURE_SET = FEATURES_SET_PA


filename_root = "all_data"
full_import_name = filename_root + "_cropped.mp4"

pickle_name = "all_data.pickle" #filename_root + ".pickle"
cap = cv2.VideoCapture("./" + full_import_name)

nexuses = [(170, 146), (329, 221)]

nexus_PA = (170, 146)
nexus_PB = (329, 221)
nexus_waiter = (nexus_PA + nexus_PB)
nexus_waiter = (nexus_waiter[0] / 2.0, nexus_waiter[1] / 2.0)

confidence_threshold = .8
NULL_POSE = [(0.0, 0.0, 0.0)]
NULL_POSE = NULL_POSE * 25


print("Looking for previously pickled file")
filepath_features = 'features_' + filename_root + '.txt'
db = {}
try:  
    db = pickle.load(open("13-17-18-21_data.pickle", "rb"))
    print("Successfully imported pickle")
except (OSError, IOError) as e:
    timeline = []
    files = ["features_8-13-18.txt", "features_8-21-18.txt", "features_8-17-18.txt", "features_8-18-18.txt", "features_8-21-18.txt"] 
    #sift_files = ["8-13-18_sift_features.txt", "8-21-18_sift_features.txt"]#"8-17-18_sift_features.txt", "8-18-18_sift_features.txt", "8-21-18_sift_features.txt"]
    for filepath_features in files:
        with open("../feature_data/" + filepath_features) as fp:
            print("Generating import file from scratch")
            print("Now with obj defs")
            input_content = fp.read()
            frames = input_content.split("Body")

            print("Number of frames:")
            print(len(frames))
            frame_counter = 0
            for i in range(len(frames)):
                frame = frames[i]
                if len(frame) > 0:
                    frame_counter = len(timeline)
                    frame_obj = RestaurantFrame(frame_counter, frame)
                    frame_index = frame_obj.get_frame_number()
                    timeline.append(frame_obj)
    db['timeline'] = timeline
    db['filename'] = filename_root
    dbfile = open("13-17-18-21_data.pickle", "ab")
    pickle.dump(db, dbfile)                                   
    dbfile.close() 
#8-9-18
"""
        with open("../feature_data/" + filepath_features) as fp:
                        print("Generating import file from scratch")
                        print("Now with obj defs")
                        input_content = fp.read()

                        frames = input_content.split("Body")

                        print("Number of frames:")
                        print(len(frames))

                        frame_counter = 0
                        for frame in frames:
                                if len(frame) > 0:
                                        frame_counter = len(timeline)
                                        frame_obj = RestaurantFrame(frame_counter, frame)
                                        frame_index = frame_obj.get_frame_number()
                                        timeline.append(frame_obj)
        db['timeline'] = timeline
        db['filename'] = filename_root
        dbfile = open("8-21-18_data" + ".pickle", 'ab')
        pickle.dump(db, dbfile)                                   
        dbfile.close() 
"""

timeline = db['timeline']
analytics_db = {}

print("Total frames")
total_frames = len(timeline)
analytics_db['total_frames'] = total_frames
print(total_frames)

processed_timeline = []
# POSTPROCESSING
prev_frame = timeline[0]

# prev_frame['pa'] = {}
# prev_frame['pb'] = {}
# prev_frame['waiters'] = {}
# prev_frame['trash_poses'] = {}
 
# dictionary of label -> pose
role_locations = []
person_id = 0

total_poses = 0
total_poses_clean = 0
pose_dist = {}
for i in range(25):
        pose_dist[i] = 0

com_dict = defaultdict(int)
all_com = []

num_person_log = []
export_counter = 0

for frame_index in range(len(timeline)):

        # Copy of a frame object
        new_frame = timeline[frame_index]
        
        # identify people in frame
        # initially, no guests in frame
        # note we have full continuity for the video
        person_poses = new_frame.poses_arrays_raw
        # frame_movement_tolerance = 3
        frame_num = new_frame.get_frame_number()

        cleaned_poses = []

        # ROLES: PA, PB, WAITER, BACKGROUND
        for person_pose in person_poses:
                total_poses += 1

                for i in range(len(person_pose)):
                        p = person_pose[i]
                        value = not (p == (0.0, 0.0, 0.0))
                        if value:
                                pose_dist[i] += 1
                                # confidence value
                                # pose_dist[i] += p[2]

                # center of mass point
                pt_com = get_main_pt(person_pose)
                com_x, com_y, com_conf = pt_com
                com_xy = (com_x, com_y)
                all_com.append(com_xy)

                is_clean_point = True
                if com_conf < confidence_threshold:
                        is_clean_point = False

                # if not com_near_nexus(com_xy, nexuses, 50):
                #       is_clean_point = False
                
                if is_clean_point:
                        cleaned_poses.append(person_pose) 

        if len(cleaned_poses) == 0:
                continue
        #new_frame.poses_arrays_cleaned = cleaned_poses
        new_frame.num_poses_cleaned = len(cleaned_poses)
        total_poses_clean += len(cleaned_poses)
        pa, pb, waiter = get_role_labels(cleaned_poses)
        new_frame.set_roles(pa, pb, waiter)
        
        num_person_log.append(new_frame.get_num_poses_clean())


        processed_timeline.append(new_frame)
        prev_frame = new_frame

print("Total poses = " + str(total_poses))
print("Cleaned poses = " + str(total_poses_clean))
dbfile = open("13-17-18-21_data_processed.pickle", 'ab')
pickle.dump(processed_timeline, dbfile)
dbfile.close()


# visualization code for weird stuff
if FLAG_EXPORT_OUTLIER_SAMPLES:
        print("Exporting strange samples")
        export_counter = 0
        for frame in processed_timeline:
                person_poses = frame.get_poses_clean()
                label_a = frame.get_label_PA()
                label_b = frame.get_label_PB()
                pose_a  = frame.get_PA()
                pose_b  = frame.get_PB()

                if label_a is 'away-from-table' and pose_a is not None:
                    export_annotated_frame(frame, LABEL_TYPE_AWAY_BUT_POSE_DETECTED)

                if label_b is 'away-from-table' and pose_b is not None:
                    export_annotated_frame(frame, LABEL_TYPE_AWAY_BUT_POSE_DETECTED)

                if EXPORT_TYPE_TOO_MANY_POSES and len(person_poses) > 3 and export_counter < 20:

                        export_counter += 1

                        frame_num = frame.get_frame_number()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame_img = cap.read()
                        for pose in person_poses:
                                pt1 = get_main_pt(pose)
                                pt2 = get_secondary_pt(pose)
                                x1, y1, c1 = pt1
                                x1 = int(x1)
                                y1 = int(y1)

                                x2, y2, c2 = pt2
                                x2 = int(x2)
                                y2 = int(y2)

                                # print("c=" + str(c1) + " and c=" + str(c2))

                                frame_img = cv2.circle(frame_img, (x1,y1), 3, (255, 0, 255), -1)
                                frame_img = cv2.circle(frame_img, (x2,y2), 3, (0, 255, 255), -1)

                                for nex in nexuses:
                                        x, y = nex
                                        frame_img = cv2.circle(frame_img, (x,y), 5, (255, 255, 0), -1)

                        # cv2.imshow('Frame', frame)
                        title = "shows_" + str(len(person_poses)) + "_f" + str(frame_num) + ".jpg"
                        cv2.imwrite('./debug_output/' + title, frame_img) 
                        print("Exported outlier " + title)

if FLAG_NUMBER_PEOPLE_DATA:

        all_0 = list(filter(lambda frame: frame.get_num_poses_clean() == 0, timeline))
        all_1 = list(filter(lambda frame: frame.get_num_poses_clean() == 1, timeline))
        all_2 = list(filter(lambda frame: frame.get_num_poses_clean() == 2, timeline))
        all_3 = list(filter(lambda frame: frame.get_num_poses_clean() == 3, timeline))
        all_4 = list(filter(lambda frame: frame.get_num_poses_clean() == 4, timeline))
        all_5 = list(filter(lambda frame: frame.get_num_poses_clean() == 5, timeline))

        num_0 = len(all_0)
        num_1 = len(all_1)
        num_2 = len(all_2)
        num_3 = len(all_3)
        num_4 = len(all_4)
        num_5 = len(all_5)

        perc_0 = float(1.0*num_0 / total_frames)
        perc_1 = float(1.0*num_1 / total_frames)
        perc_2 = float(1.0*num_2 / total_frames)
        perc_3 = float(1.0*num_3 / total_frames)
        perc_4 = float(1.0*num_4 / total_frames)
        perc_5 = float(1.0*num_5 / total_frames)

        print("Frames with 0: " + str(num_0) + " : " + str(perc_0))
        print("Frames with 1: " + str(num_1) + " : " + str(perc_1))
        print("Frames with 2: " + str(num_2) + " : " + str(perc_2))
        print("Frames with 3: " + str(num_3) + " : " + str(perc_3))
        print("Frames with 4: " + str(num_4) + " : " + str(perc_4))
        print("Frames with 5: " + str(num_5) + " : " + str(perc_5))
        t = range(len(timeline))
        y = np.zeros(len(t))

        fig, ax = plt.subplots()
        plt.scatter(t, num_person_log)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Number of people in the scene')
        ax.set_title('Number of people in scene over time')

        plt.tight_layout()
        

if FLAG_COVERAGE_DATA:
        print("Coverage Data")
        for key in pose_dist.keys():
                percent = pose_dist[key] / total_poses
                print("Coverage of " + str(percent) + " for " + keypoint_labels[int(key)])


if FLAG_DISPLAY_HOTSPOTS:
        # libraries
        import numpy as np
        import seaborn as sns
        import matplotlib.pylab as plt
        import pandas as pd

        x = []
        y = []
        for i in range(len(all_com)):
                val = all_com[i]
                x.append(val[0])
                y.append(val[1])

        x = np.array(x)
        y = np.array(y)

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')

if FLAG_PER_ACTIVITY_ANALYSIS:
        # View timeline
        same_activity = list(filter(lambda frame: frame.get_label_PA() == frame.get_label_PB(), timeline))
        percent_same_activity = len(same_activity) / total_frames
        print("Percentage of times people are performing the same activity: " + str(percent_same_activity))

        print("PER ACTIVITY ANALYSIS for person A")
        # Find average values per activity
        for activity in activitydict.keys():
                # Find averages for when pa = activity
                instances_activity_pa = list(filter(lambda frame: frame.get_label_PA() == activity, timeline))
                instances_activity_pb = list(filter(lambda frame: frame.get_label_PB() == activity, timeline))
                all_instances_activity = instances_activity_pa + instances_activity_pb

                # print(activity)
                activity_num_frames = len(all_instances_activity)


                if activity_num_frames == 0:
                        activity_num_frames = 1

                num_people_sum = 0
                for instance in all_instances_activity:
                        num_people_sum += int(instance.get_num_poses_clean())

                num_people_avg = num_people_sum / activity_num_frames

                print("Average people in scenes with activity " + str(activity) + ": " + str(num_people_avg))
                percent_total = (activity_num_frames / (total_frames * 2)) * 100
                print("Percentage of " + activity + "/total:" + "{:.2f}".format(percent_total) + "%")

print("Done with all analysis")
print("Start with classification")

#print(np.array(timeline[10000].get_poses_clean()[0][0])[:,0:2])                                                                                                                  
