# Activity Key #####
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

import cv2
import random
from collections import defaultdict
# from OPwrapper import OP
import json
import pickle
import copy
import numpy as np
import itertools
from collections import defaultdict 
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix

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


filename_root = "8-21-18"
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
# print(NULL_POSE)

# ~two highest density areas based on heatmapping of noses


class RestaurantFrames:
	poses_arrays_raw = []
	num_poses_raw = 0
	pa_label = 'NONE'
	pa_label = 'NONE'	
	frame_number = -1

	poses_arrays_cleaned = []
	num_poses_cleaned = 0

	pose_PA = None
	pose_PB = None
	# waiter is always just ONE of the waiters
	pose_waiter = None

	delta_PA = None
	delta_PB = None
	prev = None

	max_detected = 9

	def set_PA(self, pose):
		#TODO handle None case nicely
		self.pose_PA = pose

	def set_PB(self, pose):
		self.pose_PB = pose

	def set_waiter(self, pose):
		self.pose_waiter = pose

	def get_PA(self):
		return self.pose_PA

	def get_PB(self):
		return self.pose_PB

	def get_waiter(self):
		return self.pose_waiter

	def set_previous_processed_frame(self, frame):
		self.delta_PA = frame.get_PA() - self.get_PA()
		self.delta_PB = frame.get_PB() - self.get_PB()

		self.prev = frame

	def get_delta_PA(self):
		return self.delta_PA

	def get_delta_PB(self):
		return self.delta_PB

	def get_prev(self):
		return self.prev	

	def set_roles(self, pa, pb, w):
		self.set_PA(pa)
		self.set_PB(pb)
		self.set_waiter(w)	

	def set_PA(self, pa):
		self.pose_PA = pa

	def set_PB(self, pb):
		self.pose_PB = pb

	def set_waiter(self, w):
		self.pose_waiter = w


	# Extract restaurant data from text file
	def __init__(self, frame_num, s):
		self.frame_number = frame_num
		
		# 25 keyframes for body
		start = s.find(":") + len(":")
		end = s.find("LH:")
		poses = frame[start:end]
		# print(poses)
		# Overall pose info
		# list of people, with 25 3d points per person

		poses = poses[1:-1]
		pose_list = poses.split("[[")
		pose_list = pose_list[1:]

		processed_poses = []

		num_people = len(pose_list)
		num_points_total = 0

		for pose in pose_list:
			pose = pose.replace("]]", "]")
			pose = pose.rstrip(']')
			pose = pose.lstrip('[')


			pose_points = pose.split(']\n  ')
			pt_set = []
			for pt in pose_points:
				pt = pt.lstrip('[')
				pt = pt.replace('\n', '')
				pt = pt.replace(']', '')
				
				p = pt.split(" ")
				p = list(filter(None, p))
				(x,y,z) = p

				pt_set.append((float(x), float(y), float(z)))

			processed_poses.append([pt_set])
		#print(processed_poses)
		self.poses_arrays_raw = processed_poses
		self.num_poses_raw = num_people

		start = s.find("LH:") + len("LH:")
		end = s.find("RH:")
		lh = frame[start:end]
		# print(lh)
		# frame_obj['lh'] = float(lh)

		start = s.find("RH:") + len("RH:")
		end = s.find("Face:")
		rh = frame[start:end]
		# frame_obj['rh'] = float(rh)

		start = s.find("Face:") + len("Face:")
		end = s.find("PA:")
		face = frame[start:end]
		# frame_obj['face'] = float(face)

		start = s.find("PA:") + len("PA:")
		end = s.find(" PB:")
		pa = frame[start:end]
		self.pa_label = pa

		start = s.find("PB:") + len("PB:")
		pb = frame[start:].strip()
		self.pb_label = pb

	def get_poses_raw(self):
		return self.poses_arrays_raw

	def get_poses_clean(self):
		padded_array = []
		for i in range(self.max_detected):
			if len(self.poses_arrays_cleaned) > i:
				padded_array.append(self.poses_arrays_cleaned[i])
			else:
				padded_array.append(NULL_POSE)

		return padded_array

	def get_label_PA(self):
		return self.pa_label

	def get_label_PB(self):
		return self.pb_label

	def get_num_poses_raw(self):
		return self.num_poses_raw

	def get_num_poses_clean(self):
		return self.num_poses_cleaned

	def get_frame_number(self):
		return self.frame_number

"""
print("Looking for previously pickled file")
filepath_features = 'features_' + filename_root + '.txt'
db = {}
try:
	db = pickle.load(open("8-21-18_data.pickle", "rb"))
	print("Successfully imported pickle")

except (OSError, IOError) as e:
	timeline = []
	files = ["features_8-13-18.txt", "features_8-17-18.txt", "features_8-18-18.txt", "features_8-21-18.txt"] 
	for filepath_features in files:
		with open(filepath_features) as fp:
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
					frame_obj = RestaurantFram(frame_counter, frame)
					frame_index = frame_obj.get_frame_number()
					timeline.append(frame_obj)

	db['timeline'] = timeline
	db['filename'] = filename_root

	dbfile = open("13-17-18-21_data" + ".pickle", 'ab') 
	pickle.dump(db, dbfile)					  
	dbfile.close() 
	#8-9-18
	with open(filepath_features) as fp:
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


timeline = db['timeline']
analytics_db = {}

print("Total frames")
total_frames = len(timeline)
analytics_db['total_frames'] = total_frames
print(total_frames)

def get_main_pt(pt_set):
	# return the nose point
	#print(pt_set)
	return pt_set[0][1]

def get_secondary_pt(pt_set):
	# return the nose point
	return pt_set[0][8]

def com_near_nexus(com_xy, nexuses, tol):
	x1, y1 = com_xy
	for nexus in nexuses:
		x2, y2 = nexus
		dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
		if dist < tol:
			return True

	return False

def get_role_labels(cleaned_poses):
	assignments = [None, None, None]

	if FLAG_ROLE_ASSIGNMENT == ROLES_BY_BUCKET:
		role_nexuses = [nexus_PA, nexus_PB]
		if len(cleaned_poses) > len(role_nexuses):
			role_nexuses.append(nexus_waiter)


		# for each role
		for i in range(len(role_nexuses)):
			distances = []

			for pose in cleaned_poses:
				compare_pt = get_main_pt(pose)
				nex = (role_nexuses[i][0], role_nexuses[i][1], 1)

				dist = distance.euclidean(compare_pt, nex)
				distances.append(dist)

			if len(distances) > 0:
				distances_np = np.asarray(distances)
				assignment = distances_np.argmax()
			else:
				assignment = NULL_POSE

			assignments[i] = assignment #TODO change to cleaned_poses[assignment]

	else:
		print("ERROR IN ROLES")

	return assignments


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
		# 	is_clean_point = False
		
		if is_clean_point:
			cleaned_poses.append(person_pose) 


	new_frame.poses_arrays_cleaned = cleaned_poses
	new_frame.num_poses_cleaned = len(cleaned_poses)
	total_poses_clean += len(cleaned_poses)

	pa, pb, waiter = get_role_labels(cleaned_poses)
	new_frame.set_roles(pa, pb, waiter)

	num_person_log.append(new_frame.get_num_poses_clean())


	processed_timeline.append(new_frame)
	prev_frame = new_frame


print("Total poses = " + str(total_poses))
print("Cleaned poses = " + str(total_poses_clean))

# visualization code for weird stuff
if FLAG_EXPORT_OUTLIER_SAMPLES:
	print("Exporting strange samples")
	export_counter = 0
	for frame in processed_timeline:
		person_poses = frame.get_poses_clean()

		if len(person_poses) > 3 and export_counter < 20:

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
"""
def get_feature_vector(frame):
	feature_vector = []
	if FLAG_FEATURE_TYPE is FEATURES_TYPE_POSES_RAW:
		test = np.array(frame.get_poses_clean()[0][0])
		#print(test.shape)
		#raw = np.array(frame.get_poses_clean()[0][0])[:,0:2]
		if test.shape == (3,):
			return np.zeros(50), False
		raw = test[:, 0:2]	
		feature_vector.append(raw)

	elif FLAG_FEATURE_TYPE is FEATURES_TYPE_POSES_ROLES:
		if FLAG_FEATURE_SET is FEATURES_SET_PA:
			print(frame.get_PA())
			feature_vector.append(frame.get_PA())

		elif FLAG_FEATURE_SET is FEATURES_SET_PB:
			feature_vector.append(frame.get_PB())

		elif FLAG_FEATURE_SET is FEATURES_SET_BOTH:
			feature_vector.append(frame.get_PA())
			feature_vector.append(frame.get_PB())


	elif FLAG_FEATURE_TYPE is FEATURES_TYPE_ALSO_VEL:

		if FLAG_FEATURE_SET is FEATURES_SET_PA:
			feature_vector.append(frame.get_delta_PA())

		elif FLAG_FEATURE_SET is FEATURES_SET_PB:
			feature_vector.append(frame.get_label_PA())
		
		elif FLAG_FEATURE_SET is FEATURES_SET_PA:
			feature_vector.append(frame.get_label_PA())
			feature_vector.append(frame.get_label_PB())

	return np.asarray(feature_vector).flatten(), True

def get_labels_vector(frame):
	newY = []

	if FLAG_FEATURE_SET is FEATURES_SET_PA:
		newY.append(frame.get_label_PA())
	elif FLAG_FEATURE_SET is FEATURES_SET_PB:
		newY.append(frame.get_label_PA())
	elif FLAG_FEATURE_SET is FEATURES_SET_PA:
		newY.append(frame.get_label_PA())
		newY.append(frame.get_label_PB())

	return newY


def frame_to_vectors(frame):
	newX = []
	newY = []

	newY = get_labels_vector(frame)
	feature_vector, ret = get_feature_vector(frame)
	# print(feature_vector.shape)
	if not ret:
		return newX, newY, False
	#print(feature_vector.shape)
	for ys in newY:
		newX.append(feature_vector)
	newY = activitydict[newY[0]]
	#print(newY) 
	return newX, newY, True	
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))
    plt.yticks(tick_marks)
    cm = np.around(cm, decimals=1)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f').replace("0","",1),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn import svm
import random
# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import pickle
import copy
import numpy as np
import keras
class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            #clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig("LSTM_Losses70epochs.png")
            plt.close()

#fid evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 70, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print("fiting model....")
	plot_losses = TrainingPlot()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(testX, testy), callbacks=[plot_losses])
	try:
		pickle.dump(model, open("LSTM_4train-1test-1epochs.p", "wb"))
	except:
		print("error wi†h dumping model :(((((")
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
        print(scores)
        m, s = mean(scores), std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(trainX, trainy, testX, testy, repeats=1):
        # repeat experiment
        scores = list()
        for r in range(repeats):
                score = evaluate_model(trainX, trainy, testX, testy)
                score = score * 100.0
                print('>#%d: %.3f' % (r+1, score))
                scores.append(score)
        # summarize results
        summarize_results(scores)
# run the experiment
if not FLAG_SVM:
	print("training LSTM")
	
	X_train = []
	Y_train = []

	X_test = []
	Y_test = []

	#shuffled_list = copy.copy(timeline)

	shuffled_list = pickle.load(open("13-17-18-21_list.p","rb")) #copy.copy(timeline)
	#pickle.dump(shuffled_list, open("8-21-18_data_list.p", "wb"))
	test_list = pickle.load(open("8-21-18_data_list.p", "rb"))
	print("loaded pickle datasets")
	index_reduc = int(len(shuffled_list) * (0.4))
	shuffled_list = shuffled_list[:index_reduc]
	#percent_test = .2
	#index_split = int(len(shuffled_list) * (1.0 - percent_test))
	#train = shuffled_list[:index_split]
	#test = shuffled_list[index_split:]
	train = shuffled_list
	index_reduc = int(len(test_list) * (0.4))
	test = test_list[:index_reduc]
	for frame in train:
		newX, newY, ret = frame_to_vectors(frame)
		if not ret:
			continue
		X_train.extend(newX)
		Y_train.append(newY)
	for frame in test:
		newX, newY, ret = frame_to_vectors(frame)	
		if not ret:
			continue
		X_test.extend(newX)
		Y_test.append(newY)
	print("slicing...")
	window_size=128
	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)
	X_test = np.asarray(X_test)
	Y_test = np.asarray(Y_test)
	X_train_sliced = np.zeros((X_train.shape[0]-window_size, window_size, X_train.shape[1]))
	Y_train_sliced = np.zeros((Y_train.shape[0]-window_size, 1))
	X_test_sliced = np.zeros((X_test.shape[0]-window_size, window_size, X_test.shape[1]))
	Y_test_sliced = np.zeros((Y_test.shape[0]-window_size, 1))

	for idx in range(window_size, X_train.shape[0]):
		X_train_sliced[idx-window_size,:,:] = X_train[idx-window_size:idx]
	for idx in range(window_size, Y_train.shape[0]):
		Y_train_sliced[idx-window_size,0] = Y_train[idx]
	for idx in range(window_size, X_test.shape[0]):
		X_test_sliced[idx-window_size,:] = X_test[idx-window_size:idx]
	for idx in range(window_size, Y_test.shape[0]):
		Y_test_sliced[idx-window_size,0] = Y_test[idx]
	Y_train_sliced = to_categorical(Y_train_sliced)
	Y_test_sliced = to_categorical(Y_test_sliced)
	print(X_train_sliced.shape, Y_train_sliced.shape, X_test_sliced.shape, Y_test_sliced.shape)
	#pickle.dump(X_train_sliced, open("X_train_sliced.p", "wb"), protocol=2)
	#pickle.dump(X_test_sliced, open("X_test_sliced.p", "wb"), protocol=2)
	#pickle.dump(Y_train_sliced, open("Y_train_sliced.p", "wb"), protocol=2)
	#pickle.dump(Y_test_sliced, open("Y_test_sliced.p", "wb"), protocol=2)
	try:
		model = pickle.load(open("LSTMblah_4train-1test-1epochs.p", "rb"))
		#pred_Y = model.predict(X_test_sliced)
		#pickle.dump(pred_Y, open("pred_Y_LSTM70epochs.p", "wb"))
		pred_Y = pickle.load(open("pred_Y_LSTM70epochs.p", "rb"))
		decodedpredY = pred_Y.argmax(axis=1)
		decodedtestY = Y_test_sliced.argmax(axis=1)
		frequencytest = {}
		frequencypred = {}
		for num in decodedtestY:
			if num not in frequencytest.keys():
				frequencytest[num] = 1
			else:
				frequencytest[num] = frequencytest[num] + 1
		for num in decodedpredY:
                        if num not in frequencypred.keys():
                                frequencypred[num] = 1
                        else:
                                frequencypred[num] = frequencypred[num] + 1
		print("stats:")
		print(frequencytest)
		print(frequencypred)
		print(decodedpredY.shape)
		print(decodedtestY.shape)
		predPadding = []
		testPadding = []
		i = 0
		for key in activitydict.keys():
			print(key)
			predPadding.append(i)
			testPadding.append(23-i)
			i +=1
		decodedpredY = np.append(decodedpredY, predPadding)
		decodedtestY = np.append(decodedtestY, testPadding)
		print(decodedpredY.shape)
		print(decodedtestY.shape)
		cm = confusion_matrix(decodedtestY,decodedpredY)
		print("unormalized confusion matrix")
		np.set_printoptions(precision=2)
		fig, ax = plt.subplots()
		sum_of_rows = cm.sum(axis=1)
		cm = cm / (sum_of_rows[:, np.newaxis]+1e-8)
		print(cm)
		plot_confusion_matrix(cm,cmap=plt.cm.Blues)
		plt.savefig("LSTM4170epochs_confusion_mat.png")
	except:			
		print("error")
		run_experiment(X_train_sliced, Y_train_sliced, X_test_sliced, Y_test_sliced)

if FLAG_SVM:
	print("training SVM...")
	X_train = []
	Y_train = []

	X_test = []
	Y_test = []


	#shuffled_list = copy.copy(timeline)
	#random.shuffle(shuffled_list)
	shuffled_list = pickle.load(open("13-17-18-21_list.p","rb"))
	test_list = pickle.load(open("8-21-18_data_list.p", "rb"))
	#percent_test = .2
	#index_split = int(len(shuffled_list) * (1.0 - percent_test))
	index_reduc = int(len(shuffled_list) * (0.4))
	shuffled_list = shuffled_list[:index_reduc]

	#train = #shuffled_list[:index_split]
	#test = shuffled_list[index_split:]
	train = shuffled_list
	index_reduc = int(len(test_list) * (0.4))
	test = test_list[:index_reduc]
	for frame in train:
		newX, newY, ret = frame_to_vectors(frame)
		#print(newX)
		#print(newY)
		if not ret:
			continue
		X_train.extend(newX)
		Y_train.append(newY)
	for frame in test:
	        newX, newY, ret = frame_to_vectors(frame)
	        #print(newX)
	        #print(newY)
       		if not ret:
	                continue
	        X_test.extend(newX)
	        Y_test.append(newY)

	
	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)
	X_test = np.asarray(X_test)
	Y_test = np.asarray(Y_test)
	
	print("X_train")
	print(X_train.shape)
	print("Y_train")
	print(Y_train.shape)
	predicted = []
	correct = Y_test
	try:
		clf = pickle.load(open("svm41_trained.p", "rb"))
		print("Successfully imported pickle svm")
	except (OSError, IOError) as e:
		clf = svm.SVC()
		print("fitting...")
		clf.fit(X_train, Y_train)
		print("dumping...")
		pickle.dump(clf, open("svm41_trained.p", "wb"))
	
	print("Creating confusion matrix")
	# Plot non-normalized confusion matrix
	titles_options = [("Confusion matrix, without normalization", None),
	                  ("Normalized confusion matrix", 'true')]
	
	pred_y = clf.predict(X_test)
	pickle.dump(pred_y, open("svm_predy41.p", "wb"))
	print(pred_y.shape[0])
	print(correct.shape)
	count = 0
	for i in range(0, pred_y.shape[0]):
		if pred_y[i] == correct[i]:
			count += 1
	print("accuracy: " + str(count/pred_y.shape[0]))
	frequencytest = {}
	frequencypred = {}
	for num in pred_y:
		if num not in frequencytest.keys():
			frequencytest[num] = 1
		else:
			frequencytest[num] = frequencytest[num] + 1
	for num in correct:
		if num not in frequencypred.keys():
			frequencypred[num] = 1
		else:
			frequencypred[num] = frequencypred[num] + 1
	print("stats:")
	print(frequencytest)
	print(frequencypred)
	print(pred_y.shape)
	print(correct.shape)
	predPadding = []
	testPadding = []
	i = 0
	for key in activitydict.keys():
		print(key)
		predPadding.append(i)
		testPadding.append(23-i)
		i +=1
	pred_y = np.append(pred_y, predPadding)
	Y_test = np.append(correct, testPadding)
	#for title, normalize in titles_options:
	cm = confusion_matrix(Y_test, pred_y)
	np.set_printoptions(precision=2)
	print('Confusion matrix, without normalization')
	print(cm)
	fig, ax = plt.subplots()
	sum_of_rows = cm.sum(axis=1)
	cm = cm / sum_of_rows[:, np.newaxis]
	plot_confusion_matrix(cm,cmap=plt.cm.Blues)
	#plt.show()
	plt.savefig("confusion_mat.png")
	
	
print("Loaded feature utils")

