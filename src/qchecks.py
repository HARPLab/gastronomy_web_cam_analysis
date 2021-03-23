import time
import cv2
import random
import numpy as np
import imageio

# import sys
# sys.path.append(".")
import arconsts
from random import randrange

activity_labels 		= arconsts.activity_labels

CLASSIFIERS_TEMPORAL 	= arconsts.CLASSIFIERS_TEMPORAL
CLASSIFIERS_STATELESS 	= arconsts.CLASSIFIERS_STATELESS

CONST_NUM_POINTS 	= arconsts.CONST_NUM_POINTS
CONST_NUM_SUBPOINTS = arconsts.CONST_NUM_SUBPOINTS
CONST_NUM_LABEL 	= arconsts.CONST_NUM_LABEL

def add_pose_to_image(pose, img, color):
	for p in pose:
		x1, y1, c1 = p[0], p[1], p[2]
		x1 = int(x1)
		y1 = int(y1)
		frame_img = cv2.circle(img, (x1,y1), 3, color, -1)

	return frame_img

def get_frame_visualization(poses, input_labels, output_label, predicted_label, label_input, label_output):
	COLOR_NEUTRAL = (255, 255, 255)
	COLOR_A = (255, 0, 0)
	COLOR_B = (0, 0, 255)

	CONST_IMG_WIDTH = 400
	CONST_IMG_WIDTH = 450

	label_true_a, label_true_b = "-", "-"
	label_pred_a, label_pred_b = "-", "-"
	pose_a, pose_b = [], []

	FRAME_WIDTH = 400
	FRAME_HEIGHT = 450
	frame_img = np.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3), np.uint8)

	label_overall = label_input + " -> " + label_output

	# draw on the bounding boxes
	bd_box_A = ((70, 80), (200, 340))
	bd_box_B = ((230, 130), (370, 370))
	frame_img = cv2.rectangle(frame_img, bd_box_A[0], bd_box_A[1], COLOR_A, 1)
	frame_img = cv2.rectangle(frame_img, bd_box_B[0], bd_box_B[1], COLOR_B, 1)

	num_poses = int(poses.shape[0] / (CONST_NUM_POINTS * CONST_NUM_SUBPOINTS))
	poses = np.array(poses).reshape((num_poses, CONST_NUM_POINTS, CONST_NUM_SUBPOINTS))

	for pose in poses:
		if pose.shape[0] > 0:
			pose = np.array(pose).reshape((CONST_NUM_POINTS, CONST_NUM_SUBPOINTS))
			frame_img = add_pose_to_image(pose, frame_img, COLOR_NEUTRAL)
		
	# frame_img = add_pose_to_image(pose_a, frame_img, COLOR_A)
	# frame_img = add_pose_to_image(pose_b, frame_img, COLOR_B)

	halfway = int(FRAME_WIDTH / 2)

	org_a = (50, 25) 
	org_b = (45 + halfway, 25) 

	org_a2 = (50, 50) 
	org_b2 = (45 + halfway, 50) 

	font = cv2.FONT_HERSHEY_SIMPLEX 
	fontScale = 2#.6
	color = (255, 0, 0) 
	thickness = 2

	font = cv2.FONT_HERSHEY_SIMPLEX 
	org = (00, 185) 
	fontScale = .6

	output_label = int(output_label[0])

	if label_output == "a":
		label_true_a = activity_labels[output_label]
		label_pred_a = activity_labels[predicted_label]
	
		frame_img = cv2.putText(frame_img, "true: " + label_true_a, org_a, font, fontScale, color, thickness, cv2.LINE_AA)
		frame_img = cv2.putText(frame_img, "pred: " + label_pred_a, org_a2, font, fontScale, COLOR_A, thickness, cv2.LINE_AA) 
		# frame_img = cv2.putText(frame_img, "pred: " + label_pred_a, org_a2, font, fontScale, COLOR_A, thickness, cv2.LINE_AA) 

		input_labels = int(input_labels)

		try:
			addtl_b = activity_labels[input_labels]
			frame_img = cv2.putText(frame_img, "+data: " + addtl_b, org_b, font, fontScale, COLOR_NEUTRAL, thickness, cv2.LINE_AA, False)		
		except IndexError:
			pass
		
	elif label_output == 'b':
		label_true_b = activity_labels[output_label]
		label_pred_b = activity_labels[predicted_label]

		frame_img = cv2.putText(frame_img, "true: " + label_true_b, org_b, font, fontScale, COLOR_B, thickness, cv2.LINE_AA)
		frame_img = cv2.putText(frame_img, "pred: " + label_pred_b, org_b2, font, fontScale, COLOR_B, thickness, cv2.LINE_AA)

		try:
			if input_labels.shape[1] > 0:
				addtl_a = activity_labels[input_labels[0]]
				frame_img = cv2.putText(frame_img, "+data: " + addtl_a, org_a, font, fontScale, COLOR_NEUTRAL, thickness, cv2.LINE_AA, False)
		except IndexError:
			pass

	return frame_img


def export_gif_of(poses, input_labels, output_label, predicted_label, assessment_label, lookup_index, where):
	# What is the experiment being done?
	label = assessment_label[1:]
	labels = label.split("_")
	label_input, label_output = labels[0], labels[1]

	window_size = poses.shape[0]
	
	frames = []
	for fi in range(window_size):
		new_img = get_frame_visualization(poses[fi], input_labels[fi], output_label, predicted_label, label_input, label_output)
		frames.append(new_img)

	# TODO save gif to disk
	export_loco = where + assessment_label + '_at_' + str(lookup_index) + ".gif"
	imageio.mimsave(export_loco, frames, fps=30)
	print("Exported gif of results at timestep " + str(lookup_index))
	return

# def verify_input_output(X, Y):
#     # print(X.shape)
#     # print(Y.shape)
#     # print("Unique values: ")
#     unique_values = np.unique(Y)
#     if(all(x in range(len(activitydict.keys())) for x in unique_values)): 
#         # print("All good")
#         pass
#     else:
#         print("Nope- Y contains more than the valid labels")
#         np.set_printoptions(threshold=np.inf)
#         np.set_printoptions(suppress=True)
#         print(unique_values)
#         np.set_printoptions(threshold=15)
#         np.set_printoptions(suppress=False)
#         exit()

def verify_integrity_array_entry(pose1, pose2):
	if pose1.size != pose2.size:
		print("Different size poses")
		exit()

	dim = int(pose1.size)

	pose1 = pose1.reshape(dim)
	pose2 = pose2.reshape(dim)

	if not (pose1==pose2).all():
		print("Pose got deformed")
		print(pose1)
		print(pose2)
		exit()





def verify_pose(pose):
	if not isinstance(pose, np.ndarray):
		print("Pose is not an ndarray!")
		print(pose)
		exit()

	if pose.shape[0] != 25:
		print("Pose with less than 25 keypoints")
		print(pose)
		exit()


def verify_Y_valid(Y):
	unique_values = np.unique(Y)
	if(all(x in range(len(arconsts.activity_labels)) for x in unique_values)): 
		pass
	else:
		print("Y contains labels outside the correct set: verify_Y_valid")
		print(unique_values)
		exit()

def verify_io_expanded(X, Y):
	unique_values = np.unique(Y)
	if(all(x in range(len(arconsts.activity_labels_expanded)) for x in unique_values)): 
		pass
	# elif(all(x in range(len(arconsts.activity_labels_expanded)) for x in unique_values)): 
	# 	pass
	else:
		print("Y contains labels outside the correct set: verify_io_expanded")
		print(unique_values)
		exit()

def verify_input_output(X, Y, classifier_type):
	unique_values = np.unique(Y)
	good_zone = range(-1, len(arconsts.activity_labels))
	
	if(all(x in good_zone for x in unique_values)): 
		pass
	else:
		print("Y contains labels outside the correct set: verify input output")
		print(unique_values)
		exit()

# align with experimental analysis
def multiclass_to_labels(result):
	if len(result.shape) == 1:
		return result

	if len(result.shape) == 2 and result.shape[1] == 1:
		return result

	if result.shape[1] == len(activity_labels):
		decoded_result = result.argmax(axis=1)
		verify_Y_valid(decoded_result)
		return decoded_result
	
	verify_Y_valid(result)
	return result

def quality_check_output(X, Y, Y_pred, classifier_type, assessment_label, where, num_inspections = 2):
	verify_input_output(X, Y, classifier_type)
	verify_input_output(X, Y_pred, classifier_type)

	dim_X = X.shape
	dim_Y = Y.shape
	n_timesteps_X = dim_X[0]
	n_timesteps_Y = dim_Y[0]

	print(X.shape)
	print(Y.shape)
	print(Y_pred.shape)

	if classifier_type in CLASSIFIERS_TEMPORAL:
		n_window = dim_X[1]
		n_features = dim_X[2]
	elif classifier_type in CLASSIFIERS_STATELESS:
		n_window = 1
		n_features = dim_X[1]

	if n_timesteps_X != n_timesteps_Y:
		print("Misaligned number of samples in ")

	len_pose_block = CONST_NUM_POINTS*CONST_NUM_SUBPOINTS
	for i in range(num_inspections):
		lookup_index = randrange(n_timesteps_X)
		# print(lookup_index)
		X_i = X[lookup_index]
		Y_i = Y[lookup_index]

		pose_size 		= CONST_NUM_POINTS*CONST_NUM_SUBPOINTS
		label_size 		= CONST_NUM_LABEL
		poses 			= np.zeros((n_window,))
		input_labels 	= np.zeros((n_window,))
		
		# Only implementing for our primary use case,
		# the temporal classifier
		if classifier_type in CLASSIFIERS_TEMPORAL:
			if n_features == pose_size:
				poses = X_i[:, : pose_size]
			elif n_features == pose_size + label_size:
				poses = X_i[:, :pose_size]
				input_labels = X_i[:, -1]
				# input_labels.append(X_i[:, pose_size:])
			elif n_features == 2*pose_size:
				poses = X_i[:, :2*pose_size]
			else:
				print("Unrecognized dimensions of X")
				exit()

		output_label = Y[lookup_index]
		predicted_label = Y_pred[lookup_index]
		export_gif_of(poses, input_labels, output_label, predicted_label, assessment_label, lookup_index, where)


def export_qual_confusion_matrix(y1, y2, exp_batch_id, classifier_type, subexp_label, fold_id):
	save_location = "results/" + exp_batch_id + "f" + str(fold_id) + "_" + classifier_type[1:] + "_" + subexp_label + "_f" + str(fold_id)

	# print(y1)
	# print(y2)

	cm = confusion_matrix(y1.astype(int), y2.astype(int), labels=range(len(activity_labels)))

	sn.set(font_scale=2)
	sn.set_style("white",  {'figure.facecolor': 'black'})
	corr = cm
	mask = np.zeros_like(corr)
	mask[corr == 0] = True
	ax = plt.axes()
	fig = sn.heatmap(corr, cmap='Greys', mask=mask, square=True, annot=True, cbar=False, fmt='g', annot_kws={"size": 15}, ax=ax)
	ax.set_xticklabels(activity_labels, rotation=45)
	ax.set_yticklabels(activity_labels, rotation=0)
	ax.set(ylabel="Person 1", xlabel="Person 2")
	ax.set_title('Confusion Matrix for ' + classifier_type + " on " + subexp_label)
	plt.tight_layout()
	fig.get_figure().savefig(save_location + '_cm.png')
	plt.close()

	# Export stacked bar chart
	labels = activity_labels
	correct_labels = []
	incorrect_labels = []
	percent_labels = []
	width = 0.8	   # the width of the bars: can also be len(x) sequence

	for idx, cls in enumerate(activity_labels):
		# True negatives are all the samples that are not our current GT class (not the current row) 
		# and were not predicted as the current class (not the current column)
		true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
		
		# True positives are all the samples of our current GT class that were predicted as such
		true_positives = cm[idx, idx]

		correct 	= true_positives
		incorrect 	= np.sum(cm[:,idx]) - correct
		percent = (correct / (correct + incorrect))*100.0
		percent = "{:.4}".format(percent) + "%"
		
		# The accuracy for the current class is ratio between correct predictions to all predictions
		# per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)

		correct_labels.append(correct)
		incorrect_labels.append(incorrect)
		percent_labels.append(percent)


	fig, ax = plt.subplots()
	le_font_size = 10.0

	ax.bar(labels, correct_labels, width, align="center", label='Correct Labels')
	ax.bar(labels, incorrect_labels, width, align="center", bottom=correct_labels,
		   label='Incorrect Labels')

	ax.set_ylabel('Number of Samples', fontsize=le_font_size)
	ax.set_ylabel('Predicted Label', fontsize=le_font_size)
	ax.set_xticklabels(activity_labels, rotation=90, fontsize=le_font_size)
	ax.yaxis.set_tick_params(labelsize=le_font_size)

	ax.set_title('Per-Class Classification Accuracy for ' + subexp_label, fontsize=le_font_size)
	ax.legend(fontsize=le_font_size)

	# for p in ax.patches:
	# 	width, height = p.get_width(), p.get_height()
	# 	x, y = p.get_xy() 
	# 	ax.text(x+width/2, 
	# 			y+height/2, 
	# 			'{:.0f} %'.format(height), 
	# 			horizontalalignment='center', 
	# 			verticalalignment='center')

	# set individual bar lables using above list
	counter = 0
	for i in ax.patches:
		if counter % 2 == 0 and int(counter / 2) < len(percent_labels):
			lookup_index = int(counter / 2)
			label = percent_labels[lookup_index]
			counter += 1
			# get_x pulls left or right; get_height pushes up or down
			ax.text(i.get_x()+.12, i.get_height()+(60*le_font_size), label, fontsize=(le_font_size),
					color='black', rotation=90)
		counter += 1


	# for i in range(len(correct_labels)): 
	#	 label = percent_labels[i]
	#	 plt.annotate(label, xy=(i, 0), color='white')

	plt.tight_layout()
	plt.savefig(save_location + '_graphs.png')
	plt.close()
