import time
import cv2
import random
import numpy as np

activity_labels = ['NONE', 'away-from-table', 'idle', 'eating', 'talking', 'talk:waiter', 'looking:window', 
					'reading:bill', 'reading:menu', 'paying:check', 'using:phone', 'obj:wildcard', 'standing']

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

	label_true_a, label_true_b = "", ""
	label_pred_a, label_pred_b = "", ""
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

	org_a = (50, 0) 
	org_b = (45 + halfway, 0) 

	org_a2 = (50, 50) 
	org_b2 = (45 + halfway, 50) 

	font = cv2.FONT_HERSHEY_SIMPLEX 
	fontScale = .6
	color = (255, 0, 0) 
	thickness = 2


	if label_output == "a":
		label_true_a = output_label
		label_pred_a = predicted_label
	
		# frame_img = cv2.putText(frame_img, "TEST", org_a, font, fontScale, COLOR_A, thickness, cv2.LINE_AA) 
		# frame_img = cv2.putText(frame_img, "pred: " + label_pred_a, org_a2, font, fontScale, COLOR_A, thickness, cv2.LINE_AA) 

	elif label_output == 'b':
		label_true_b = output_label
		label_pred_b = predicted_label

		# frame_img = cv2.putText(frame_img, label_true_b, org_b, font, fontScale, COLOR_B, thickness, cv2.LINE_AA)
		# frame_img = cv2.putText(frame_img, "pred: " + label_pred_b, org_b2, font, fontScale, COLOR_B, thickness, cv2.LINE_AA)

	return frame_img


def export_gif_of(poses, input_labels, output_label, predicted_label, assessment_label, lookup_index, where):
	# What is the experiment being done?
	label = assessment_label[1:]
	labels = label.split("_")
	label_input, label_output = labels[0], labels[1]

	print(input_labels.shape)

	window_size = poses.shape[0]
	print(window_size)
	print(input_labels.shape[0])

	frames = []
	for fi in range(window_size):
		dummy = poses[fi]
		dummy = input_labels[fi]
		new_img = get_frame_visualization(poses[fi], input_labels[fi], output_label, predicted_label, label_input, label_output)
		frames.append(new_img)

	# TODO save gif to disk
	export_loco = where + assessment_label + '_at_' + str(lookup_index) + ".gif"
	imageio.mimsave(export_loco, frames, fps=30)
	print("Exported gif of results at timestep " + str(lookup_index))
	return




def verify_input_output(X, Y, classifier_type):
	unique_values = np.unique(Y)
	if(all(x in range(len(activity_labels)) for x in unique_values)): 
		pass
	else:
		print("Y contains labels outside the correct set")
		print(unique_values)
		exit()

# def quality_check_input(X, Y, classifier_type, assessment_label, where, num_inspections = 2):


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
		print(lookup_index)
		X_i = X[lookup_index]
		Y_i = Y[lookup_index]

		pose_size 		= CONST_NUM_POINTS*CONST_NUM_SUBPOINTS
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

