import os
import json
import numpy as np
from collections import defaultdict 
import pickle
import pandas as pd

import sys
sys.path.append("..")
import qchecks
import arconsts

# Lookup table for OpenPose keypoint indices
keypoint_labels = arconsts.keypoint_labels

filenames_all = arconsts.filenames_all
max_poses = 5
prefix_output = "output-vectors/raws/"

def get_file_frame_index(file_title):
	index_length = len('000000000000')
	start_index = file_title.index('_cropped_') + len('_cropped_')
	return int(file_title[start_index: start_index + index_length])

def get_json_raw_to_tuples(json_raw):
	pose = []
	num_pts = 25 #len(json_raw) / 3
	for i in range(num_pts):
		pt = json_raw[i*3 : (i*3) + 3]
		pose.append(pt)

	return pose


def in_bd_box(bd_box, p):
	pX, pY = p[0], p[1]
	return pX > bd_box[0][0] and pX < bd_box[1][0] and pY < bd_box[1][1] and pY > bd_box[0][1]

# based on get_role_labels(cleaned_poses) in feature_utils
def get_role_assignments(all_poses_in_frame):
	best_num_a = 0
	best_num_b = 0
	best_pose_a = np.zeros((25, 3))
	best_pose_b = np.zeros((25, 3))

	bd_box_A = arconsts.bd_box_A
	bd_box_B = arconsts.bd_box_B

	for pose in all_poses_in_frame:
		num_a, num_b = 0, 0

		for pt in pose:
			if in_bd_box(bd_box_A, pt):
				num_a += 1

			if in_bd_box(bd_box_B, pt):
				num_b += 1

		if num_a > best_num_a:
			best_num_a = num_a
			best_pose_a = pose

		if num_b > best_num_b:
			best_num_b = num_b
			best_pose_b = pose

	# TODO alternate take where we check if contiguous with previous?
	return best_pose_a, best_pose_b

def get_vectorized(pose):
	flat_list = []

	for pt in pose:
		# ditch the confidence percent
		flat_list.append(pt[0])
		flat_list.append(pt[1])

	return flat_list
        
def process_vectors_for_filename(group_name):
	print("Analyzing " + group_name)
	prefix = '../../Annotations/json/'
	entries = os.listdir(prefix)

	# get all the input files from this video
	entries = list(filter(lambda k: group_name in k, entries))
	
	if len(entries) < 1:
		print("No entries found, skipping")
		return

	start_index = entries[0].index('_cropped_') + len('_cropped_')

	indices = [get_file_frame_index(e) for e in entries]
	max_frame = max(indices) + 1

	# Save a numpy array of all of it concatenated together
	output_vector_raw = defaultdict(list)
	output_vector_roles = np.ndarray((max_frame, 25 * 2, 3))
	pandas_data = []


	filehandler = open(prefix_output + group_name + '_test.p', 'wb') 
	pickle.dump(output_vector_raw, filehandler)


	for entry in entries:
		json_file = open(prefix + entry) 
		raw_json = json.load(json_file)
		frame_index = get_file_frame_index(entry)

		best_a = None
		best_b = None


		people = raw_json['people']
		row_raw_poses = []
		
		all_poses_in_frame = []
		for person in people:
			json_raw_pose = person['pose_keypoints_2d']
			pose = get_json_raw_to_tuples(json_raw_pose)


			row_raw_poses.append(json_raw_pose)
			all_poses_in_frame.append(pose)

		pose_a, pose_b = get_role_assignments(all_poses_in_frame)
		pose_ab = np.concatenate((np.array(pose_a), np.array(pose_b)))

		# Add findings to appropriate data structures
		output_vector_raw[frame_index] = row_raw_poses
		output_vector_roles[frame_index] = pose_ab


		datum = [group_name, frame_index, pose_a, pose_b, row_raw_poses]
		pandas_data.append(datum)

		# if (frame_index % 1000 == 0):
		# 	print(frame_index)



	# Get some stats on things as we go
	df = pd.DataFrame(pandas_data, columns = ['meal', 'frame_index', 'pose_a', 'pose_b', 'poses_all']) 
	df.to_csv(prefix_output + group_name + '.csv')

	filehandler = open(prefix_output + group_name + '_raw_X.p', 'wb') 
	pickle.dump(output_vector_raw, filehandler)
	filehandler.close()

	filehandler = open(prefix_output + group_name + '_roles_X.p', 'wb') 
	pickle.dump(output_vector_roles, filehandler)
	filehandler.close()


for group_name in filenames_all:
	process_vectors_for_filename(group_name)


# # TODO adjust to be the correct slice
# filehandler = open(prefix_output + group_name + '_A_X.p', 'wb') 
# pickle.dump(output_vector_roles[:25], filehandler)
# filehandler.close()

# # TODO adjust to be the correct slice
# filehandler = open(prefix_output + group_name + '_B_X.p', 'wb') 
# pickle.dump(output_vector_roles[25:], filehandler)
# filehandler.close()











