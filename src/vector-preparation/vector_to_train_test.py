import time
import cv2
import random
import pickle
import pandas as pd
import numpy as np
import json
import copy
import os

import sys
sys.path.append("..")
import arconsts

activitydict = arconsts.activitydict
activity_labels = arconsts.activity_labels

filenames_all = arconsts.filenames_all
prefix_qc = './quality-checks/'
prefix_vectors_out = './output-vectors/'

INDEX_PA = 0
INDEX_PB = 1

BATCH_ID_STATELESS 	= arconsts.BATCH_ID_STATELESS
BATCH_ID_TEMPORAL 	= arconsts.BATCH_ID_TEMPORAL

BATCH_ID_MEALWISE_STATELESS = arconsts.BATCH_ID_MEALWISE_STATELESS
BATCH_ID_MEALWISE_TEMPORAL 	= arconsts.BATCH_ID_MEALWISE_TEMPORAL

BATCH_ID_TEMPORAL_SPARE = 'temporal_sparse'

GROUPING_MEALWISE 	= arconsts.GROUPING_MEALWISE
GROUPING_RANDOM 	= arconsts.GROUPING_RANDOM

def reduce_labels(Y_array):
	# activity_from_key = {0:'away-from-table', 1:'idle', 2:'eating', 3: 'drinking', 4: 'talking', 5: 'ordering', 6: 'standing',
	# 					7: 'talking:waiter', 8: 'looking:window', 9: 'looking:waiter', 10: 'reading:bill', 11: 'reading:menu',
	# 					12: 'paying:check', 13: 'using:phone', 14: 'using:napkin', 15: 'using:purse', 16: 'using:glasses',
	# 					17: 'using:wallet', 18: 'looking:PersonA', 19: 'looking:PersonB', 20: 'takeoutfood', 21: 'leaving-table', 22: 'cleaning-up', 23: 'NONE'}

	# activity_labels = [0: 'NONE', 1: 'away-from-table', 2: 'idle', 3: 'eating', 4: 'talking', 5:'talking:waiter', 6: 'looking:window', 
	# 7: 'reading:bill', 8: 'reading:menu', 9: 'paying:check', 10: 'using:phone', 11: 'using:objs', 12: 'standing']
	
	ACT_NONE 			= 0
	ACT_AWAY_FROM_TABLE = 1
	ACT_IDLE			= 2
	ACT_EATING			= 3
	ACT_TALKING			= 4
	ACT_WAITER			= 5
	ACT_LOOKING_WINDOW	= 6
	ACT_READING_BILL	= 7
	ACT_READING_MENU	= 8
	ACT_PAYING_CHECK	= 9
	ACT_USING_PHONE		= 10
	ACT_OBJ_WILDCARD 	= 11
	ACT_STANDING		= 12


	Y_new = np.empty_like(Y_array)
	Y_new = np.where(Y_array==23, ACT_NONE, 		Y_new)
	Y_new = np.where(Y_array==22, ACT_NONE, 		Y_new)
	Y_new = np.where(Y_array==21, ACT_STANDING, 	Y_new) 
	Y_new = np.where(Y_array==20, ACT_OBJ_WILDCARD, Y_new)
	Y_new = np.where(Y_array==19, ACT_IDLE, 		Y_new)
	Y_new = np.where(Y_array==18, ACT_IDLE, 		Y_new)
	Y_new = np.where(Y_array==17, ACT_PAYING_CHECK, Y_new)
	Y_new = np.where(Y_array==16, ACT_OBJ_WILDCARD, Y_new)
	Y_new = np.where(Y_array==15, ACT_OBJ_WILDCARD, Y_new)
	Y_new = np.where(Y_array==14, ACT_OBJ_WILDCARD, Y_new)
	Y_new = np.where(Y_array==13, ACT_USING_PHONE, 	Y_new)
	Y_new = np.where(Y_array==12, ACT_PAYING_CHECK, Y_new)
	Y_new = np.where(Y_array==11, ACT_READING_MENU, Y_new)
	Y_new = np.where(Y_array==10, ACT_READING_BILL, Y_new)
	Y_new = np.where(Y_array==9, ACT_WAITER, 		Y_new) 
	Y_new = np.where(Y_array==8, ACT_LOOKING_WINDOW,Y_new)
	Y_new = np.where(Y_array==7, ACT_WAITER, 		Y_new)
	Y_new = np.where(Y_array==6, ACT_STANDING, 		Y_new)
	Y_new = np.where(Y_array==5, ACT_WAITER, 		Y_new)
	Y_new = np.where(Y_array==4, ACT_TALKING, 		Y_new)
	Y_new = np.where(Y_array==3, ACT_EATING, 		Y_new)
	Y_new = np.where(Y_array==2, ACT_EATING, 		Y_new)
	Y_new = np.where(Y_array==1, ACT_IDLE, 			Y_new)
	Y_new = np.where(Y_array==0, ACT_AWAY_FROM_TABLE, Y_new)

	return Y_new


def unison_shuffled_copies(a, b, seed_val):
	assert len(a) == len(b)
	p = np.random.RandomState(seed=seed_val).permutation(len(a))
	return a[p], b[p]

def unison_shuffled_copies_three(a, b, c, seed_val):
	assert len(a) == len(b)
	p = np.random.RandomState(seed=seed_val).permutation(len(a))
	return a[p], b[p], c[p]

def export_each_fold_to_individual_chunks(filename, test_size, X_shuffled, Y_shuffled, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed):
	chunk_size = int(len(X_shuffled) * (test_size))
	num_folds = int(1.0 / test_size)

	label_testsize	  = str(int(test_size * 100)) + "_test"
	label_trainsize	 = str(int((1.0 - test_size) * 100)) + "_train"
	label_random_seed   = str(seed)

	for f in range(num_folds):
		test_start, test_end = f*chunk_size, f*chunk_size + chunk_size
		print("Exporting " + batch_id + " fold " + str(f) + " from " + str(test_start) + " to " + str(test_end))

		test_X	  = X_shuffled[test_start : test_end]
		test_Y	  = Y_shuffled[test_start : test_end]
		
		train_chunk_x1 = X_shuffled[0 : test_start]
		train_chunk_x2 = X_shuffled[test_end : ]

		train_chunk_y1 = Y_shuffled[0 : test_start]
		train_chunk_y2 = Y_shuffled[test_end : ]

		train_X	 = np.concatenate((train_chunk_x1, train_chunk_x2), axis=0)
		train_Y	 = np.concatenate((train_chunk_y1, train_chunk_y2), axis=0)

		# print(test_X.shape)
		# print(test_Y.shape)

		# print(total_test_X[f].shape)
		# print(total_test_Y[f].shape)

		total_test_X[f] = np.concatenate((total_test_X[f], test_X))
		total_test_Y[f] = np.concatenate((total_test_Y[f], test_Y))

		total_train_X[f] = np.concatenate((total_train_X[f], train_X))
		total_train_Y[f] = np.concatenate((total_train_Y[f], train_Y))

		# print(test_size)
		# print(chunk_size)
		# print(len(X_shuffled))
		# print(train_X.shape)
		# print(test_X.shape)

		core_name   = "/" + batch_id + "/" + filename + "_id=" + str(batch_id) + "_s" + label_random_seed + "_f" + str(f) + "_"
		test_name   = core_name + label_testsize
		train_name  = core_name + label_trainsize

		filehandler = open(prefix_vectors_out + train_name + "_X.p", "wb")
		pickle.dump(train_X, filehandler)
		filehandler.close()

		filehandler = open(prefix_vectors_out + train_name + "_Y.p", "wb")
		pickle.dump(train_Y, filehandler)
		filehandler.close()

		filehandler = open(prefix_vectors_out + test_name + "_X.p", "wb")
		pickle.dump(test_X, filehandler)
		filehandler.close()

		filehandler = open(prefix_vectors_out + test_name + "_X.p", "wb")
		pickle.dump(test_Y, filehandler)
		filehandler.close()

	print()

	return total_train_X, total_train_Y, total_test_X, total_test_Y

def verify_input_output(X, Y):
	# print(X.shape)
	# print(Y.shape)
	# print("Unique values: ")
	unique_values = np.unique(Y)
	if(all(x in range(len(activity_labels)) for x in unique_values)): 
		# print("All good")
		pass
	else:
		print("Nope- Y contains more than the valid labels")
		print(unique_values)
		exit()


def make_slices(X, Y, window_size, overlap_percent):
	shift_stepsize = int(window_size * overlap_percent)

	X_slices_list 	= []
	Y_slices_list 	= []
	Y_solo_list		= []

	num_steps = int(X.shape[0] / shift_stepsize)

	for i in range(num_steps):
		slice_start, slice_end = int(i*shift_stepsize), int(i*shift_stepsize + window_size)
		if slice_end < Y.shape[0]:
			X_slice = copy.copy(X[slice_start:slice_end])
			Y_slice = copy.copy(Y[slice_start:slice_end])
			Y_solo 	= copy.copy(Y[slice_end])

			X_slices_list.append(X_slice)
			Y_slices_list.append(Y_slice)
			Y_solo_list.append(Y_solo)

	X_slices_list = np.array(X_slices_list)
	Y_slices_list = np.array(Y_slices_list)
	Y_solo_list = np.array(Y_solo_list)

	return X_slices_list, Y_slices_list, Y_solo_list


def make_slices_sparse(X, Y, window_size, overlap_percent):
	pass

		
def export_folds_aggregate(test_size, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed, groupings_type):
	num_folds = int(1.0 / test_size)
	print(test_size)
	print("num_folds=" + str(num_folds))
	label_testsize	  = str(int(test_size * 100)) + "_test"
	label_trainsize	 = str(int((1.0 - test_size) * 100)) + "_train"
	label_random_seed   = str(seed)

	for f in range(num_folds):
		fold_train_X = total_train_X[f]
		fold_train_Y = total_train_Y[f]
		fold_test_X = total_test_X[f]
		fold_test_Y = total_test_Y[f]

		# print(np.histogram(fold_train_Y, bins=len(activitydict)))
		# print(np.histogram(fold_test_Y, bins=len(activitydict)))
		
		print("These dimensions should be consistent across folds:")
		print("trainshape=" + str(fold_train_X.shape))
		print("testshape=" + str(fold_test_X.shape))

		core_name   = "/" + batch_id + "/" + "total" + "_" + str(batch_id) + groupings_type + "_s" + label_random_seed + "_f" + str(f) + "_"
		test_name   = core_name + label_testsize
		train_name  = core_name + label_trainsize

		filehandler = open(prefix_vectors_out + train_name + "_X.p", "wb")
		pickle.dump(fold_train_X, filehandler)
		filehandler.close()

		filehandler = open(prefix_vectors_out + train_name + "_Y.p", "wb")
		pickle.dump(fold_train_Y, filehandler)
		filehandler.close()

		filehandler = open(prefix_vectors_out + test_name + "_X.p", "wb")
		pickle.dump(fold_test_X, filehandler)
		filehandler.close()

		filehandler = open(prefix_vectors_out + test_name + "_Y.p", "wb")
		pickle.dump(fold_test_Y, filehandler)
		filehandler.close()


def export_folds_temporal(filenames_all, prefix_vectors_out, window_size, overlap_percent, test_size, seed):
	print("Creating temporal folds")
	batch_id = BATCH_ID_TEMPORAL
	num_folds = int(1.0 / test_size)

	total_test_X = {}
	total_test_Y = {}

	total_train_X = {}
	total_train_Y = {}

	# Set up the input fold arrays
	for f in range(num_folds):
		input_row = np.zeros((0,128,50,3))
		output_row = np.zeros((0,128,2))

		total_test_X[f] = input_row
		total_test_Y[f] = output_row
		total_train_X[f] = input_row
		total_train_Y[f] = output_row

	# for each of the importable meals
	for filename in filenames_all:
		print("Adding vectors from meal " + filename)
		core_name = prefix_vectors_out + "trimmed/trimmed_" + filename

		filehandler = open(core_name + "_X.p", "rb")
		X_all = pickle.load(filehandler)
		filehandler.close()

		filehandler = open(core_name + "_Y.p", "rb")
		Y_all = pickle.load(filehandler)
		filehandler.close()

		Y_all = reduce_labels(Y_all)

		# print(X_all.shape)
		# print(X_all.shape)

		X_all_slices, Y_all_slices, Y_solo_labels = make_slices(X_all, Y_all, window_size, overlap_percent)
		# print(X_all_slices.shape)
		# print(Y_all_slices.shape)

		verify_input_output(X_all_slices, Y_all_slices)

		X_shuffled, Y_shuffled, Y_solo = unison_shuffled_copies_three(X_all_slices, Y_all_slices, Y_solo_labels, seed)

		verify_input_output(X_shuffled, Y_shuffled)

		# Note that we're passing just the labels in Y
		total_train_X, total_train_Y, total_test_X, total_test_Y = \
			export_each_fold_to_individual_chunks(filename, test_size, X_shuffled, Y_shuffled, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed)

	export_folds_aggregate(test_size, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed, GROUPING_RANDOM)

def export_folds_mealwise_stateless(filenames_all, prefix_vectors_out, test_size, seed):
	print("Mealwise stateless")
	batch_id = BATCH_ID_MEALWISE_STATELESS
	num_folds = int(1.0 / test_size)

	total_test_X = {}
	total_test_Y = {}
	total_train_X = {}
	total_train_Y = {}

	for f in range(num_folds):
		input_row = np.zeros((0,50,3))
		output_row = np.zeros((0,2))

		total_test_X[f] 	= input_row
		total_test_Y[f] 	= output_row
		total_train_X[f]	= input_row
		total_train_Y[f]	= output_row

		for filename in filenames_all:
			# print("Adding vectors from meal " + filename)
			core_name = prefix_vectors_out + "/trimmed/trimmed_" + filename

			index = filenames_all.index(filename)

			filehandler = open(core_name + "_X.p", "rb")
			X_all = pickle.load(filehandler)
			filehandler.close()

			filehandler = open(core_name + "_Y.p", "rb")
			Y_all = pickle.load(filehandler)
			filehandler.close()

			# if this is the fold, that meal is the test
			if index == f:
				# print("this is our test set")
				total_test_X[f] 	= np.concatenate((total_test_X[f], X_all), axis=0)
				total_test_Y[f] 	= np.concatenate((total_test_Y[f], Y_all), axis=0)
			else:
				# print("this is our train set")
				total_train_X[f]	= np.concatenate((total_train_X[f], X_all), axis=0)
				total_train_Y[f]	= np.concatenate((total_train_Y[f], Y_all), axis=0)



	# for f in range(num_folds):
	# 	print()

	export_folds_aggregate(test_size, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed, GROUPING_MEALWISE)

	print("\n")


def export_folds_mealwise_temporal(filenames_all, prefix_vectors_out, test_size, seed, window_size, overlap_percent):
	print("Export folds mealwise temporal")
	batch_id = BATCH_ID_MEALWISE_TEMPORAL
	num_folds = int(1.0 / test_size)

	total_test_X = {}
	total_test_Y = {}
	total_train_X = {}
	total_train_Y = {}

	for f in range(num_folds):
		input_row = np.zeros((0,128,50,3))
		output_row = np.zeros((0,2))

		total_test_X[f] 	= input_row
		total_test_Y[f] 	= output_row
		total_train_X[f]	= input_row
		total_train_Y[f]	= output_row

		for filename in filenames_all:
			# print("Adding vectors from meal " + filename)
			core_name = prefix_vectors_out + "/trimmed/trimmed_" + filename

			index = filenames_all.index(filename)

			filehandler = open(core_name + "_X.p", "rb")
			X_all = pickle.load(filehandler)
			filehandler.close()

			filehandler = open(core_name + "_Y.p", "rb")
			Y_all = pickle.load(filehandler)
			filehandler.close()

			X_all, Y_all_verbose, Y_all = make_slices(X_all, Y_all, window_size, overlap_percent)

			# if this is the fold, that meal is the test
			if index == f:
				# print("this is our test set")
				total_test_X[f] 	= np.concatenate((total_test_X[f], X_all), axis=0)
				total_test_Y[f] 	= np.concatenate((total_test_Y[f], Y_all), axis=0)
			else:
				# print("this is our train set")
				total_train_X[f]	= np.concatenate((total_train_X[f], X_all), axis=0)
				total_train_Y[f]	= np.concatenate((total_train_Y[f], Y_all), axis=0)



	# for f in range(num_folds):
	# 	print()

	export_folds_aggregate(test_size, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed, GROUPING_MEALWISE)

	print("\n")



def export_folds_stateless(filenames_all, prefix_vectors_out, test_size, seed):
	print("Export folds stateless")
	batch_id = BATCH_ID_STATELESS
	num_folds = int(1.0 / test_size)

	total_test_X = {}
	total_test_Y = {}
	total_train_X = {}
	total_train_Y = {}

	for f in range(num_folds):
		input_row = np.zeros((0,50,3))
		output_row = np.zeros((0,2))

		total_test_X[f] 	= input_row
		total_test_Y[f] 	= output_row
		total_train_X[f]	= input_row
		total_train_Y[f]	= output_row

	for filename in filenames_all:
		print("Adding vectors from meal " + filename)
		core_name = prefix_vectors_out + "/trimmed/trimmed_" + filename

		filehandler = open(core_name + "_X.p", "rb")
		X_all = pickle.load(filehandler)
		filehandler.close()

		filehandler = open(core_name + "_Y.p", "rb")
		Y_all = pickle.load(filehandler)
		filehandler.close()

		X_shuffled, Y_shuffled = unison_shuffled_copies(X_all, Y_all, seed)

		total_train_X, total_train_Y, total_test_X, total_test_Y = \
			export_each_fold_to_individual_chunks(filename, test_size, X_shuffled, Y_shuffled, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed)


	export_folds_aggregate(test_size, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed, GROUPING_RANDOM)

	print("\n")

def make_directories(type_list):
	for batch_id in type_list:
		try:
			os.mkdir(prefix_vectors_out + batch_id)
		except OSError as error:  
			print("Directory " + batch_id + " already exists; will overwrite contents")

def export_folds(filenames_all, prefix_vectors_out, seed):
	print("Creating fold sets for all files")
	window_size = 128
	overlap_percent = .2
	test_percent = .2

	vector_types = [BATCH_ID_STATELESS, BATCH_ID_TEMPORAL, BATCH_ID_MEALWISE_TEMPORAL, BATCH_ID_MEALWISE_STATELESS]
	make_directories(vector_types)

	export_folds_mealwise_stateless(filenames_all, prefix_vectors_out, test_percent, seed)
	export_folds_mealwise_temporal(filenames_all, prefix_vectors_out, test_percent, seed, window_size, overlap_percent)
	export_folds_stateless(filenames_all, prefix_vectors_out, test_percent, seed)
	export_folds_temporal(filenames_all, prefix_vectors_out, window_size, overlap_percent, test_percent, seed)


def export_all_folds():
	seed = 111
	export_folds(filenames_all, prefix_vectors_out, seed)


export_all_folds()




