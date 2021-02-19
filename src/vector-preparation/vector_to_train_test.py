import time
import cv2
import random
import pickle
import pandas as pd
import numpy as np
import json
import copy


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


filenames_all = ['8-13-18', '8-18-18', '8-17-18', '8-21-18', '8-9-18']
prefix_qc = './quality-checks/'
prefix_vectors_out = './output-vectors/'

INDEX_PA = 0
INDEX_PB = 1

BATCH_ID_STATELESS 	= 'stateless'
BATCH_ID_TEMPORAL 	= 'temporal'

BATCH_ID_MEALWISE_STATELESS 	= 'mwstateless'
BATCH_ID_MEALWISE_TEMPORAL 	= 'mwtemporal'

BATCH_ID_TEMPORAL_SPARE = 'temporal_sparse'

GROUPING_MEALWISE = '_g-mw'
GROUPING_RANDOM = "_g-rand"


def unison_shuffled_copies(a, b, seed_val):
	assert len(a) == len(b)
	p = np.random.RandomState(seed=seed_val).permutation(len(a))
	return a[p], b[p]

def unison_shuffled_copies_three(a, b, c, seed_val):
	assert len(a) == len(b)
	p = np.random.RandomState(seed=seed_val).permutation(len(a))
	return a[p], b[p], c[p]

# def slice_vectors(X_train, Y_train, training_suff, logfile, window_size=128, X_test=None, Y_test=None, overlap_percent=.5, percent_test=.2):
#     test_exclusive = False
#     test_set_provided = X_test is not None and Y_test is not None
    
#     x_sliced_list = []
#     y_sliced_list = []
    
#     # prepare test set for queries outside of its range
#     if test_set_provided:
#         X_test_sliced = np.zeros((X_test.shape[0]-window_size, window_size, X_test.shape[1]))
#         Y_test_sliced = np.zeros((Y_test.shape[0]-window_size, 1))
    
#     # counter for frequency of each label
#     label_freqs = {}
#     for i in range(0, 25):
#         label_freqs[i] = 0

#     # offset by one frame
#     for idx in range(window_size, X_train.shape[0]):
#         x_sliced_list.append(X_train[idx - window_size:idx].tolist())
    
#     # slice Y in the same way
#     X_train_sliced = np.array(x_sliced_list)
#     for idx in range(window_size, Y_train.shape[0]):
#         label_freqs[Y_train[idx]] += 1
#         y_sliced_list.append(Y_train[idx].tolist())
    
#     # turn back into array
#     Y_train_sliced = np.array(y_sliced_list)
    
#     # verify dimensions
#     print(X_train_sliced.shape)
#     print(Y_train_sliced.shape)
#     print(label_freqs)

#     # export class frequencies for this label of output vectors
#     logfile.write("class freqs: " + str(label_freqs))
    
#     # split the test set in the same way as the train set
#     if test_set_provided:
#         for idx in range(window_size, X_test.shape[0]):
#             X_test_sliced[idx-window_size,:] = X_test[idx-window_size:idx]
#         for idx in range(window_size, Y_test.shape[0]):
#             Y_test_sliced[idx-window_size,0] = Y_test[idx]
    
#     if test_exclusive:
#         test_set_provided = True
#         x_sliced_test_list = []
#         y_sliced_test_list = []
#         for idx in range(window_size+2, X_train.shape[0]):
#             x_sliced_test_list.append(X_train[idx-window_size:idx].tolist())
#         for idx in range(window_size+2, Y_train.shape[0]):
#             y_sliced_test_list.append(Y_train[idx].tolist())
#         Y_test_sliced = np.array(y_sliced_test_list)
#         X_test_sliced = np.array(x_sliced_test_list)
#     Y_train_sliced = to_categorical(Y_train_sliced)

#     if test_set_provided:
#         Y_test_sliced = to_categorical(Y_test_sliced)
#         X_test_sliced, Y_test_sliced = shuffle_in_unison(X_test_sliced, Y_test_sliced)

#     X_sliced, Y_sliced = shuffle_in_unison(X_train_sliced, Y_train_sliced)
    
#     if test_set_provided:
#         # export the pool of all vectors, not yet split into test train
#         print("test_set_provided")
#         train_list = [(X_sliced,Y_sliced)]
#         test_list = [(X_test_sliced, Y_test_sliced)]
        
#         pickle.dump(train_list, open("training_sets/train_list_" + training_suff +".p", "wb"),protocol=4)
#         pickle.dump(test_list, open("training_sets/test_list_" + training_suff +".p", "wb"),protocol=4)
#         return train_list, test_list
    
#     train_list = []
#     test_list = []

#     num_folds = int(1.0/percent_test)
#     for fold in range(num_folds):    
#         index_split = int(len(X_sliced) * (1.0 - percent_test))
#         lower = int(fold*percent_test*len(X_sliced))
#         upper = lower + int(percent_test*len(X_sliced))
#         X_train_sliced = np.concatenate((X_sliced[0:lower], X_sliced[upper:]), axis=0)
#         Y_train_sliced = np.concatenate((Y_sliced[0:lower], Y_sliced[upper:]), axis=0)
#         X_test_sliced = X_sliced[lower:upper]
#         Y_test_sliced = Y_sliced[lower:upper]
#         train_list.append((X_train_sliced,Y_train_sliced))
#         test_list.append((X_test_sliced,Y_test_sliced))

#         print(X_train_sliced.shape, Y_train_sliced.shape, X_test_sliced.shape, Y_test_sliced.shape)

# 	    pickle.dump(train_list, open("training_sets/train_list_" + training_suff +".p", "wb"),protocol=4)
# 	    pickle.dump(test_list, open("training_sets/test_list_" + training_suff +".p", "wb"),protocol=4)

#     return train_list, test_list


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

		core_name   = "/" + batch_id + "/" + "total" + "_id=" + str(batch_id) + groupings_type + "_s" + label_random_seed + "_f" + str(f) + "_"
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
		output_row = np.zeros((0,2))

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

		# print(X_all.shape)
		# print(X_all.shape)

		X_all_slices, Y_all_slices, Y_solo_labels = make_slices(X_all, Y_all, window_size, overlap_percent)
		# print(X_all_slices.shape)
		# print(Y_all_slices.shape)

		X_shuffled, Y_shuffled, Y_solo = unison_shuffled_copies_three(X_all_slices, Y_all_slices, Y_solo_labels, seed)

		# Note that we're passing just the labels in Y
		total_train_X, total_train_Y, total_test_X, total_test_Y = \
			export_each_fold_to_individual_chunks(filename, test_size, X_shuffled, Y_solo, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed)

	export_folds_aggregate(test_size, batch_id, total_train_X, total_train_Y, total_test_X, total_test_Y, seed, GROUPING_RANDOM)

def export_folds_mealwise_stateless(filenames_all, prefix_vectors_out, test_size, seed):
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

def export_folds(filenames_all, prefix_vectors_out, seed):
	print("Creating fold sets for all files")
	window_size = 128
	overlap_percent = .2
	test_percent = .2

	export_folds_mealwise_stateless(filenames_all, prefix_vectors_out, test_percent, seed)
	export_folds_mealwise_temporal(filenames_all, prefix_vectors_out, test_percent, seed, window_size, overlap_percent)
	export_folds_stateless(filenames_all, prefix_vectors_out, test_percent, seed)
	export_folds_temporal(filenames_all, prefix_vectors_out, window_size, overlap_percent, test_percent, seed)


def export_all_folds():
	seed = 111
	export_folds(filenames_all, prefix_vectors_out, seed)


export_all_folds()




