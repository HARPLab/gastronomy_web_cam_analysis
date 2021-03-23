import os
import pickle
from sklearn import svm
import time
import numpy as np
import cv2
import pandas as pd
np.set_printoptions(suppress=True)

from sklearn.metrics import confusion_matrix
from dictdiffer import diff
import seaborn as sn
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
# from tensorflow import keras
# from keras.models import Sequential
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

import sys
sys.path.append("..")
import qchecks
import arconsts

def get_prefix_export_result(exp_batch_id, classifier_type, feature_type, grouping_type, fold_id, seed):
	return 'results/' + str(exp_batch_id) + str(classifier_type) + str(feature_type) + str(grouping_type) + '_i' + str(fold_id) + '_s' + str(seed)

def get_prefix_export_truth(exp_batch_id, classifier_type, feature_type, grouping_type, fold_id, seed):
	return 'results/' + str(exp_batch_id) + str(classifier_type) + str(feature_type) + str(grouping_type) + '_i' + str(fold_id) + '_s' + str(seed)


# def get_prefix_export_result(unique_title, exp_batch_id, classifier_type, fold_id, grouping_type):
# 	return 'results/' + exp_batch_id + unique_title + "_f" + str(fold_id) + "_" + grouping_type + classifier_type

# def get_prefix_export_truth(unique_title, exp_batch_id, fold_id, grouping_type):
# 	return 'results/' + exp_batch_id + unique_title + "_f" + str(fold_id) + "_" + grouping_type


def get_slices_from_recipe(df, df_r):
	time_start = time.perf_counter()
	# print(df.shape)
	KEY_START = arconsts.PD_INDEX_START
	KEY_END = arconsts.PD_INDEX_END
	KEY_MEAL = arconsts.PD_MEALID
	KEY_YA = arconsts.PD_LABEL_A_RAW
	KEY_YB = arconsts.PD_LABEL_B_RAW

	df_length = df_r.shape[0]

	# print("df cols")
	# print(df.columns)

	# print("df_recipe cols")
	# print(df_recipe.columns)

	df_single = df.iloc[1]
	X_pa = df_single.get(arconsts.PD_POSE_A_RAW)
	X_pa = np.concatenate(X_pa, axis=0)
	n_features = int(X_pa.size)

	# default X value is 0
	X 	= np.full((df_length, 128, n_features * 2), 0)
	# default Y value is unlabeled
	Y 	= np.full((df_length, 2), -1)

	print("Target = " + str(df_length) + " slices... \t", end='')

	# Cache the original vectors to improve performance
	meals = {}
	for meal_id in arconsts.filenames_all:
		meals[meal_id] = df[df[arconsts.PD_MEALID].str.lower().str.contains(meal_id)]

	i = 0
	for start, end, meal_id in zip(df_r[KEY_START], df_r[KEY_END], df_r[KEY_MEAL]): #, df_r[KEY_YA], df_r[KEY_YB]
		# time_start = arconsts.get_start_time()
		# print(start)
		# print(end)

		df_t = meals[meal_id]
		# print("mealid")
		# arconsts.print_time(time_start)
		df_t = df_t[df_t[arconsts.PD_FRAMEID].between(start, end, inclusive=True)]
		# arconsts.print_time(time_start)
		# print('slice')
		target = df_t[df_t[arconsts.PD_FRAMEID].between(end, end, inclusive=True)]

		Y_pa = target[arconsts.PD_LABEL_A_CODE]
		Y_pb = target[arconsts.PD_LABEL_B_CODE]
		
		X_pa = df_t.get(arconsts.PD_POSE_A_RAW)
		X_pb = df_t.get(arconsts.PD_POSE_B_RAW)
		# print("get pose")
		# arconsts.print_time(time_start)

		X_pa = X_pa.to_numpy() #.values
		X_pb = X_pb.to_numpy() #.values
		# arconsts.print_time(time_start)

		X_pa = np.concatenate(X_pa, axis=0)
		X_pb = np.concatenate(X_pb, axis=0)
		# arconsts.print_time(time_start)

		n_features = int(X_pa.size / 128)
		X_pa = X_pa.reshape(128, n_features)
		X_pb = X_pb.reshape(128, n_features)
		# arconsts.print_time(time_start)

		X_row = np.hstack((X_pa, X_pb))

		X[i] = X_row
		Y[i] = [Y_pa, Y_pb]
		# arconsts.print_time(time_start)

		i += 1
		# if (i % 100 == 0):
		# 	print("Collected " + str(i) + " slices")
		
	print("Prepped slices complete! ", end="")
	# print(X.shape)
	# print(Y.shape)
	return X, Y

def get_temporal_vectors(df_transformed, exp_batch_id, fold_id, grouping_type, seed):
	df = df_transformed
	df_recipe = import_recipe()

	if grouping_type == arconsts.GROUPING_RANDOM:
		df_test_recipe 	= df_recipe.loc[df_recipe[arconsts.PD_TEST_SET] == fold_id]
		df_train_recipe = df_recipe.loc[df_recipe[arconsts.PD_TEST_SET] != fold_id]

	if grouping_type == arconsts.GROUPING_MEALWISE:
		test_meal = arconsts.filenames_all[fold_id]
		df_test_recipe 	= df_recipe.loc[df_recipe[arconsts.PD_MEALID] == test_meal]
		df_train_recipe = df_recipe.loc[df_recipe[arconsts.PD_TEST_SET] != test_meal]

	print("Slicing test vectors... \t", end='')
	X_test, Y_test 		= get_slices_from_recipe(df_transformed, df_test_recipe)
	print("Done")
	print("Slicing train vectors... \t", end='')
	X_train, Y_train 	= get_slices_from_recipe(df_transformed, df_train_recipe)
	print("Done")


	# for row in df_train_recipe:
	# 	start = row[PD_INDEX_START]
	# 	end   = row[PD_INDEX_END]

	# 	df_train.append( : row[PD_INDEX_END])


	fold_set = {}
	fold_set['xtest'] 	= X_test
	fold_set['xtrain'] 	= X_train
	fold_set['ytest'] 	= Y_test
	fold_set['ytrain'] 	= Y_train
	fold_set['pd_test'] = df_test_recipe
	fold_set['pd_train']= df_train_recipe
	return fold_set

def no_prob_col(value):
	value.reshape(25,3)
	return value[:,:2]

def pandas_remove_probs_a(row):
	return no_prob_col(row[arconsts.PD_POSE_A_RAW])

def pandas_remove_probs_b(row):
	return no_prob_col(row[arconsts.PD_POSE_B_RAW])

def remove_probabilities(df):
	df[arconsts.PD_POSE_A_RAW] = df.apply(pandas_remove_probs_a, axis=1)
	df[arconsts.PD_POSE_B_RAW] = df.apply(pandas_remove_probs_b, axis=1)
	return df

def points_to_angles(df):
	return df

def points_to_offsets(df):
	return df



def transform_features(df, feature_type):
	print("Transforming features to type " + str(feature_type))
	if feature_type == arconsts.FEATURES_VANILLA:
		print("Vanilla features")
		df = arconsts.reduce_Y_labels(df)
	elif feature_type == arconsts.FEATURES_OFFSET:
		print("Offset features")
		df = arconsts.reduce_Y_labels(df)		

	elif feature_type == arconsts.FEATURES_ANGLES:
		print("Angle features")
		df = arconsts.reduce_Y_labels(df)

	elif feature_type == arconsts.FEATURES_NO_PROB:
		print("Removing probability...", end='')
		df = arconsts.reduce_Y_labels(df)
		df = remove_probabilities(df)
		print("Done")
		

	elif feature_type == arconsts.FEATURES_LABELS_FULL:
		print("Expanded labels")
		df = arconsts.dont_reduce_Y_labels(df)
	else:
		print("SAD DAY, NO MATCH FOR " + str(feature_type))

	return df

def import_recipe():
	prefix_vectors = '../vector-preparation/output-vectors/slices/'
	entries = os.listdir(prefix_vectors)
	entries = list(filter(lambda k: 'all_slices.p' in k, entries))

	if len(entries) > 1:
		print("Error in import: multiple matching batches for this unique key")
		print("Please provide a key that aligns with only one of the following")
		print(entries)
		return
	if len(entries) == 0:
		print("Sorry, no import matching found")
		print(entries)
		return

	entry = entries[0]
	df = pd.read_pickle(prefix_vectors + entry)

	return df

def import_vectors():
	prefix_vectors = '../vector-preparation/output-vectors/trimmed/'
	entries = os.listdir(prefix_vectors)
	entries = list(filter(lambda k: 'all' in k, entries))

	if len(entries) > 1:
		print("Error in import: multiple matching batches for this unique key")
		print("Please provide a key that aligns with only one of the following")
		print(entries)
		return
	if len(entries) == 0:
		print("Sorry, no import matching found")
		print(entries)
		return

	entry = entries[0]
	df_XY = pd.read_pickle(prefix_vectors + entry)

	return df_XY



# def import_vectors_temporal_mealwise(unique_title, prefix, fold_id, grouping_type):


# def import_vectors_temporal_folds(unique_title, prefix, fold_id, grouping_type):
# 	entries = os.listdir(prefix)
# 	# print(prefix)
# 	# get all the input files from this video
# 	entries = list(filter(lambda k: unique_title in k, entries))
# 	entries = list(filter(lambda k: grouping_type in k, entries))
# 	entries = list(filter(lambda k: 'total' in k, entries))
# 	# print(entries)

# 	fold_group = "f" + str(fold_id) + "_"
# 	fold_entries = list(filter(lambda k: fold_group in k, entries))

# 	test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
# 	train 	= list(filter(lambda k: 'train' in k, fold_entries))

# 	X_test_label 	= list(filter(lambda k: '_X' 	in k, test))
# 	X_train_label 	= list(filter(lambda k: '_X' 	in k, train))
# 	Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
# 	Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))

# 	if len(X_test_label) > 1 or len(Y_test_label) > 1  or len(X_train_label) > 1  or len(Y_train_label) > 1:
# 		print("Error in import: multiple matching batches for this unique key")
# 		print("Please provide a key that aligns with only one of the following")
# 		print(X_test_label)
# 	if len(X_test_label) == 0 or len(Y_test_label) == 0  or len(X_train_label) == 0  or len(Y_train_label) == 0:
# 		print("Sorry, no import matching found")
# 		print(fold_group)

# 	X_test_label 	= X_test_label[0]
# 	X_train_label 	= X_train_label[0]
# 	Y_test_label 	= Y_test_label[0]
# 	Y_train_label 	= Y_train_label[0]

# 	X_test 		= pickle.load(open(prefix + X_test_label, 'rb'))
# 	X_train 	= pickle.load(open(prefix + X_train_label, 'rb'))
# 	Y_test 		= pickle.load(open(prefix + Y_test_label, 'rb'))
# 	Y_train 	= pickle.load(open(prefix + Y_train_label, 'rb'))
	
# 	return X_train, X_test, Y_train, Y_test







# # Given a file location, return the four test/train vectors
# def import_vectors(unique_title, prefix, fold_id, grouping_type):
# 	entries = os.listdir(prefix)
# 	# print(prefix)
# 	# get all the input files from this video
# 	entries = list(filter(lambda k: unique_title in k, entries))
# 	entries = list(filter(lambda k: grouping_type in k, entries))
# 	entries = list(filter(lambda k: 'total' in k, entries))
# 	# print(entries)

# 	fold_group = "f" + str(fold_id) + "_"
# 	fold_entries = list(filter(lambda k: fold_group in k, entries))

# 	test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
# 	train 	= list(filter(lambda k: 'train' in k, fold_entries))

# 	X_test_label 	= list(filter(lambda k: '_X' 	in k, test))
# 	X_train_label 	= list(filter(lambda k: '_X' 	in k, train))
# 	Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
# 	Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))

# 	if len(X_test_label) > 1 or len(Y_test_label) > 1  or len(X_train_label) > 1  or len(Y_train_label) > 1:
# 		print("Error in import: multiple matching batches for this unique key")
# 		print("Please provide a key that aligns with only one of the following")
# 		print(X_test_label)
# 	if len(X_test_label) == 0 or len(Y_test_label) == 0  or len(X_train_label) == 0  or len(Y_train_label) == 0:
# 		print("Sorry, no import matching found")
# 		print(fold_group)

# 	X_test_label 	= X_test_label[0]
# 	X_train_label 	= X_train_label[0]
# 	Y_test_label 	= Y_test_label[0]
# 	Y_train_label 	= Y_train_label[0]

# 	X_test 		= pickle.load(open(prefix + X_test_label, 'rb'))
# 	X_train 	= pickle.load(open(prefix + X_train_label, 'rb'))
# 	Y_test 		= pickle.load(open(prefix + Y_test_label, 'rb'))
# 	Y_train 	= pickle.load(open(prefix + Y_train_label, 'rb'))
	
# 	return X_train, X_test, Y_train, Y_test

# def get_stateless_vectors(folds, unique_title, exp_batch_id, grouping_type, seed=42):
# 	# These variables are set for a given import
# 	# different seeds, different values
	
# 	prefix = '../vector-preparation/output-vectors/stateless/'
# 	n_features = 2*CONST_NUM_POINTS*CONST_NUM_SUBPOINTS

# 	if grouping_type == GROUPING_RANDOM:
# 		grouping_type = BATCH_ID_STATELESS[1:]
# 	elif grouping_type == GROUPING_MEALWISE:
# 		grouping_type = BATCH_ID_MEALWISE_STATELESS[1:]

# 	exp_sets = {}
# 	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
# 	for fold_id in range(folds):
# 		long_prefix = get_prefix_export_truth(unique_title, exp_batch_id, fold_id, grouping_type)

# 		print("Geting stateless data for fold " + str(fold_id))
# 		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id, grouping_type)

# 		# print(X_test.shape)
# 		# print(X_train.shape)

# 		X_test 		= X_test.reshape(X_test.shape[0], n_features)
# 		X_train 	= X_train.reshape(X_train.shape[0], n_features)

# 		# print(X_test.shape)
# 		# print(X_train.shape)

# 		export_result(Y_train, 	long_prefix + '_Ytruetrain')
# 		export_result(Y_test, 	long_prefix + '_Ytruetest')

# 		exp_sets[fold_id] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

# 	print()
# 	return exp_sets

# def get_temporal_vectors(folds, unique_title, exp_batch_id, grouping_type, seed=42):
# 	# These variables are set for a given import
# 	# different seeds, different values
# 	prefix = '../vector-preparation/output-vectors/temporal/'
	
# 	dimension_X_row = (128,2*25*3)
# 	window_size = 128
# 	n_features = 2*25*3

# 	if grouping_type == GROUPING_RANDOM:
# 		grouping_type = BATCH_ID_TEMPORAL[1:]
# 	elif grouping_type == GROUPING_MEALWISE:
# 		grouping_type = BATCH_ID_MEALWISE_TEMPORAL[1:]

# 	exp_sets = {}
# 	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
# 	for fold_id in range(folds):
# 		long_prefix = get_prefix_export_truth(unique_title, exp_batch_id, fold_id, grouping_type)
# 		print("Getting temporal data for fold " + str(fold_id))

# 		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id, grouping_type)

# 		X_test 		= X_test.reshape(X_test.shape[0], window_size, n_features)
# 		X_train 	= X_train.reshape(X_train.shape[0], window_size, n_features) # dimension_X_row)

# 		export_result(Y_train, 	long_prefix + '_Ytruetrain') 
# 		export_result(Y_test, 	long_prefix + '_Ytruetest')	
# 		exp_sets[fold_id] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

# 	print()
# 	return exp_sets