import os
import pickle
from sklearn import svm
import time
import numpy as np
import cv2
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
import experiment_io

FLAG_TEST_MODE = False

CLASSIFIER_ADABOOST = arconsts.CLASSIFIER_ADABOOST
CLASSIFIER_SGDC 	= arconsts.CLASSIFIER_SGDC
CLASSIFIER_SVM 		= arconsts.CLASSIFIER_SVM
CLASSIFIER_KNN9 	= arconsts.CLASSIFIER_KNN9
CLASSIFIER_KNN5 	= arconsts.CLASSIFIER_KNN5
CLASSIFIER_KNN3 	= arconsts.CLASSIFIER_KNN3
CLASSIFIER_DecisionTree = arconsts.CLASSIFIER_DecisionTree

CLASSIFIER_LSTM = arconsts.CLASSIFIER_LSTM
CLASSIFIER_LSTM_BIGGER = arconsts.CLASSIFIER_LSTM_BIGGER
CLASSIFIER_LSTM_BIGGEST = arconsts.CLASSIFIER_LSTM_BIGGEST
CLASSIFIER_LSTM_TINY	= arconsts.CLASSIFIER_LSTM_TINY
CLASSIFIER_CRF = arconsts.CLASSIFIER_CRF

GROUPING_MEALWISE = arconsts.GROUPING_MEALWISE
GROUPING_RANDOM = arconsts.GROUPING_RANDOM

BATCH_ID_STATELESS 	= arconsts.BATCH_ID_STATELESS
BATCH_ID_TEMPORAL 	= arconsts.BATCH_ID_TEMPORAL
BATCH_ID_MEALWISE_STATELESS = arconsts.BATCH_ID_MEALWISE_STATELESS
BATCH_ID_MEALWISE_TEMPORAL 	= arconsts.BATCH_ID_MEALWISE_TEMPORAL


CLASSIFIERS_TEMPORAL = arconsts.CLASSIFIERS_TEMPORAL
CLASSIFIERS_STATELESS = arconsts.CLASSIFIERS_STATELESS

activity_labels = arconsts.activity_labels

FEATURES_VANILLA 	= arconsts.FEATURES_VANILLA
FEATURES_OFFSET 	= arconsts.FEATURES_OFFSET
FEATURES_ANGLES		= arconsts.FEATURES_ANGLES
FEATURES_NO_PROB	= arconsts.FEATURES_NO_PROB

CONST_NUM_POINTS 	= arconsts.CONST_NUM_POINTS
CONST_NUM_SUBPOINTS = arconsts.CONST_NUM_SUBPOINTS
CONST_NUM_LABEL 	= arconsts.CONST_NUM_LABEL

EPOCHS_SHORTEST = 5
EPOCHS_STANDARD = 500

def get_LSTM(trainX, trainY, scale=1):
	# print(trainY.shape)rint(trainX.shape)
	# p
	# an input layer that expects 1 or more samples, 50 time steps, and 2 features
	n_outputs = qchecks.get_num_outputs(trainY)
	print("num_outputs " + str(n_outputs))
	n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
	input_shape=(n_timesteps, n_features)

	# print("input_shape: " + str(input_shape))
	# print("ought to be (128 , (25*3)*#poses)")

	dropout = 0.1
	batch_size = 256
	hidden_dim = int(80*scale)
	print("Hidden dimension is: " + str(hidden_dim))
	model = Sequential()
	model.add(LSTM(hidden_dim, input_shape=input_shape))
	model.add(Dropout(dropout))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print("fiting model...." + model_name)
	#plot_losses = TrainingPlot()

	# TODO Export model and notes
	# Export loss over time graph

	return model

def get_CRF():
	pass
	return model

def get_classifier(key, X, Y):
	if key == CLASSIFIER_ADABOOST:
		return AdaBoostClassifier()
	if key == CLASSIFIER_SGDC:
		return SGDClassifier()
	if key == CLASSIFIER_SVM:
		return LinearSVC()
	if key == CLASSIFIER_KNN3:
		return KNeighborsClassifier(n_neighbors=3)
	if key == CLASSIFIER_KNN5:
		return KNeighborsClassifier(n_neighbors=5)
	if key == CLASSIFIER_KNN9:
		return KNeighborsClassifier(n_neighbors=9)
	if key == CLASSIFIER_DecisionTree:
		return DecisionTreeClassifier()
	if key == CLASSIFIER_LSTM:
		return get_LSTM(X, Y)
	if key == CLASSIFIER_LSTM_BIGGER:
		return get_LSTM(X, Y, scale=2)
	if key == CLASSIFIER_LSTM_BIGGEST:
		return get_LSTM(X, Y, scale=4)
	if key == CLASSIFIER_LSTM_TINY:
		return get_LSTM(X, Y, scale = .25)
	return None

def classifier_train(X, Y, classifier_key, prefix_where):
	Y = Y.astype(int).ravel()
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Building " + classifier_key + " at " + str(current_time))

	# classifier = KNeighborsClassifier(n_neighbors=9)
	classifier = get_classifier(classifier_key, X, Y)

	time_start = time.perf_counter()
	
	if classifier_key in CLASSIFIERS_TEMPORAL:
		# Y.shape[]
		dropout = 0.1
		batch_size = 256
		# epochs = 7

		# unique_values = np.unique(Y)
		# num_unique_values = len(unique_values)
		Y = to_categorical(Y)
		
		if FLAG_TEST_MODE == True:
			epochs = EPOCHS_SHORTEST
		else:
			epochs = EPOCHS_STANDARD
		
		epochs = 5#00 #5 #EPOCHS_SHORTEST

		print("epochs = " + str(epochs))
		print("Fitting to...")
		print(X.shape)
		print(Y.shape)
		history = classifier.fit(X, Y, batch_size=batch_size, verbose=0, epochs=epochs)
	else:
		history = classifier.fit(X, Y)

	time_end = time.perf_counter()
	time_diff = time_end - time_start

	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Time elapsed: " + str(time_diff) + " ending at " + str(current_time))

	experiment_io.export_classifier(classifier, prefix_where)
	experiment_io.export_classifier_history(history, prefix_where)

	return classifier

# align with experimental analysis
def get_single_vector_of_multiclass_result(result):
	decoded_result = result.argmax(axis=1)
	return decoded_result

def classifier_test(classifier, X, Y, classifier_type, prefix_where):
	print(X.shape)
	print("Predicting...")
	time_start = time.perf_counter()
	result = classifier.predict(X)
	# print(result.shape)

	if classifier_type in CLASSIFIERS_TEMPORAL:
		result = get_single_vector_of_multiclass_result(result)

	# print(result)
	# print(result.shape)
	time_end = time.perf_counter()
	time_diff = time_end - time_start
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Time elapsed: " + str(time_diff) + " at "  + str(current_time))

	experiment_io.export_test_result(result, Y, prefix_where)

	print("Done with predictions\n")
	return result


def unpack_dict(input_set):
	X_test 	= input_set['xtest']
	X_train = input_set['xtrain']
	Y_test 	= input_set['ytest']
	Y_train = input_set['ytrain']

	return X_train, X_test, Y_train, Y_test

def get_AlB(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		Xa = X_array[:, :half_dim_X]
		lb = Y_array[:, 	half_dim_Y:]
		Y_out = Y_array[:, :half_dim_Y]

	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)
		Xa = X_array[:, :, :half_dim_X]
		lb = Y_array[:, :, half_dim_Y:]
		# Y_out.reshape((Y_out.shape[0], 1))

		Y_out = Y_array[:, -1, :half_dim_Y]

	X_alb = np.concatenate((Xa, lb), axis=2)
	return X_alb, Y_out

def get_BlA(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape


	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)

		Xb = X_array[:, half_dim_X:]
		la = Y_array[:, :half_dim_Y]
		Y_out = Y_array[:, half_dim_Y:]
	
	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)

		Xb = X_array[:, :, half_dim_X:]
		la = Y_array[:, :, :half_dim_Y]

		Y_out = Y_array[:, -1, half_dim_Y:]
	

	X_bla = np.concatenate((Xb, la), axis=2)
	return X_bla, Y_out

def get_lA(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)

		Y_out = Y_array[:, 	:half_dim_Y]

	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)

		Y_out = Y_array[:, -1,	:half_dim_Y]

	return Y_out

def get_lB(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)

		Y_out = Y_array[:, half_dim_Y:]
	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)

		Y_out = Y_array[:, -1, half_dim_Y:]

	return Y_out


def get_A(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, :half_dim_X]
		Y_out = Y_array[:, 	:half_dim_Y]

	elif c_type in CLASSIFIERS_TEMPORAL:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)

		X_out = X_array[:, :, :half_dim_X]
		Y_out = Y_array[:, -1, :half_dim_Y]

	else:
		print("What kind of classifier is this?")
		exit()

	qchecks.verify_input_output(X_out, Y_out, c_type)
	return X_out, Y_out

def get_B(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, half_dim_X:]
		Y_out = Y_array[:, 	half_dim_Y:]
	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)
		X_out = X_array[:, :, half_dim_X:]
		Y_out = Y_array[:, -1, half_dim_Y:]

	qchecks.verify_input_output(X_out, Y_out, c_type)
	return X_out, Y_out

def experiment_swapped_poses(fold_id, input_set, classifier_type, long_prefix):
	print("Experiment: Poses Swapped")
	label_b_a = "_b_a"
	label_a_b = "_a_b"

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	X_train_A, 	Y_train_A 	= get_A(X_train_AB, Y_train_AB, classifier_type)
	X_test_A, 	Y_test_A 	= get_A(X_test_AB, Y_test_AB, classifier_type)

	X_train_B, 	Y_train_B 	= get_B(X_train_AB, Y_train_AB, classifier_type)
	X_test_B, 	Y_test_B 	= get_B(X_test_AB, Y_test_AB, classifier_type)

	print("a_b")
	clf_a_b = classifier_train(X_train_A, Y_train_B, classifier_type, long_prefix + label_a_b)
	result_a_b = classifier_test(clf_a_b, X_test_A, Y_test_B, classifier_type, long_prefix + label_a_b)
	qchecks.quality_check_output(X_test_A, Y_test_B, result_a_b, classifier_type, label_a_b, long_prefix, num_inspections = 30)

	print("b_a")
	clf_b_a = classifier_train(X_train_B, Y_train_A, classifier_type, long_prefix + label_b_a)
	result_b_a = classifier_test(clf_b_a, X_test_B, Y_test_A, classifier_type, long_prefix + label_b_a)
	qchecks.quality_check_output(X_test_B, Y_test_A, result_b_a, classifier_type, label_b_a, long_prefix, num_inspections = 30)


def experiment_label_to_label(fold_id, input_set, classifier_type, long_prefix):
	print("Experiment: Label to Label")
	label_lb_la = "_lb_la"
	label_la_lb = "_la_lb"
	
	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	Y_train_A 	= get_lA(X_train_AB, Y_train_AB, classifier_type)
	Y_test_A 	= get_lA(X_test_AB, Y_test_AB, classifier_type)

	Y_train_B 	= get_lB(X_train_AB, Y_train_AB, classifier_type)
	Y_test_B 	= get_lB(X_test_AB, Y_test_AB, classifier_type)

	# export_confusion_matrix(Y_train_A, Y_train_B, exp_batch_id, classifier_type, "label_to_label", fold_id)
	print("la_lb")
	clf_la_lb = classifier_train(Y_train_A, Y_train_B, classifier_type, long_prefix + label_la_lb)
	result_la_lb = classifier_test(clf_la_lb, Y_test_A, Y_test_B, classifier_type, long_prefix + label_la_lb)

	print("lb_la")
	clf_lb_la = classifier_train(Y_train_B, Y_train_A, classifier_type, long_prefix + label_lb_la)
	result_lb_la = classifier_test(clf_lb_la, Y_test_B, Y_test_A, classifier_type, long_prefix + label_lb_la)

def experiment_pose_vs_poseauxlabel(fold_id, input_set, classifier_type, long_prefix):
	print("Experiment: pose_vs_poseauxlabel")
	label_alb_a = "_alb_a"
	label_bla_b = "_bla_b"

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	X_train_AlB, Y_train_A 	= get_AlB(X_train_AB, Y_train_AB, classifier_type)
	X_test_AlB, Y_test_A 	= get_AlB(X_test_AB, Y_test_AB, classifier_type)

	X_train_BlA, Y_train_B 	= get_BlA(X_train_AB, Y_train_AB, classifier_type)
	X_test_BlA, Y_test_B 	= get_BlA(X_test_AB, Y_test_AB, classifier_type)

	print("alb_a")
	clf_alb_a = classifier_train(X_train_AlB, Y_train_A, classifier_type, long_prefix + label_alb_a)
	result_alb_a = classifier_test(clf_alb_a, X_test_AlB, Y_test_A, classifier_type, long_prefix + label_alb_a)
	qchecks.quality_check_output(X_test_AlB, Y_test_A, result_alb_a, classifier_type, label_alb_a, long_prefix, num_inspections = 30)
	
	print("bla_b")
	clf_bla_b = classifier_train(X_train_BlA, Y_train_B, classifier_type, long_prefix + label_bla_b)
	result_bla_b = classifier_test(clf_bla_b, X_test_BlA, Y_test_B, classifier_type, long_prefix + label_bla_b)
	qchecks.quality_check_output(X_test_BlA, Y_test_B, result_bla_b, classifier_type, label_bla_b, long_prefix, num_inspections = 30)
	

def experiment_duo_vs_solo(fold_id, input_set, classifier_type, long_prefix):
	print("Experiment: Duo vs Solo")
	label_a_a = "_a_a"
	label_b_b = "_b_b"

	label_ab_a = "_ab_a"
	label_ab_b = "_ab_b"

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	X_train_A, Y_train_A 	= get_A(X_train_AB, Y_train_AB, classifier_type)
	X_test_A, Y_test_A 		= get_A(X_test_AB, Y_test_AB, classifier_type)

	X_train_B, Y_train_B 	= get_B(X_train_AB, Y_train_AB, classifier_type)
	X_test_B, Y_test_B 		= get_B(X_test_AB, Y_test_AB, classifier_type)

	qchecks.verify_input_output(X_train_AB, Y_train_AB, classifier_type)
	qchecks.verify_input_output(X_test_AB, Y_test_AB, classifier_type)

	print("a_a")
	clf_a_a = classifier_train(X_train_A, Y_train_A, classifier_type, long_prefix + label_a_a)
	result_a_a = classifier_test(clf_a_a, X_test_A, Y_test_A, classifier_type, long_prefix + label_a_a)
	qchecks.quality_check_output(X_test_A, Y_test_A, result_a_a, classifier_type, label_a_a, long_prefix, num_inspections = 30)
	
	print("b_b")
	clf_b_b = classifier_train(X_train_B, Y_train_B, classifier_type, long_prefix + label_b_b)
	result_b_b = classifier_test(clf_b_b, X_test_B, Y_test_B, classifier_type, long_prefix + label_b_b)
	qchecks.quality_check_output(X_test_B, Y_test_B, result_b_b, classifier_type, label_b_b, long_prefix, num_inspections = 30)

	print("ab_a")
	clf_ab_a = classifier_train(X_train_AB, Y_train_A, classifier_type, long_prefix + label_ab_a)
	result_ab_a = classifier_test(clf_ab_a, X_test_AB, Y_test_A, classifier_type, long_prefix + label_ab_a)
	qchecks.quality_check_output(X_test_AB, Y_test_A, result_ab_a, classifier_type, label_ab_a, long_prefix, num_inspections = 30)

	print("ab_b")
	clf_ab_b = classifier_train(X_train_AB, Y_train_B, classifier_type, long_prefix + label_ab_b)
	result_ab_b = classifier_test(clf_ab_b, X_test_AB, Y_test_B, classifier_type, long_prefix + label_ab_b)
	qchecks.quality_check_output(X_test_AB, Y_test_B, result_ab_b, classifier_type, label_ab_b, long_prefix, num_inspections = 30)



def run_experiments(exp_batch_id):
	num_folds = 1
	seed = 111
	unique_title = 's' + str(seed)
	exp_batch_id = "exp_" + str(exp_batch_id) + "/"
	prefix_export = 'results/' + exp_batch_id

	folds = range(num_folds)

	try:
		os.mkdir(prefix_export)  
	except OSError as error:  
		print("This directory already exists; do you want a fresh experiment ID?")

	print("Experimental results being saved to " + str(prefix_export))

	grouping_type = GROUPING_RANDOM

	#  all_stateless_vectors 	= get_stateless_vectors(num_folds, unique_title, exp_batch_id, grouping_type, seed)
	
	# exp_types = [CLASSIFIER_KNN3, CLASSIFIER_DecisionTree, CLASSIFIER_ADABOOST, CLASSIFIER_KNN5, CLASSIFIER_KNN9]#, CLASSIFIER_LSTM, CLASSIFIER_LSTM_BIGGER, CLASSIFIER_LSTM_BIGGEST]#, CLASSIFIER_SGDC, CLASSIFIER_SVM]
	# exp_types = [CLASSIFIER_DecisionTree, CLASSIFIER_KNN3, CLASSIFIER_KNN5, CLASSIFIER_KNN9, CLASSIFIER_ADABOOST, CLASSIFIER_SVM]
	exp_types 		= [CLASSIFIER_LSTM]
	feature_types 	= [arconsts.FEATURES_VANILLA, arconsts.FEATURES_OFFSET, arconsts.FEATURES_NO_PROB, arconsts.FEATURES_LABELS_FULL]

	for i in range(len(exp_types)): 
		classifier_type 	= exp_types[i]
		df_vectors 	= experiment_io.import_vectors()
		print("Imported vector set")

		for feature_type in feature_types:
			df_transformed = experiment_io.transform_features(df_vectors, feature_type)
		
			for fold_id in folds:
				
				if classifier_type in CLASSIFIERS_TEMPORAL:
					input_set = experiment_io.get_temporal_vectors(df_transformed, exp_batch_id, fold_id, grouping_type, prefix_export, seed)
				elif classifier_type in CLASSIFIERS_STATELESS:
					train_test_vectors = all_stateless_vectors
				else:
					print("Classifier type and corresponding vectors for " + classifier_type + " not found!")
					exit()
			

				if classifier_type != CLASSIFIER_LSTM:
					pass

				long_prefix = experiment_io.get_prefix_export_result(exp_batch_id, classifier_type, feature_type, grouping_type, fold_id, seed)
				# experiment_duo_vs_solo(fold_id, input_set, classifier_type, long_prefix)
				# experiment_pose_vs_poseauxlabel(fold_id, input_set, classifier_type, long_prefix)
				experiment_swapped_poses(fold_id, input_set, classifier_type, long_prefix)
				# experiment_label_to_label(fold_id, input_set, classifier_type, long_prefix)


def main():
	exp_batch_id = 30
	prefix_export = 'results/' + str(exp_batch_id)
	FLAG_TEST_MODE = True
	run_experiments(exp_batch_id)


main()





