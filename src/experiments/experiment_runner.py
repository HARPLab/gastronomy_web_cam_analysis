import os
import pickle
from sklearn import svm
import time
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


CLASSIFIER_ADABOOST = '_adaboost'
CLASSIFIER_SGDC = '_sgdc'
CLASSIFIER_SVM = '_svm'
CLASSIFIER_KNN9 = '_kNN9'
CLASSIFIER_KNN5 = '_kNN5'
CLASSIFIER_KNN3 = '_kNN3'
CLASSIFIER_DecisionTree = '_dectree'

def get_classifier(key):
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

	return None

def export_result(obj, long_prefix, label):
	filehandler = open(long_prefix  + label  + "_resultY.p", "wb")
	pickle.dump(obj, filehandler)
	filehandler.close()
	print("Exported to " + long_prefix + label)

def classifier_train(X, Y, classifier_key):
	Y = Y.astype(int).ravel()
	print(X.shape)
	print(Y.shape)

	print(Y)
	print("Building " + classifier_key)
	# classifier = KNeighborsClassifier(n_neighbors=9)
	classifier = get_classifier(classifier_key)

	time_start = time.perf_counter()
	classifier.fit(X, Y)

	# clf.fit(X, Y)
	time_end = time.perf_counter()
	time_diff = time_end - time_start
	print("Time elapsed: " + str(time_diff))

	return classifier

def classifier_test(classifier, X, Y):
	print(X.shape)
	print("Predicting...")
	time_start = time.perf_counter()
	result = classifier.predict(X)
	time_end = time.perf_counter()
	time_diff = time_start - time_end
	print("Time elapsed: " + str(time_diff))

	print("Done with predictions")
	return result


def unpack_dict(input_set):
	X_test 	= input_set['xtest']
	X_train = input_set['xtrain']
	Y_test 	= input_set['ytest']
	Y_train = input_set['ytrain']

	return X_train, X_test, Y_train, Y_test


def get_AlB(X_array, Y_array):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	half_dim_X = int(og_dim_X[1] / 2)
	half_dim_Y = int(og_dim_Y[1] / 2)

	Xa = X_array[:, :half_dim_X]
	lb = Y_array[:, 	half_dim_Y:]

	X_alb = np.hstack((Xa, lb))

	return X_alb, Y_array[:, :half_dim_Y]

def get_BlA(X_array, Y_array):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	half_dim_X = int(og_dim_X[1] / 2)
	half_dim_Y = int(og_dim_Y[1] / 2)

	Xb = X_array[:, half_dim_X:]
	la = Y_array[:, 	:half_dim_Y]

	X_bla = np.hstack((Xb, la))

	return X_bla, Y_array[:, :half_dim_Y]


def get_A(X_array, Y_array):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	half_dim_X = int(og_dim_X[1] / 2)
	half_dim_Y = int(og_dim_Y[1] / 2)

	return X_array[:, :half_dim_X], Y_array[:, 	:half_dim_Y]

def get_B(X_array, Y_array):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	half_dim_X = int(og_dim_X[1] / 2)
	half_dim_Y = int(og_dim_Y[1] / 2)

	return X_array[:, half_dim_X:], Y_array[:, 	half_dim_Y:]



def experiment_duo_vs_solo_just_labels(vector_dict, unique_title, classifier_type, exp_batch_id):
	prefix_export = 'results/' + exp_batch_id
	label_alb_a = "_alb_a"
	label_bla_b = "_bla_b"
	long_prefix = prefix_export + exp_batch_id + unique_title + '_f' + str(key_group) + classifier_type

	experiment_blob_all = {}

	for key_group in [5]:
		experiment_blob = {}
		input_set = vector_dict[key_group]

		X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

		X_train_AlB, Y_train_A 	= get_AlB(X_train_AB, Y_train_AB)
		X_test_AlB, Y_test_A 		= get_AlB(X_test_AB, Y_test_AB)

		X_train_BlA, Y_train_B 	= get_BlA(X_train_AB, Y_train_AB)
		X_test_BlA, Y_test_B 		= get_BlA(X_test_AB, Y_test_AB)

		svm_alb_a = classifier_train(X_train_AlB, Y_train_A, classifier_type)
		result_alb_a = classifier_test(svm_alb_a, X_test_AlB, Y_test_A)
		export_result(result_alb_a, long_prefix, label_alb_a, exp_batch_id)

		svm_bla_b = classifier_train(X_train_BlA, Y_train_B, classifier_type)
		result_bla_b = classifier_test(svm_bla_b, X_test_BlA, Y_test_B)
		export_result(result_bla_b, long_prefix, label_bla_b, exp_batch_id)

		# store for future examination
		experiment_blob['alb_a_predict'] = result_alb_a
		experiment_blob['alb_a_correct'] = Y_test_A
		experiment_blob['bla_b_predict'] = result_bla_b
		experiment_blob['bla_b_correct'] = Y_test_B

		experiment_blob_all[key_group] = experiment_blob

	filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + classifier_type + "_results.p", "wb")
	pickle.dump(experiment_blob_all, filehandler)
	filehandler.close()



def experiment_duo_vs_solo_svm(vector_dict, unique_title, classifier_type, exp_batch_id):
	prefix_export = 'results/' + exp_batch_id
	label_a_a = "_a_a"
	label_b_b = "_b_b"

	label_ab_a = "_ab_a"
	label_ab_b = "_ab_b"

	experiment_blob_all = {}

	for key_group in [5]:
		experiment_blob = {}
		input_set = vector_dict[key_group]

		X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

		X_train_A, Y_train_A 	= get_A(X_train_AB, Y_train_AB)
		X_test_A, Y_test_A 		= get_A(X_test_AB, Y_test_AB)

		X_train_B, Y_train_B 	= get_B(X_train_AB, Y_train_AB)
		X_test_B, Y_test_B 		= get_B(X_test_AB, Y_test_AB)


		svm_a_a = classifier_train(X_train_A, Y_train_A, classifier_type)
		result_a_a = classifier_test(svm_a_a, X_test_A, Y_test_A)
		export_result(result_a_a, long_prefix, label_a_a, exp_batch_id)

		svm_b_b = classifier_train(X_train_B, Y_train_B, classifier_type)
		result_b_b = classifier_test(svm_b_b, X_test_B, Y_test_B)
		export_result(result_b_b, long_prefix, label_b_b, exp_batch_id)

		svm_ab_a = classifier_train(X_train_AB, Y_train_A, classifier_type)
		result_ab_a = classifier_test(svm_ab_a, X_test_AB, Y_test_A)
		export_result(result_ab_a, long_prefix, label_ab_a, exp_batch_id)


		svm_ab_b = classifier_train(X_train_AB, Y_train_B, classifier_type)
		result_ab_b = classifier_test(svm_ab_b, X_test_AB, Y_test_B)
		export_result(result_ab_b, long_prefix, label_ab_b, exp_batch_id)

		
		# store for future examination
		experiment_blob['a_a_predict'] = result_a_a
		experiment_blob['a_a_correct'] = Y_test_A
		experiment_blob['b_b_predict'] = result_b_b
		experiment_blob['b_b_correct'] = Y_test_B

		experiment_blob['ab_a_predict'] = result_ab_a
		experiment_blob['ab_a_correct'] = Y_test_A
		experiment_blob['ab_b_predict'] = result_ab_b
		experiment_blob['ab_b_correct'] = Y_test_B
		experiment_blob_all[key_group] = experiment_blob

		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + classifier_type + "_results.p", "wb")
		pickle.dump(experiment_blob_all, filehandler)
		filehandler.close()


# Given a file location, return the four test/train vectors
def import_vectors(unique_title, prefix, fold_id):
	entries = os.listdir(prefix)

	# get all the input files from this video
	entries = list(filter(lambda k: unique_title in k, entries))

	fold_group = "f" + str(fold_id) + "_"
	fold_entries = list(filter(lambda k: fold_group in k, entries))

	test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
	train 	= list(filter(lambda k: 'train' in k, fold_entries))

	X_test_label 	= list(filter(lambda k: '_X' 	in k, test))
	X_train_label 	= list(filter(lambda k: '_X' 	in k, train))
	Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
	Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))

	if len(X_test_label) > 1 or len(Y_test_label) > 1  or len(X_train_label) > 1  or len(Y_train_label) > 1:
		print("Error in import: multiple matching batches for this unique key")
		print("Please provide a key that aligns with only one of the following")
		print(X_test)

	X_test_label 	= X_test_label[0]
	X_train_label 	= X_train_label[0]
	Y_test_label 	= Y_test_label[0]
	Y_train_label 	= Y_train_label[0]

	X_test 		= pickle.load(open(prefix + X_test_label, 'rb'))
	X_train 	= pickle.load(open(prefix + X_train_label, 'rb'))
	Y_test 		= pickle.load(open(prefix + Y_test_label, 'rb'))
	Y_train 	= pickle.load(open(prefix + Y_train_label, 'rb'))

	X_test 		= X_test.reshape(X_test.shape[0], 50*3)
	X_train 	= X_train.reshape(X_train.shape[0], 50*3)
	
	return X_train, X_test, Y_train, Y_test


# Returns each set of 
def get_svm_vectors(folds, unique_title, seed=42):
	# These variables are set for a given import
	# different seeds, different values
	
	prefix = '../vector-preparation/output-vectors/for_svm/'
	
	exp_sets = {}
	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
	for fold_id in range(folds):
		print("Geting svm data for fold " + str(fold_id))
		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id)

		exp_sets[folds] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

	return exp_sets

def run_experiments():
	folds = 5
	unique_title = 'total_forsvm_s42_'
	all_svm_vectors = get_svm_vectors(folds, unique_title)

	exp_batch_id = 2
	exp_batch_id = "exp_" + str(exp_batch_id) + "/"
	prefix_export = 'results/' + exp_batch_id

	try:
		os.mkdir(prefix_export)  
	except OSError as error:  
		print("This directory already exists; do you want a fresh experiment ID?")

	exp_types = [CLASSIFIER_KNN3, CLASSIFIER_SGDC, CLASSIFIER_SVM, CLASSIFIER_KNN3]
	for i in range(len(exp_types)):
		classifier_type = exp_types[i]
		experiment_duo_vs_solo_svm(all_svm_vectors, unique_title, classifier_type, exp_batch_id)
		experiment_duo_vs_solo_just_labels(all_svm_vectors, unique_title, classifier_type, exp_batch_id)


def main():
    run_experiments()


main()





