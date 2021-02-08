import os
import pickle
from sklearn import svm
import time
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

# experiment runner
# 	for each type of trial
# 		
#		looks for appropriate vectors from to-vectors for the calls on vector train to test 
# 		

# class Experiment:
#     def __init__(self):
# 		pass

def classifier_train(X, Y):
	clf = svm.SVC()
	Y = Y.astype(int).ravel()
	print(X.shape)
	print(Y.shape)

	print(Y)
	print("Building AB")
	# classifier = KNeighborsClassifier(n_neighbors=9)
	classifier = AdaBoostClassifier()

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



def experiment_duo_vs_solo_just_labels(vector_dict, unique_title):
	prefix_export = 'results/'
	label_alb_a = "_alb_a"
	label_bla_b = "_bla_b"

	exp_id = "_adaboost"


	experiment_blob_all = {}

	for key_group in vector_dict.keys():
		experiment_blob = {}
		input_set = vector_dict[key_group]

		X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

		X_train_AlB, Y_train_A 	= get_AlB(X_train_AB, Y_train_AB)
		X_test_AlB, Y_test_A 		= get_AlB(X_test_AB, Y_test_AB)

		X_train_BlA, Y_train_B 	= get_BlA(X_train_AB, Y_train_AB)
		X_test_BlA, Y_test_B 		= get_BlA(X_test_AB, Y_test_AB)

		svm_alb_a = classifier_train(X_train_AlB, Y_train_A)
		result_alb_a = classifier_test(svm_alb_a, X_test_AlB, Y_test_A)

		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id  + label_alb_a  + "_resultY.p", "wb")
		pickle.dump(result_alb_a, filehandler)
		filehandler.close()

		svm_bla_b = classifier_train(X_train_BlA, Y_train_B)
		result_bla_b = classifier_test(svm_bla_b, X_test_BlA, Y_test_B)

		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id  + label_bla_b  + "_resultY.p", "wb")
		pickle.dump(result_bla_b, filehandler)
		filehandler.close()

		# store for future examination
		experiment_blob['alb_a_predict'] = result_alb_a
		experiment_blob['alb_a_correct'] = Y_test_A
		experiment_blob['bla_b_predict'] = result_bla_b
		experiment_blob['bla_b_correct'] = Y_test_B

		experiment_blob_all[key_group] = experiment_blob

	filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id + "_results.p", "wb")
	pickle.dump(experiment_blob_all, filehandler)
	filehandler.close()



def experiment_duo_vs_solo_svm(vector_dict, unique_title):
	prefix_export = 'results/'

	experiment_blob_all = {}

	for key_group in vector_dict.keys():
		experiment_blob = {}
		input_set = vector_dict[key_group]

		X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

		X_train_A, Y_train_A 	= get_A(X_train_AB, Y_train_AB)
		X_test_A, Y_test_A 		= get_A(X_test_AB, Y_test_AB)

		X_train_B, Y_train_B 	= get_B(X_train_AB, Y_train_AB)
		X_test_B, Y_test_B 		= get_B(X_test_AB, Y_test_AB)

		svm_a_a = classifier_train(X_train_A, Y_train_A)
		result_a_a = classifier_test(svm_a_a, X_test_A, Y_test_A)

		svm_b_b = classifier_train(X_train_B, Y_train_B)
		result_b_b = classifier_test(svm_b_b, X_test_B, Y_test_B)

		svm_ab_a = classifier_train(X_train_AB, Y_train_A)
		result_ab_a = classifier_test(svm_ab_a, X_test_AB, Y_test_A)

		svm_ab_b = classifier_train(X_train_AB, Y_train_B)
		result_ab_b = classifier_test(svm_ab_b, X_test_AB, Y_test_B)

		label_a_a = "_a_a"
		label_b_b = "_b_b"

		label_ab_a = "_ab_a"
		label_ab_b = "_ab_b"

		exp_id = "_adaboost"


		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id + label_a_a + "_resultY.p", "wb")
		pickle.dump(result_a_a, filehandler)
		filehandler.close()

		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id  + label_b_b  + "_resultY.p", "wb")
		pickle.dump(result_b_b, filehandler)
		filehandler.close()

		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id  + label_ab_a  + "_resultY.p", "wb")
		pickle.dump(result_ab_a, filehandler)
		filehandler.close()

		filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id  + label_ab_b  + "_resultY.p", "wb")
		pickle.dump(result_ab_b, filehandler)
		filehandler.close()

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

	filehandler = open(prefix_export + unique_title + '_f' + str(key_group) + exp_id + "_results.p", "wb")
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

	experiment_duo_vs_solo_svm(all_svm_vectors, unique_title)
	experiment_duo_vs_solo_just_labels(all_svm_vectors, unique_title)

	# experiment_duo_vs_solo_lstm()
	# experiment_duo_vs_solo_just_label_lstm()

def main():
    run_experiments()


main()





