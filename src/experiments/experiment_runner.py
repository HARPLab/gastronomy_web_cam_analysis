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

activity_labels = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']

CLASSIFIER_ADABOOST = '_adaboost'
CLASSIFIER_SGDC = '_sgdc'
CLASSIFIER_SVM = '_svm'
CLASSIFIER_KNN9 = '_kNN9'
CLASSIFIER_KNN5 = '_kNN5'
CLASSIFIER_KNN3 = '_kNN3'
CLASSIFIER_DecisionTree = '_dectree'

CLASSIFIER_LSTM = '_lstm'

CLASSIFIERS_TEMPORAL = [CLASSIFIER_LSTM]
CLASSIFIERS_STATELESS = [CLASSIFIER_KNN3, CLASSIFIER_KNN5, CLASSIFIER_KNN9, CLASSIFIER_SVM, CLASSIFIER_SGDC, CLASSIFIER_ADABOOST, CLASSIFIER_DecisionTree]

LSTM_NUM_LABELS = len(activity_labels)

def get_LSTM(trainX, trainY):
	print(trainX.shape)
	print(trainY.shape)
	# an input layer that expects 1 or more samples, 50 time steps, and 2 features
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], LSTM_NUM_LABELS
	input_shape=(n_timesteps, n_features)

	print("input_shape: " + str(input_shape))
	print("ought to be (128 , (25*3)*#poses)")

	dropout = 0.1
	batch_size = 256
	hidden_dim = 80*2
	model = Sequential()
	model.add(LSTM(hidden_dim, input_shape=input_shape))
	model.add(Dropout(dropout))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print("fiting model...." + model_name)
	#plot_losses = TrainingPlot()
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
	classifier = get_classifier(classifier_key, X, Y)

	time_start = time.perf_counter()
	
	if classifier_key == CLASSIFIER_LSTM:
		# Y.shape[]
		dropout = 0.1
		batch_size = 256
		hidden_dim = 80
		epochs = 70
		Y = to_categorical(Y, num_classes=len(activity_labels))
		classifier.fit(X, Y, batch_size=batch_size, verbose=0)
	else:
		classifier.fit(X, Y)




	# clf.fit(X, Y)
	time_end = time.perf_counter()
	time_diff = time_end - time_start
	print("Time elapsed: " + str(time_diff))

	return classifier

def get_single_vector_of_multiclass_result(result):
	decoded_result = result.argmax(axis=1)
	return decoded_result

#     frequencytest = {}
#     frequencypred = {}
#     for num in decodedtestY:
#             if num not in frequencytest.keys():
#                     frequencytest[num] = 1
#             else:
#                     frequencytest[num] = frequencytest[num] + 1
#     for num in decodedpredY:
#             if num not in frequencypred.keys():
#                     frequencypred[num] = 1
#             else:
#                     frequencypred[num] = frequencypred[num] + 1
#     print("stats:")
#     print(frequencytest)
#     print(frequencypred)
#     self.logfile.write("cm stats:\n")
#     self.logfile.write(str(frequencytest) + "\n")
#     self.logfile.write(str(frequencypred) + "\n")
#     #print(decodedpredY.shape)
#     #print(decodedtestY.shape)
#     predPadding = []
#     testPadding = []
#     i = 0
#     for key in activitydict.keys():
#             #print(key)
#             predPadding.append(i)
#             testPadding.append(23-i)
#             i +=1
#     decodedpredY = np.append(decodedpredY, predPadding)
#     decodedtestY = np.append(decodedtestY, testPadding)
#     cm = confusion_matrix(decodedtestY,decodedpredY)
#     np.set_printoptions(precision=2)
#     fig, ax = plt.subplots()
#     sum_of_rows = cm.sum(axis=1)
#     cm = cm / (sum_of_rows[:, np.newaxis]+1e-8)
#     p, r = calc_precision_recall(cm,self.logfile)
#     pickle.dump(cm, open(filename +"cm_mat.p", "wb"))
#     plot_confusion_matrix(cm,cmap=plt.cm.Blues)
#     plt.savefig(filename + "cm.png")
#     plt.close()

# 	return result

def classifier_test(classifier, X, Y, classifier_type):
	print(X.shape)
	print("Predicting...")
	time_start = time.perf_counter()
	result = classifier.predict(X)
	print(result.shape)

	if classifier_type == CLASSIFIER_LSTM:
		result = get_single_vector_of_multiclass_result(result)

	# print(result)
	# print(result.shape)
	time_end = time.perf_counter()
	time_diff = time_end - time_start
	print("Time elapsed: " + str(time_diff))

	print("Done with predictions")
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

	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		Xa = X_array[:, :, :half_dim_X]
		lb = Y_array[:, 	half_dim_Y:]

	X_alb = np.hstack((Xa, lb))
	Y_out = Y_array[:, :half_dim_Y]

	return X_alb, Y_out

def get_BlA(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape


	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)

		Xb = X_array[:, half_dim_X:]
		la = Y_array[:, :half_dim_Y]
	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)

		Xb = X_array[:, :, half_dim_X:]
		la = Y_array[:, :half_dim_Y]
		print("IMPLEMENT")

	X_bla = np.hstack((Xb, la))
	Y_out = Y_array[:, :half_dim_Y]
	return X_bla, Y_out


def get_A(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, :half_dim_X]

	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, :, :half_dim_X]

	return X_out, Y_array[:, 	:half_dim_Y]

def get_B(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, half_dim_X:]

	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, :, half_dim_X:]

	return X_out, Y_array[:, half_dim_Y:]



def experiment_duo_vs_solo_swapped(fold_id, input_set, unique_title, classifier_type, exp_batch_id):
	prefix_export = 'results/' + exp_batch_id
	label_b_a = "_b_a"
	label_a_b = "_a_b"
	
	experiment_blob = {}
	long_prefix = prefix_export + unique_title + '_f' + str(fold_id) + classifier_type

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	X_train_A, 	Y_train_A 	= get_A(X_train_AB, Y_train_AB, classifier_type)
	X_test_A, 	Y_test_A 	= get_A(X_test_AB, Y_test_AB, classifier_type)

	X_train_B, 	Y_train_B 	= get_B(X_train_AB, Y_train_AB, classifier_type)
	X_test_B, 	Y_test_B 	= get_B(X_test_AB, Y_test_AB, classifier_type)

	svm_a_b = classifier_train(X_train_A, Y_train_B, classifier_type)
	result_a_b = classifier_test(svm_a_b, X_test_A, Y_test_B, classifier_type)
	export_result(result_a_b, long_prefix, label_a_b)

	svm_b_a = classifier_train(X_train_B, Y_train_A, classifier_type)
	result_b_a = classifier_test(svm_b_a, X_test_B, Y_test_A, classifier_type)
	export_result(result_b_a, long_prefix, label_b_a)

	# store for future examination
	experiment_blob['a_b_predict'] = result_a_b
	experiment_blob['a_b_correct'] = Y_test_B
	experiment_blob['b_a_predict'] = result_b_a
	experiment_blob['b_a_correct'] = Y_test_A

	filehandler = open(prefix_export + unique_title + '_f' + str(fold_id) + classifier_type + "_results.p", "wb")
	pickle.dump(experiment_blob, filehandler)
	filehandler.close()


def experiment_label_to_label(fold_id, input_set, unique_title, classifier_type, exp_batch_id):
	prefix_export = 'results/' + exp_batch_id
	label_lb_la = "_lb_la"
	label_la_lb = "_la_lb"
	
	experiment_blob_all = {}

	experiment_blob = {}
	long_prefix = prefix_export + unique_title + '_f' + str(fold_id) + classifier_type

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	X_train_A, 	Y_train_A 	= get_A(X_train_AB, Y_train_AB, classifier_type)
	X_test_A, 	Y_test_A 	= get_A(X_test_AB, Y_test_AB, classifier_type)

	X_train_B, 	Y_train_B 	= get_B(X_train_AB, Y_train_AB, classifier_type)
	X_test_B, 	Y_test_B 	= get_B(X_test_AB, Y_test_AB, classifier_type)

	svm_la_lb = classifier_train(Y_train_A, Y_train_B, classifier_type)
	result_la_lb = classifier_test(svm_la_lb, Y_test_A, Y_test_B, classifier_type)
	export_result(result_la_lb, long_prefix, label_la_lb)

	svm_lb_la = classifier_train(Y_train_B, Y_train_A, classifier_type)
	result_lb_la = classifier_test(svm_lb_la, Y_test_B, Y_test_A, classifier_type)
	export_result(result_lb_la, long_prefix, label_lb_la)

	# store for future examination
	experiment_blob['la_lb_predict'] = result_la_lb
	experiment_blob['la_lb_correct'] = Y_test_B
	experiment_blob['lb_la_predict'] = result_lb_la
	experiment_blob['lb_la_correct'] = Y_test_A

	experiment_blob_all[key_group] = experiment_blob

	filehandler = open(prefix_export + unique_title + '_f' + str(fold_id) + classifier_type + "_results.p", "wb")
	pickle.dump(experiment_blob_all, filehandler)
	filehandler.close()


def experiment_duo_vs_solo_just_labels(fold_id, input_set, unique_title, classifier_type, exp_batch_id):
	prefix_export = 'results/' + exp_batch_id
	label_alb_a = "_alb_a"
	label_bla_b = "_bla_b"

	experiment_blob_all = {}

	long_prefix = prefix_export + unique_title + '_f' + str(fold_id) + classifier_type
	experiment_blob = {}

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	X_train_AlB, Y_train_A 	= get_AlB(X_train_AB, Y_train_AB, classifier_type)
	X_test_AlB, Y_test_A 	= get_AlB(X_test_AB, Y_test_AB, classifier_type)

	X_train_BlA, Y_train_B 	= get_BlA(X_train_AB, Y_train_AB, classifier_type)
	X_test_BlA, Y_test_B 	= get_BlA(X_test_AB, Y_test_AB, classifier_type)

	svm_alb_a = classifier_train(X_train_AlB, Y_train_A, classifier_type)
	result_alb_a = classifier_test(svm_alb_a, X_test_AlB, Y_test_A, classifier_type)
	export_result(result_alb_a, long_prefix, label_alb_a)

	svm_bla_b = classifier_train(X_train_BlA, Y_train_B, classifier_type)
	result_bla_b = classifier_test(svm_bla_b, X_test_BlA, Y_test_B, classifier_type)
	export_result(result_bla_b, long_prefix, label_bla_b)

	# store for future examination
	experiment_blob['alb_a_predict'] = result_alb_a
	experiment_blob['alb_a_correct'] = Y_test_A
	experiment_blob['bla_b_predict'] = result_bla_b
	experiment_blob['bla_b_correct'] = Y_test_B

	experiment_blob_all[fold_id] = experiment_blob

	filehandler = open(prefix_export + unique_title + '_f' + str(fold_id) + classifier_type + "_results.p", "wb")
	pickle.dump(experiment_blob_all, filehandler)
	filehandler.close()



def experiment_duo_vs_solo(fold_id, input_set, unique_title, classifier_type, exp_batch_id):
	prefix_export = 'results/' + exp_batch_id

	label_a_a = "_a_a"
	label_b_b = "_b_b"

	label_ab_a = "_ab_a"
	label_ab_b = "_ab_b"

	experiment_blob_all = {}

	long_prefix = prefix_export + unique_title + '_f' + str(fold_id) + classifier_type

	experiment_blob = {}

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	export_result(Y_train_AB, 	prefix_export, 'Ytruetrain_f' + str(fold_id))
	export_result(Y_test_AB, 	prefix_export, 'Ytruetest_f' + str(fold_id))

	X_train_A, Y_train_A 	= get_A(X_train_AB, Y_train_AB, classifier_type)
	X_test_A, Y_test_A 		= get_A(X_test_AB, Y_test_AB, classifier_type)

	X_train_B, Y_train_B 	= get_B(X_train_AB, Y_train_AB, classifier_type)
	X_test_B, Y_test_B 		= get_B(X_test_AB, Y_test_AB, classifier_type)


	svm_a_a = classifier_train(X_train_A, Y_train_A, classifier_type)
	result_a_a = classifier_test(svm_a_a, X_test_A, Y_test_A, classifier_type)
	export_result(result_a_a, long_prefix, label_a_a)

	svm_b_b = classifier_train(X_train_B, Y_train_B, classifier_type)
	result_b_b = classifier_test(svm_b_b, X_test_B, Y_test_B, classifier_type)
	export_result(result_b_b, long_prefix, label_b_b)

	svm_ab_a = classifier_train(X_train_AB, Y_train_A, classifier_type)
	result_ab_a = classifier_test(svm_ab_a, X_test_AB, Y_test_A, classifier_type)
	export_result(result_ab_a, long_prefix, label_ab_a)


	svm_ab_b = classifier_train(X_train_AB, Y_train_B, classifier_type)
	result_ab_b = classifier_test(svm_ab_b, X_test_AB, Y_test_B, classifier_type)
	export_result(result_ab_b, long_prefix, label_ab_b)

	
	# store for future examination
	experiment_blob['a_a_predict'] = result_a_a
	experiment_blob['a_a_correct'] = Y_test_A
	experiment_blob['b_b_predict'] = result_b_b
	experiment_blob['b_b_correct'] = Y_test_B

	experiment_blob['ab_a_predict'] = result_ab_a
	experiment_blob['ab_a_correct'] = Y_test_A
	experiment_blob['ab_b_predict'] = result_ab_b
	experiment_blob['ab_b_correct'] = Y_test_B
	experiment_blob_all[fold_id] = experiment_blob

	filehandler = open(prefix_export + unique_title + '_f' + str(fold_id) + classifier_type + "_results.p", "wb")
	pickle.dump(experiment_blob_all, filehandler)
	filehandler.close()


# Given a file location, return the four test/train vectors
def import_vectors(unique_title, prefix, fold_id):
	entries = os.listdir(prefix)

	# get all the input files from this video
	entries = list(filter(lambda k: unique_title in k, entries))

	entries = list(filter(lambda k: 'total' in k, entries))

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
	if len(X_test_label) == 0 or len(Y_test_label) == 0  or len(X_train_label) == 0  or len(Y_train_label) == 0:
		print("Sorry, no import matching found")
		print(fold_group)

	X_test_label 	= X_test_label[0]
	X_train_label 	= X_train_label[0]
	Y_test_label 	= Y_test_label[0]
	Y_train_label 	= Y_train_label[0]

	X_test 		= pickle.load(open(prefix + X_test_label, 'rb'))
	X_train 	= pickle.load(open(prefix + X_train_label, 'rb'))
	Y_test 		= pickle.load(open(prefix + Y_test_label, 'rb'))
	Y_train 	= pickle.load(open(prefix + Y_train_label, 'rb'))
	
	return X_train, X_test, Y_train, Y_test

def get_stateless_vectors(folds, unique_title, exp_batch_id, seed=42):
	# These variables are set for a given import
	# different seeds, different values
	
	prefix = '../vector-preparation/output-vectors/stateless/'
	prefix_export = 'results/' + exp_batch_id
	
	n_features = 2*3*25

	exp_sets = {}
	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
	for fold_id in range(folds):
		print("Geting svm data for fold " + str(fold_id))
		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id)

		X_test 		= X_test.reshape(X_test.shape[0], n_features)
		X_train 	= X_train.reshape(X_train.shape[0], n_features)

		export_result(Y_train, 	prefix_export, 'Ytruetrain_f' + str(fold_id))
		export_result(Y_test, 	prefix_export, 'Ytruetest_f' + str(fold_id))

		exp_sets[fold_id] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

	print()
	return exp_sets

def get_temporal_vectors(folds, unique_title, exp_batch_id, seed=42):
	# These variables are set for a given import
	# different seeds, different values
	prefix = '../vector-preparation/output-vectors/temporal/'
	
	dimension_X_row = (128,2*25*3)
	window_size = 128
	n_features = 2*25*3


	exp_sets = {}
	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
	for fold_id in range(folds):
		print("Geting temporal data for fold " + str(fold_id))

		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id)

		X_test 		= X_test.reshape(X_test.shape[0], window_size, n_features)
		X_train 	= X_train.reshape(X_train.shape[0], window_size, n_features) # dimension_X_row)

		export_result(Y_train, 	'results/' + exp_batch_id, 'Ytruetrain_f' + str(fold_id))
		export_result(Y_test, 	'results/' + exp_batch_id, 'Ytruetest_f' + str(fold_id))	

		exp_sets[fold_id] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

	print()
	return exp_sets


def run_experiments():
	num_folds = 5
	unique_title = 's42_'
	exp_batch_id = 5
	exp_batch_id = "exp_" + str(exp_batch_id) + "/"
	prefix_export = 'results/' + exp_batch_id

	try:
		os.mkdir(prefix_export)  
	except OSError as error:  
		print("This directory already exists; do you want a fresh experiment ID?")

	all_temporal_vectors 	= get_temporal_vectors(num_folds, unique_title, exp_batch_id)
	all_svm_vectors 		= get_stateless_vectors(num_folds, unique_title, exp_batch_id)


	# exp_types = [CLASSIFIER_DecisionTree, CLASSIFIER_KNN3, CLASSIFIER_KNN9, CLASSIFIER_ADABOOST, CLASSIFIER_SGDC]
	# exp_types = [CLASSIFIER_LSTM]
	exp_types = [CLASSIFIER_LSTM, CLASSIFIER_DecisionTree]

	for i in range(len(exp_types)):		
		classifier_type = exp_types[i]

		train_test_vectors = None
		if classifier_type in CLASSIFIERS_TEMPORAL:
			train_test_vectors = all_temporal_vectors
		elif classifier_type in CLASSIFIERS_STATELESS:
			train_test_vectors = all_svm_vectors
		else:
			print("Classifier type and corresponding vectors for " + classifier_type + " not found!")
			exit()

		for fold_id in range(num_folds):

			fold_data = train_test_vectors[fold_id]

			experiment_duo_vs_solo_swapped(fold_id, fold_data, unique_title, classifier_type, exp_batch_id)
			# experiment_duo_vs_solo_just_labels(fold_id, fold_data, unique_title, classifier_type, exp_batch_id)
			# experiment_label_to_label(fold_id, fold_data, unique_title, classifier_type, exp_batch_id)
			experiment_duo_vs_solo(fold_id, fold_data, unique_title, classifier_type, exp_batch_id)



def main():
	run_experiments()


main()





