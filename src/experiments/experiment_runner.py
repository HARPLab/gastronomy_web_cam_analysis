import os
import pickle
from sklearn import svm
import time
import numpy as np

from sklearn.metrics import confusion_matrix
from dictdiffer import diff
import seaborn as sn
import matplotlib.pyplot as plt

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

# activity_labels = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
# 					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
# 					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
# 					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']

activity_labels = ['NONE', 'away-from-table', 'idle', 'eating', 'talking', 'talk:waiter', 'looking:window', 
					'reading:bill', 'reading:menu', 'paying:check', 'using:phone', 'obj:wildcard', 'standing']

# activitydict = {0: 'NONE', 1: 'away-from-table', 2: 'idle', 3: 'eating', 4: 'talking', 5:'talking:waiter', 6: 'looking:window', 
# 	7: 'reading:bill', 8: 'reading:menu', 9: 'paying:check', 10: 'using:phone', 11: 'using:objs', 12: 'standing'}

CLASSIFIER_ADABOOST = '_adaboost'
CLASSIFIER_SGDC = '_sgdc'
CLASSIFIER_SVM = '_svm'
CLASSIFIER_KNN9 = '_kNN9'
CLASSIFIER_KNN5 = '_kNN5'
CLASSIFIER_KNN3 = '_kNN3'
CLASSIFIER_DecisionTree = '_dectree'

CLASSIFIER_LSTM = '_lstm-og'
CLASSIFIER_LSTM_BIGGER = '_lstm-big'
CLASSIFIER_LSTM_BIGGEST = '_lstm-biggest'
CLASSIFIER_CRF = '_crf'

GROUPING_MEALWISE = '_g-mw'
GROUPING_RANDOM = "_g-rand"

CLASSIFIERS_TEMPORAL = [CLASSIFIER_LSTM, CLASSIFIER_LSTM_BIGGER, CLASSIFIER_LSTM_BIGGEST, CLASSIFIER_CRF]
CLASSIFIERS_STATELESS = [CLASSIFIER_KNN3, CLASSIFIER_KNN5, CLASSIFIER_KNN9, CLASSIFIER_SVM, CLASSIFIER_SGDC, CLASSIFIER_ADABOOST, CLASSIFIER_DecisionTree]

LSTM_NUM_LABELS = len(activity_labels)

def get_LSTM(trainX, trainY, scale=1):
	print(trainX.shape)
	print(trainY.shape)
	# an input layer that expects 1 or more samples, 50 time steps, and 2 features
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], LSTM_NUM_LABELS
	input_shape=(n_timesteps, n_features)

	print("input_shape: " + str(input_shape))
	print("ought to be (128 , (25*3)*#poses)")

	dropout = 0.1
	batch_size = 256
	hidden_dim = 80*scale
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



	return model

def export_confusion_matrix(y1, y2, exp_batch_id, classifier_type, subexp_label, fold_id):
	save_location = "results/" + exp_batch_id + "f" + str(fold_id) + "_" + classifier_type[1:] + "_" + subexp_label + "_f" + str(fold_id)

	print(y1)
	print(y2)

	cm = confusion_matrix(y1.astype(int), y2.astype(int), labels=range(len(activity_labels)))

	sn.set(font_scale=2)
	plt.subplots(figsize=(22,22))
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
	

	return None

def export_result(obj, long_prefix, label):
	filehandler = open(long_prefix  + label  + "_resultY.p", "wb")

	print(long_prefix + label)
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
	
	if classifier_key in CLASSIFIERS_TEMPORAL:
		# Y.shape[]
		dropout = 0.1
		batch_size = 256
		# epochs = 7
		Y = to_categorical(Y, num_classes=len(activity_labels))
		epochs = 500
		print("Fitting to...")
		print(X.shape)
		print(Y.shape)
		classifier.fit(X, Y, batch_size=batch_size, verbose=0, epochs=epochs)
	else:
		classifier.fit(X, Y)




	# clf.fit(X, Y)
	time_end = time.perf_counter()
	time_diff = time_end - time_start
	print("Time elapsed: " + str(time_diff) + " ending at " + str(time_end))

	return classifier

# align with experimental analysis
def get_single_vector_of_multiclass_result(result):
	decoded_result = result.argmax(axis=1)
	return decoded_result

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

		Y_out = Y_array[:, :,	:half_dim_Y]

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

		Y_out = Y_array[:, :, half_dim_Y:]

	return Y_out


def get_A(X_array, Y_array, c_type):
	og_dim_X = X_array.shape
	og_dim_Y = Y_array.shape

	if c_type in CLASSIFIERS_STATELESS:
		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)
		X_out = X_array[:, :half_dim_X]
		Y_out = Y_array[:, 	:half_dim_Y]

	else:
		half_dim_X = int(og_dim_X[2] / 2)
		half_dim_Y = int(og_dim_Y[2] / 2)
		X_out = X_array[:, :, :half_dim_X]
		Y_out = Y_array[:, -1, :half_dim_Y]

	print(Y_out.shape)		
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

	print(Y_out.shape)
	return X_out, Y_out



def experiment_swapped_poses(fold_id, input_set, unique_title, classifier_type, exp_batch_id, grouping_type):
	print("Experiment: Poses Swapped")
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


def experiment_label_to_label(fold_id, input_set, unique_title, classifier_type, exp_batch_id, grouping_type):
	print("Experiment: Label to Label")
	prefix_export = 'results/' + exp_batch_id
	label_lb_la = "_lb_la"
	label_la_lb = "_la_lb"
	
	experiment_blob_all = {}

	experiment_blob = {}
	long_prefix = prefix_export + unique_title + '_f' + str(fold_id) + classifier_type

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	Y_train_A 	= get_lA(X_train_AB, Y_train_AB, classifier_type)
	Y_test_A 	= get_lA(X_test_AB, Y_test_AB, classifier_type)

	Y_train_B 	= get_lB(X_train_AB, Y_train_AB, classifier_type)
	Y_test_B 	= get_lB(X_test_AB, Y_test_AB, classifier_type)

	# export_confusion_matrix(Y_train_A, Y_train_B, exp_batch_id, classifier_type, "label_to_label", fold_id)

	svm_la_lb = classifier_train(Y_train_A, Y_train_B, classifier_type)
	result_la_lb = classifier_test(svm_la_lb, Y_test_A, Y_test_B, classifier_type)
	export_result(result_la_lb, long_prefix, label_la_lb)

	svm_lb_la = classifier_train(Y_train_B, Y_train_A, classifier_type)
	result_lb_la = classifier_test(svm_lb_la, Y_test_B, Y_test_A, classifier_type)
	export_result(result_lb_la, long_prefix, label_lb_la)

def experiment_pose_vs_poseauxlabel(fold_id, input_set, unique_title, classifier_type, exp_batch_id, grouping_type):
	print("Experiment: pose_vs_poseauxlabel")
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

	if classifier_type != CLASSIFIER_LSTM:
		svm_alb_a = classifier_train(X_train_AlB, Y_train_A, classifier_type)
		result_alb_a = classifier_test(svm_alb_a, X_test_AlB, Y_test_A, classifier_type)
		export_result(result_alb_a, long_prefix, label_alb_a)

	svm_bla_b = classifier_train(X_train_BlA, Y_train_B, classifier_type)
	result_bla_b = classifier_test(svm_bla_b, X_test_BlA, Y_test_B, classifier_type)
	export_result(result_bla_b, long_prefix, label_bla_b)


def experiment_duo_vs_solo(fold_id, input_set, unique_title, classifier_type, exp_batch_id, grouping_type):
	print("Experiment: Duo vs Solo")

	prefix_export = 'results/' + exp_batch_id

	label_a_a = "_a_a"
	label_b_b = "_b_b"

	label_ab_a = "_ab_a"
	label_ab_b = "_ab_b"

	experiment_blob_all = {}

	long_prefix = prefix_export + unique_title + '_f' + str(fold_id) + classifier_type

	experiment_blob = {}

	X_train_AB, X_test_AB, Y_train_AB, Y_test_AB = unpack_dict(input_set)

	# export_result(Y_train_AB, 	prefix_export, 'Ytruetrain_f' + str(fold_id))
	# export_result(Y_test_AB, 	prefix_export, 'Ytruetest_f' + str(fold_id))

	X_train_A, Y_train_A 	= get_A(X_train_AB, Y_train_AB, classifier_type)
	X_test_A, Y_test_A 		= get_A(X_test_AB, Y_test_AB, classifier_type)

	X_train_B, Y_train_B 	= get_B(X_train_AB, Y_train_AB, classifier_type)
	X_test_B, Y_test_B 		= get_B(X_test_AB, Y_test_AB, classifier_type)

	print("a_a")
	svm_a_a = classifier_train(X_train_A, Y_train_A, classifier_type)
	result_a_a = classifier_test(svm_a_a, X_test_A, Y_test_A, classifier_type)
	export_result(result_a_a, long_prefix, label_a_a)

	print("b_b")
	svm_b_b = classifier_train(X_train_B, Y_train_B, classifier_type)
	result_b_b = classifier_test(svm_b_b, X_test_B, Y_test_B, classifier_type)
	export_result(result_b_b, long_prefix, label_b_b)

	print("ab_a")
	svm_ab_a = classifier_train(X_train_AB, Y_train_A, classifier_type)
	result_ab_a = classifier_test(svm_ab_a, X_test_AB, Y_test_A, classifier_type)
	export_result(result_ab_a, long_prefix, label_ab_a)

	print("ab_b")
	svm_ab_b = classifier_train(X_train_AB, Y_train_B, classifier_type)
	result_ab_b = classifier_test(svm_ab_b, X_test_AB, Y_test_B, classifier_type)
	export_result(result_ab_b, long_prefix, label_ab_b)


# Given a file location, return the four test/train vectors
def import_vectors(unique_title, prefix, fold_id, grouping_type):
	entries = os.listdir(prefix)
	# print(entries)
	# print(prefix)
	# get all the input files from this video
	entries = list(filter(lambda k: unique_title in k, entries))
	# print(entries)
	entries = list(filter(lambda k: grouping_type in k, entries))
	entries = list(filter(lambda k: 'total' in k, entries))
	# print(entries)


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
		print(X_test_label)
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

def get_stateless_vectors(folds, unique_title, exp_batch_id, grouping_type, seed=42):
	# These variables are set for a given import
	# different seeds, different values
	
	prefix = '../vector-preparation/output-vectors/stateless/'
	prefix_export = 'results/' + exp_batch_id
	
	n_features = 2*3*25

	exp_sets = {}
	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
	for fold_id in range(folds):
		print("Geting stateless data for fold " + str(fold_id))
		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id, grouping_type)

		print(X_test.shape)
		print(X_train.shape)

		X_test 		= X_test.reshape(X_test.shape[0], n_features)
		X_train 	= X_train.reshape(X_train.shape[0], n_features)

		print(X_test.shape)
		print(X_train.shape)

		export_result(Y_train, 	prefix_export, 'Ytruetrain_stateless' + grouping_type + '_f' + str(fold_id))
		export_result(Y_test, 	prefix_export, 'Ytruetest_stateless' + grouping_type + '_f' + str(fold_id))

		exp_sets[fold_id] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

	print()
	return exp_sets

def get_temporal_vectors(folds, unique_title, exp_batch_id, grouping_type, seed=42):
	# These variables are set for a given import
	# different seeds, different values
	prefix = '../vector-preparation/output-vectors/temporal/'
	
	dimension_X_row = (128,2*25*3)
	window_size = 128
	n_features = 2*25*3


	exp_sets = {}
	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
	for fold_id in range(folds):
		print("Getting temporal data for fold " + str(fold_id))

		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id, grouping_type)

		X_test 		= X_test.reshape(X_test.shape[0], window_size, n_features)
		X_train 	= X_train.reshape(X_train.shape[0], window_size, n_features) # dimension_X_row)

		export_result(Y_train, 	'results/' + exp_batch_id, 'Ytruetrain_temporal' + grouping_type + '_f' + str(fold_id))
		export_result(Y_test, 	'results/' + exp_batch_id, 'Ytruetest_temporal' + grouping_type + '_f' + str(fold_id))	

		exp_sets[fold_id] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

	print()
	return exp_sets


def run_experiments():
	num_folds = 1
	unique_title = 's111_'
	exp_batch_id = 14
	exp_batch_id = "exp_" + str(exp_batch_id) + "/"
	prefix_export = 'results/' + exp_batch_id

	try:
		os.mkdir(prefix_export)  
	except OSError as error:  
		print("This directory already exists; do you want a fresh experiment ID?")

	grouping_type = GROUPING_RANDOM

	all_svm_vectors 		= get_stateless_vectors(num_folds, unique_title, exp_batch_id, grouping_type)
	all_temporal_vectors 	= get_temporal_vectors(num_folds, unique_title, exp_batch_id, grouping_type)
	
	print("Time elapsed: " + str("hi"))

	# exp_types = [CLASSIFIER_KNN3, CLASSIFIER_DecisionTree, CLASSIFIER_ADABOOST, CLASSIFIER_KNN5, CLASSIFIER_KNN9]#, CLASSIFIER_LSTM, CLASSIFIER_LSTM_BIGGER, CLASSIFIER_LSTM_BIGGEST]#, CLASSIFIER_SGDC, CLASSIFIER_SVM]
	# exp_types = [CLASSIFIER_DecisionTree, CLASSIFIER_KNN3, CLASSIFIER_KNN5, CLASSIFIER_KNN9, CLASSIFIER_ADABOOST, CLASSIFIER_SVM]
	# exp_types = [CLASSIFIER_LSTM]
	exp_types = [CLASSIFIER_LSTM, CLASSIFIER_LSTM_BIGGER, CLASSIFIER_LSTM_BIGGEST, CLASSIFIER_DecisionTree]

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

			if classifier_type != CLASSIFIER_LSTM:
				experiment_duo_vs_solo(fold_id, fold_data, unique_title, classifier_type, exp_batch_id, grouping_type)
				experiment_pose_vs_poseauxlabel(fold_id, fold_data, unique_title, classifier_type, exp_batch_id, grouping_type)
			experiment_swapped_poses(fold_id, fold_data, unique_title, classifier_type, exp_batch_id, grouping_type)

			# experiment_label_to_label(fold_id, fold_data, unique_title, classifier_type, exp_batch_id, grouping_type)



def main():
	run_experiments()


main()





