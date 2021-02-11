import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report

LABEL_ADABOOST = '_adaboost'
LABEL_SGDC = '_sgdc'
LABEL_SVM = '_svm'
LABEL_KNN9 = '_kNN9'
LABEL_KNN5 = '_kNN5'
LABEL_KNN3 = '_kNN3'
LABEL_DecisionTree = '_dectree'

activity_labels = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']


def analyze_results(Ytrue_train, Ytrue_test, results_dict, exp_batch_id, classifier_type):
	Y_correct_a = Ytrue_test[:,:1]
	Y_correct_b = Ytrue_test[:,1:]

	sub_experiments = list(results_dict.keys())
	if '' in sub_experiments:
		sub_experiments.remove('')
	if 'results' in sub_experiments:
		sub_experiments.remove('results')

	for subexp_label in sub_experiments:

		Y_test = results_dict[subexp_label]

		if '_a' in subexp_label:
			Y_correct = Y_correct_a
		elif '_b' in subexp_label:
			Y_correct = Y_correct_b
		elif '_la' in subexp_label:
			Y_correct = Y_correct_a
		elif '_lb' in subexp_label:
			Y_correct = Y_correct_b
		else:
			print("Error, no correct set found for " + subexp_label)

		Y_correct = Y_correct.astype(int).ravel()
		print("Analyzing: " + classifier_type + "\t on " + subexp_label)
		# labels=activity_labels
		report = classification_report(Y_correct, Y_test, output_dict=True)
		df = pd.DataFrame(report).T


		save_location = "results-analysis/" + exp_batch_id + classifier_type[1:] + "_" + subexp_label
		df.to_csv(save_location + ".csv")



def import_results(unique_title, prefix, fold_id, classifier_type):
	result_dict = {}

	# Given a file location, return the four test/train vectors
	entries = os.listdir(prefix)
	
	# get all the input files from this video
	entries = list(filter(lambda k: classifier_type in k, entries))
	fold_group = "f" + str(fold_id) + "_"
	fold_entries = list(filter(lambda k: fold_group in k, entries))
	
	# test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
	# train 	= list(filter(lambda k: 'train' in k, fold_entries))

	# Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
	# Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))

	for item in fold_entries:
		start = item.find(classifier_type) + len(classifier_type) + len("_")
		label = item[start : item.rfind("_")]
		
		Y_test 		= pickle.load(open(prefix + item, 'rb'))
		result_dict[label] = Y_test
	
	return result_dict

def import_original_vectors(unique_title, prefix, fold_id):
	# Given a file location, return the four test/train vectors
	entries = os.listdir(prefix)

	# get all the input files from this video
	# true is the keyword for the correct vectors
	entries = list(filter(lambda k: 'true' in k, entries))
	
	fold_group = "f" + str(fold_id) + "_"
	fold_entries = list(filter(lambda k: fold_group in k, entries))
	
	test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
	train 	= list(filter(lambda k: 'train' in k, fold_entries))
	
	if len(test) > 1  or len(train) > 1:
		print("Error in import: multiple matching batches for this unique key")
		print("Please provide a key that aligns with only one of the following")


	Y_test_label 	= test[0]
	Y_train_label 	= train[0]

	Y_test 		= pickle.load(open(prefix + Y_test_label, 'rb'))
	Y_train 	= pickle.load(open(prefix + Y_train_label, 'rb'))
	
	return Y_train, Y_test

def main():
	folds = 5
	unique_title = 'total_forsvm_s42_'

	LABEL_ADABOOST = '_adaboost'
	LABEL_SGDC = '_sgdc'
	LABEL_SVM = '_svm'
	LABEL_KNN9 = '_kNN9'
	LABEL_KNN5 = '_kNN5'
	LABEL_KNN3 = '_kNN3'
	LABEL_DecisionTree = '_dectree'

	experiment_titles = [LABEL_DecisionTree, LABEL_KNN9, LABEL_ADABOOST, LABEL_KNN3, LABEL_KNN5, LABEL_SGDC, LABEL_SVM]
	# experiment_titles = [LABEL_SVM]
	
	exp_batch_id = 1
	exp_batch_id = "exp_" + str(exp_batch_id) + "/"
	prefix_import = 'results/' + exp_batch_id
	fold_id = 5

	# Import set of results
	Ytrue_train, Ytrue_test = import_original_vectors(unique_title, prefix_import, fold_id)

	for classifier_type in experiment_titles:
		print("Getting results for " + classifier_type)
		results_dict = import_results(unique_title, prefix_import, fold_id, classifier_type)

		# Note that basedon the label suffix, the correct train and test files will be pulled
		analyze_results(Ytrue_train, Ytrue_test, results_dict, exp_batch_id, classifier_type)



	# Compare with appropriate Y values
	# Accuracy and stats overall
	# accuracy and stats per category



main()