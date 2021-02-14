import os
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from dictdiffer import diff

LABEL_ADABOOST = '_adaboost'
LABEL_SGDC = '_sgdc'
LABEL_SVM = '_svm'
LABEL_KNN9 = '_kNN9'
LABEL_KNN5 = '_kNN5'
LABEL_KNN3 = '_kNN3'
LABEL_DecisionTree = '_dectree'
LABEL_LSTM = '_lstm'


LABEL_A_A = 'a_a'
LABEL_B_B = 'b_b'
LABEL_AB_B = 'ab_b'
LABEL_AB_A = 'ab_a'

LABEL_BLA_B = 'bla_b'
LABEL_ALB_A = 'alb_a'

LABEL_LA_LB = 'la_lb'
LABEL_LB_LA = 'lb_la'

LABEL_B_A = 'b_a'
LABEL_A_B = 'a_b'

CSV_ORDER = ['a_a', 'b_b', 'ab_b', 'ab_a', 'bla_b', 'alb_a', 'la_lb', 'lb_la', 'b_a', 'a_b']


LABEL_RANDOM_CHANCE_UNIFORM_A = 'random_chance_uniform_a'
LABEL_RANDOM_CHANCE_UNIFORM_B = 'random_chance_uniform_b'

LABEL_RANDOM_CHANCE_CLASSCHANCE_A = 'random_chance_distributions_a'
LABEL_RANDOM_CHANCE_CLASSCHANCE_B = 'random_chance_distributions_b'

LABEL_RANDOM_CHANCE_MOST_COMMON_A = 'random_chance_most_common_a'
LABEL_RANDOM_CHANCE_MOST_COMMON_B = 'random_chance_most_common_b'

GENERATED_BENCHMARKS		= [LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_RANDOM_CHANCE_UNIFORM_B]
# [LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_RANDOM_CHANCE_CLASSCHANCE_B]

HYPOTH_VANILLA_RATE 			= 'hypothesis_differences_between_people'
HYPOTH_SOLO_DUO_POSES 			= 'hypothesis_duopose_to_target'
HYPOTH_SOLO_DUO_POSELABEL 		= 'hypothesis_solopose_auxlabel_vs_duo'
HYPOTH_AUXPOSE_TO_TARGET		= 'hypothesis_auxpose_to_target'
HYPOTH_AUX_LABEL_TO_TARGET 		= 'hypothesis_auxlabel_to_target'

HYPOTH_FALLOUT					= 'hypothesis_missing_data'
HYPOTH_BODYPART_FALLOUT			= 'hypothesis_bodypart_fallout'

ANALYSIS_OVERALL_ACCURACY		= 'analysis:overall-accuracy'
ANALYSIS_BEST_CLASSES			= 'analysis:best_classes'
ANALYSIS_WORST_CLASSES			= 'analysis:worst_classes'
ANALYSIS_CLASS_PERFORMANCE 		= 'analysis:class_performance'

COMPARISON_TABLE_ACCURACY		= 'comparisons_accuracy'
COMPARISONS 					= [COMPARISON_TABLE_ACCURACY]


activity_labels = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']

class Hypothesis:
	comparison_groups = COMPARISONS

	def __init__(self, hypothesis_label):
		self.hypothesis_label = hypothesis_label
		comparison_groups 	= []

		if hypothesis_label == HYPOTH_SOLO_DUO_POSES:
			comparison_groups.append([LABEL_A_A, LABEL_AB_A])
			comparison_groups.append([LABEL_B_B, LABEL_AB_B])
			self.analysis_types = [ANALYSIS_OVERALL_ACCURACY, ANALYSIS_CLASS_PERFORMANCE]

		elif hypothesis_label == HYPOTH_SOLO_DUO_POSELABEL:
			self.analysis_types = [ANALYSIS_OVERALL_ACCURACY, ANALYSIS_CLASS_PERFORMANCE]

			comparison_groups.append([LABEL_A_A, LABEL_ALB_A])
			comparison_groups.append([LABEL_B_B, LABEL_BLA_B])
			
		elif hypothesis_label == HYPOTH_AUXPOSE_TO_TARGET:
			self.analysis_types = [ANALYSIS_OVERALL_ACCURACY, ANALYSIS_CLASS_PERFORMANCE]

			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_B_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_A_B])
			
			# comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_B_A])
			# comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_B, LABEL_A_B])
		
		elif hypothesis_label == HYPOTH_AUX_LABEL_TO_TARGET:
			self.analysis_types = [ANALYSIS_OVERALL_ACCURACY, ANALYSIS_CLASS_PERFORMANCE]

			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_B_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_A_B])
			# comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_LB_LA])
			# comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_B, LABEL_LA_LB])

		elif hypothesis_label == HYPOTH_VANILLA_RATE:
			self.analysis_types = [ANALYSIS_OVERALL_ACCURACY]

			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_A_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_B_B])
			# comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_A_A])
			# comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_B, LABEL_B_B])
		
		else:
			print("Hypothesis not recognized!")

		self.comparison_groups = comparison_groups

	def get_comparison_groups(self):
		return self.comparison_groups

	def get_hypothesis_label(self):
		return self.hypothesis_label

	def analysis_compare_overall_accuracy(self, first_test, first_truth, second_test, second_truth):
		accuracy_first 	= accuracy_score(first_truth, first_test)
		accuracy_second = accuracy_score(second_truth, second_test)
		delta = accuracy_second - accuracy_first
		output_string = ""

		output_string += str(accuracy_first) + " ::: " + str(accuracy_second) + "\n"
		output_string += "Accuracy \u0394: " + str(delta) + "\n"

		return output_string

	def get_per_class_accuracies(self, cm):
		per_class_accuracies = {}
		# print(cm.shape[0])

		if cm.shape[0] < len(activity_labels):
			print("Limited predictions to subset of length: " + str(cm.shape[0]))
			# return per_class_accuracies

		# https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
		# Calculate the accuracy for each one of our classes
		for idx, cls in enumerate(activity_labels):
		    # True negatives are all the samples that are not our current GT class (not the current row) 
		    # and were not predicted as the current class (not the current column)
		    true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
		    
		    # True positives are all the samples of our current GT class that were predicted as such
		    true_positives = cm[idx, idx]
		    
		    # The accuracy for the current class is ratio between correct predictions to all predictions
		    per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)

		return per_class_accuracies

	def analysis_class_performance(self, first_test, first_truth, second_test, second_truth):

		output_string = ""
		labels_range = range(len(activity_labels))
		cm1 = confusion_matrix(first_truth, first_test, labels_range)
		cm2 = confusion_matrix(second_truth, second_test, labels_range)

		per_class1 = self.get_per_class_accuracies(cm1)
		per_class2 = self.get_per_class_accuracies(cm2)


		# #Now the normalize the diagonal entries
		# cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
		# #The diagonal entries are the accuracies of each class
		# perf1 = cm1.diagonal()

		# cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
		# perf2 = cm2.diagonal()

		delta_performance = {x: per_class2[x] - per_class1[x] for x in per_class2.keys() if x in per_class1.keys()}
		
		per_class1_by_best = sorted( ((v,k) for k,v in per_class1.items()), reverse=True)
		per_class2_by_best = sorted( ((v,k) for k,v in per_class2.items()), reverse=True)
		delta_perf_by_best = sorted( ((v,k) for k,v in delta_performance.items()), reverse=True)

		output_string += "first: \t" + str(per_class1) + "\n\n"
		output_string += "second: \t" + str(per_class2) + "\n\n"
		output_string += "\u0394:\t" + str(delta_performance) + "\n"

		# top 5
		# missing classes
		# most improved


		return output_string


	def get_generated_benchmark(self, label, all_results_dict):
		# find the correct output dimensions
		key_pool = get_key_pool(all_results_dict)

		if label == LABEL_RANDOM_CHANCE_CLASSCHANCE_A or label == LABEL_RANDOM_CHANCE_UNIFORM_A:
			res = {item for item in key_pool if item.endswith('_a')}

		elif label == LABEL_RANDOM_CHANCE_CLASSCHANCE_B or label == LABEL_RANDOM_CHANCE_UNIFORM_B:
			res = {item for item in key_pool if item.endswith('_b')}

		if len(res) < 1:
			print("Unable to match array for assesment " + label)

		template_key 	= (list(res)[0], 'test')
		truth_key 	= (list(res)[0], 'truth')
		truth_array = all_results_dict[truth_key]
		template_vector = all_results_dict[template_key]
		matching_shape 	= template_vector.shape
		output_array = None
		random.seed(42)

		if label == LABEL_RANDOM_CHANCE_CLASSCHANCE_A:
			# look for suffix of "_a"
			pass
		elif label == LABEL_RANDOM_CHANCE_CLASSCHANCE_B:
			pass

		elif label == LABEL_RANDOM_CHANCE_UNIFORM_A:
			output_array = np.random.randint(len(activity_labels), size=matching_shape, dtype=int)
		elif label == LABEL_RANDOM_CHANCE_UNIFORM_B:
			output_array = np.random.randint(len(activity_labels), size=matching_shape, dtype=int)


		# todo adjust for fewer activity labels
		return output_array, truth_array


	def get_experimental_inputs_from_label(self, label, all_results_dict):
		if label in GENERATED_BENCHMARKS:
			test, truth = self.get_generated_benchmark(label, all_results_dict)
		else:
			test 	= all_results_dict[(label, 'truth')]
			truth 	= all_results_dict[(label, 'test')]

		return test, truth

	def verify_experimental_input_available(self, label, all_results_dict):
		key_pool = get_key_pool(all_results_dict)
		output_string = ""
		is_found = True

		if label not in key_pool and label not in GENERATED_BENCHMARKS:
			status = False
			if label not in key_pool:
				output_string += "Missing required label: {" +  label + "}\n"
				is_found = False
			else :
				output_string += "Missing required benchmark method: " + label + "\n"

		return is_found, output_string

	def run_analyses(self, all_results_dict):
		output_string = ""
		for pair in self.comparison_groups:
			first, second = pair
			output_string += "Comparing {" + first + "} VS {" + second + "} \n"

			is_first, outstr_first = self.verify_experimental_input_available(first, all_results_dict)
			is_second, outstr_second = self.verify_experimental_input_available(second, all_results_dict)

			if not is_first or not is_second:
				output_string += "FAIL on TESTING " + self.get_hypothesis_label() + "\n"
				print("MISSING DATA for TEST -> " + self.get_hypothesis_label())
				output_string += outstr_first
				output_string += outstr_second
				print(outstr_first + outstr_second)
				continue

			first_test, first_truth 	= self.get_experimental_inputs_from_label(first, all_results_dict)
			second_test, second_truth 	= self.get_experimental_inputs_from_label(second, all_results_dict)			

			if ANALYSIS_OVERALL_ACCURACY in self.analysis_types:
				print("Comparing overall accuracy")
				output_string += self.analysis_compare_overall_accuracy(first_test, first_truth, second_test, second_truth) + "\n"

			if ANALYSIS_CLASS_PERFORMANCE in self.analysis_types:
				print("Comparing class performance")
				output_string += self.analysis_class_performance(first_test, first_truth, second_test, second_truth) + "\n"




		print("Analysis is: ")
		print(output_string)
		return output_string

def get_key_pool(all_results_dict):
	key_pool = list(all_results_dict.keys())
	key_pool = [k[0] for k in key_pool]
	return key_pool

def get_random_chance_benchmark_uniform():
	pass

def get_random_chance_benchmark_distribution():
	pass

def export_hypothesis_analysis_report(report, exp_batch_id, classifier_type):
	print(report)

	save_location = "results-analysis/" + exp_batch_id + classifier_type[1:] + "_hypotheses.txt"
	with open(save_location, "w") as text_file:
	    text_file.write(report)

	
def export_raw_classification_report(report, exp_batch_id, classifier_type, subexp_label):
	df = pd.DataFrame(report).T
	save_location = "results-analysis/" + exp_batch_id + classifier_type[1:] + "_" + subexp_label
	df.to_csv(save_location + ".csv")


def get_comparison(cg, key, all_results_dict):
	value = float('NaN')
	if cg == COMPARISON_TABLE_ACCURACY:
		test = all_results_dict[(key, 'test')]
		true = all_results_dict[(key, 'truth')]
		value = accuracy_score(true, test)
	else:
		print("COMPARISON TYPE NOT YET SUPPORTED " + cg)
		exit()

	return value


def meta_analysis_from_classifier_data(all_results_dict, hypothesis_list):
	output_string = ""
	
	# Run pairwise hypotheses
	for hypothesis in hypothesis_list:
		output_string += hypothesis.run_analyses(all_results_dict)

	comparison_row_dict = {}
	# TODO make this less sloppily passed in
	for cg in COMPARISONS:
		this_cg = {}

		# get this stat for this pair, and add row to comparison row dict
		for key in get_key_pool(all_results_dict):
			value = get_comparison(cg, key, all_results_dict)
			this_cg[key] = value

		comparison_row_dict[cg] = this_cg
	
	return comparison_row_dict, output_string

	

def analyze_results(Ytrue_train, Ytrue_test, results_dict, exp_batch_id, classifier_type, hypothesis_list):
	Y_correct_a = Ytrue_test[:,:1]
	Y_correct_b = Ytrue_test[:,1:]

	sub_experiments = list(results_dict.keys())
	if '' in sub_experiments:
		sub_experiments.remove('')
	if 'results' in sub_experiments:
		sub_experiments.remove('results')

	results_lookup = {}
	print("Loading and generating classification report for: {", end='')
	for subexp_label in sub_experiments:

		Y_test = results_dict[subexp_label]
		# identify the correct test set from label
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
		print(subexp_label + " ", end='')
		
		results_lookup[(subexp_label, 'truth')] = Y_correct
		results_lookup[(subexp_label, 'test')] 	= Y_test

		# labels=activity_labels
		report = classification_report(Y_correct, Y_test, output_dict=True, labels=range(len(activity_labels)), target_names=activity_labels)
		export_raw_classification_report(report, exp_batch_id, classifier_type, subexp_label)

	print("}")
	print("\n\nRunning analysis for this classifier's results: ")
	return meta_analysis_from_classifier_data(results_lookup, hypothesis_list)


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

	print(prefix)

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

def export_comparisons(all_comparisons, exp_batch_id):
	# dictionary for each classifier, which contains comparisons per label
	# returns: rows of labels, columns of classifier types

	all_stat_types = COMPARISONS
	df_stacks = {}
	for comparison_type in COMPARISONS:
		df_stacks[comparison_type] = {}

	for classifier_type in all_comparisons.keys():
		for comparison_type in all_comparisons[classifier_type]:
			comparison_row = all_comparisons[classifier_type][comparison_type]
			df_stacks[comparison_type][classifier_type] = comparison_row

	for comparison_type in COMPARISONS:
		df = pd.DataFrame.from_dict(df_stacks[comparison_type])

		save_location = "results-analysis/" + exp_batch_id + '_overview_comparisons_' + comparison_type + ".csv"
		df.to_csv(save_location)

	

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

	# experiment_titles = [LABEL_DecisionTree, LABEL_KNN9, LABEL_ADABOOST, LABEL_KNN3, LABEL_KNN5, LABEL_SGDC, LABEL_SVM]
	experiment_titles = [LABEL_LSTM]
	
	exp_batch_id = 4
	exp_batch_id = "exp_" + str(exp_batch_id) + "/"
	prefix_import = 'results/' + exp_batch_id
	fold_id = 5

	try:
		os.mkdir(prefix_export)
		print("Please add a readme with experimental notes!")
	except OSError as error:  
		print("This directory already exists; do you want a fresh experiment ID?")

	# Import set of results
	Ytrue_train, Ytrue_test = import_original_vectors(unique_title, prefix_import, fold_id)
	hypothesis_list = []
	hypothesis_list.append(Hypothesis(HYPOTH_AUX_LABEL_TO_TARGET))
	hypothesis_list.append(Hypothesis(HYPOTH_SOLO_DUO_POSELABEL))
	hypothesis_list.append(Hypothesis(HYPOTH_SOLO_DUO_POSES))
	hypothesis_list.append(Hypothesis(HYPOTH_AUXPOSE_TO_TARGET))

	all_comparisons = {}
	for classifier_type in experiment_titles:
		print("Getting results for " + classifier_type)
		results_dict = import_results(unique_title, prefix_import, fold_id, classifier_type)

		# Note that basedon the label suffix, the correct train and test files will be pulled
		comparisons_to_log, results = analyze_results(Ytrue_train, Ytrue_test, results_dict, exp_batch_id, classifier_type, hypothesis_list)
		export_hypothesis_analysis_report(results, exp_batch_id, classifier_type)
		all_comparisons[classifier_type] = comparisons_to_log

	# Compare with appropriate Y values
	# Accuracy and stats overall
	# accuracy and stats per category

	export_comparisons(all_comparisons, exp_batch_id)

main()