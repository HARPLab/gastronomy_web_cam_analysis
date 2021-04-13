import os
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import matplotlib.mlab as mlab
from dictdiffer import diff
import seaborn as sn
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import qchecks
import arconsts
import experiment_io

LABEL_ADABOOST 	= arconsts.CLASSIFIER_ADABOOST
LABEL_SGDC 		= arconsts.CLASSIFIER_SGDC
LABEL_SVM 		= arconsts.CLASSIFIER_SVM
LABEL_KNN9 		= arconsts.CLASSIFIER_KNN9
LABEL_KNN5 		= arconsts.CLASSIFIER_KNN5
LABEL_KNN3 		= arconsts.CLASSIFIER_KNN3
LABEL_DecisionTree = arconsts.CLASSIFIER_DecisionTree

LABEL_LSTM 			= arconsts.CLASSIFIER_LSTM
LABEL_LSTM_BIGGER 	= arconsts.CLASSIFIER_LSTM_BIGGER
LABEL_LSTM_BIGGEST 	= arconsts.CLASSIFIER_LSTM_BIGGEST
LABEL_LSTM_TINY		= arconsts.CLASSIFIER_LSTM_TINY
LABEL_CRF 			= arconsts.CLASSIFIER_CRF

GROUPING_MEALWISE 	= arconsts.GROUPING_MEALWISE
GROUPING_RANDOM 	= arconsts.GROUPING_RANDOM

BATCH_ID_STATELESS 	= arconsts.BATCH_ID_STATELESS
BATCH_ID_TEMPORAL 	= arconsts.BATCH_ID_TEMPORAL
BATCH_ID_MEALWISE_STATELESS = arconsts.BATCH_ID_MEALWISE_STATELESS
BATCH_ID_MEALWISE_TEMPORAL 	= arconsts.BATCH_ID_MEALWISE_TEMPORAL


LABELS_TEMPORAL 	= arconsts.CLASSIFIERS_TEMPORAL
LABELS_STATELESS 	= arconsts.CLASSIFIERS_STATELESS

activity_labels 	= arconsts.activity_labels

LSTM_NUM_LABELS 	= len(activity_labels)
CONST_NUM_POINTS 	= arconsts.CONST_NUM_POINTS
CONST_NUM_SUBPOINTS = arconsts.CONST_NUM_SUBPOINTS
CONST_NUM_LABEL 	= arconsts.CONST_NUM_LABEL


LABEL_A_A = arconsts.LABEL_A_A
LABEL_B_B = arconsts.LABEL_B_B
LABEL_AB_B = arconsts.LABEL_AB_B
LABEL_AB_A = arconsts.LABEL_AB_A

LABEL_BLA_B = arconsts.LABEL_BLA_B
LABEL_ALB_A = arconsts.LABEL_ALB_A

LABEL_LA_LB = arconsts.LABEL_LA_LB
LABEL_LB_LA = arconsts.LABEL_LB_LA

LABEL_B_A = arconsts.LABEL_B_A
LABEL_A_B = arconsts.LABEL_A_B

CSV_ORDER = arconsts.CSV_ORDER


LABEL_RANDOM_CHANCE_UNIFORM_A = 'random_chance_uniform_a'
LABEL_RANDOM_CHANCE_UNIFORM_B = 'random_chance_uniform_b'

LABEL_RANDOM_CHANCE_CLASSCHANCE_A = 'random_chance_distributions_a'
LABEL_RANDOM_CHANCE_CLASSCHANCE_B = 'random_chance_distributions_b'

LABEL_RANDOM_CHANCE_MOST_COMMON_A = 'random_chance_most_common_a'
LABEL_RANDOM_CHANCE_MOST_COMMON_B = 'random_chance_most_common_b'

GENERATED_BENCHMARKS		= [LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_RANDOM_CHANCE_CLASSCHANCE_B]
# [LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_RANDOM_CHANCE_CLASSCHANCE_B]

HYPOTH_VANILLA_RATE 			= 'hypothesis_differences_between_people'
HYPOTH_SOLO_DUO_POSES 			= 'hypothesis_duopose_to_target'
HYPOTH_SOLO_DUO_POSELABEL 		= 'hypothesis_solopose_auxlabel_vs_duo'
HYPOTH_AUXPOSE_TO_TARGET		= 'hypothesis_auxpose_to_target'
HYPOTH_AUX_LABEL_TO_TARGET 		= 'hypothesis_auxlabel_to_target'

HYPOTH_FALLOUT					= 'hypothesis_missing_data'
HYPOTH_BODYPART_FALLOUT			= 'hypothesis_bodypart_fallout'

ANALYSIS_OVERALL_ACCURACY		= 'analysis:overall-accuracy'
ANALYSIS_CLASS_PERFORMANCE 		= 'analysis:class_performance'
ANALYSIS_MCC 					= 'analysis:mcc'

default_analysis = [ANALYSIS_OVERALL_ACCURACY, ANALYSIS_CLASS_PERFORMANCE, ANALYSIS_MCC]

COMPARISONS = arconsts.COMPARISONS


# ACT_NONE 			= 0
# ACT_AWAY_FROM_TABLE = 1
# ACT_IDLE			= 2
# ACT_EATING			= 3
# ACT_TALKING			= 4
# ACT_WAITER			= 5
# ACT_LOOKING_WINDOW	= 6
# ACT_READING_BILL	= 7
# ACT_READING_MENU	= 8
# ACT_PAYING_CHECK	= 9
# ACT_USING_PHONE		= 10
# ACT_OBJ_WILDCARD 	= 11
# ACT_STANDING		= 12


# activity_labels = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
# 					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
# 					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
# 					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']
activity_labels = ['NONE', 'away-from-table', 'idle', 'eating', 'talking', 'talk:waiter', 'looking:window', 
					'reading:bill', 'reading:menu', 'paying:check', 'using:phone', 'obj:wildcard', 'standing']

class Hypothesis:
	comparison_groups = arconsts.COMPARISONS

	def __init__(self, hypothesis_label):
		self.hypothesis_label = hypothesis_label
		comparison_groups 	= []

		if hypothesis_label == HYPOTH_SOLO_DUO_POSES:
			comparison_groups.append([LABEL_A_A, LABEL_AB_A])
			comparison_groups.append([LABEL_B_B, LABEL_AB_B])
			self.analysis_types = default_analysis

		elif hypothesis_label == HYPOTH_SOLO_DUO_POSELABEL:
			self.analysis_types = default_analysis

			comparison_groups.append([LABEL_A_A, LABEL_ALB_A])
			comparison_groups.append([LABEL_B_B, LABEL_BLA_B])
			
		elif hypothesis_label == HYPOTH_AUXPOSE_TO_TARGET:
			self.analysis_types = default_analysis

			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_B_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_A_B])
			
			comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_B_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_B, LABEL_A_B])
		
		elif hypothesis_label == HYPOTH_AUX_LABEL_TO_TARGET:
			self.analysis_types = default_analysis

			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_B_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_A_B])
			comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_LB_LA])
			comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_B, LABEL_LA_LB])

		elif hypothesis_label == HYPOTH_VANILLA_RATE:
			self.analysis_types = default_analysis

			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_A, LABEL_A_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_UNIFORM_B, LABEL_B_B])
			comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_A, LABEL_A_A])
			comparison_groups.append([LABEL_RANDOM_CHANCE_CLASSCHANCE_B, LABEL_B_B])
		
		else:
			print("Hypothesis not recognized!")

		self.comparison_groups = comparison_groups

	def get_comparison_groups(self):
		return self.comparison_groups

	def get_hypothesis_label(self):
		return self.hypothesis_label

	def analysis_mcc(self, first_test, first_truth, second_test, second_truth):
		mcc_first 	= accuracy_score(first_truth, first_test)
		mcc_second = accuracy_score(second_truth, second_test)
		delta = mcc_second - mcc_first
		output_string = ""

		output_string += str(mcc_first) + " ::: " + str(mcc_second) + "\n"
		output_string += "MCC del: " + str(delta) + "\n"

		return output_string



	def analysis_compare_overall_accuracy(self, first_test, first_truth, second_test, second_truth):
		accuracy_first 	= accuracy_score(first_truth, first_test)
		accuracy_second = accuracy_score(second_truth, second_test)
		delta = accuracy_second - accuracy_first
		output_string = ""

		# del = \u0394
		output_string += str(accuracy_first) + " ::: " + str(accuracy_second) + "\n"
		output_string += "Accuracy del: " + str(delta) + "\n"

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

	def get_per_class_mcc(self, true, predicted):
		per_class_mcc = {}
		for idx, cls in enumerate(activity_labels):
			per_class_mcc[cls] = matthews_corrcoef(true, predicted)

		return per_class_mcc

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

		output_string += "---ACCURACY--- \n"
		output_string += "first: \t" + str(per_class1) + "\n\n"
		output_string += "second: \t" + str(per_class2) + "\n\n"
		output_string += "del:\t" + str(delta_performance) + "\n"

		per_class1 = self.get_per_class_mcc(first_truth, first_test)
		per_class2 = self.get_per_class_mcc(second_truth, second_test)
		delta_performance = {x: per_class2[x] - per_class1[x] for x in per_class2.keys() if x in per_class1.keys()}

		output_string += "---MCC--- \n"
		output_string += "first: \t" + str(per_class1) + "\n\n"
		output_string += "second: \t" + str(per_class2) + "\n\n"
		output_string += "del:\t" + str(delta_performance) + "\n"

		# top 5
		# missing classes
		# most improved


		return output_string


	def get_generated_benchmark(self, label, subexps_dict):
		# find the correct output dimensions
		key_pool = subexps_dict.keys()

		if label == LABEL_RANDOM_CHANCE_CLASSCHANCE_A or label == LABEL_RANDOM_CHANCE_UNIFORM_A:
			res = {item for item in key_pool if item.endswith('_a')}

		elif label == LABEL_RANDOM_CHANCE_CLASSCHANCE_B or label == LABEL_RANDOM_CHANCE_UNIFORM_B:
			res = {item for item in key_pool if item.endswith('_b')}

		if len(res) < 1:
			print("Unable to match array for assesment " + label)

		input_template = list(res)[0]
		key, e_result, e_truth = subexps_dict[input_template][0]
		truth_array 	= e_truth
		matching_shape 	= e_truth.shape
		output_array = None
		random.seed(42)

		distribution = truth_array

		if label == LABEL_RANDOM_CHANCE_CLASSCHANCE_A:
			values, counts = np.unique(distribution, return_counts=True)
			majority_label = np.argmax(counts)
			output_array = np.full(matching_shape, majority_label)
		elif label == LABEL_RANDOM_CHANCE_CLASSCHANCE_B:
			values, counts = np.unique(distribution, return_counts=True)
			majority_label = np.argmax(counts)
			output_array = np.full(matching_shape, majority_label)

		elif label == LABEL_RANDOM_CHANCE_UNIFORM_A:
			output_array = np.random.randint(len(activity_labels), size=matching_shape, dtype=int)
		elif label == LABEL_RANDOM_CHANCE_UNIFORM_B:
			output_array = np.random.randint(len(activity_labels), size=matching_shape, dtype=int)

		return output_array, truth_array


	def get_experimental_inputs_from_label(self, label, subexps_dict):
		if label in GENERATED_BENCHMARKS:
			test, truth = self.get_generated_benchmark(label, subexps_dict)
		else:
			key, test, truth = subexps_dict[label][0]
		return test, truth

	def verify_experimental_input_available(self, label, all_results_dict):
		subexp_dict = experiment_io.get_subexp_labeled_dict(all_results_dict)
		key_pool = subexp_dict.keys()

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
		subexp_dict = experiment_io.get_subexp_labeled_dict(all_results_dict)
		key_pool = subexp_dict.keys()

		output_string = ""
		for pair in self.comparison_groups:
			first, second = pair
			output_string += "Comparing {" + first + "} VS {" + second + "} \n"

			is_first, outstr_first 		= self.verify_experimental_input_available(first, all_results_dict)
			is_second, outstr_second 	= self.verify_experimental_input_available(second, all_results_dict)

			if not is_first or not is_second:
				output_string += "FAIL on TESTING " + self.get_hypothesis_label() + "\n"
				# print("MISSING DATA for TEST -> " + self.get_hypothesis_label())
				output_string += outstr_first
				output_string += outstr_second
				# print(outstr_first + outstr_second)
				continue

			first_test, first_truth 	= self.get_experimental_inputs_from_label(first, subexp_dict)
			second_test, second_truth 	= self.get_experimental_inputs_from_label(second, subexp_dict)			

			if ANALYSIS_OVERALL_ACCURACY in self.analysis_types:
				# print("Comparing overall accuracy")
				output_string += self.analysis_compare_overall_accuracy(first_test, first_truth, second_test, second_truth) + "\n"

			if ANALYSIS_CLASS_PERFORMANCE in self.analysis_types:
				# print("Comparing class performance")
				output_string += self.analysis_class_performance(first_test, first_truth, second_test, second_truth) + "\n"

			if ANALYSIS_MCC in self.analysis_types:
				# print("Comparing mcc values")
				output_string += self.analysis_mcc(first_test, first_truth, second_test, second_truth) + "\n"




		# print("Analysis is: ")
		# print(output_string)
		return output_string

def get_key_pool(all_results_dict):
	key_pool = list(all_results_dict.keys())
	key_pool = [k[0] for k in key_pool]
	return key_pool

def get_random_chance_benchmark_uniform():
	pass

def get_random_chance_benchmark_distribution():
	pass

def get_comparison(cg, key, all_results_dict):
	value = float('NaN')
	test = all_results_dict[key][0]
	true = all_results_dict[key][1]

	if cg == arconsts.COMPARISON_TABLE_ACCURACY:
		value = accuracy_score(true, test)
	elif cg == arconsts.COMPARISON_TABLE_MCC:
		value = matthews_corrcoef(true, test)
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
		for key in all_results_dict.keys():
			value = get_comparison(cg, key, all_results_dict)
			this_cg[key] = value

		comparison_row_dict[cg] = this_cg
	
	return comparison_row_dict, output_string

# align with experimental runner
def get_single_vector_of_multiclass_result(result):
	decoded_result = result.argmax(axis=1)
	return decoded_result
	
# Run on each individual pair of results in directory
def analyze_results(Y_true, Y_result, key, hypothesis_list):
	results_lookup = {}
	# print(results_dict.keys())
	print("Loading and generating classification report for: {", end='')
	# for subexp_label in sub_experiments:

	Y_test = Y_result
	Y_test = qchecks.multiclass_to_labels(Y_test)

	Y_correct = Y_true.astype(int).ravel()
	Y_test = Y_test.astype(int).ravel()
	
	exp_batch_id, classifier_type, feature_type, grouping_type, fold_id, seed, input_label, output_label = key
	subexp_label 	= experiment_io.get_subexp_label(key)
	print("_".join(key) + "}")

	num_outputs 	= qchecks.get_num_outputs(Y_correct)
	output_labels 	= qchecks.get_output_set(Y_correct)
	
	report = classification_report(Y_correct, Y_test, output_dict=True, labels=range(num_outputs), target_names=output_labels)
	experiment_io.export_raw_classification_report(report, key)	
	experiment_io.export_confusion_matrix(Y_correct, Y_test, key, output_labels)
	return report

def main():
	num_folds = 1
	seed = 56
	unique_title = '_s111_'

	print("Beginning analysis")

	exp_batch_id = 35 
	exp_batch_id = "exp_" + str(exp_batch_id) + "_server"+ "/"
	prefix_import = 'results/' + exp_batch_id
	prefix_export = 'results-analysis/' + exp_batch_id
	
	try:
		os.mkdir(prefix_export)
		print("Please add a readme with experimental notes!")
	except OSError as error:  
		print("This directory already exists; do you want a fresh experiment ID?")

	# Import set of results
	hypothesis_list = []
	hypothesis_list.append(Hypothesis(HYPOTH_AUX_LABEL_TO_TARGET))
	hypothesis_list.append(Hypothesis(HYPOTH_SOLO_DUO_POSELABEL))
	hypothesis_list.append(Hypothesis(HYPOTH_SOLO_DUO_POSES))
	hypothesis_list.append(Hypothesis(HYPOTH_AUXPOSE_TO_TARGET))

	# Pair all hypotheses true and test for analysis
	all_results_dict, filenames_dict = experiment_io.find_all_results(prefix_import)
	all_comparisons 	= {}

	# for each of these pairs, 
	for key in all_results_dict.keys():
		print(key)
		# for possible combo
		# look if it exists
		Y_result, Y_true = all_results_dict[key]

		# Judge individual pair for accuracy, etc
		analyze_results(Y_true, Y_result, key, hypothesis_list)


	comparison_dict, hypothesis_log = meta_analysis_from_classifier_data(all_results_dict, hypothesis_list)
	# Now that all values are logged, attempt hypothesis analysis
	experiment_io.export_hypothesis_analysis_report(hypothesis_log, prefix_export)
	# Now that all data is collected export cross-group comparison tables
	experiment_io.export_all_comparisons(comparison_dict, all_results_dict, prefix_export)

main()