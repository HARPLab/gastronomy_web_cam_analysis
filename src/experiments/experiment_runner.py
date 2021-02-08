import os
import pickle

# experiment runner
# 	for each type of trial
# 		
#		looks for appropriate vectors from to-vectors for the calls on vector train to test 
# 		

# class Experiment:
#     def __init__(self):
# 		pass

def experiment_duo_vs_solo_just_label_svm(all_svm_vectors):
	pass


def experiment_duo_vs_solo_svm(vector_dict):

	for key_group in vector_dict.keys():
		input_set = vector_dict[key_group]

		X_test_AB 	= input_set['xtest']
		X_train_AB 	= input_set['xtrain']
		Y_test_AB 	= input_set['ytest']
		Y_train_AB 	= input_set['ytrain']

		og_dim_X = X_test_AB.shape
		og_dim_Y = Y_test_AB.shape

		half_dim_X = int(og_dim_X[1] / 2)
		half_dim_Y = int(og_dim_Y[1] / 2)

		X_test_A 	= X_test_AB[:, 	:half_dim_X, :]
		X_train_A 	= X_train_AB[:, :half_dim_X, :]
		Y_test_A 	= Y_test_AB[:, 	:half_dim_Y]
		Y_train_A 	= Y_train_AB[:, :half_dim_Y]

		X_test_B 	= X_test_AB[:, 	half_dim_X:, :]
		X_train_B 	= X_train_AB[:, half_dim_X:, :]
		Y_test_B 	= Y_test_AB[:, 	half_dim_Y:]
		Y_train_B 	= Y_train_AB[:, half_dim_Y:]

		print()
		svm_train_test(X_train_AB, Y_test_AB)

		# run the experiment


		# report the results
		# store for future inspection


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

	return X_train, X_test, Y_train, Y_test


# Returns each set of 
def get_svm_vectors(seed=42):
	# These variables are set for a given import
	# different seeds, different values
	folds = 5
	unique_title = 'total_forsvm_s42_'
	prefix = '../vector-preparation/output-vectors/for_svm/'
	
	exp_sets = {}
	# exp_sets['all'] = import_vectors(unique_title, prefix, -1)
	for fold_id in range(folds):
		print("Geting svn data for fold " + str(fold_id))
		X_train, X_test, Y_train, Y_test = import_vectors(unique_title, prefix, fold_id)

		exp_sets[folds] = {'xtest': X_test, 'xtrain': X_train, 'ytest': Y_test, 'ytrain': Y_train}

	return exp_sets

def run_experiments():
	all_svm_vectors = get_svm_vectors()

	experiment_duo_vs_solo_svm(all_svm_vectors)

	# experiment_duo_vs_solo_lstm()

	experiment_duo_vs_solo_just_label_svm(all_svm_vectors)
	# experiment_duo_vs_solo_just_label_lstm()

def main():
    run_experiments()


main()





