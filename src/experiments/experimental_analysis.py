LABEL_ADABOOST = '_adaboost'
LABEL_SGDC = '_sgdc'
LABEL_SVM = '_svm'
LABEL_KNN9 = '_kNN9'
LABEL_KNN5 = '_kNN5'
LABEL_KNN3 = '_kNN3'
LABEL_DecisionTree = '_dectree'


# entries = os.listdir(prefix)
# entries = list(filter(lambda k: unique_title in k, entries))

# fold_group = "f" + str(fold_id) + "_"
# fold_entries = list(filter(lambda k: fold_group in k, entries))

# test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
# train 	= list(filter(lambda k: 'train' in k, fold_entries))

# X_test_label 	= list(filter(lambda k: '_X' 	in k, test))
# X_train_label 	= list(filter(lambda k: '_X' 	in k, train))
# Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
# Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))


def import_original_vectors(unique_title, prefix, fold_id):
	# Given a file location, return the four test/train vectors
	entries = os.listdir(prefix)

	# get all the input files from this video
	entries = list(filter(lambda k: unique_title in k, entries))

	fold_group = "f" + str(fold_id) + "_"
	fold_entries = list(filter(lambda k: fold_group in k, entries))

	test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
	train 	= list(filter(lambda k: 'train' in k, fold_entries))

	Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
	Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))

	if len(Y_test_label) > 1  or len(Y_train_label) > 1:
		print("Error in import: multiple matching batches for this unique key")
		print("Please provide a key that aligns with only one of the following")
		print(X_test)

	Y_test_label 	= Y_test_label[0]
	Y_train_label 	= Y_train_label[0]

	Y_test 		= pickle.load(open(prefix + Y_test_label, 'rb'))
	Y_train 	= pickle.load(open(prefix + Y_train_label, 'rb'))
	
	return Y_train, Y_test

def import_original_vectors(unique_title, prefix, fold_id):
	# Given a file location, return the four test/train vectors
	entries = os.listdir(prefix)

	# get all the input files from this video
	entries = list(filter(lambda k: unique_title in k, entries))

	fold_group = "f" + str(fold_id) + "_"
	fold_entries = list(filter(lambda k: fold_group in k, entries))

	test 	= list(filter(lambda k: 'test' 	in k, fold_entries))
	train 	= list(filter(lambda k: 'train' in k, fold_entries))

	Y_test_label 	= list(filter(lambda k: '_Y' 	in k, test))
	Y_train_label 	= list(filter(lambda k: '_Y' 	in k, train))

	if len(Y_test_label) > 1  or len(Y_train_label) > 1:
		print("Error in import: multiple matching batches for this unique key")
		print("Please provide a key that aligns with only one of the following")
		print(X_test)

	Y_test_label 	= Y_test_label[0]
	Y_train_label 	= Y_train_label[0]

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

	experiment_titles = [LABEL_DecisionTree, LABEL_KNN9, LABEL_ADABOOST]

	# Import set of results
	Ytrue_train, Ytrue_test = import_original_vectors(unique_title, prefix, fold_id)
	for i in experiment_titles:
		Yexp_train, Yexp_test = import_results(unique_title, prefix, fold_id, exp_id)
		analyze_resutls(Ytrue_train, Ytrue_test, Yexp_train, Y_exp_test)



	# Compare with appropriate Y values
	# Accuracy and stats overall
	# accuracy and stats per category



main()