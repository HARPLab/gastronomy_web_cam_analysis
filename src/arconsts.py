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
CLASSIFIER_LSTM_TINY	= '_lstm_tiny'
CLASSIFIER_CRF = '_crf'

GROUPING_MEALWISE = '_g-mw'
GROUPING_RANDOM = "_g-rand"

BATCH_ID_STATELESS 	= '_stateless'
BATCH_ID_TEMPORAL 	= '_temporal'
BATCH_ID_MEALWISE_STATELESS 	= '_mwstateless'
BATCH_ID_MEALWISE_TEMPORAL 	= '_mwtemporal'


CLASSIFIERS_TEMPORAL = [CLASSIFIER_LSTM, CLASSIFIER_LSTM_BIGGER, CLASSIFIER_LSTM_BIGGEST, CLASSIFIER_CRF, CLASSIFIER_LSTM_TINY]
CLASSIFIERS_STATELESS = [CLASSIFIER_KNN3, CLASSIFIER_KNN5, CLASSIFIER_KNN9, CLASSIFIER_SVM, CLASSIFIER_SGDC, CLASSIFIER_ADABOOST, CLASSIFIER_DecisionTree]\

# activity_labels = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
# 					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
# 					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
# 					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']

activity_labels = ['NONE', 'away-from-table', 'idle', 'eating', 'talking', 'talk:waiter', 'looking:window', 
					'reading:bill', 'reading:menu', 'paying:check', 'using:phone', 'obj:wildcard', 'standing']

# activitydict = {0: 'NONE', 1: 'away-from-table', 2: 'idle', 3: 'eating', 4: 'talking', 5:'talking:waiter', 6: 'looking:window', 
# 	7: 'reading:bill', 8: 'reading:menu', 9: 'paying:check', 10: 'using:phone', 11: 'using:objs', 12: 'standing'}

CONST_NUM_LABELS = len(activity_labels)
CONST_NUM_POINTS = 25
CONST_NUM_SUBPOINTS = 3
CONST_NUM_LABEL = 1