# TO ADD THESE CONSTANTS FOR USE IN A FILE, ADD THE FOLLOWING
# import sys
# sys.path.append("..")
# import qchecks
# import arconsts


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

FEATURES_VANILLA      = '_fvanilla'
FEATURES_OFFSET       = '_foffset'
FEATURES_ANGLES       = '_fangles'
FEATURES_NO_PROB      = '_fnoprob'
FEATURES_LABELS_FULL    = '_ffull'
FEATURES_LABELS_REDUCED = '_freduced'

CONST_WINDOW_SIZE = 128

CLASSIFIERS_TEMPORAL = [CLASSIFIER_LSTM, CLASSIFIER_LSTM_BIGGER, CLASSIFIER_LSTM_BIGGEST, CLASSIFIER_CRF, CLASSIFIER_LSTM_TINY]
CLASSIFIERS_STATELESS = [CLASSIFIER_KNN3, CLASSIFIER_KNN5, CLASSIFIER_KNN9, CLASSIFIER_SVM, CLASSIFIER_SGDC, CLASSIFIER_ADABOOST, CLASSIFIER_DecisionTree]\

activity_labels_expanded = ['away-from-table', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 
					'talking:waiter', 'looking:window', 'looking:waiter', 'reading:bill', 'reading:menu',
					'paying:check', 'using:phone', 'using:napkin', 'using:purse', 'using:glasses',
					'using:wallet', 'looking:PersonA', 'looking:PersonB', 'takeoutfood', 'leaving-table', 'cleaning-up', 'NONE']

activity_labels = ['NONE', 'away-from-table', 'idle', 'eating', 'talking', 'talk:waiter', 'looking:window', 
					'reading:bill', 'reading:menu', 'paying:check', 'using:phone', 'obj:wildcard', 'standing', 'leaving', 'drinking']

activitydict = {0: 'NONE', 1: 'away-from-table', 2: 'idle', 3: 'eating', 4: 'talking', 5:'talking:waiter', 6: 'looking:window', 
	7: 'reading:bill', 8: 'reading:menu', 9: 'paying:check', 10: 'using:phone', 11: 'using:objs', 12: 'standing', 13: 'leaving', 14: 'drinking'}

activity_from_key = {0:'away-from-table', 1:'idle', 2:'eating', 3: 'drinking', 4: 'talking', 5: 'ordering', 6: 'standing',
                7: 'talking:waiter', 8: 'looking:window', 9: 'looking:waiter', 10: 'reading:bill', 11: 'reading:menu',
                12: 'paying:check', 13: 'using:phone', 14: 'using:napkin', 15: 'using:purse', 16: 'using:glasses',
                17: 'using:wallet', 18: 'looking:PersonA', 19: 'looking:PersonB', 20: 'takeoutfood', 21: 'leaving-table', 22: 'cleaning-up', 23: 'NONE'}

activitydict_full_text_to_id = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'NONE': 21}

activity_from_key_full = activity_from_key

CONST_NUM_LABELS = len(activity_labels)
CONST_NUM_POINTS = 25
CONST_NUM_SUBPOINTS = 3
CONST_NUM_LABEL = 1

keypoint_labels = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
                                "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
                                "LEar","LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background", '']

filenames_all = ['8-13-18', '8-18-18', '8-17-18', '8-21-18', '8-9-18']
filenames_shifted = ['8-9-18']

# offset dicts are applied directly to the openpose info, so it's a 1x3 vector applied to all (x, y, confidence) points
offset_dict = {}
offset_dict['8-13-18']  = [0, 0, 0] # 50, 300
offset_dict['8-18-18']  = [0, 0, 0] # 50, 300
offset_dict['8-17-18']  = [0, 0, 0] # 50, 295
offset_dict['8-21-18']  = [0, 0, 0] # 50, 300
offset_dict['8-9-18']   = [0, 50, 0] # 50, 250


PD_MEALID   = 'meal_id'
PD_FRAMEID  = 'frame_id'
PD_LABEL_A_RAW    = 'label_A_raw'
PD_LABEL_B_RAW    = 'label_B_raw'
PD_LABEL_A_CODE    = 'label_A_code'
PD_LABEL_B_CODE    = 'label_B_code'
PD_POSE_A_RAW     = 'pose_A_raw'
PD_POSE_B_RAW     = 'pose_B_raw'
PD_VALIDITY_STATE = 'validity_state'

PD_POSE_A_WINDOWED    = 'pose_A_windowed'
PD_POSE_B_WINDOWED    = 'pose_B_windowed'
PD_INDEX_START        = 'index_start'
PD_INDEX_END        = 'index_end'
PD_TEST_SET         = 'test_set'

PD_COLS_FEAT_QUAL_CHECKS    = [PD_MEALID, PD_FRAMEID, PD_LABEL_A_RAW, PD_LABEL_B_RAW, PD_LABEL_A_CODE, PD_LABEL_B_CODE, PD_POSE_A_RAW, PD_POSE_B_RAW, PD_VALIDITY_STATE]
PD_COLS_SLICES            = [PD_MEALID, PD_LABEL_A_RAW, PD_LABEL_B_RAW, PD_INDEX_START, PD_INDEX_END, PD_TEST_SET]


bd_box_A = ((70, 80), (200, 340))
bd_box_B = ((230, 130), (370, 370))


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

CSV_ORDER = ['a_a', 'ab_a', 'alb_a', 'b_a', 'b_b', 'ab_b', 'bla_b', 'a_b', 'la_lb', 'lb_la']

OG_LABEL_NA                   = -1
OG_LABEL_AWAY_FROM_TABLE      = 0
OG_LABEL_IDLE                 = 1
OG_LABEL_EATING               = 2
OG_LABEL_DRINKING             = 3
OG_LABEL_TALKING              = 4
OG_LABEL_ORDERING             = 5
OG_LABEL_STANDING             = 6
OG_LABEL_TALKING_WAITER       = 7
OG_LABEL_LOOKING_WINDOW       = 8
OG_LABEL_LOOKING_WAITER       = 9
OG_LABEL_READING_BILL         = 10
OG_LABEL_READING_MENU         = 11
OG_LABEL_PAYING_CHECK         = 12
OG_LABEL_USING_PHONE          = 13
OG_LABEL_USING_NAPKIN         = 14
OG_LABEL_USING_PURSE          = 15
OG_LABEL_USING_GLASSES        = 16
OG_LABEL_USING_WALLET         = 17
OG_LABEL_LOOKING_PERSON_A     = 18
OG_LABEL_LOOKING_PERSON_B     = 19
OG_LABEL_TAKEOUT_FOOD         = 20
OG_LABEL_LEAVING_TABLE        = 21
OG_LABEL_CLEANING_UP          = 22
OG_LABEL_NONE                 = 23


ACT_NONE                = 0
ACT_AWAY_FROM_TABLE     = 1
ACT_IDLE                = 2
ACT_EATING              = 3
ACT_TALKING             = 4
ACT_WAITER              = 5
ACT_LOOKING_WINDOW      = 6
ACT_READING_BILL        = 7
ACT_READING_MENU        = 8
ACT_PAYING_CHECK        = 9
ACT_USING_PHONE         = 10
ACT_OBJ_WILDCARD        = 11
ACT_STANDING            = 12
ACT_LEAVING             = 13
ACT_DRINKING            = 14

LEN_REDUCED = 15
LEN_OG_ALL = 24
RANGE_REDUCED = range(LEN_REDUCED)
RANGE_OG_ALL = range(LEN_OG_ALL)

transform_dict = {}
transform_dict[OG_LABEL_NA]               = ACT_NONE
transform_dict[OG_LABEL_AWAY_FROM_TABLE]  = ACT_NONE    
transform_dict[OG_LABEL_IDLE]             = ACT_IDLE
transform_dict[OG_LABEL_EATING]           = ACT_EATING
transform_dict[OG_LABEL_DRINKING]         = ACT_DRINKING
transform_dict[OG_LABEL_TALKING]          = ACT_TALKING
transform_dict[OG_LABEL_ORDERING]         = ACT_WAITER
transform_dict[OG_LABEL_STANDING]         = ACT_STANDING
transform_dict[OG_LABEL_TALKING_WAITER]   = ACT_WAITER
transform_dict[OG_LABEL_LOOKING_WINDOW]   = ACT_LOOKING_WINDOW
transform_dict[OG_LABEL_LOOKING_WAITER]   = ACT_WAITER
transform_dict[OG_LABEL_READING_BILL]     = ACT_READING_BILL
transform_dict[OG_LABEL_READING_MENU]     = ACT_READING_MENU
transform_dict[OG_LABEL_PAYING_CHECK]     = ACT_PAYING_CHECK
transform_dict[OG_LABEL_USING_PHONE]      = ACT_USING_PHONE
transform_dict[OG_LABEL_USING_NAPKIN]     = ACT_OBJ_WILDCARD
transform_dict[OG_LABEL_USING_PURSE]      = ACT_OBJ_WILDCARD
transform_dict[OG_LABEL_USING_GLASSES]    = ACT_OBJ_WILDCARD
transform_dict[OG_LABEL_USING_WALLET]     = ACT_OBJ_WILDCARD
transform_dict[OG_LABEL_LOOKING_PERSON_A] = ACT_IDLE
transform_dict[OG_LABEL_LOOKING_PERSON_B] = ACT_IDLE
transform_dict[OG_LABEL_TAKEOUT_FOOD]     = ACT_OBJ_WILDCARD
transform_dict[OG_LABEL_LEAVING_TABLE]    = ACT_LEAVING
transform_dict[OG_LABEL_CLEANING_UP]      = ACT_LEAVING
transform_dict[OG_LABEL_NONE]             = ACT_NONE


def label_encode(label_string):
    if label_string in activitydict_full_text_to_id:
        return activitydict_full_text_to_id[label_string]

    return -1


# def reduce_labels(Y_array, feature_type):
#     # activity_from_key = {0:'away-from-table', 1:'idle', 2:'eating', 3: 'drinking', 4: 'talking', 5: 'ordering', 6: 'standing',
#     #                     7: 'talking:waiter', 8: 'looking:window', 9: 'looking:waiter', 10: 'reading:bill', 11: 'reading:menu',
#     #                     12: 'paying:check', 13: 'using:phone', 14: 'using:napkin', 15: 'using:purse', 16: 'using:glasses',
#     #                     17: 'using:wallet', 18: 'looking:PersonA', 19: 'looking:PersonB', 20: 'takeoutfood', 21: 'leaving-table', 22: 'cleaning-up', 23: 'NONE'}

#     # activity_labels = [0: 'NONE', 1: 'away-from-table', 2: 'idle', 3: 'eating', 4: 'talking', 5:'talking:waiter', 6: 'looking:window', 
#     # 7: 'reading:bill', 8: 'reading:menu', 9: 'paying:check', 10: 'using:phone', 11: 'using:objs', 12: 'standing']
    
#     ACT_NONE            = 0
#     ACT_AWAY_FROM_TABLE     = 1
#     ACT_IDLE            = 2
#     ACT_EATING          = 3
#     ACT_TALKING         = 4
#     ACT_WAITER          = 5
#     ACT_LOOKING_WINDOW    = 6
#     ACT_READING_BILL      = 7
#     ACT_READING_MENU      = 8
#     ACT_PAYING_CHECK      = 9
#     ACT_USING_PHONE       = 10
#     ACT_OBJ_WILDCARD      = 11
#     ACT_STANDING        = 12
#     ACT_DRINKING          = 13

#     Y_new = np.empty_like(Y_array)
#     Y_new = np.where(Y_array==23, ACT_NONE,       Y_new)
#     Y_new = np.where(Y_array==22, ACT_NONE,       Y_new)
#     Y_new = np.where(Y_array==21, ACT_STANDING,     Y_new) 
#     Y_new = np.where(Y_array==20, ACT_OBJ_WILDCARD, Y_new)
#     Y_new = np.where(Y_array==19, ACT_IDLE,       Y_new)
#     Y_new = np.where(Y_array==18, ACT_IDLE,       Y_new)
#     Y_new = np.where(Y_array==17, ACT_PAYING_CHECK, Y_new)
#     Y_new = np.where(Y_array==16, ACT_OBJ_WILDCARD, Y_new)
#     Y_new = np.where(Y_array==15, ACT_OBJ_WILDCARD, Y_new)
#     Y_new = np.where(Y_array==14, ACT_OBJ_WILDCARD, Y_new)
#     Y_new = np.where(Y_array==13, ACT_USING_PHONE,  Y_new)
#     Y_new = np.where(Y_array==12, ACT_PAYING_CHECK, Y_new)
#     Y_new = np.where(Y_array==11, ACT_READING_MENU, Y_new)
#     Y_new = np.where(Y_array==10, ACT_READING_BILL, Y_new)
#     Y_new = np.where(Y_array==9, ACT_WAITER,      Y_new) 
#     Y_new = np.where(Y_array==8, ACT_LOOKING_WINDOW,Y_new)
#     Y_new = np.where(Y_array==7, ACT_WAITER,      Y_new)
#     Y_new = np.where(Y_array==6, ACT_STANDING,        Y_new)
#     Y_new = np.where(Y_array==5, ACT_WAITER,      Y_new)
#     Y_new = np.where(Y_array==4, ACT_TALKING,         Y_new)
#     Y_new = np.where(Y_array==3, ACT_EATING,      Y_new)
#     Y_new = np.where(Y_array==2, ACT_EATING,      Y_new)
#     Y_new = np.where(Y_array==1, ACT_IDLE,            Y_new)
#     Y_new = np.where(Y_array==0, ACT_AWAY_FROM_TABLE, Y_new)

#     return Y_new

def reduce_Y_labels(df):
    # print("Pre")
    # print(df[PD_LABEL_A_CODE].unique())
    # print(df[PD_LABEL_B_CODE].unique())

    df[PD_LABEL_A_CODE] = df[PD_LABEL_A_CODE].map(transform_dict)
    df[PD_LABEL_B_CODE] = df[PD_LABEL_B_CODE].map(transform_dict)

    # print("post")
    # print(df[PD_LABEL_A_CODE].unique())
    # print(df[PD_LABEL_B_CODE].unique())

    return df

def dont_reduce_Y_labels(df):
    # print("Pre")
    # print(df[PD_LABEL_A_CODE].unique())
    # print(df[PD_LABEL_B_CODE].unique())

    mini_transform = {OG_LABEL_NA: OG_LABEL_NONE}
    df = df.replace({PD_LABEL_A_CODE: mini_transform})
    df = df.replace({PD_LABEL_B_CODE: mini_transform})

    # print("post")
    # print(df[PD_LABEL_A_CODE].unique())
    # print(df[PD_LABEL_B_CODE].unique())

    return df


def get_start_time():
    time_start = time.perf_counter()
    return time_start

def print_time(time_start):
    time_end = time.perf_counter()
    time_diff = time_end - time_start

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Time elapsed: " + str(time_diff) + " ending at " + str(current_time))