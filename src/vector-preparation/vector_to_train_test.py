import time
import cv2
import random
import pickle
import pandas as pd
import numpy as np
import json


activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}

activity_from_key = {0:'away-from-table', 1:'idle', 2:'eating', 3: 'drinking', 4: 'talking', 5: 'ordering', 6: 'standing',
                        7: 'talking:waiter', 8: 'looking:window', 9: 'looking:waiter', 10: 'reading:bill', 11: 'reading:menu',
                        12: 'paying:check', 13: 'using:phone', 14: 'using:napkin', 15: 'using:purse', 16: 'using:glasses',
                        17: 'using:wallet', 18: 'looking:PersonA', 19: 'looking:PersonB', 20: 'takeoutfood', 21: 'leaving-table', 22: 'cleaning-up', 23: 'NONE'}


# Lookup table for OpenPose keypoint indices
keypoint_labels = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
                                                "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
                                                "LEar","LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background", '']


filenames_all = ['8-13-18', '8-18-18', '8-17-18', '8-21-18', '8-9-18']
prefix_qc = './quality-checks/'
prefix_vectors_out = './output-vectors/'

INDEX_PA = 0
INDEX_PB = 1



def unison_shuffled_copies(a, b, seed_val):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=seed_val).permutation(len(a))
    return a[p], b[p]

def export_folds_svm(X_final, Y_final, filename, prefix_vectors_out, seed, test_size=.2):
    num_folds = int(1.0 / test_size)
    X_shuffled, Y_shuffled = unison_shuffled_copies(X_final, Y_final, seed)

    for f in range(num_folds):
        chunk_size = int(len(X_shuffled) * (test_size))
        print("Exporting fold " + str(f) + " from " + str(f*chunk_size) + " to " + str(f*chunk_size + chunk_size))
        
        test_start, test_end = f*chunk_size, f*chunk_size + chunk_size

        test_X      = X_shuffled[test_start : test_end]
        test_Y      = Y_shuffled[test_start : test_end]
        
        train_chunk_x1 = X_shuffled[0 : test_start]
        train_chunk_x2 = X_shuffled[test_end : ]

        train_chunk_y1 = Y_shuffled[0 : test_start]
        train_chunk_y2 = Y_shuffled[test_end : ]

        train_X     = np.concatenate((train_chunk_x1, train_chunk_x2), axis=0)
        train_Y     = np.concatenate((train_chunk_y1, train_chunk_y2), axis=0)

        # print("Fold " + str(f))
        # print(train_X.shape)
        # print(test_X.shape)
        

        label_testsize      = str(int(test_size * 100))
        label_trainsize     = str(int((1.0 - test_size) * 100))
        label_random_seed   = str(seed)

        core_name   = "/for_svm/" + filename + "_forsvm" + "_s" + label_random_seed + "_f" + str(f) + "_"
        test_name   = core_name + label_testsize
        train_name  = core_name + label_trainsize

        filehandler = open(prefix_vectors_out + train_name + "_X.p", "wb")
        pickle.dump(train_X, filehandler)
        filehandler.close()

        filehandler = open(prefix_vectors_out + train_name + "_Y.p", "wb")
        pickle.dump(train_Y, filehandler)
        filehandler.close()

        filehandler = open(prefix_vectors_out + test_name + "_X.p", "wb")
        pickle.dump(test_X, filehandler)
        filehandler.close()

        filehandler = open(prefix_vectors_out + test_name + "_X.p", "wb")
        pickle.dump(test_Y, filehandler)
        filehandler.close()

    print("\n")

def export_folds(X_final, Y_final, filename, prefix_vectors_out, seed):
    export_folds_svm(X_final, Y_final, filename, prefix_vectors_out, seed)




for filename in filenames_all:
	print("Creating fold sets for " + filename)
	seed = 42
	core_name = prefix_vectors_out + "/trimmed/trimmed_" + filename

	filehandler = open(core_name + "_X.p", "rb")
	print(core_name + "_X.p")
	X_all = pickle.load(filehandler)
	filehandler.close()

	filehandler = open(core_name + "_Y.p", "rb")
	Y_all = pickle.load(filehandler)
	filehandler.close()

	export_folds(X_all, Y_all, filename, prefix_vectors_out, seed)







