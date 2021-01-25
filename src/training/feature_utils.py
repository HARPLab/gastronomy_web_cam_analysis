import random
from collections import defaultdict
# from OPwrapper import OP
import json
import pickle
import copy
import numpy as np
import itertools
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn import svm
import random
# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.utils import to_categorical
from matplotlib import pyplot
import pickle
import copy
import numpy as np
from tensorflow import keras

activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}

# Lookup table for OpenPose keypoint indices
keypoint_labels = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
                                                "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
                                                "LEar","LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background", '']

NULL_POSE = [(0.0, 0.0, 0.0)]

ROLES_BY_BUCKET = 0
FEATURES_TYPE_POSES_RAW = 10
FEATURES_TYPE_POSES_ROLES = 11
FEATURES_TYPE_ALSO_VEL = 12
FEATURES_SET_PA = 0
FEATURES_SET_PB = 1
FEATURES_SET_BOTH = 2
LABELS_SET_PA = 0
LABELS_SET_PB = 1
FLAG_FEATURE_SET = FEATURES_SET_BOTH
FLAG_FEATURE_TYPE = FEATURES_TYPE_POSES_ROLES
FLAG_ROLE_ASSIGNMENT = ROLES_BY_BUCKET
FLAG_LABEL_SET = LABELS_SET_PA

nexuses = [(170, 146), (329, 221)]

nexus_PA = (170, 146)
nexus_PB = (329, 221)
nexus_waiter = (nexus_PA + nexus_PB)
nexus_waiter = (nexus_waiter[0] / 2.0, nexus_waiter[1] / 2.0)

confidence_threshold = .8
NULL_POSE = [(0.0, 0.0, 0.0)]
NULL_POSE = NULL_POSE * 25


def get_main_pt(pt_set):
        # return the nose point
        #print(pt_set)
        return pt_set[0][1]

def get_secondary_pt(pt_set):
        # return the nose point
        return pt_set[0][8]

def com_near_nexus(com_xy, nexuses, tol):
        x1, y1 = com_xy
        for nexus in nexuses:
                x2, y2 = nexus
                dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                if dist < tol:
                        return True

        return False
def in_bd_box(bd_box, point):
        return point[0] > bd_box[0][0] and point[0] < bd_box[0][1] and point[1] < bd_box[1][1] and point[1] > bd_box[1][0]

def get_role_labels(cleaned_poses):
        if len(cleaned_poses) == 0:
                return None, None, None
        bd_box_B = ((90,350),(250,450))
        bd_box_A = ((90,350),(70,230))
        num_points_in_Abox = []
        num_points_in_Bbox = []
        for pose in cleaned_poses:
                num_pts_in_A = 0
                num_pts_in_B = 0
                for r in range(len(pose[0])):
                        if in_bd_box(bd_box_A, pose[0][r]):
                                num_pts_in_A += 1
                        if in_bd_box(bd_box_B, pose[0][r]):
  	                        num_pts_in_B += 1
                num_points_in_Abox.append(num_pts_in_A) 
                num_points_in_Bbox.append(num_pts_in_B)
        max_A_idx = 0
        max_pts_A = -1
        max_B_idx = 0
        max_pts_B = -1
        for i in range(len(num_points_in_Abox)):
                if num_points_in_Abox[i] > max_pts_A:
                        max_A_idx = i
                        max_pts_A = num_points_in_Abox[i]
                if num_points_in_Bbox[i] > max_pts_B:
                        max_B_idx = i
                        max_pts_B = num_points_in_Bbox[i]
        return cleaned_poses[max_A_idx], cleaned_poses[max_B_idx], None
"""
        assignments = [None, None, None]
        if FLAG_ROLE_ASSIGNMENT == ROLES_BY_BUCKET:
                role_nexuses = [nexus_PA, nexus_PB]
                if len(cleaned_poses) > len(role_nexuses):
                        role_nexuses.append(nexus_waiter)


                # for each role
                for i in range(len(role_nexuses)):
                        distances = []

                        for pose in cleaned_poses:
                                compare_pt = get_main_pt(pose)
                                nex = (role_nexuses[i][0], role_nexuses[i][1], 1)

                                dist = distance.euclidean(compare_pt, nex)
                                distances.append(dist)

                        if len(distances) > 0:
                                distances_np = np.asarray(distances)
                                assignment = distances_np.argmin()
                        else:
                                assignment = NULL_POSE

                        assignments[i] = cleaned_poses[assignment] #TODO change to cleaned_poses[assignment] PA->index of closest pos in cleaned poses, PB, Waiter

        else:
                print("ERROR IN ROLES")

        return assignments
"""

def get_feature_vector(frame, feature_set=FLAG_FEATURE_SET): # TODO add cleaned feature type here
        # TODO make this more flexible?
        FLAG_FEATURE_SET = feature_set
        feature_vector = []

        if FLAG_FEATURE_TYPE is FEATURES_TYPE_POSES_RAW:
                if len(frame.get_poses_raw()) > 1:
                    test = np.array(frame.get_poses_raw()[0][0])
                else:
                    return [np.zeros(50)], [np.zeros(50)],  False
                #print(test)
                #raw = np.zeros(50)
                #for cleaned_pose in frame.get_poses_clean():
                #    if np.array(cleaned_pose[0]).shape != (3,):
                #        raw = np.array(cleaned_pose[0])[:, 0:2]
                #        break
                #raw = np.array(frame.get_poses_clean()[0][0])[:,0:2]
                if test.shape == (3,):
                        return [np.zeros(50)], [np.zeros(50)], False
                raw = test[:, 0:2]
                #print(raw)
                feature_vector.append(raw.flatten())

        elif FLAG_FEATURE_TYPE is FEATURES_TYPE_POSES_ROLES:
                if FLAG_FEATURE_SET is FEATURES_SET_PA:
                        #print("PA: " + str(frame.get_PA()))
                        feature_vector.append(np.array(frame.get_PA()[0])[:,0:2].flatten())
                        # print("feature A")
                elif FLAG_FEATURE_SET is FEATURES_SET_PB:
                        feature_vector.append(np.array(frame.get_PB()[0])[:,0:2].flatten())
                        # print("feature B")
                elif FLAG_FEATURE_SET is FEATURES_SET_BOTH:
                        b_feats = np.array(frame.get_PB()[0])[:,0:2].flatten()
                        a_feats = np.array(frame.get_PA()[0])[:,0:2].flatten()
                        both_feats = np.concatenate((a_feats, b_feats), axis=None)
                        feature_vector.append(both_feats)
                        # print("feature both")

        elif FLAG_FEATURE_TYPE is FEATURES_TYPE_ALSO_VEL:

                if FLAG_FEATURE_SET is FEATURES_SET_PA:
                        feature_vector.append(frame.get_delta_PA())

                elif FLAG_FEATURE_SET is FEATURES_SET_PB:
                        feature_vector.append(frame.get_label_PB())

                elif FLAG_FEATURE_SET is FEATURES_SET_PA:
                        feature_vector.append(frame.get_label_PA())
                        feature_vector.append(frame.get_label_PB())
        #rev_temp = copy.deepcopy(feature_vector)
        #rev_temp.reverse()
        return feature_vector, True#, rev_temp, True #list of flattend poses, up to two poses per list for PA and PB

def get_labels_vector(frame):
        newY = []

        if FLAG_FEATURE_SET is FEATURES_SET_PA:
                newY.append(frame.get_label_PA())
                # print("a label")
        elif FLAG_FEATURE_SET is FEATURES_SET_PB:
                newY.append(frame.get_label_PB())
                # print("b label")
        elif FLAG_FEATURE_SET is FEATURES_SET_BOTH:
                if FLAG_LABEL_SET is LABELS_SET_PA:
                        newY.append(frame.get_label_PA())
                        # print("a label both")
                if FLAG_LABEL_SET is LABELS_SET_PB:
                        newY.append(frame.get_label_PB())
                        # print("b label both")
        #rev_temp = copy.deepcopy(newY)
        #rev_temp.reverse()
        return newY#, rev_temp #up to two labels per list

# TODO: verify
def frame_to_vectors(frame):
        newX = []
        newrX = []
        newY = []

        newY = get_labels_vector(frame)
        feature_vector, ret = get_feature_vector(frame)
       # print(newY)
        #print("feat:" + str(feature_vector))
        assert len(newY) == len(feature_vector)
        #print(feature_vector.shape)
        #print(newY)
        if not ret:
                return newX, newY, False
        #print(feature_vector.shape)
        for i in range(len(newY)):
                newX.append(feature_vector[i])
                #newrX.append(rev_feature_vector[i])
        newY = [activitydict[ny] for ny in newY]
        #newrY = [activitydict[ny] for ny in revY] 
        #print("nexX: " + str(newX))
        return newX, newY, True # ith label in newY corresponds to ith featurevector in newX.

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))
    plt.yticks(tick_marks)
    cm = np.around(cm, decimals=1)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f').replace("0","",1),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class TrainingPlot(keras.callbacks.Callback):
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            # Clear the previous plot
            #clear_output(wait=True)
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig("LSTM_Losses.png")
            plt.close()

