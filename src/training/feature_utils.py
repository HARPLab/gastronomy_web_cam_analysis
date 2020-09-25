import cv2
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import pickle
import copy
import numpy as np
import keras
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
                        'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
                        'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                        'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'leaving-table':21, 'cleaning-up':22, 'NONE':23}

# Lookup table for OpenPose keypoint indices
keypoint_labels = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
                                                "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar",
                                                "LEar","LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background", '']

NULL_POSE = [(0.0, 0.0, 0.0)]

FEATURES_TYPE_POSES_RAW = 10
FEATURES_TYPE_POSES_ROLES = 11
FEATURES_TYPE_ALSO_VEL = 12
FEATURES_SET_PA = 0
FEATURES_SET_PB = 1
FEATURES_SET_BOTH = 2
FLAG_FEATURE_SET = FEATURES_SET_PA
FLAG_FEATURE_TYPE = FEATURES_TYPE_POSES_RAW

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

def get_role_labels(cleaned_poses):
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
                                assignment = distances_np.argmax()
                        else:
                                assignment = NULL_POSE

                        assignments[i] = assignment #TODO change to cleaned_poses[assignment]

        else:
                print("ERROR IN ROLES")

        return assignments
def get_feature_vector(frame): # TODO add cleaned feature type here
        feature_vector = []
        if FLAG_FEATURE_TYPE is FEATURES_TYPE_POSES_RAW:
                if len(frame.get_poses_raw()) > 0:
                    test = np.array(frame.get_poses_raw()[0][0])
                else:
                    return np.zeros(50), False
                #print(test)
                #raw = np.zeros(50)
                #for cleaned_pose in frame.get_poses_clean():
                #    if np.array(cleaned_pose[0]).shape != (3,):
                #        raw = np.array(cleaned_pose[0])[:, 0:2]
                #        break
                #raw = np.array(frame.get_poses_clean()[0][0])[:,0:2]
                if test.shape == (3,):
                        return np.zeros(50), False
                raw = test[:, 0:2]
                #print(raw)
                feature_vector.append(raw)

        elif FLAG_FEATURE_TYPE is FEATURES_TYPE_POSES_ROLES:
                if FLAG_FEATURE_SET is FEATURES_SET_PA:
                        #iprint(frame.get_PA())
                        feature_vector.append(frame.get_PA())

                elif FLAG_FEATURE_SET is FEATURES_SET_PB:
                        feature_vector.append(frame.get_PB())

                elif FLAG_FEATURE_SET is FEATURES_SET_BOTH:
                        feature_vector.append(frame.get_PA())
                        feature_vector.append(frame.get_PB())


        elif FLAG_FEATURE_TYPE is FEATURES_TYPE_ALSO_VEL:

                if FLAG_FEATURE_SET is FEATURES_SET_PA:
                        feature_vector.append(frame.get_delta_PA())

                elif FLAG_FEATURE_SET is FEATURES_SET_PB:
                        feature_vector.append(frame.get_label_PA())

                elif FLAG_FEATURE_SET is FEATURES_SET_PA:
                        feature_vector.append(frame.get_label_PA())
                        feature_vector.append(frame.get_label_PB())

        return np.asarray(feature_vector).flatten(), True

def get_labels_vector(frame):
        newY = []

        if FLAG_FEATURE_SET is FEATURES_SET_PA:
                newY.append(frame.get_label_PA())
        elif FLAG_FEATURE_SET is FEATURES_SET_PB:
                newY.append(frame.get_label_PA())
        elif FLAG_FEATURE_SET is FEATURES_SET_BOTH:
                newY.append(frame.get_label_PA())
                newY.append(frame.get_label_PB())

        return newY


def frame_to_vectors(frame):
        newX = []
        newY = []

        newY = get_labels_vector(frame)
        feature_vector, ret = get_feature_vector(frame)
        #print(feature_vector.shape)
        #print(newY)
        if not ret:
                return newX, newY, False
        #print(feature_vector.shape)
        for ys in newY:
                newX.append(feature_vector)
        newY = activitydict[newY[0]]
        #print(newY) 
        return newX, newY, True
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
            plt.savefig("LSTM_Losses70epochs.png")
            plt.close()

