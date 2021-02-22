##### Activity Key #####
# 0 away-from-table
# 1 idle
# 2 eating
# 3 drinking
# 4 talking
# 5 ordering
# 6 standing
# 7 talking:waiter
# 8 looking:window
# 9 looking:waiter
# 10 reading:bill
# 11 reading:menu
# 12 paying:check
# 13 using:phone
# 14 using:napkin
# 15 using:purse
# 16 using:glasses
# 17 using:wallet
# 18 looking:PersonA
# 19 looking:PersonB
# 20 take-out-food
######################
import xml.etree.ElementTree as ET
import cv2
from collections import defaultdict
import json
import copy
import pandas as pd
import os

import seaborn as sns
from matplotlib import cm
import matplotlib
import numpy as np
import pickle

prefix_output = "output-vectors/raws/"

def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

filename_root = "8-21-18"
filenames_all = ['8-9-18', '8-13-18', '8-17-18', '8-18-18', '8-21-18']

timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'NONE': 21}

def getFeatureObj(currentframe):
    feature_data = {}
    feature_data['frame'] = currentframe   
    return feature_data

def addActivityToFeatureObj(feature_obj, personLabel, activity):
    feature_obj[personLabel] = activity
    return feature_obj

def verify_input_output(X, Y):
    # print(X.shape)
    # print(Y.shape)
    # print("Unique values: ")
    
    unique_values = np.unique(Y)
    if(all(x in range(len(activitydict.keys())) for x in unique_values)): 
        print("OK! Values in reasonable range")
    else:
        print("Nope- Y contains more than the valid labels")
        np.set_printoptions(threshold=np.inf)
        # np.set_printoptions(suppress=True)
        print(unique_values)
        np.set_printoptions(threshold=15)
        # np.set_printoptions(suppress=False)
        # exit()


## database initialization
frame_to_poseact ={} #f_id -> (posedata, personAactivity, personBactivity)
frame_to_poseact ={} #f_id -> (personAactivity, personBactivity)
frames = {}

log = {}
# log[(mealid, index)] = {customerTransition, }

LABEL_NONE = "NONE"

BLANK_LABELS = [LABEL_NONE, LABEL_NONE, LABEL_NONE]
TYPE_WAITER = 'waiter_action'
TYPE_CUSTOMER_STATE = 'customer_state'

# INDEX_FID = 0
INDEX_A = 0
INDEX_B = 1

overall_flow = []
waiter_events = []
customer_states = []

warned_of = []

def get_file_frame_index(file_title):
    start_index = file_title.index('_cropped_') + len('_cropped_')
    return int(file_title[start_index: start_index + index_length])



for meal in filenames_all:
    root = parseXML('../../Annotations/' + meal + '-michael.eaf')
    print("Processing meal annotations for " + meal)
    timeline = []

    # Find the latest frame in the series from examining the outputs
    prefix = '../../Annotations/json/'
    entries = os.listdir(prefix)
    entries = list(filter(lambda k: meal in k, entries))
    # print(entries)
    start_index = entries[0].index('_cropped_') + len('_cropped_')
    index_length = len('000000000000')

    indices = [get_file_frame_index(e) for e in entries]
    max_frame = max(indices) + 1

    print(max_frame)

    timeline_Y = np.full((max_frame, 2), activitydict[LABEL_NONE])

    ## start looping through annotation labels
    for child in root:
        if child.tag == 'TIME_ORDER':
            for times in child:
                timedict[times.attrib['TIME_SLOT_ID']] = times.attrib['TIME_VALUE']

        # elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'Waiter':
        #     for annotation in child:
        #         # print(annotation)
        #         label = ""
        #         for temp in annotation:
        #             beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
        #             ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
        #             # print(beginning_frame, ending_frame)
                   
        #             label = LABEL_NONE
        #             # for each of the labeled activities in this block of time
        #             for anno in temp: ## another single iteration loop
        #                 label = anno.text

        #             overall_flow.append((meal, beginning_frame, label, TYPE_WAITER))
        #             waiter_events.append(label)


        # # initial setup happens here
        # elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'CustomerTransitions':
        #     # print("opening table state tier")
        #     for annotation in child:
        #         # print(annotation)
        #         label = ""
        #         for temp in annotation:
        #             beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
        #             ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
        #             # print(beginning_frame, ending_frame)
                   
        #             label = LABEL_NONE
        #             # for each of the labeled activities in this block of time
        #             for anno in temp: ## another single iteration loop
        #                 label = anno.text

        #             for f_id in range(beginning_frame, ending_frame):
        #                 if (meal, f_id) not in log.keys():
        #                     log[(meal, f_id)] = copy.copy(BLANK_LABELS)
                        
        #                 log[(meal, f_id)][0] = label

        #             overall_flow.append((meal, beginning_frame, label, TYPE_CUSTOMER_STATE))
        #             customer_states.append(label)



        elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonA':
            for annotation in child:
                # print("adding PersonA's annotations...")
                for temp in annotation: ## this should only loop once, couldnt figure out how to access a child xml tag without looping
                    #print(temp.attrib['TIME_SLOT_REF1'])
                    ## beginning frame
                    beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                    ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
                    
                    activity = LABEL_NONE
                    # for each of the labeled activities in this block of time
                    for anno in temp: ## another single iteration loop
                        activity=anno.text

                    for f_id in range(beginning_frame, ending_frame):
                        if activity in activitydict.keys():
                            timeline_Y[f_id][INDEX_A] = activitydict[activity]
                            # timeline_Y[f_id][INDEX_FID] = f_id
                        else:
                            if activity not in warned_of:
                                print("Strange label: " + activity)
                                warned_of.append(activity)


        elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonB':
             for annotation in child:
                # print("adding PersonB's annotations...")
                for temp in annotation: ## this should only loop once, couldnt figure out how to access a child xml tag without looping
                    ## beginning frame
                    beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                    ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)

                    activity = LABEL_NONE
                    for anno in temp: ## another single iteration loop
                        activity=anno.text

                    # for every ms, add to the listing
                    for f_id in range(beginning_frame, ending_frame):
                        if activity in activitydict.keys():
                            timeline_Y[f_id][INDEX_B] = activitydict[activity]
                            # timeline_Y[f_id][INDEX_FID] = f_id
                        else:
                            if activity not in warned_of:
                                print("Strange label: " + activity)
                                warned_of.append(activity)

    verify_input_output(None, timeline_Y)



    filehandler = open(prefix_output + meal + '_raw_Y.p', 'wb') 
    pickle.dump(timeline_Y, filehandler)
    filehandler.close()
                     



