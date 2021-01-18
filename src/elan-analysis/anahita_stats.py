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

def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

filename_root = "8-21-18"
filenames_all = ['8-9-19', '8-13-18', '8-17-18', '8-18-18', '8-21-18']

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



## database initialization
frame_to_poseact ={} #f_id -> (posedata, personAactivity, personBactivity)
frame_to_poseact ={} #f_id -> (personAactivity, personBactivity)
frames = {}

log = {}
# log[(mealid, index)] = {customerTransition, }

LABEL_NONE = "NONE"

BLANK_LABELS = [LABEL_NONE, LABEL_NONE, LABEL_NONE]


for meal in filenames_all:
    root = parseXML('../../Annotations/' + meal + '-michael.eaf')
    print("Processing meal annotations for " + meal)
    timeline = []

    ## start looping through annotation labels
    for child in root:
        if child.tag == 'TIME_ORDER':
            for times in child:
                timedict[times.attrib['TIME_SLOT_ID']] = times.attrib['TIME_VALUE']

        elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'Waiter':
            pass


        # initial setup happens here
        elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'CustomerTransitions':
            # print("opening table state tier")
            for annotation in child:
                # print(annotation)
                label = ""
                for temp in annotation:
                    beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                    ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
                    # print(beginning_frame, ending_frame)
                   
                    label = LABEL_NONE
                    # for each of the labeled activities in this block of time
                    for anno in temp: ## another single iteration loop
                        label = anno.text

                    for f_id in range(beginning_frame, ending_frame):
                        if (meal, f_id) not in log.keys():
                            log[(meal, f_id)] = copy.copy(BLANK_LABELS)
                        
                        log[(meal, f_id)][0] = label



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

                        if (meal, f_id) not in log.keys():
                            log[(meal, f_id)] = copy.copy(BLANK_LABELS)

                        log[(meal, f_id)][1] = activity

                        # frame_to_poseact[f_id] = feature_data
                        # print("added personA " + str(f_id))


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

                        if (meal, f_id) not in log.keys():
                            log[(meal, f_id)] = copy.copy(BLANK_LABELS)
                        
                        log[(meal, f_id)][2] = activity

                            # print(log[meal, f_id])

                        
# print(log)

data = []
for entry in log.keys():
    datum = []
    value = log[entry] 
    # value = value / sum(value)
    datum = [entry[0], entry[1], value[0], value[1], value[2]]
    data.append(datum)


# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Meal ID', 'timestamp', 'table-state', 'person-A', 'person-B']) 


# POST ANALYSIS
table_state_labels = df['table-state'].unique()
print(table_state_labels)


data = []
table_state_emissions = {}
activity_labels = activitydict.keys()
print(activity_labels)


for ts in table_state_labels:
    datum = [ts]
    total = df.loc[(df['table-state'] == ts)]
    total = len(total)

    for activity in activity_labels:
        entries_A = df.loc[(df['person-A'] == activity) & (df['table-state'] == ts)]
        entries_B = df.loc[(df['person-B'] == activity) & (df['table-state'] == ts)]
        num_entries = len(entries_A) + len(entries_B)

        if num_entries != 0:
            value = (num_entries / (1.0 * total))
        else:
            value = 0


        table_state_emissions[activity] = value
        datum.append(value)

    data.append(datum)


cols_emi = ["table-state"] + list(activity_labels)
# print(cols_emi)



d_emi = pd.DataFrame(data, columns = cols_emi) 
d_emi.to_csv('observerations.csv')





# exit()
df.to_csv('all_data.csv')




