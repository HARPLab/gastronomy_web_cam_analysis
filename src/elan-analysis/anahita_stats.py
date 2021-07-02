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

import seaborn as sns
from matplotlib import cm
import matplotlib



def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

filename_root = "8-21-18"
filenames_all = ['8-13-18', '8-17-18', '8-18-18', '8-21-18', '8-9-18']

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
TYPE_WAITER = 'waiter_action'
TYPE_CUSTOMER_STATE = 'customer_state'



overall_flow = []
waiter_events = []
customer_states = []

simple_timeline = {}

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

                    overall_flow.append((meal, beginning_frame, ending_frame, label, TYPE_WAITER))
                    waiter_events.append(label)


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

                    overall_flow.append((meal, beginning_frame, ending_frame, label, TYPE_CUSTOMER_STATE))
                    customer_states.append(label)



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

# print(overall_flow)
# Process the overall flow
# list of (meal_id, timestamp, label)

print(overall_flow)
flow_per_meal = {}
for meal in filenames_all:
    flow_per_meal[meal] =[]

for meal, start, end, event, e_type in overall_flow:
    flow_per_meal[meal].append([start, end, event, e_type])

for meal in filenames_all:
    df = pd.DataFrame(flow_per_meal[meal], columns = ['start_time', 'end_time', 'event', 'event_type']) 
    df.to_csv('outputs/readable_highlevel_' + str(meal) + '.csv')


waiter_events = list(set(waiter_events))
customer_states = list(set(customer_states))

no_op = (meal, None, 'NOP', TYPE_WAITER)

data_all = []
data_individual_meals = []

for meal in filenames_all:
    print("Adding timeline info for " + meal)
    timeline = list(filter(lambda entry: entry[0] == meal, overall_flow))
    timeline.sort(key=lambda x: x[1])
    prev_event = (meal, 0, 'NONE', TYPE_CUSTOMER_STATE)
    prev_action = no_op
    data = []

    INDEX_MEALID = 0
    INDEX_TIMESTAMP = 1
    INDEX_LABEL = 2
    INDEX_TYPE = 3

    for event in timeline:
        # if the waiter took a novel action, ex not a NO-OP, then hold onto it
        if prev_event[INDEX_TYPE] == TYPE_CUSTOMER_STATE and event[INDEX_TYPE] == TYPE_WAITER:
            # if it's the unique waiter event
            if event[INDEX_LABEL] not in ['arriving', 'leaving']:
                prev_action = event

        # if we have a transition between two events
        elif prev_event[INDEX_TYPE] == TYPE_CUSTOMER_STATE and event[INDEX_TYPE] == TYPE_CUSTOMER_STATE:
            datum = [meal, prev_event[INDEX_LABEL], prev_action[INDEX_LABEL], event[INDEX_LABEL], prev_event[INDEX_TIMESTAMP], event[INDEX_TIMESTAMP]]
            data.append(datum)
            # print("added " + str(prev_event[INDEX_LABEL]) + " --" + prev_action[INDEX_LABEL] + "--> " + event[INDEX_LABEL])

            prev_event = event
            prev_action = no_op

        elif prev_event[INDEX_LABEL] == 'NONE':
            prev_event = event

        else:
            print("ERR")
            print(prev_event)
            print(prev_action)
            print(event)
            print("~~~")

    data_individual_meals.append(data)
    data_all.extend(data)

transition_log = pd.DataFrame(data_all, columns = ['Meal ID', 'before', 'operation', 'after', 'bt', 'at'])

# Make nice graph
import pydot_ng as pydot
from pydot_ng import Dot, Edge,Node

def make_graph(data, graph_label, customer_states):
    data_overview = []

    g = Dot()
    g.set_node_defaults(color='lightgray',
                        style='filled',
                        shape='box',
                        fontname='Courier',
                        fontsize='10',
                        fontcolor='white')
    colors_viridis = cm.get_cmap('viridis', len(customer_states))

    for i in range(len(customer_states)):
        label = customer_states[i]
        label = label.replace(':', "-")

        new_node = Node(label)
        col = colors_viridis(i)
        col = matplotlib.colors.rgb2hex(col)
        new_node.set_color(col)
        g.add_node(new_node)
        # print("Add node " + str(customer_states[i]))

    transition_types = defaultdict(list)

    # Consolidate labels
    for link in data:
        m, la, op, lb, at, bt = link
        la = la.replace(':', "-")
        lb = lb.replace(':', "-")
        op = op.replace(':', "-")

        transition_types[(la, op)].append(lb)

    checklist = set()
    for link in data:
        m, la, op, lb, at, bt = link
        la = la.replace(':', "-")
        lb = lb.replace(':', "-")
        op = op.replace(':', "-")

        this_edge = (la, op, lb)
        if this_edge not in checklist:
            checklist.add(this_edge)

            edge = pydot.Edge(la, lb)
            
            # print("edge from " + la + " to " + lb + " via " + op)
            # prob_label = transmats[i_a][j_b]
            # if (prob_label,b) in best:
            #     prob_label = float('%.3g' % prob_label)
            #     prob_label = str(prob_label)
            #     # print(prob_label)

            results = transition_types[(la, op)]
            prob = results.count(lb) / len(results)
            prob = float('%.3g' % prob)

            datum = [la, op, lb, prob]
            data_overview.append(datum)

            edge.set_label(op + "\nP=" + str(prob))
            g.add_edge(edge)


    
    g.write_png("graphs/ada_table_states-" + graph_label + ".png")
    print("Output graph " + graph_label)

    df = pd.DataFrame(data_overview, columns = ['before', 'operation', 'after', 'probability']) 
    df.to_csv('outputs/ada_table_states-' + graph_label + '.csv')



graph_data = data_individual_meals
graph_data.append(data_all)

graph_names = filenames_all
graph_names.append("all")

for i in range(len(graph_data)):
    data_list = graph_data[i]
    graph_name = graph_names[i]
    make_graph(data_list, graph_name, customer_states)


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
# print(table_state_labels)


data = []
table_state_emissions = {}
activity_labels = activitydict.keys()
# print(activity_labels)


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

d_emi = pd.DataFrame(data, columns = cols_emi) 
d_emi.to_csv('outputs/observerations.csv')





# exit()
df.to_csv('outputs/all_data.csv')




