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
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import arconsts


def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)

# Import the five annotated files
filename_root = "8-21-18"
filenames_all = ['8-13-18', '8-17-18', '8-18-18', '8-21-18', '8-9-18']

export_prefix = 'outputs-2022/'

timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'NONE': 21}


activity_shorterdict = {'away-from-table': 'away', 'idle': 'idle', 'eating': 'eating', 'drinking': 'drinking', 'talking': 'talking', 'ordering': 'ordering', 
            'standing':'standing','talking:waiter': 'talk:waiter', 'looking:window': 'look:window', 'looking:waiter': 'look:waiter', 
            'reading:bill':'read:bill', 'reading:menu': 'read:menu',
            'paying:check': 'pay:check', 'using:phone': 'use:phone', 'using:napkin': 'use:napkin', 'using:purse': 'use:purse', 'using:glasses': 'use:glasses',
            'using:wallet': 'use:wallet', 'looking:PersonA': 'look:partner', 'looking:PersonB':'look:partner', 'takeoutfood':'takeout', 'NONE': 'NONE'}

activitydict_display = ['away', 'idle', 'eating', 'drinking', 'talking', 'ordering', 'standing', 'talk:waiter', 'look:window',
            'look:waiter', 'read:bill', 'read:menu', 'pay:check', 'use:phone', 'use:napkin', 'use:purse', 'use:glasses',
            'use:wallet', 'look:partner', 'takeout', 'NONE']

def getFeatureObj(currentframe):
    feature_data = {}
    feature_data['frame'] = currentframe   
    return feature_data

def addActivityToFeatureObj(feature_obj, personLabel, activity):
    feature_obj[personLabel] = activity
    return feature_obj

def get_label_from_block(temp):
    activity_label = 'NONE'
    for anno in temp: ## another single iteration loop
        activity_label = anno.text

    return activity_label

def import_meals():
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


            # LOGGING PERSON A AND B ACTIVITIES
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
                        activity = get_label_from_block(temp)

                        for anno in temp: ## another single iteration loop
                            activity = anno.text

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
                            activity = anno.text

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

    # print(overall_flow)
    flow_per_meal = {}
    for meal in filenames_all:
        flow_per_meal[meal] =[]

    for meal, start, end, event, e_type in overall_flow:
        flow_per_meal[meal].append([start, end, event, e_type])

    for meal in filenames_all:
        df = pd.DataFrame(flow_per_meal[meal], columns = ['start_time', 'end_time', 'event', 'event_type']) 
        df.to_csv(export_prefix + 'readable_highlevel_' + str(meal) + '.csv')


    waiter_events = list(set(waiter_events))
    customer_states = list(set(customer_states))

    no_op = (meal, None, 'NOP', TYPE_WAITER)




    data_individual_meals = []
    data_all = []

    # Import all the meals on the list
    for meal in filenames_all:
        print("Adding timeline info for " + meal)
        timeline = list(filter(lambda entry: entry[0] == meal, overall_flow))
        timeline.sort(key=lambda x: x[1])
        prev_event = (meal, 0, 'NONE', TYPE_CUSTOMER_STATE)
        prev_action = no_op
        data = []

        # Constants for reading/interpreting each of the entries
        INDEX_MEALID = 0
        INDEX_TIMESTAMP = 1
        INDEX_LABEL = 2
        INDEX_TYPE = 3

        for event in timeline:
            print(event)
            # if the waiter took a novel action, ex not a NO-OP, then hold onto it
            if prev_event[INDEX_TYPE] == TYPE_CUSTOMER_STATE and event[INDEX_TYPE] == TYPE_WAITER:
                # if it's the unique waiter event
                if event[INDEX_LABEL] not in ['arriving', 'leaving']:
                    prev_action = event

            # if we have a transition between two events
            elif prev_event[INDEX_TYPE] == TYPE_CUSTOMER_STATE and event[INDEX_TYPE] == TYPE_CUSTOMER_STATE:
                # read it off and record it 
                # into a list of tuples, for future parsing
                datum = [meal, prev_event[INDEX_LABEL], prev_action[INDEX_LABEL], event[INDEX_LABEL], prev_event[INDEX_TIMESTAMP], event[INDEX_TIMESTAMP]]
                data.append(datum)
                # print("added " + str(prev_event[INDEX_LABEL]) + " --" + prev_action[INDEX_LABEL] + "--> " + event[INDEX_LABEL])

                prev_event = event
                prev_action = no_op

            elif prev_event[INDEX_LABEL] == 'NONE':
                prev_event = event

            else:
                # print("ERR")
                # print(prev_event)
                # print(prev_action)
                # print(event)
                # print("~~~")
                pass

        data_individual_meals.append(data)
        data_all.extend(data)

    # transform these readings into a dataframe
    # this allows filtering by meal or events

    transition_log = pd.DataFrame(data_all, columns = ['Meal ID', 'before', 'operation', 'after', 'bt', 'at'])

    print("data")
    print(data)
    print("transition_log")
    print(transition_log)
    print("data individual meals")
    print(data_individual_meals)
    print("data all")
    print(data_all)
    return transition_log, data_individual_meals, data_all, customer_states, log



def cm_analysis(y_true, y_pred, title, labels, ymap=None, figsize=(14,10), normalize_by=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """

    filename = export_prefix + 'cm-' + title

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    y_true = y_true.values
    y_pred = y_pred.values

    sns.set(font_scale=1.5)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    normal_axis = None
    if normalize_by == None:
        normal_axis = None
    elif normalize_by == 'row':
        normal_axis = 1
    elif normalize_by == 'col':
        normal_axis = 0

    # What are we going to normalize by? The rows?
    if normal_axis != None:
        cm_sum = np.sum(cm, axis=normal_axis, keepdims=True)
    else:
        cm_sum = np.sum(cm, axis=normal_axis)
    
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                # s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                if not np.isnan(p) and p != 0:
                    annot[i, j] = '%.1f%%' % (p)
                else:
                    annot[i, j] = ''
            elif c == 0:
                annot[i, j] = ''
            else:
                if p is not np.nan:
                    annot[i, j] = '%.1f%%' % (p)
                else:
                    annot[i,j] = ''
                # annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Person A'
    cm.columns.name = 'Person B'
    fig, ax = plt.subplots(figsize=figsize)

    print(annot)
    # print(max(annot))
    # vmin=0, vmax=100
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, square=True, annot_kws={"size": 10}, cbar=False)

    ax.set_title(title)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    plt.clf()


def activity_fingerprint(df, labels, title, ymap=None, figsize=(14,10)):
    # Update to normalize
    # https://stackoverflow.com/questions/35692781/python-plotting-percentage-in-seaborn-bar-plot

    sns.set(font_scale=2)
    filename = export_prefix + 'act-bars-' + title
    # df.columns = ['Meal ID', 'timestamp', 'table-state', 'person-A', 'person-B']

    # sns.set_palette(sns.color_palette("colorblind", len(labels)))

    df = pd.melt(df, id_vars=['Meal ID', 'timestamp'], value_vars=['person-A', 'person-B'])
    # print(df.columns)
    # print(df)

    # sns.countplot(df['person-A'])
    # sns.countplot(df['person-B'])
    histo = df['value'].value_counts().reindex(labels, fill_value=0)

    # print(histo)
    # print("~~~~~~")
    f = open(filename + "_histo.txt", "w")
    f.write(str(histo))
    f.close()

    # df['value'].value_counts().plot(kind='bar', rot=0)

    ax = sns.countplot(x='value', order=labels, data=df)
    ax.set_xticklabels(rotation=90, labels=labels)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Activity Label")
    plt.tight_layout()
    plt.savefig(filename + "_count.png")
    plt.clf()


    # sns.catplot(x="value", y=['person-A', 'person-B'], hue="variable", kind="box", data=df)
        # sns.catplot(x="value", y="variable", hue="smoker", kind="box", data=df)
    # fig, ax = plt.subplots()
    # ax.set_title(title)
    # plt.title("Activity Distribution Per Category")
    # ax.xlabel("Activity Label")
    # ax.ylabel("# of Frames")

    # sns.barplot(x = "class", y = "survived", hue = "embark_town", data = titanic_dataset)

# Make nice graph
# import pydot_ng as pydot
# from pydot_ng import Dot, Edge,Node

def make_graph(data, graph_label, customer_states):
    print(data)
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


    
    g.write_png("graphs/table_states-" + graph_label + ".png")
    print("Output graph " + graph_label)

    df = pd.DataFrame(data_overview, columns = ['before', 'operation', 'after', 'probability']) 
    df.to_csv(export_prefix + 'table_states-' + graph_label + '.csv')

def clean_df(df):
    # columns = ['Meal ID', 'timestamp', 'table-state', 'person-A', 'person-B']
    # print(df[1000:1010])
    # df = reduce_Y_labels(df)

    person_a_code = 'person-A'
    person_b_code = 'person-B'
    tag_look_partner = 'look:partner'

    # mini_transform = {'looking:PersonA': tag_look_partner, 'looking:PersonB': tag_look_partner}
    df = df.replace({person_a_code: activity_shorterdict})
    df = df.replace({person_b_code: activity_shorterdict})

    return df

if __name__ == "__main__":
    # transition log columns = ['Meal ID', 'before', 'operation', 'after', 'bt', 'at']
    transition_log, data_individual_meals, data_all, customer_states, log = import_meals()

    graph_data = data_individual_meals
    graph_data.append(data_all)

    graph_names = filenames_all
    graph_names.append("all")

    for i in range(len(graph_data)):
        data_list = graph_data[i]
        graph_name = graph_names[i]
        # make_graph(data_list, graph_name, customer_states)


    data = []
    for entry in log.keys():
        datum = []
        value = log[entry] 
        # value = value / sum(value)
        # ['Meal ID', 'timestamp', 'table-state', 'person-A', 'person-B']
        datum = [entry[0], entry[1], value[0], value[1], value[2]]
        data.append(datum)


    # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ['Meal ID', 'timestamp', 'table-state', 'person-A', 'person-B']) 
    df = clean_df(df)

    # POST ANALYSIS
    table_state_labels = df['table-state'].unique()
    # print all the unique table state labels found
    # print(table_state_labels)


    data = []
    table_state_emissions = {}
    activity_labels = activitydict_display
    labels = list(activity_labels)

    cm_analysis(df['person-A'], df['person-B'], 'all', labels)

    # Operations per-table-state
    for ts in table_state_labels:
        datum = [ts]

        df_ts = df.loc[(df['table-state'] == ts)]
        total = len(df_ts)

        cm_analysis(df_ts['person-A'], df_ts['person-B'], ts, labels)
        activity_fingerprint(df_ts, labels, ts)

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
    # print(d_emi.columns)
    d_emi.to_csv(export_prefix + 'observations.csv')

    df.to_csv(export_prefix + 'all_data.csv')


    print("Done")

