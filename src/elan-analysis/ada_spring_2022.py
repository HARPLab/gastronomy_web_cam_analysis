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

FLAG_EXPORT_ACTIVITY_LENGTH_STATS   = True
FLAG_EXPORT_ACTIVITY_SCATTER        = False
FLAG_EXPORT_CM                      = True

# Import the five annotated files
filename_root = "8-21-18"
filenames_all = ['8-13-18', '8-17-18', '8-18-18', '8-21-18', '8-9-18']

export_prefix = 'outputs-2022/'

time_multiplier_val = 33.3333

timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'NONE': 21}


activity_shorterdict = {'away-from-table': 'away', 'idle': 'idle', 'eating': 'eating', 'drinking': 'drinking', 'talking': 'talking', 'ordering': 'ordering', 
            'standing':'standing','talking:waiter': 'talk:waiter', 'looking:window': 'look:window', 'looking:waiter': 'look:waiter', 
            'reading:bill':'read:bill', 'reading:menu': 'read:menu',
            'paying:check': 'pay:check', 'using:phone': 'use:phone', 'using:napkin': 'use:napkin', 'using:purse': 'use:purse', 'using:glasses': 'use:glasses',
            'using:wallet': 'use:wallet', 'looking:PersonA': 'look:partner', 'looking:PersonB':'look:partner', 'takeoutfood':'takeout', 'NONE': 'NONE', 'cleaning-up':'NONE', 'leaving-table':'NONE'}

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

def to_3sf(number):
    to_str = "{:.2e}".format(number)
    expon = int(to_str.split("e")[1])
    if expon in [-1, -2, 0, 1]:
        to_str = "{:.3f}".format(number)
    if expon in [2, 3, 4, 5, 6, 7, 8, 9]:
        to_str = to_str.split("e")[0] + "e" + str(expon)
    if expon in [-2, -3, -4, -5, -6, -7, -8, -9]:
        to_str = to_str.split("e")[0] + "e" + str(expon)
    return to_str

def to_1sf(number):
    # to_str = "{:.1e}".format(number)
    to_str = "{0:.1f}".format(number)
    # expon = int(to_str.split("e")[1])
    # if expon in [-1, -2, 0, 1]:
    #     to_str = "{:.3f}".format(number)
    # if expon in [2, 3, 4, 5, 6, 7, 8, 9]:
    #     to_str = to_str.split("e")[0] + "e" + str(expon)
    # if expon in [-2, -3, -4, -5, -6, -7, -8, -9]:
    #     to_str = to_str.split("e")[0] + "e" + str(expon)
    return to_str

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
    TYPE_CUSTOMER_A = 'customer_a'
    TYPE_CUSTOMER_B = 'customer_b'



    overall_flow = []
    overall_flow_customer_events = []
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
                        beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//time_multiplier_val)
                        ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//time_multiplier_val)
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
                        beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//time_multiplier_val)
                        ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//time_multiplier_val)
                        # print(beginning_frame, ending_frame)
                       
                        label = LABEL_NONE
                        # for each of the labeled activities in this block of time
                        for anno in temp: ## another single iteration loop
                            label = anno.text

                        # fill in with blank notes if we don't have something here
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
                        beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//time_multiplier_val)
                        ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//time_multiplier_val)
                        
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

                        overall_flow_customer_events.append((meal, beginning_frame, ending_frame, activity, TYPE_CUSTOMER_A))


            elif child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonB':
                 for annotation in child:
                    # print("adding PersonB's annotations...")
                    for temp in annotation: ## this should only loop once, couldnt figure out how to access a child xml tag without looping
                        ## beginning frame
                        beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//time_multiplier_val)
                        ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//time_multiplier_val)
                        activity = LABEL_NONE
                        for anno in temp: ## another single iteration loop
                            activity = anno.text

                        # for every ms, add to the listing
                        for f_id in range(beginning_frame, ending_frame):

                            if (meal, f_id) not in log.keys():
                                log[(meal, f_id)] = copy.copy(BLANK_LABELS)
                            
                            log[(meal, f_id)][2] = activity

                                # print(log[meal, f_id])

                        overall_flow_customer_events.append((meal, beginning_frame, ending_frame, activity, TYPE_CUSTOMER_B))

                            
    # print(log)

    # print(overall_flow)
    # Process the overall flow
    # list of (meal_id, timestamp, label)

    # Fill in weird gaps in overall_flow
    # where there's places with no middle section between customer_states

    overall_flow = sorted(overall_flow)
    BLANK_SLATE_CUSTOMER_STATE = (meal, 0, 0, LABEL_NONE, TYPE_CUSTOMER_STATE)
    BLANK_SLATE_WAITER_STATE = (meal, 0, 0, LABEL_NONE, TYPE_WAITER)

    prev_meal = None
    prev_customer_state = None

    meal_prev, start_prev, end_prev, event_prev, e_type_prev = BLANK_SLATE_CUSTOMER_STATE
    meal_waiter_prev, start_waiter_prev, end_waiter_prev, event_waiter_prev, e_type_waiter_prev = BLANK_SLATE_WAITER_STATE

    new_overall_flow = []
    for o in overall_flow:
        is_no_edit = True

        m_current, start_current, end_current, event_current, e_type_current = o

        if m_current != prev_meal:
            print("New meal to parse!")
            prev_meal = m_current
            prev_customer_state = BLANK_SLATE_CUSTOMER_STATE

        if e_type_current == TYPE_WAITER:
            meal_waiter_prev, start_waiter_prev, end_waiter_prev, event_waiter_prev, e_type_waiter_prev = o

        elif e_type_current == TYPE_CUSTOMER_STATE:
            meal_cs_prev, start_cs_prev, end_cs_prev, event_cs_prev, e_type_cs_prev = prev_customer_state

            # print()
            # print(event_cs_prev + " -> " + event_current)
            # print(e_type_current + "\t: " + event_current)
            
            if event_current == 'table-waiting':
                is_no_edit = False

                if event_cs_prev == 'reading-menus':
                    new_event_name = 'ready-to-order'
                elif event_cs_prev == 'ready-to-order':
                    new_event_name = 'waiting-for-food'
                elif event_cs_prev == 'ready-for-cleanup':
                    new_event_name = 'waiting-for-bill'    
                elif event_cs_prev == 'ready-for-bill':
                    new_event_name = 'paying-bill'
                elif event_cs_prev == 'eating':
                    new_event_name = 'ready-for-cleanup'
                elif event_cs_prev == 'ready-for-final-check':
                    new_event_name = 'paying-final-check'
                elif event_cs_prev == 'ready-to-leave':
                    new_event_name = 'leaving'
                else:
                    new_event_name = 'testy'
                    print("Needs a change!")

                customer_states.append(new_event_name)

                new_o = (m_current, start_current, end_current, new_event_name, e_type_current)
                prev_customer_state = new_o
            
            else:
                prev_customer_state = o

            if False:
                gap_size = start_current - end_prev
                if gap_size > 1000:
                    print("Hey we need something here!")
                    print(gap_size)
                    print(e_type_current)
                    print(event_prev + " -> " + event_current)
                    print(str(start_current) + " -> " + str(end_prev))

        if is_no_edit:
            new_overall_flow.append(o)
        else:
            # print("Added new guy!")
            new_overall_flow.append(new_o)


    # print("-------------------------------")
    
    print("***Labeling unlabeled table-waiting events")
    overall_flow = new_overall_flow
    
    # for o in overall_flow:
    #     m_current, start_current, end_current, event_current, e_type_current = o
    #     if e_type_current == TYPE_CUSTOMER_STATE:
    #         meal_cs_prev, start_cs_prev, end_cs_prev, event_cs_prev, e_type_cs_prev = prev_customer_state
    #         # print(o)
    #     else:
    #         # print(e_type_current + "\t: " + event_current + "\t" + str(start_current) + " - " + str(end_current))
        
    #         # print(event_cs_prev + " -> " + event_current)
    #         # print(e_type_current + "\t: " + event_current)
    #         pass

    # print(overall_flow)
    flow_per_meal = {}
    for meal in filenames_all:
        flow_per_meal[meal] = []

    for meal, start, end, event, e_type in overall_flow:
        flow_per_meal[meal].append([start, end, event, e_type])



    # for meal in filenames_all:
    #     df = pd.DataFrame(flow_per_meal[meal], columns = ['start_time', 'end_time', 'event_label', 'event_of']) 
    #     df.to_csv(export_prefix + 'readable_highlevel_' + str(meal) + '.csv')

    # # CUSTOMER EVENT LENGTH LOG
    # customer_events_per_meal = {}
    # for meal in filenames_all:
    #     customer_events_per_meal[meal] =[]

    # for meal, start, end, event, e_type in overall_flow:
    #     customer_events_per_meal[meal].append([start, end, event, e_type])

    # for meal in filenames_all:
    #     df_events = pd.DataFrame(customer_events_per_meal[meal], columns = ['start_time', 'end_time', 'event_label', 'event_of', 'group_state']) 
    #     df_events.to_csv(export_prefix + 'all_events_' + str(meal) + '.csv')


    waiter_events = list(set(waiter_events))
    customer_states = list(set(customer_states))

    print(waiter_events)
    print(customer_states)

    no_op = 'NOP'


    data_individual_meals = []
    data_all = []

    # Import all the meals on the list
    for meal in filenames_all:
        print("Adding timeline info for " + meal)
        timeline = list(filter(lambda entry: entry[0] == meal, overall_flow))
        timeline.sort(key=lambda x: x[1])
        prev_event = (meal, 0, 0, 'NONE', TYPE_CUSTOMER_STATE)
        prev_action = (meal, 0, 0, no_op, TYPE_WAITER)
        data = []

        # Constants for reading/interpreting each of the entries
        INDEX_MEALID = 0
        INDEX_TIMESTAMP_START = 1
        INDEX_TIMESTAMP_END = 2
        INDEX_LABEL = 3
        INDEX_TYPE = 4

        for event in timeline:
            # print(event)
            # if the waiter took a novel action, ex not a NO-OP, then hold onto it
            print("===")
            print(prev_event)
            print(prev_action)
            print(event)
            

            if prev_event[INDEX_TYPE] == TYPE_CUSTOMER_STATE and event[INDEX_TYPE] == TYPE_WAITER:
                # if it's the unique waiter event
                if event[INDEX_LABEL] not in ['arriving', 'leaving']:
                    prev_action = event
                    datum = [meal, prev_event[INDEX_LABEL], prev_action[INDEX_LABEL], prev_event[INDEX_LABEL], event[INDEX_TIMESTAMP_START], event[INDEX_TIMESTAMP_END]]
                    data.append(datum)
                    print("!!1 pushed back waiter event")

            # if we have a transition between two events
            elif prev_event[INDEX_TYPE] == TYPE_CUSTOMER_STATE and event[INDEX_TYPE] == TYPE_CUSTOMER_STATE:
                # read it off and record it 
                # into a list of tuples, for future parsing

                #TODO verify the stamps are all good
                datum = [meal, prev_event[INDEX_LABEL], prev_action[INDEX_LABEL], event[INDEX_LABEL], prev_event[INDEX_TIMESTAMP_END], event[INDEX_TIMESTAMP_START]]
                data.append(datum)
                print("added " + str(prev_event[INDEX_LABEL]) + " --" + prev_action[INDEX_LABEL] + "--> " + event[INDEX_LABEL])

                prev_event  = event
                new_nop     = (meal, start, end, no_op, TYPE_WAITER)
                prev_action = new_nop

            elif prev_action[INDEX_TYPE] == TYPE_WAITER and event[INDEX_TYPE] == TYPE_WAITER:
                if event[INDEX_LABEL] not in ['arriving', 'leaving']:
                    prev_action = event
                    #TODO record a no-op
                    print("!!2 pushed back waiter event")
            else:
                print("ERR")
                print(prev_event)
                print(prev_action)
                print(event)
                print("~~~")
                pass

            if prev_event[INDEX_LABEL] == 'NONE' and event[INDEX_TYPE] == TYPE_CUSTOMER_STATE:
                prev_event = event
                print("Replaced event NONE")


        print("~~~~~~~~~~~~~~")
        print(data)
        # exit()
        data_individual_meals.append(data)
        data_all.extend(data)

    # transform these readings into a dataframe
    # this allows filtering by meal or events

    transition_log = pd.DataFrame(data_all, columns = ['Meal ID', 'before', 'operation', 'after', 'bt', 'at'])

    if False:
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

    # print(annot)
    # print(max(annot))
    # vmin=0, vmax=100
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, square=True, annot_kws={"size": 10}, cbar=False)

    ax.set_title(title)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    plt.clf()


def to_seconds(value):
    return (value * time_multiplier_val) / 1000.0

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
    # cm_sum = np.sum(cm, axis=normal_axis)

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
import pydot as pydot
# from pydot import Dot, Edge, Node

def make_graph(data, graph_label, customer_states):
    # print(data)
    data_overview = []

    g = pydot.Dot()
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

        new_node = pydot.Node(label)
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

    unique_states_a = df['person-A'].unique()
    unique_states_b = df['person-B'].unique()

    for state in unique_states_a:
        if state not in activitydict_display:
            print("Extra states in A " + state)
    
    for state in unique_states_b:
        if state not in activitydict_display:
            print("Extra states in B " + state)


    return df

def make_scatter_of_var(df_events, x, y, activity, fname):
    activity_fn = activity.replace(":", "-")

    type_of_graph = x

    # print(df_events.columns)

    plot = df_events.plot.scatter(x=type_of_graph, y=y)#, c='table-state');
    # y1, r, *_ = np.polyfit(df_events[type_of_graph], df_events['norm_length'], 1)
    # poly = np.poly1d(y1)
    # x = np.linspace(df_events[type_of_graph].min(), df_events[type_of_graph].max())
    # plt.plot(x, poly(x),"r--", label=f"linear (y = {y1:0.2f})")

    plot.set_ylabel("Time compared to mean")
    plot.set_xlabel("Time from end to end of interval")
    plot.set_title("Event Length compared to Period Length\n" + activity)
    plt.xticks(rotation=90)
    plt.gca().invert_xaxis()

    plt.savefig(export_prefix + fname + activity_fn + '.png', bbox_inches='tight', pad_inches=0.01)
    plt.clf()

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
        make_graph(data_list, graph_name, customer_states)


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
    print(table_state_labels)
    # print all the unique table state labels found
    # print(table_state_labels)

    data = []
    table_state_emissions = {}
    activity_labels = activitydict_display
    labels = list(activity_labels)

    df_events = copy.copy(df)
    df_events['ts_subgroup'] = (df_events['table-state'] != df_events['table-state'].shift(1)).cumsum()

    tablestate_subgroup_endtimes = {}

    for subgroup in df_events['ts_subgroup'].unique():
        block = df_events[df_events['ts_subgroup'] == subgroup]
        act_len = block.shape[0]
        event_datum = block.iloc[-1].to_dict()

        tablestate_subgroup_endtimes[subgroup] = event_datum['timestamp']
    
    # make analysis of lengths of different types of events
    # and how those change within different group states

    df_events_a = copy.copy(df_events)
    df_events_b = copy.copy(df_events)

    # TODO add cuts for meals
    # https://datascience.stackexchange.com/questions/41428/how-to-find-the-count-of-consecutive-same-string-values-in-a-pandas-dataframe/41431
    df_events_a['subgroup'] = (df_events_a['person-A'] != df_events_a['person-A'].shift(1)).cumsum()
    df_events_b['subgroup'] = (df_events_b['person-B'] != df_events_b['person-B'].shift(1)).cumsum()
    
    events_a = []
    events_b = []

    event_datums = []
    # event_datum = [meal, start, length, person, group_state]
    for subgroup in df_events_a['subgroup'].unique():
        block = df_events_a[df_events_a['subgroup'] == subgroup]
        act_len = block.shape[0]
        event_datum = block.iloc[0].to_dict()
        event_datum['length'] = act_len
        event_datum['activity'] = event_datum['person-A']
        tste = tablestate_subgroup_endtimes[event_datum['ts_subgroup']] - event_datum['timestamp']
        tete = tablestate_subgroup_endtimes[event_datum['ts_subgroup']] - event_datum['timestamp'] + act_len
        if tete < 0:
            tete = 0
        if tste < 0:
            tste = 0

        event_datum['time_from_start_to_end_of_groupstate'] = tste
        event_datum['time_from_end_to_end_of_groupstate']   = tete

        # removing key for B since this just addresses a streak of person a
        event_datum.pop('person-B')
       
        event_datums.append(event_datum)

    for subgroup in df_events_b['subgroup'].unique():
        block = df_events_b[df_events_b['subgroup'] == subgroup]
        act_len = block.shape[0]
        event_datum = block.iloc[0].to_dict()
        event_datum['length'] = act_len
        event_datum['activity'] = event_datum['person-B']
        tste = tablestate_subgroup_endtimes[event_datum['ts_subgroup']] - event_datum['timestamp']
        tete = tablestate_subgroup_endtimes[event_datum['ts_subgroup']] - event_datum['timestamp'] + act_len
        if tete < 0:
            tete = 0
        if tste < 0:
            tste = 0

        # TODO verify why this wrap is needed
        event_datum['time_from_start_to_end_of_groupstate'] = tste
        event_datum['time_from_end_to_end_of_groupstate']   = tete

        # removing key for A since this just addresses a streak of person b
        event_datum.pop('person-A')

        event_datums.append(event_datum)

    df_events = pd.DataFrame(event_datums)

    activity_length_dict = {}
    for activity in activity_labels:
        single_activity = df_events.loc[(df_events['activity'] == activity)]
        activity_length_dict[activity] = single_activity['length'].mean()
    
    df_events['norm_length'] = df_events.apply(lambda x: x['length'] / activity_length_dict[x['activity']], axis=1)
    # print(df_events['norm_length'])

    if FLAG_EXPORT_ACTIVITY_LENGTH_STATS:
        # https://towardsdatascience.com/violin-strip-swarm-and-raincloud-plots-in-python-as-better-sometimes-alternatives-to-a-boxplot-15019bdff8f8
        # boxplot = df_events.boxplot(column=['length'], by=['activity']) #, sort=False)
        boxplot = sns.stripplot(y='length', x='activity', data=df_events) 
        boxplot.set_ylabel("Time in ms")
        boxplot.set_xlabel("Activity")
        boxplot.set_title("Lengths of Events")
        plt.xticks(rotation=90)
        plt.savefig(export_prefix + 'activity_length_histo.png', bbox_inches='tight', pad_inches=0.01)
        plt.clf()

        boxplot = sns.stripplot(y='norm_length', x='activity', data=df_events) 
        boxplot.set_ylabel("Time in ratio")
        boxplot.set_xlabel("Activity")
        boxplot.set_title("Lengths of Events")
        plt.xticks(rotation=90)
        plt.savefig(export_prefix + 'activity_norm_histo.png', bbox_inches='tight', pad_inches=0.01)
        plt.clf()

        boxplot = df_events.boxplot(column=['length'], by=['activity']) #, sort=False)
        boxplot.set_ylabel("Time in ms")
        boxplot.set_xlabel("Activity")
        boxplot.set_title("Lengths of Events")
        plt.xticks(rotation=90)
        plt.savefig(export_prefix + 'activity_length_boxplot.png', bbox_inches='tight', pad_inches=0.01)
        plt.clf()
        print("EXPORTED activity length graphs")

        # combinations of activity lengths and group states
        plot = df_events.plot.scatter(x='time_from_start_to_end_of_groupstate', y='norm_length')#, c='table-state');
        plot.set_ylabel("Time compared to mean")
        plot.set_xlabel("Time from start to end of interval")
        plot.set_title("Event Length compared to Period Length")
        plt.xticks(rotation=90)
        plt.savefig(export_prefix + 'time_start_to_end_scatter.png', bbox_inches='tight', pad_inches=0.01)
        plt.clf()

        # combinations of activity lengths and group states
        plot = df_events.plot.scatter(x='time_from_end_to_end_of_groupstate', y='norm_length') #, c='table-state');
        plot.set_ylabel("Time compared to mean")
        plot.set_xlabel("Time from end to end of interval")
        plot.set_title("Event Length compared to Period Length")
        plt.xticks(rotation=90)
        plt.savefig(export_prefix + 'time_end_to_end_scatter.png', bbox_inches='tight', pad_inches=0.01)
        plt.clf()

        total_time = df_events['length'].sum()
        all_means = []
        all_percents = []

        for activity in activity_labels:
            df_single_activity = df_events.loc[(df_events['activity'] == activity)]
            print(activity)
            
            mean_time = to_seconds(df_single_activity['length'].mean())
            all_means.append(mean_time)
            mean_time = to_1sf(mean_time)

            # print("Avg time: ")
            # print(str(mean_time) + " s")
            # print(df_single_activity.shape)
            percent_of_total = df_single_activity['length'].sum() / total_time
            percent_of_total = percent_of_total * 100.0
            all_percents.append(percent_of_total)
            percent_of_total = to_1sf(percent_of_total)
            # print(str(percent_of_total) + "\\%")

            num_events = len(df_single_activity['Meal ID'].unique())

            # output for latex
            print(mean_time + "s" + "\t& " + percent_of_total + "\\% \t&" + str(num_events) + "\\\\")

        print("TOTAL PERCENTS: " + str(sum(all_percents)))

        if FLAG_EXPORT_ACTIVITY_SCATTER:
            for activity in activity_labels:
                # combinations of activity lengths and group states
                type_of_graph = 'time_from_start_to_end_of_groupstate'
                fname = 'tste-act-'
                make_scatter_of_var(df_single_activity, type_of_graph, 'norm_length', activity, fname)

                # combinations of activity lengths and group states
                type_of_graph = 'time_from_end_to_end_of_groupstate'
                fname = 'tete-act-'
                make_scatter_of_var(df_single_activity, type_of_graph, 'norm_length', activity, fname)

        print("EXPORTED scatter of time before end")
        # exit()

    types_of_metric = ['norm_length', 'length']
    type_of_y = ['time_from_start_to_end_of_groupstate', 'time_from_end_to_end_of_groupstate']

    if FLAG_EXPORT_ACTIVITY_SCATTER:
        all_datum = []
        for ts in df_events['table-state'].unique():
            for activity in activity_labels:
                df_target = df_events.loc[(df_events['activity'] == activity) & (df_events['table-state'] == ts)]

                # type_of_graph = 'time_from_start_to_end_of_groupstate'
                # fname = 'cross-tste-'
                # make_scatter_of_var(df_target, type_of_graph, 'norm_length', activity, fname)

                # # combinations of activity lengths and group states
                # type_of_graph = 'time_from_end_to_end_of_groupstate'
                # fname = 'cross-tete-' + ts + "-" + activity.replace(":", '-')
                # make_scatter_of_var(df_target, type_of_graph, 'norm_length', activity, fname)

                for metric in types_of_metric:
                    for y_type in type_of_y:
                        try:
                            y1, r, *_ = np.polyfit(df_target[y_type], df_target[metric], 1)
                        except TypeError:
                            y1 = 0

                        polarity = int(0)
                        if y1 < 0:
                            polarity = int(-1)
                        if y1 > 0:
                            polarity = int(+1)

                        datum = [ts, activity, metric, y_type, y1, polarity]
                        all_datum.append(datum)

        df_trends = pd.DataFrame(all_datum, columns = ['table-state', 'activity', 'metric', 'y_type', 'y1', 'polarity'])

        # coloring = sns.color_palette("vlag", as_cmap=True)
        coloring = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
        # coloring = sns.color_palette("coolwarm", as_cmap=True)
        # coloring = sns.color_palette("Spectral", as_cmap=True)
        # coloring = "YlOrRd_r"
        # coloring = sns.color_palette("RdYlGn")
        # https://stackoverflow.com/questions/56536419/how-to-set-center-color-in-heatmap

        for metric in types_of_metric:
            for y_type in type_of_y:
                df_target = df_trends.loc[(df_trends['metric'] == metric) & (df_trends['y_type'] == y_type)]
                df_target = df_target.pivot(index='activity', columns='table-state', values='polarity')

                # cmap=coloring

                res_labels = sns.heatmap(df_target, annot=True, square=True, annot_kws={"size": 6}, cbar=False)
                for t in res_labels.texts: 
                    if t.get_text() == "0":
                        t.set_text("")

                res_labels.xaxis.tick_top() # x axis on top
                res_labels.xaxis.set_label_position('top')
                res_labels.set_xticklabels(res_labels.get_xmajorticklabels(), rotation=90)

                # plt.set_title("Trends over time")
                plt.savefig(export_prefix + metric + y_type + "_impact.png", bbox_inches='tight', pad_inches=0.01)
                plt.clf()


    if FLAG_EXPORT_CM:
        target_aux1 = copy.copy(df)
        target_aux2 = copy.copy(df)
        target_aux2['person-A'] = df['person-B']
        target_aux2['person-B'] = df['person-A']
        
        target_aux = target_aux1.append(target_aux2)

        cm_analysis(df['person-A'], df['person-B'], 'all-ab', labels)
        cm_analysis(target_aux['person-A'], target_aux['person-B'], 'all-targetaux', labels)

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
    print("Exported csvs of all")


    print("Done")

