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
import pprint

import seaborn as sns
from matplotlib import cm
import matplotlib

# Make nice graph
import pydot_ng as pydot
from pydot_ng import Dot, Edge,Node




def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
    print(root.tag)



FLAG_CONSOLIDATE_WAITER_EVENTS = False #True
FLAG_TRANSLATE_WAITING_TYPES = False


filename_root = "8-21-18"
filenames_all = ['8-21-18'] #['8-13-18', '8-17-18', '8-18-18', '8-21-18', '8-9-18']

timedict = {}
activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5, 'standing':6,
            'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill':10, 'reading:menu': 11,
            'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
            'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB':19, 'takeoutfood':20, 'NONE': 21}




## database initialization
frame_to_poseact ={} #f_id -> (posedata, personAactivity, personBactivity)
frame_to_poseact ={} #f_id -> (personAactivity, personBactivity)
frames = {}

log = {}
# log[(mealid, index)] = {customerTransition, }

LABEL_NONE = ""

E_MEALID    = 0
E_START     = 1
E_END       = 2
E_LABEL     = 3
E_TYPE      = 4


TYPE_WAITER         = 'waiter-action'
TYPE_CUSTOMER_STATE = 'customer_state'

#Event Format = (meal, beginning_frame, ending_frame, label, TYPE_WAITER)
BLANK_EVENT_TABLE   = [LABEL_NONE,      0,      0, 'EMPTY', TYPE_CUSTOMER_STATE]
BLANK_EVENT_WAITER  = [LABEL_NONE,      0,      0, 'NOP',   TYPE_WAITER]

LOG_INDEX_TABLESTATE    = 0
LOG_INDEX_PERSON_A      = 1
LOG_INDEX_PERSON_B      = 2
LOG_INDEX_WAITER_STATE  = 3

waiter_ignore_list = ['arriving', 'leaving']
# waiter_action_hierarchy = ['take:info']

# WAITER EVENTS
# ['take:order', 'take:info', 'clear:table', 'bring:food', 
# 'take:dishes', 'bring:check', 'take:bill', 'take:check', 
# 'bring:menus', 'bring:bill', 'bring:drinks']
wevent_priority = {}
wevent_priority['bring:to-seat']   = -1
wevent_priority['take:info']    = 100
wevent_priority['bring:drinks'] = 80
wevent_priority['take:dishes']  = 1
wevent_priority['bring:menus']  = 1

wevent_priority['take:order']   = 90
wevent_priority['clear:table']  = 1
wevent_priority['bring:food']   = 1

wevent_priority['bring:check']  = 1
wevent_priority['take:bill']    = 1
wevent_priority['take:check']   = 1
wevent_priority['bring:bill']   = 1


wait_type_event = {}
wait_type_event['reading-menus'] = 'XXXwaiting-for-food'
wait_type_event['eating'] = 'waiting-for-cleanup'
wait_type_event['NONE'] = 'waiting-?'
wait_type_event['ready-to-order'] = 'waiting-for-food'
wait_type_event['ready-for-cleanup'] = 'waiting-5'
wait_type_event['ready-for-bill'] = 'waiting-for-final-check'
wait_type_event['ready-for-final-check'] = 'waiting-to-leave?'
wait_type_event['ready-to-leave'] = 'waiting-8'

def get_com(e):
    return int((e[E_START] + e[E_END]) / 2.0)

def consolidate_event_list(old_timeline):
    timeline = copy.copy(old_timeline)

    # sort them sequentially
    # keep meals together, but sort by starting timestamp
    timeline.sort(key=lambda x: (x[E_MEALID], x[E_START]))

    # if this is a list of customer states,
    # then add a first stage of "empty" at the beginning timestamp
    first = timeline[0]
    timeline_type = first[E_TYPE]
    if timeline_type == TYPE_CUSTOMER_STATE:
        print("First item is a customer action")

        if first[E_LABEL] != BLANK_EVENT_TABLE[E_LABEL]:
            customer_start = BLANK_EVENT_TABLE
            customer_start[E_MEALID]    = first[E_MEALID]
            customer_start[E_START]     = first[E_START]
            customer_start[E_END]       = first[E_START]

            timeline = [customer_start] + timeline

    new_timeline = []
    prev_event = BLANK_EVENT_TABLE

    # merge sequential identical groupings
    for datum in timeline:
        if timeline_type == TYPE_CUSTOMER_STATE and datum[E_LABEL] == 'table-waiting':
            if FLAG_TRANSLATE_WAITING_TYPES:
                prev = new_timeline[-1]
                label = wait_type_event[prev[E_LABEL]]
                datum[E_LABEL] = label

            length = datum[E_END] - datum[E_START]
            print("length: " + length)


        # if they're identical, merge them
        if datum[E_LABEL] == prev_event[E_LABEL] and datum[E_LABEL] != BLANK_EVENT_TABLE[E_LABEL]:
            prev_unit = new_timeline.pop()
            datum = [datum[0], int(prev_unit[1]), int(datum[2]), datum[3], datum[4]]

        new_timeline.append(datum)
        prev_event = datum

    pprint.pprint(new_timeline)

    # now that adjacent same-events consolidated, look for super close events
    if timeline_type == TYPE_WAITER:
        merged_timeline = []
        prev_event = BLANK_EVENT_WAITER
        for t in new_timeline:
            overlap = (t[E_START] - prev_event[E_END])
            if overlap < 1000 and prev_event != BLANK_EVENT_WAITER:

                print("\nmerge events?")
                print("overlap = " + str(overlap))
                
                prev_event = merged_timeline.pop()
                try:
                    p_prev = wevent_priority[prev_event[E_LABEL]]
                    p_this = wevent_priority[t[E_LABEL]]

                    better_label = ''
                    if p_prev < p_this:
                        better_label = prev_event[E_LABEL]
                    elif p_this < p_prev:
                        better_label = t[E_LABEL]
                    else:
                        better_label = prev_event[E_LABEL] + ',' + t[E_LABEL]
                except KeyError:
                    if t[E_LABEL] not in better_label:
                        better_label = prev_event[E_LABEL] + ',' + t[E_LABEL]

                print(prev_event)
                print(t)

                

                print("winner: " + better_label)
                datum = [t[0], int(prev_event[1]), int(t[2]), better_label, t[4]]
                print(datum)


                merged_timeline.append(datum)
                prev_event = datum

            else:
                merged_timeline.append(t)
                prev_event = t

        print("MERGED TIMELINE")
        pprint.pprint(merged_timeline)

        new_timeline = merged_timeline

    return new_timeline

def combine_wait_and_ts(w_events, ts_events):
    all_e = copy.copy(w_events)
    all_e.extend(copy.copy(ts_events))
    all_e.sort(key=lambda x: (x[E_MEALID], x[E_START]))
   


    return all_e

def generate_links(w_events, ts_events):
    ts_prev = BLANK_EVENT_TABLE
    w_prev = BLANK_EVENT_WAITER
    all_links = []

    all_e = combine_wait_and_ts(w_events, ts_events) 

    pprint.pprint(all_e)
    print("----LINK FORMAT----")

    # review and collate the state transitions to get the percentages
    for e in all_e:
        if e[E_TYPE] == TYPE_CUSTOMER_STATE:

            # if we are moving between two events
            if e[E_LABEL] != ts_prev[E_LABEL] and e[E_MEALID] == ts_prev[E_MEALID]:

                # log the transition
                # if waiter action is not NOP, use it, and wipe to default NOP
                label_op = w_prev[E_LABEL]
                w_prev = BLANK_EVENT_WAITER

                label_a = ts_prev[E_LABEL]
                label_b = e[E_LABEL]

                # TODO: decide if these times are appropriate
                # TODO !!!
                time_a = w_prev[E_START]
                time_b = e[E_END]

                link = [meal, label_a, label_op, label_b, time_a, time_b]
                all_links.append(link)

            
            # If we are moving to the next meal, 
            # don't count the transition, reset the previous event
            if e[E_MEALID] != ts_prev[E_MEALID]:
                ts_event = BLANK_EVENT_TABLE

            ts_prev = e

        # if this is a waiter-type event
        elif e[E_TYPE] == TYPE_WAITER:
            # check if this is within an event, or between events
            if ts_prev[E_START] < e[E_START] and ts_prev[E_END] > e[E_END]:
                label_a = ts_prev[E_LABEL]
                label_b = ts_prev[E_LABEL]
                label_op = e[E_LABEL]
                time_a = ts_prev[E_START]
                time_b = e[E_END]

                link = [meal, label_a, label_op, label_b, time_a, time_b]
                all_links.append(link)
            else:
                w_prev = e
        else:
            print("ALERT: Incorrectly typed event " + e[E_TYPE])


    all_links.sort(key=lambda x: x[4])
    pprint.pprint(all_links, width=320)
    return all_links

def make_graph(w_events, ts_events, meal_id):

    # mix together and order the combined waiter and customer events

    graph_label = meal_id

    waiter_action_set   = set()
    table_states_set    = set()

    for w in w_events:
        waiter_action_set.add(w[E_LABEL])

    for ts in ts_events:
        table_states_set.add(ts[E_LABEL])

    waiter_action_set   = list(waiter_action_set)
    table_states_set    = list(table_states_set)

    all_links = generate_links(w_events, ts_events)


    data_overview = []
    g = Dot()
    g.set_node_defaults(color='lightgray',
                        style='filled',
                        shape='box',
                        fontname='Courier',
                        fontsize='10',
                        fontcolor='white')
    colors_viridis = cm.get_cmap('viridis', len(table_states_set))

    # add nodes for all the relevant states
    for i in range(len(table_states_set)):
        label = table_states_set[i]
        label = label.replace(':', "-")

        new_node = Node(label)
        col = colors_viridis(i)
        col = matplotlib.colors.rgb2hex(col)
        new_node.set_color(col)
        g.add_node(new_node)
 
    #data_hl = [start_time, end_time, label, type]
    # Consolidate labels
    transition_types = defaultdict(list)

    # Calculate the percentage transitions
    for link in all_links:
        m, la, op, lb, at, bt = link

        label = op
        if label == 'arriving' or label == 'leaving':
            pass
        else:
            # m, la, op, lb, at, bt
            la = la.replace(':', "-")
            lb = lb.replace(':', "-")
            op = op.replace(':', "-")

            transition_types[(la, op)].append(lb)

    checklist = set()

    for link in all_links:
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
    print("\t-> Output graph " + graph_label)

    df = pd.DataFrame(data_overview, columns = ['before', 'operation', 'after', 'probability']) 
    df.to_csv('outputs/ada_table_states-' + graph_label + '.csv')


overall_flow = []
waiter_events = []
customer_states = []

simple_timeline = {}

waiter_events_dict      = {}
tablestate_events_dict    = {}


# scan for timeslots first
for meal in filenames_all:
    root = parseXML('../../../Annotations/' + meal + '-michael.eaf')
    timeline = []
    ## start looping through annotation labels
    for child in root:
        if child.tag == 'TIME_ORDER':
            for times in child:
                timedict[times.attrib['TIME_SLOT_ID']] = times.attrib['TIME_VALUE']


# df_log      = pd.DataFrame(data, columns = ['meal_id', 'timestamp', 'TableState', 'PersonA', 'PersonB', 'WaiterAction'])

for meal in filenames_all:
    waiter_events = []
    tablestate_events = []

    root = parseXML('../../../Annotations/' + meal + '-michael.eaf')
    print("Processing meal annotations for " + meal)
    timeline = []

    ## start looping through annotation labels
    for child in root:
        if child.tag == 'TIER' and child.attrib['TIER_ID'] == 'Waiter':
            for annotation in child:
                # print(annotation)
                label = ""
                for temp in annotation:
                    beginning_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF1']])//33.3333)
                    ending_frame = int(int(timedict[temp.attrib['TIME_SLOT_REF2']])//33.3333)
                   
                    label = LABEL_NONE
                    # for each of the labeled activities in this block of time
                    for anno in temp: ## another single iteration loop
                        if anno.text != 'NONE':
                            label += anno.text

                    if label not in waiter_ignore_list:
                        # print("Check")
                        # print("-" + label + "-")
                        if FLAG_CONSOLIDATE_WAITER_EVENTS:
                            label = 'waiter-visit'
                        
                        waiter_events.append([meal, beginning_frame, ending_frame, label, TYPE_WAITER])


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
                        if anno.text != 'NONE':
                            label = anno.text

                    tablestate_events.append([meal, beginning_frame, ending_frame, label, TYPE_CUSTOMER_STATE])

    #merge duplicate events
    waiter_events       = consolidate_event_list(waiter_events)
    tablestate_events   = consolidate_event_list(tablestate_events)

    # pprint.pprint(waiter_events)
    # pprint.pprint(tablestate_events)

    waiter_events_dict[meal]        = waiter_events
    tablestate_events_dict[meal]    = tablestate_events

    make_graph(waiter_events, tablestate_events, meal)

all_waiter_events = []
all_tablestate_events = []

for i in waiter_events_dict.keys():
    # print(i)
    # print(len(tablestate_events_dict[i]))
    # print(len(waiter_events_dict[i]))

    all_waiter_events.extend(waiter_events_dict[i])
    all_tablestate_events.extend(tablestate_events_dict[i])
    # print(len(tablestate_events_dict[i]))
    # print(len(waiter_events_dict[i]))

# print(len(tablestate_events_dict[i]))
# print(len(waiter_events_dict[i]))
exit()
make_graph(all_waiter_events, all_tablestate_events, 'all')


for meal in filenames_all:
    root = parseXML('../../../Annotations/' + meal + '-michael.eaf')
    print("Processing meal annotations for " + meal)
    for child in root:
        if child.tag == 'TIER' and child.attrib['TIER_ID'] == 'PersonA':
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

                        log[(meal, f_id)][LOG_INDEX_PERSON_A] = activity

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
                        
                        log[(meal, f_id)][LOG_INDEX_PERSON_B] = activity






