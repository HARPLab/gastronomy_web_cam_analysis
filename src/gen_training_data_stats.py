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
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys



def parseXML(elanfile):
    tree = ET.parse(elanfile)
    root = tree.getroot()
    return root
class TrainingDataStats :
    def __init__(self, graph_dir, tier_list, distfilename, avgfilename, annotations_root, annotation_files, activitydict):
        self.timedict = {}
        self.activitydict = activitydict
        self.inv_actdict = {v: k for k, v in self.activitydict.items()}
        self.graph_dir = graph_dir
        self.distfilename = distfilename
        self.avgfilename = avgfilename
        self.annotations_root = annotations_root
        self.annotation_names = annotation_files
        self.delta = 1e-8
        self.tier_list = tier_list

    def generate_stats(self):
        activitiesToTotalLength = defaultdict(int)
        activitiesToTotalOccurences = defaultdict(int)
        activitiesToAverageLength = defaultdict(int)
        actToNextActDist = np.zeros((len(self.activitydict.keys()), len(self.activitydict.keys())))
        for annotation_name in self.annotation_names:
            print("parsing data from " + annotation_name + "...")
            root = parseXML(self.annotations_root + annotation_name)
            prev_activity = "NONE"
            for child in root:
                if child.tag == 'TIME_ORDER':
                    for times in child:
                        self.timedict[times.attrib['TIME_SLOT_ID']] = times.attrib['TIME_VALUE']

                elif child.tag == 'TIER' and child.attrib['TIER_ID'] in self.tier_list:
                    for annotation in child:
                        for temp in annotation:  ## this should only loop once, couldnt figure out how to access a child xml tag without looping
                            ## beginning frame
                            beginning_frame = int(int(self.timedict[temp.attrib['TIME_SLOT_REF1']]) // 33.3333)
                            ending_frame = int(int(self.timedict[temp.attrib['TIME_SLOT_REF2']]) // 33.3333)
                            for anno in temp:  ## another single iteration loop
                                activity = anno.text
                            activitiesToTotalLength[activity] += (ending_frame - beginning_frame)
                            activitiesToTotalOccurences[activity] += 1
                            actToNextActDist[self.activitydict[prev_activity]][self.activitydict[activity]] += 1
                            prev_activity = activity
        for activity in activitiesToTotalOccurences.keys():
            activitiesToAverageLength[activity] = activitiesToTotalLength[activity] / activitiesToTotalOccurences[activity]
        actToNextActDist = actToNextActDist / (self.delta + actToNextActDist.sum(axis=1)[:, None])
        return activitiesToAverageLength, actToNextActDist

    def generate_avg_graph(self,activitiesToAverageLength):
        activity_buckets = activitiesToAverageLength.keys()
        avg_length_vals = activitiesToAverageLength.values()
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(activity_buckets, avg_length_vals)
        plt.title("Average Length of Activities")
        plt.xticks(rotation=90)
        plt.xlabel("Activity")
        plt.ylabel("Average Number of Frames per Occurence")
        plt.savefig(graph_dir + self.avgfilename, bbox_inches='tight')
        plt.close()

    def generate_dist_graphs(self,actToNextActDist):
        for i in range(0, actToNextActDist.shape[0]):
            curr_act = self.inv_actdict[i]
            print("generating graph for: " + curr_act )
            act_list = []
            dist_list = []
            for j in range(0, actToNextActDist.shape[1]):
                if actToNextActDist[i][j] > 0 :
                    act_list.append(self.inv_actdict[j])
                    dist_list.append(actToNextActDist[i][j])
            if len(act_list) > 0 :
                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                ax.bar(act_list, dist_list)
                plt.xticks(rotation=90)
                plt.title("Next Activity Distribution for " + curr_act)
                plt.xlabel("Activity")
                plt.ylabel("Probability")
                plt.savefig(graph_dir + curr_act + "-" + self.distfilename, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    timedict = {}
    activitydict = {'away-from-table': 0, 'idle': 1, 'eating': 2, 'drinking': 3, 'talking': 4, 'ordering': 5,
                    'standing': 6,
                    'talking:waiter': 7, 'looking:window': 8, 'looking:waiter': 9, 'reading:bill': 10,
                    'reading:menu': 11,
                    'paying:check': 12, 'using:phone': 13, 'using:napkin': 14, 'using:purse': 15, 'using:glasses': 16,
                    'using:wallet': 17, 'looking:PersonA': 18, 'looking:PersonB': 19, 'takeoutfood': 20,
                    'leaving-table': 21, 'cleaning-up': 22, 'NONE': 23}
    groupactdict = {'NONE': 7, 'reading-menus': 0, 'ready-to-order': 1, 'eating':2, 'ready-for-cleanup':3, 'ready-for-bill':4, 'ready-for-final-check':5, 'ready-to-leave':6}
    inv_actdict = {v: k for k, v in activitydict.items()}
    graph_dir = "/home/mghuang/spring2020_gastronomy_analysis/gastronomy_web_cam_analysis/src/graphs/"
    distfilename = 'groupActDist.png'
    outfilename = 'groupAvgLength.png'
    annotations_root = '/home/mghuang/spring2020_gastronomy_analysis/gastronomy_web_cam_analysis/Annotations/'
    annotation_names = ['8-9-18-michael.eaf', '8-13-18-michael.eaf', '8-17-18-michael.eaf', '8-18-18-michael.eaf',
                        '8-21-18-michael.eaf']
    delta = 1e-8
    tier_list = ["CustomerTransitions"]
    args = sys.argv

    statobj = TrainingDataStats(graph_dir,tier_list,distfilename,outfilename,annotations_root,annotation_names, groupactdict)
    ## start looping through annotation labels
    activitiesToAverageLength, actToNextActDist = statobj.generate_stats()
    print("finished generating stats, starting visualizations")
    statobj.generate_avg_graph(activitiesToAverageLength)
    print("finished average graph")
    statobj.generate_dist_graphs(actToNextActDist)
