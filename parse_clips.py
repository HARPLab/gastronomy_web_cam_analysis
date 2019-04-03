import cv2
import numpy as np
from tensorflow_human_detection import DetectorAPI
import datetime, os
from OPwrapper import OP #openpose wrapper for convenience


NUM_DINERS_INFO_PATH = "/mnt/harpdata/gastronomy_clips/extracted_clips"
diners_file_dst = os.path.join(NUM_DINERS_INFO_PATH, 'diner_nums.txt')

def get_num_people(frame, openpose_wrapper=None):
    if(openpose_wrapper==None):
        openpose_wrapper = OP()
    d = openpose_wrapper.getOpenposeDataFrom(frame=frame)
    return d.poseKeypoints.shape[0]

def is_relevant_scene(frame, confidence_threshold = 0.7):
    #input:
        #frame - an image
    #outputs:
        #boolean denoting whether input image is something we want in an extracted clip
    return DetectorAPI.get_human_count(frame, confidence_threshold) > 0

def initialize_region(clip_info, region_name, width, height, x0, y0):
    clip_info[region_name] = dict()
    clip_info[region_name]["width"] = width
    clip_info[region_name]["height"] = height
    clip_info[region_name]["x0"] = x0
    clip_info[region_name]["y0"] = y0
    clip_info[region_name]["recording"] = False
    clip_info[region_name]["clip"] = None
    clip_info[region_name]["clip_num"] = 0
    #info to help with documenting number of diners in scene
    clip_info[region_name]["people_in_scene"] = 0
    clip_info[region_name]["times_checked"] = 0
    return

def extract_relevant_clips(source="", dest=""):
    timeFromMilliseconds = lambda x: str(datetime.timedelta(milliseconds=x))
    vid = cv2.VideoCapture(source)
    input_fps = vid.get(cv2.CAP_PROP_FPS)
    input_fps = 30 #the input fps is far higher than it actually is, have to change it manually
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_dims = (int(width), int(height))
    # variable denoting the number of frames we want to skip before performing
        #our 'relevant_scene' check
    frames_to_skip = 360*10

    #variables for recording/saving relevant clips
    recording_clip = False
    cur_clip = None
    clip_num = 0

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')

    num_diners_file = open(diners_file_dst, 'a+')
    print("now writing the number of diners corresponding to each parsed clip into {}".format(diners_file_dst))

    #######################################################################
    ### Initializing sub-region specific dimensions and recording flag
    #######################################################################
    clip_info = dict()
    # initiliaze_region(x0=220, y0=500, width=)
    x0 = 1665; y0 = 570; width=240; height=450
    #clip_info["bottom_left"] = dict()
    #clip_info["bottom_left"]["width"] = 550
    #clip_info["bottom_left"]["height"] = 580
    #clip_info["bottom_left"]["x0"] = 220
    #clip_info["bottom_left"]["y0"] = 500
    #clip_info["bottom_left"]["recording"] = False
    #clip_info["bottom_left"]["clip"] = None
    #clip_info["bottom_left"]["clip_num"] = 0
    initialize_region(clip_info, 
        region_name="bottom_left",
        width=550, height=580,
        x0=220, y0=500)


    # clip_info["middle_right"] = dict()
    initialize_region(clip_info, 
        region_name="middle_right",
        width=400, height=450,
        x0=1300, y0=450)
    # clip_info["middle_right"]["width"] = 400
    # clip_info["middle_right"]["height"] = 450
    # clip_info["middle_right"]["x0"] = 1300
    # clip_info["middle_right"]["y0"] = 450
    # clip_info["middle_right"]["recording"] = False
    # clip_info["middle_right"]["clip"] = None
    # clip_info["middle_right"]["clip_num"] = 0

    #next_region="middle_over"
    #x0 = 1565; y0 = 570; width=340; height=450
    initialize_region(clip_info, 
        region_name="middle_over",
        width=340, height=450,
        x0=1565, y0=570)
    #clip_info[next_region] = dict()
    #clip_info[next_region]["width"] = width
    #clip_info[next_region]["height"] = height
    #clip_info[next_region]["x0"] = x0
    #clip_info[next_region]["y0"] = y0
    #clip_info[next_region]["recording"] = False
    #clip_info[next_region]["clip"] = None
    #clip_info[next_region]["clip_num"] = 0

    # next_region="central"
    # x0 = 0; y0 = 200; width=290; height=240
#    initialize_region(clip_info, 
#        region_name="central",
#        width=290, height=240,
#        x0=0, y0=200)

    #next_region="middle"
    #clip_info[next_region] = dict()
    #x0 = 575; y0 = 250; width=250; height=545
    #clip_info[next_region]["width"] = width
    #clip_info[next_region]["height"] = height
    #clip_info[next_region]["x0"] = x0
    #clip_info[next_region]["y0"] = y0
    #clip_info[next_region]["recording"] = False
    #clip_info[next_region]["clip"] = None
    #clip_info[next_region]["clip_num"] = 0
    # counter representing the frame number we're on
    i = 0

    #initialize openpose_wrapper for determining number of people in frame
    openpose_wrapper = OP()
    while (vid.isOpened()):
        valid, frame = vid.read()
        if (not valid): break

        # print "showing frame {}".format(i)

        #recording subregions now, rather than entire frame
        for subregion in clip_info:
            subregion_info = clip_info[subregion]
            recording_clip = subregion_info["recording"]
            if (recording_clip):
                time_string = timeFromMilliseconds(vid.get(cv2.CAP_PROP_POS_MSEC))
                print("recording frame {}, {}, {}".format(i, valid, time_string))
                # add frame to clip
                clip_obj = subregion_info["clip"]
                x0, y0 = subregion_info["x0"], subregion_info["y0"]
                sub_width, sub_height = subregion_info["width"], subregion_info["height"]
                sub_frame = frame[y0:(y0 + sub_height), x0:(x0 + sub_width)]

                clip_obj.write(sub_frame)



        confidence_threshold = 0.7 # for use in the ppl counter
        if (i % frames_to_skip == 0):
            time_string = timeFromMilliseconds(vid.get(cv2.CAP_PROP_POS_MSEC))
            print('frame {} completed; time (hh:mm:ss): {}'.format(i, time_string))
            for subregion in clip_info:
                subregion_info = clip_info[subregion]
                x0, y0 = subregion_info["x0"], subregion_info["y0"]
                sub_width, sub_height = subregion_info["width"], subregion_info["height"]
                sub_frame = frame[y0:(y0 + sub_height), x0:(x0 + sub_width)]

                clip_num = subregion_info["clip_num"]

                recording_clip = subregion_info["recording"]
                people_count = get_num_people(sub_frame, openpose_wrapper)
                #people_count = DetectorAPI.get_human_count(sub_frame, confidence_threshold)
                if (people_count > 0):
                    #if scene is to be recorded, you either want to start a new clip
                        # or add to existing clip, depending on whether you're recording

                    if (not recording_clip):
                        #if we aren't yet recording we'd like to set the recording flag
                            #and initialize the new clip
                        new_clip_name = dest + "/clip_{}_{}.avi".format(subregion, clip_num)
                        cur_dims = (sub_width, sub_height)
                        subregion_info["clip"] = cv2.VideoWriter(new_clip_name, fourcc, input_fps, cur_dims)
                        subregion_info["recording"] = True
                    #either way, we want to update num diners in scene info
                    subregion_info["people_in_scene"] += people_count
                    subregion_info["times_checked"] += 1
                elif (recording_clip):
                    #if the frame isn't relevant and we are currently recording, we want
                        #to stop recording and save it
                    avg_num_diners = float(subregion_info["people_in_scene"]) / float(subregion_info["times_checked"])
                    print("clip {} completed for region: {}\n\t avg number of diners was: {}".format(clip_num, subregion, avg_num_diners))
                    num_diners_file.write("{}\n\t{}\n".format(os.path.join(dest, "clip_{}_{}.avi".format(subregion, subregion_info["clip_num"])), avg_num_diners))
                    subregion_info["recording"] = False
                    subregion_info["clip_num"] += 1

                    subregion_info["clip"].release()
                    subregion_info["clip"] = None
        i += 1

    # if any clips are still recording be sure to release them
    for subregion in clip_info:
        subregion_info = clip_info[subregion]
        if subregion_info["clip"] != None:
            subregion_info["clip"].release()
    vid.release()
    num_diners_file.close()
import subprocess
# #TODO: update so file can be passed as cmd line arg
# openface_dir = os.path.join("~/dev/OpenFace/build")
# execute_instr = os.path.join(openface_dir, "bin/FaceLandmarkVid")
# print execute_instr
def view_clips(base="/mnt/harpdata/gastronomy_clips"):
    while True:
        inp = raw_input("Next Video pls:\n\t-->")
        mode = raw_input("\tSingle or Double? -->").lower()
        #single face
        print("processing {}!".format(mode))
        if (mode=="single"):
            subprocess.call('~/dev/OpenFace/build/bin/FaceLandmarkVid -f "{}"'.format(inp), shell=True)
        elif(mode == "double"):
            #multiple faces
            subprocess.call('~/dev/OpenFace/build/bin/FaceLandmarkVidMulti -f "{}" -out_dir ~/Downloads -of good_one_8_remote.avi -tracked -vis-track -wild'.format(inp), shell=True)
        else:
            print("Sorry, mode '{}' is not recognized! Please try again")
#~/dev/OpenFace/build/bin/FaceLandmarkVidMulti -f "/mnt/harpdata/gastronomy_clips/extracted_clips/3-3_15:0/clip_middle_0.avi" -out_dir ~/Downloads -of good_one_7_remote.avi -tracked -vis-track -wild


def list_clips(base="/mnt/harpdata/gastronomy_clips"):
    for f in os.listdir(base):
        if (f.endswith(".ts") ):
            dst=os.path.join(base, "extracted_clips", f[:f.find(".ts")])
            if not os.path.exists(dst):
                print("{} does not exist!")
            else:
                print("in {}:".format(dst))
                for vid in os.listdir(dst):
                    is_middle_over = "middle_over" in vid
                    is_middle = ("middle" in vid) and (not ("middle_over" in vid)) and (not ("middle_right" in vid))
                    if (is_middle or is_middle_over):
                        print("\t{}".format(os.path.join(dst,vid)))
def parse_dirs(base="/mnt/harpdata/gastronomy_clips"):
    now = datetime.datetime.now()
    log = os.path.join(os.getcwd(), "logs", "{}-{}__{}:{}.txt".format(now.month, now.day, now.hour, now.minute))
    for f in os.listdir(base):
        if (f.endswith(".ts")):
            log_file = open(log, 'a+')
            dst=os.path.join(base, "extracted_clips", f[:f.find(".ts")])
            if not os.path.exists(dst):
                os.makedirs(dst)
            next_vid_path = os.path.join(base, f)
            log_file.write("Started parsing {}\n".format(next_vid_path))
            print("Started for {}!\n".format(f))
            extract_relevant_clips(source=next_vid_path, dest=dst)
            print("Finished for {}!\n".format(f))
            log_file.write("Finished parsing {}\n".format(next_vid_path))
            log_file.close()
def play(fname=None):
    openpose_wrapper = OP()
    while(True):
        #fname=raw_input("What file would you like to play?\n\t-->")
        fname=input("What file would you like to play?\n\t-->")
        cap = cv2.VideoCapture(fname)
    #x0 = 0; y0 = 200; width=445; height=320
    #x0 = 0; y0 = 200; width=445; height=240
    #x0 = 0; y0 = 200; width=290; height=240
        print("playing {}".format(fname))
        while(cap.isOpened()):
            ret, frame = cap.read()
        #sub_frame = frame[y0:(y0 + height), x0:(x0 + width)]

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            d = openpose_wrapper.getOpenposeDataFrom(frame=frame)
            print("{} people in frame".format(d.poseKeypoints.shape[0]))
            #print(d.poseKeypoints3D)
            #print(d)
            cv2.imshow('frame', frame)
            cv2.imshow('openpose_frame', d.cvOutputData)
        #cv2.imshow('sub_frame', sub_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("done playing {}!".format(fname))

    cap.release()
    cv2.destroyAllWindows()
def sharpen(fname=None):
    if (fname==None):
        fname=raw_input("What file would you like to sharpen?\n\t-->")
    cap = cv2.VideoCapture(fname)
    # x0 = 1565; y0 = 570; width=340; height=450
    base, tail = os.path.split(fname)
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    outfile_name = os.path.join(base,tail[:tail.find('.avi')] + "_sharpened" + ".avi")
    input_fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Writing to {}".format(outfile_name))
    out_vid = cv2.VideoWriter(outfile_name, codec, input_fps, (width, height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharper = cv2.filter2D(frame, -1, k)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_vid.write(sharper)

        cv2.imshow('frame', frame)
        cv2.imshow('sharper?', sharper)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
#play()
# play("ridtydz2.mp4")
#play("/mnt/harpdata/gastronomy_clips/extracted_clips/3-2_13:32/clip_middle_over_0_sharpened.avi")
#sharpen("/mnt/harpdata/gastronomy_clips/extracted_clips/3-2_13:32/clip_middle_over_0.avi")
# extract_relevant_clips(source="ridtydz2.mp4", dest="./extracted_clips")
#extract_relevant_clips(source="/home/rkaufman/Downloads/vid3.mp4", dest="/mnt/harpdata/gastronomy_clips/tmp_demo")
# extract_relevant_clips("./extracted_clips/clip_0.avi")
parse_dirs()
