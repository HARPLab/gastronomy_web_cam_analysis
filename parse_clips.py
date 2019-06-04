import cv2
import numpy as np
from tensorflow_human_detection import DetectorAPI
import datetime, os
from OPwrapper import OP #openpose wrapper for convenience
import subprocess
from SQL_DB.ClassDeclarations import Clip
from SQL_DB.DBWrapper import DBWrapper


NUM_DINERS_INFO_PATH = "/mnt/harpdata/gastronomy_clips/extracted_clips"
diners_file_dst = os.path.join(NUM_DINERS_INFO_PATH, 'diner_nums.txt')

def get_num_people(frame, openpose_wrapper=None):
    if(openpose_wrapper==None):
        openpose_wrapper = OP()
    d = openpose_wrapper.getOpenposeDataFrom(frame=frame)
    return 0 if (len(d.poseKeypoints.shape) == 0) else d.poseKeypoints.shape[0]
    #return (d.poseKeypoints.shape)

def get_num_people_tf(frame, threshold=0.7):
    return DetectorAPI.get_human_count(frame, threshold)

def is_relevant_scene(frame, confidence_threshold = 0.7):
    #input:
        #frame - an image
    #outputs:
        #boolean denoting whether input image is something we want in an extracted clip
    return DetectorAPI.get_human_count(frame, confidence_threshold) > 0

def initialize_region(clip_info=None, region_name=None, width=None, height=None, x0=None, y0=None):
    clip_info[region_name] = dict()
    clip_info[region_name]["width"] = width
    clip_info[region_name]["height"] = height
    clip_info[region_name]["x0"] = x0
    clip_info[region_name]["y0"] = y0
    clip_info[region_name]["recording"] = False
    clip_info[region_name]["clip"] = None
    clip_info[region_name]["clip_num"] = 0
    clip_info[region_name]["cur_clip_duration"] = None
    clip_info[region_name]["num_frames"] = 0
    return

def end_and_save(dest=None, frames_to_skip=None, clip_dict=None, subregion=None, parent_clip_path=None, db_session=None):
    # ends and saves relevant clips. If they're too short to be actual dining scenes,
        # deletes them
    formatStr = "%m-%d_%H:%M.ts"
    _, parent_clip_filename = os.path.split(parent_clip_path)

    subregion_info = clip_dict[subregion]

    # determining info necessary to store Clip object into database
    start_time = datetime.datetime.strptime(parent_clip_filename, formatStr)
    duration_in_secs = subregion_info["num_frames"] / 30.0
    end_time = start_time + datetime.timedelta(seconds=duration_in_secs)
    num_frames = subregion_info["num_frames"]

    subregion_info["recording"] = False
    clip_num = subregion_info["clip_num"]
    clip_fname = dest + "/clip_{}_{}.avi".format(subregion, clip_num)
    duration = subregion_info["cur_clip_duration"]
    clip_num = subregion_info["clip_num"]
    subregion_info["clip_num"] += 1
    subregion_info["clip"].release()
    subregion_info["clip"] = None
    print("clip {} completed, with duration {}".format(clip_fname, duration))
    if (duration < 4 * frames_to_skip):
        print("\tDELETED {}".format(clip_fname))
        subprocess.run(["rm", clip_fname])
    else: #only want to save metadata to db if we're going to save the video
        new_clip = Clip(num_frames=num_frames, start_time=start_time, end_time=end_time, parent_clip_path=parent_clip_path, clip_path=clip_fname, processed=False)
        db_session.add(new_clip)
        db_session.commit()
        #num_diners_file.write("{}\n\t{}\n".format(os.path.join(dest, "clip_{}_{}.avi".format(subregion, clip_num)), avg_num_diners))

def extract_relevant_clips(source="", dest=""):
    timeFromMilliseconds = lambda x: str(datetime.timedelta(milliseconds=x))
    vid = cv2.VideoCapture(source)
    input_fps = vid.get(cv2.CAP_PROP_FPS)
    input_fps = 30 #the input fps that is read is far higher than it actually is, have to change it manually
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_dims = (int(width), int(height))
    # variable denoting the number of frames we want to skip before performing
        #our 'relevant_scene' check
    frames_to_skip = 30*120#only check frame every 2 minutes

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    print("now writing the number of diners corresponding to each parsed clip into {}".format(diners_file_dst))

    #######################################################################
    ### Initializing sub-region specific dimensions and recording flag
    #######################################################################
    clip_info = dict()
    # initiliaze_region(x0=220, y0=500, width=)
    # x0 = 1665; y0 = 570; width=240; height=450
    initialize_region(clip_info, 
        region_name="bottom_left",
        width=550, height=580,
        x0=220, y0=500)


    # clip_info["middle_right"] = dict()
    initialize_region(clip_info, 
        region_name="middle_right",
        width=400, height=450,
        x0=1300, y0=450)

    #next_region="middle_over"
    #x0 = 1565; y0 = 570; width=340; height=450
    initialize_region(clip_info, 
        region_name="middle_over",
        width=340, height=450,
        x0=1565, y0=570)

    # next_region="central"
    # x0 = 0; y0 = 200; width=290; height=240
#    initialize_region(clip_info, 
#        region_name="central",
#        width=290, height=240,
#        x0=0, y0=200)

    # counter representing the frame number we're on
    i = 0

    #initialize openpose_wrapper for determining number of people in frame
    openpose_wrapper = OP()
    db_wrapper = DBWrapper()
    db_session = db_wrapper.get_session()
    confidence_threshold = 0.4 # for use in filtering poses by confidence
    while (vid.isOpened()):
        valid, frame = vid.read()
        if (not valid): break

        # print "showing frame {}".format(i)

        #recording subregions now, rather than entire frame
        for subregion in clip_info:
            subregion_info = clip_info[subregion]
            recording_clip = subregion_info["recording"]
            if (recording_clip):
                # add frame to clip
                clip_obj = subregion_info["clip"]
                x0, y0 = subregion_info["x0"], subregion_info["y0"]
                sub_width, sub_height = subregion_info["width"], subregion_info["height"]
                sub_frame = frame[y0:(y0 + sub_height), x0:(x0 + sub_width)]
                subregion_info["cur_clip_duration"] += 1

                clip_obj.write(sub_frame)
                subregion_info["num_frames"] += 1


        if (i % frames_to_skip == 0):
            print('frame {} completed'.format(i))
            for subregion in clip_info:
                subregion_info = clip_info[subregion]
                x0, y0 = subregion_info["x0"], subregion_info["y0"]
                sub_width, sub_height = subregion_info["width"], subregion_info["height"]
                sub_frame = frame[y0:(y0 + sub_height), x0:(x0 + sub_width)]

                clip_num = subregion_info["clip_num"]

                recording_clip = subregion_info["recording"]

                #logic to determine number of people detected in frame
                cv2.imwrite('s.jpg', sub_frame)
                sub_frame = cv2.imread('s.jpg')
                d = openpose_wrapper.getOpenposeDataFrom(frame=sub_frame)
                real_poses = list(filter(lambda x: x > confidence_threshold, np.atleast_1d(d.poseScores)))
                people_count = len(real_poses)

                #print("\tseeing {} people!\n\tin {}\n\tfrom {}".format(people_count, real_poses, np.atleast_1d(d.poseScores)))
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
                        subregion_info["cur_clip_duration"] = 0
                        subregion_info["num_frames"] = 0
                elif (recording_clip):
                    #if the frame isn't relevant and we are currently recording, we want
                        #to stop recording and save it
                    end_and_save(dest=dest, frames_to_skip=frames_to_skip, clip_dict=clip_info, subregion=subregion, parent_clip_path=source, db_session=db_session)
        i += 1

    # if any clips are still recording be sure to release them
    for subregion in clip_info:
        subregion_info = clip_info[subregion]
        if subregion_info["clip"] != None:
            end_and_save(dest=dest, frames_to_skip=frames_to_skip, clip_dict=clip_info, subregion=subregion, parent_clip_path=source, db_session=db_session)
    vid.release()


#utility function that runs and renders openface output (on a single or multiple faces)
def view_clips(base="/mnt/harpdata/gastronomy_clips"):
    while True:
        inp = raw_input("Next Video pls:\n\t-->")
        mode = raw_input("\tSingle or Double? -->").lower()
        print("processing {}!".format(mode))
        if (mode=="single"):
            #single face
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

#Input: String of directory 
#Operation: Searches directory for all occurences of ".ts" files and calls `extract_relevant_clips` function on each such file
def parse_dirs(base="/mnt/harpdata/gastronomy_clips/"):
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

#utility function that plays a file of interest (I usually use this for rendering the segmented output from mask rcnn or the pose data from openpose)
def play(fname=None):
    confidence_threshold=0.4
    confidence_threshold_weak = 0.2
    #openpose_wrapper = OP()
    detector = DetectorAPI()
    while(True):
        # if python3, use input (but if python2, use raw_input)
        fname = input("What file would you like to play?\n\t-->")
        cap = cv2.VideoCapture(fname)
        print("playing {}".format(fname))
        mins=5; fps=30;i=0;
        frames_to_skip = mins*60*fps*0
        segment_flag = False
        pause_flag = False
        while(cap.isOpened()):
            if(pause_flag):
                pass
            else:
                ret, frame = cap.read()
            if (i < frames_to_skip):
                i+=1; continue;
            if (not ret):
                print("all done!")
                return
            sub_frame=frame

            # segment frame
            if (segment_flag):
                frame = detector.segment(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                segment_flag = not segment_flag
            if cv2.waitKey(1) & 0xFF == ord('p'):
                pause_flag = not pause_flag
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("done playing {}!".format(fname))

        cap.release()
        cv2.destroyAllWindows()
play()
# play("ridtydz2.mp4")
#play("/mnt/harpdata/gastronomy_clips/extracted_clips/3-2_13:32/clip_middle_over_0_sharpened.avi")
#sharpen("/mnt/harpdata/gastronomy_clips/extracted_clips/3-2_13:32/clip_middle_over_0.avi")
# extract_relevant_clips(source="ridtydz2.mp4", dest="./extracted_clips")
#extract_relevant_clips(source="/home/rkaufman/Downloads/vid3.mp4", dest="/mnt/harpdata/gastronomy_clips/tmp_demo")
# extract_relevant_clips("./extracted_clips/clip_0.avi")
#extract_relevant_clips("/mnt/harpdata/gastronomy_clips/2-28_19:10.ts")
#parse_dirs()