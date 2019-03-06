import cv2
import numpy as np
from tensorflow_human_detection import DetectorAPI
import datetime, os


def is_relevant_scene(frame, confidence_threshold = 0.7):
    #input:
        #frame - an image
    #outputs:
        #boolean denoting whether input image is something we want in an extracted clip
    return DetectorAPI.get_human_count(frame, confidence_threshold) > 0

def extract_relevant_clips(source="", dest=""):
    print (source, dest)
    timeFromMilliseconds = lambda x: str(datetime.timedelta(milliseconds=x))
    vid = cv2.VideoCapture(source)
    input_fps = vid.get(cv2.CAP_PROP_FPS)
    ## print input_fps
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')


    #######################################################################
    ### Initializing sub-region specific dimensions and recording flag
    #######################################################################
    clip_info = dict()
    clip_info["bottom_left"] = dict()
    clip_info["bottom_left"]["width"] = 550
    clip_info["bottom_left"]["height"] = 580
    clip_info["bottom_left"]["x0"] = 220
    clip_info["bottom_left"]["y0"] = 500
    clip_info["bottom_left"]["recording"] = False
    clip_info["bottom_left"]["clip"] = None
    clip_info["bottom_left"]["clip_num"] = 0


    clip_info["middle_right"] = dict()
    clip_info["middle_right"]["width"] = 400
    clip_info["middle_right"]["height"] = 450
    clip_info["middle_right"]["x0"] = 1300
    clip_info["middle_right"]["y0"] = 450
    clip_info["middle_right"]["recording"] = False
    clip_info["middle_right"]["clip"] = None
    clip_info["middle_right"]["clip_num"] = 0
    # counter representing the frame number we're on
    i = 0


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
                print "recording frame {}, {}, {}".format(i, valid, time_string)
                # add frame to clip
                clip_obj = subregion_info["clip"]
                x0, y0 = subregion_info["x0"], subregion_info["y0"]
                sub_width, sub_height = subregion_info["width"], subregion_info["height"]
                sub_frame = frame[y0:(y0 + sub_height), x0:(x0 + sub_width)]

                clip_obj.write(sub_frame)



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

                if (is_relevant_scene(sub_frame)):
                    #if scene is to be recorded, you either want to start a new clip
                        # or add to existing clip, depending on whether you're recording
                    if (not recording_clip):
                        #if we aren't yet recording we'd like to set the recording flag
                            #and initialize the new clip
                        new_clip_name = dest + "/clip_{}_{}.avi".format(subregion, clip_num)
                        cur_dims = (sub_width, sub_height)
                        subregion_info["clip"] = cv2.VideoWriter(new_clip_name, fourcc, input_fps, cur_dims)
                        subregion_info["recording"] = True
                elif (recording_clip):
                    #if the frame isn't relevant and we are currently recording, we want
                        #to stop recording and save it
                    print("clip {} completed for region: {}".format(clip_num, subregion))
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
import subprocess
#TODO: update so file can be passed as cmd line arg
base = "/mnt/harpdata/gastronomy_clips"
now = datetime.datetime.now()
log = os.path.join(os.getcwd(), "logs", "{}-{}__{}:{}.txt".format(now.month, now.day, now.hour, now.minute))
openface_dir = os.path.join("~/dev/OpenFace/build")
execute_instr = os.path.join(openface_dir, "bin/FaceLandmarkVid")
print execute_instr
#for f in os.listdir(base):
while True:
    inp = raw_input("Next Video pls:\n\t-->")
    subprocess.call('~/dev/OpenFace/build/bin/FaceLandmarkVid -f "{}"'.format(inp), shell=True)
    #if (f.endswith(".ts")):
    #    dst=os.path.join(base, "extracted_clips", f[:f.find(".ts")])
#	if not os.path.exists(dst):
#            print "{} does not exist!"
#        else:
#	     print "in {}:".format(dst)
#	     for vid in os.listdir(dst):
#	         print "\t{}".format(os.path.join(dst,vid))
#./bin/FaceLandmarkVid -f "../samples/changeLighting.wmv" -f "../samples/2015-10-15-15-14.avi"

#for f in os.listdir(base):
#    if (f.endswith(".ts")):
#	log_file = open(log, 'a+')
#        dst=os.path.join(base, "extracted_clips", f[:f.find(".ts")])
#	if not os.path.exists(dst):
#	    os.makedirs(dst)
#        next_vid_path = os.path.join(base, f)
#        log_file.write("Started parsing {}\n".format(next_vid_path))
#        print "Started for {}!\n".format(f)
#	extract_relevant_clips(source=next_vid_path, dest=dst)
       # print "Finished for {}!\n".format(f)
       # log_file.write("Finished parsing {}\n".format(next_vid_path))
#	log_file.close()
#extract_relevant_clips(source="ridtydz2.mp4", dest="./extracted_clips")
# extract_relevant_clips("./extracted_clips/clip_0.avi")
