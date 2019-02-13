import cv2
import numpy as np
from tensorflow_human_detection import DetectorAPI
import datetime


def is_relevant_scene(frame, confidence_threshold = 0.7):
    #input:
        #frame - an image
    #outputs:
        #boolean denoting whether input image is something we want in an extracted clip
    return DetectorAPI.get_human_count(frame, confidence_threshold) > 0

def extract_relevant_clips(video_file_path):
    timeFromMilliseconds = lambda x: str(datetime.timedelta(milliseconds=x))
    vid = cv2.VideoCapture(video_file_path)
    input_fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    input_dims = (int(width), int(height))
    # variable denoting the number of frames we want to skip before performing
        #our 'relevant_scene' check
    frames_to_skip = 360*10

    #variables for recording/saving relevant clips
    recording_clip = False
    cur_clip = None
    clip_num = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')


    #######################################################################
    ### Initializing sub-region specific dimensions and recording flag
    #######################################################################
    new_clip_name = "./extracted_clips/clip_69.avi"
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
        time_string = timeFromMilliseconds(vid.get(cv2.CAP_PROP_POS_MSEC))
        print "at frame {}, {}, {}".format(i, valid, time_string)
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
                        new_clip_name = "./extracted_clips/clip_{}_{}.avi".format(subregion, clip_num)
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

#TODO: update so file can be passed as cmd line arg
extract_relevant_clips("ridtydz2.mp4")
# extract_relevant_clips("./extracted_clips/clip_0.avi")
