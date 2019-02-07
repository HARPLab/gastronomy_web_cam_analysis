import cv2
import numpy as np
from tensorflow_human_detection import DetectorAPI


def is_relevant_scene(frame, confidence_threshold = 0.7):
    #input:
        #frame - an image
    #outputs:
        #boolean denoting whether input image is something we want in an extracted clip
    return DetectorAPI.get_human_count(frame, confidence_threshold) > 0
    
def extract_relevant_clips(video_file_path):
    vid = cv2.VideoCapture(video_file_path)
    input_fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    input_dims = (int(width), int(height))
    # counter representing the frame number we're on
    i = 0
    # variable denoting the number of frames we want to skip before performing
        #our 'relevant_scene' check
    frames_to_skip = 60

    #variables for recording/saving relevant clips
    recording_clip = False
    cur_clip = None
    clip_num = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')


    while (vid.isOpened()):
        valid, frame = vid.read()
        if (not valid): continue

        if (recording_clip):
            #add frame to clip
            cur_clip.write(frame)
            print "recording frame {}".format(i)


        if (i % frames_to_skip == 0):
            if (is_relevant_scene(frame)):
                #if scene is to be recorded, you either want to start a new clip
                    # or add to existing clip, depending on whether you're recording
                if (not recording_clip):
                    #if we aren't yet recording we'd like to set the recording flag
                        #and initialize the new clip
                    new_clip_name = "./extracted_clips/clip_{}.avi".format(clip_num)
                    cur_clip = cv2.VideoWriter(new_clip_name, fourcc, input_fps, input_dims)
                    recording_clip = True
            elif (recording_clip):
                #if the frame isn't relevant and we are currently recording, we want
                    #to stop recording and save it
                print("clip {} completed!".format(clip_num))
                recording_clip = False
                clip_num += 1
                cur_clip.release()

        i += 1
    vid.release()

#TODO: update so file can be passed as cmd line arg
extract_relevant_clips("ex.mov")
