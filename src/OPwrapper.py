# From Python 
# It requires OpenCV installed for Python 
import sys 
import cv2 
import os 
from sys import platform 
import argparse 
 
# Import Openpose (Windows/Ubuntu/OSX) 
dir_path = os.path.dirname(os.path.realpath(__file__)) 
try: 
    # Windows Import 
    if platform == "win32": 
        # Change these variables to point to the correct folder (Release/x64 etc.)  
        sys.path.append(dir_path + '/../../python/openpose/Release'); 
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;' 
        import pyopenpose as op 
    else: 
# Change these variables to point to the correct folder (Release/x64 etc.)  
#sys.path.append('../../python'); 
#        sys.path.append('/home/mghuang/spring2020_gastronomy_analysis/gastronomy_web_cam_analysis/openpose/build/python/'); 
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it. 
        sys.path.append('/usr/local/python');
        from openpose import pyopenpose as op 
except ImportError as e: 
	print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?') 
	raise e 
 
# Flags 
#parser = argparse.ArgumentParser() 
#parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).") 
#args = parser.parse_known_args() 
 
# Custom Params (refer to include/openpose/flags.hpp for more parameters) 
#params["model_folder"] = "../../../models/" 
 
# Add others in path? 
#for i in range(0, len(args[1])): 
#    curr_item = args[1][i] 
#    if i != len(args[1])-1: next_item = args[1][i+1] 
#    else: next_item = "1" 
#    if "--" in curr_item and "--" in next_item: 
#        key = curr_item.replace('-','') 
#        if key not in params:  params[key] = "1" 
#    elif "--" in curr_item and "--" not in next_item: 
#        key = curr_item.replace('-','') 
#        if key not in params: params[key] = next_item 
 
# Construct it from system arguments 
# op.init_argv(args[1]) 
# oppython = op.OpenposePython() 
 
 
# Process Image 
#datum = op.Datum() 
#imageToProcess = cv2.imread(args[0].image_path) 
#datum.cvInputData = imageToProcess 
#opWrapper.emplaceAndPop([datum]) 
 
# Display Image 
#print("Body keypoints: \n" + str(datum.poseKeypoints)) 
#cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData) 
class OP():
    def __init__(self):
        params = dict() 
        params["model_folder"] = "/home/mghuang/gastronomy_web_cam_analysis/openpose/models/"
        self.opWrapper = op.WrapperPython() 
        self.opWrapper.configure(params) 
        self.opWrapper.start() 
         
         
    def getOpenposeDataFrom(self, frame=None):
        if (frame is None):
            raise Exception("NoneType passed into OP wrapper class")
        datum = op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        return datum

clip_paths = [ "/home/rkaufman/Downloads/restaurant_footage/IMG_8500.MOV",
"/home/rkaufman/Downloads/restaurant_footage/IMG_8502.MOV",
"/home/rkaufman/Downloads/restaurant_footage/IMG_2074.MOV",
"/home/rkaufman/Downloads/restaurant_footage/IMG_8495.MOV",
"/home/rkaufman/Downloads/restaurant_footage/IMG_8497.MOV",
"/home/rkaufman/Downloads/restaurant_footage/IMG_8501.MOV"]

def create_clips(file_name=None):
    # Starting OpenPose 
    params = dict() 
    params["model_folder"] = "/home/rkaufman/dev/openpose/models" 
    opWrapper = op.WrapperPython() 
    opWrapper.configure(params) 
    opWrapper.start() 
    out_dir = "/home/rkaufman/Downloads/restaurant_footage_openposified"
    #while True:
    #    file_name = input("Next clip please.\n\t-->")
    for file_name in clip_paths[3:]:
    #while(True):
    #    file_name=input("next video please:\n\t-->")
        cap = cv2.VideoCapture(file_name)
        base, tail = os.path.split(file_name)
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        outfile_name = os.path.join(out_dir,tail[:tail.find('.')] + "_openpose" + ".avi")
        input_fps = 30
#        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Writing to {}".format(outfile_name))
        out_vid = cv2.VideoWriter(outfile_name, codec, input_fps, (width, height))
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            print("\titeration: {}".format(i))
            if (not ret): break
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            op_image = datum.cvOutputData
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_vid.write(op_image)
            #cv2.imshow('frame', frame)
            #cv2.imshow('openpose', op_image)
            i+=1
         #   if cv2.waitKey(1) & 0xFF == ord('q'):
         #       break
        print("Successfully wrote to {}\n".format(outfile_name))

        cap.release()
        out_vid.release()
        #cv2.destroyAllWindows()
#create_clips() 
