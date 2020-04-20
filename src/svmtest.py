from sklearn import svm
import cv2
import os
import sys
from sys import platform


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append('../../python');
        sys.path.append('/home/rkaufman/dev/openpose/build/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

inputfeatures = []
labels = []
pose_features = []
pose_feature_data = "../data/"
action_label_data = "../data/"
pose_feature_data_test = "test_data.txt"

directory = os.fsencode(pose_feature_data)

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith("pose.txt"):
         # print(os.path.join(directory, filename))
         with open(filename, 'r') as f:
             #each line will be a frame, containing two peoples data split by semicolon
             for line in f:
                 keypoints = line.split(";")
                 pose_features.append(keypoints[0].split(","))
                 pose_features.append(keypoints[1].split(","))
         continue
     elif filename.endswith("action.txt"):
         with open(filename, 'r') as f:
             #each line will be a frame
             for line in f:
                 keypoints = line.split(";") # two actions per line
                 labels.append(keypoints[0])
                 labels.append(keypoints[1])
         continue
input_features = pose_features
lin_clf = svm.LinearSVC()
lin_clf.fit(input_features, labels)

with open(pose_feature_data_test, 'r') as f:
    for line in f:
        keypoints = line.split(";")
        print(lin_clf.predict(keypoints[0].split(',')))

