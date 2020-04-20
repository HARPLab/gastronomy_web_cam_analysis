import sys
try:
        sys.path.append('/Users/michaelhuang/PycharmProjects/gast_analyisis_new/openpose/build/python');
        from openpose import pyopenpose as op
except:
        print(":(")
import cv2

import OPwrapper
def main():

        #Constructing OpenPose object allocates GPU memory
        openpose = OPwrapper.OP()

        #Opening OpenCV stream
        stream = cv2.VideoCapture("../videos/9-10-18_cropped.mp4")
        stream.set(cv2.CAP_PROP_POS_FRAMES, 28800)
        font = cv2.FONT_HERSHEY_SIMPLEX

        #while True:

        img = cv2.imread("../videos/test_person.jpg")

        # Output keypoints and the image with the human skeleton blended on it
        datum = openpose.getOpenposeDataFrom(img)

        print(str(datum.poseKeypoints))
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)

        cv2.waitKey(0)

         #       if key==ord('q'):
         #               break

        stream.release()
        cv2.destroyAllWindows()

main()