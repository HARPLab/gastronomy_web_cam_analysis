import sys
import cv2

import OPwrapper
def main():

        #Constructing OpenPose object allocates GPU memory
        openpose = OPwrapper.OP()

        #Opening OpenCV stream
        stream = cv2.VideoCapture("../videos/9-10-18_cropped.mp4")
        stream.set(cv2.CAP_PROP_POS_FRAMES, 28800)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = stream.read()
        #while True:

        #img = cv2.imread("test.jpg")

        # Output keypoints and the image with the human skeleton blended on it
        datum = openpose.getOpenposeDataFrom(frame)

        print(str(datum.poseKeypoints))
        cv2.imwrite("pose.png", datum.cvOutputData)

        cv2.waitKey(0)

         #       if key==ord('q'):
         #               break

        stream.release()
        cv2.destroyAllWindows()

main()
