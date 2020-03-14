#######################
# code below adapted from https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
#######################

# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import os
import random
def make_random_color():
    return (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
colors = [make_random_color() for i in range(90)]
labels_path = "./faster_rcnn_inception_v2_coco_2018_01_28/coco_classes.txt"
LABELS = open(labels_path).read().strip().split("\n")

'''
Code below all taken from https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
'''
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
maskThreshold = 0.3  # Mask threshold 
# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    cv = cv2
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
     
    # Print a label of class.
    label = '%.2f: %s'% (conf, LABELS[classId])
    # if classes:
    #     assert(classId < len(classes))
    #     label = '%s:%s' % (classes[classId], label)
    round = lambda x: int(x)
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
 
    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom+1, left:right+1][mask]
 
    color = colors[classId%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    #colorIndex = random.randint(0, len(colors)-1)
    #color = colors[colorIndex]
 
    frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
 
    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)
# For each frame, extract the bounding box and mask for each detected object
def postprocess(frame, boxes, masks):
    cv = cv2
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]
     
    frameH = frame.shape[0]
    frameW = frame.shape[1]
     
    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
             
            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])
             
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))
             
            # Extract the mask for the object
            classMask = mask[classId]
             
            # Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, left, top, right, bottom, classMask)

''' 
code above all taken from https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
'''
class DetectorAPI:
    def __init__(self, path_to_ckpt='./faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', config_path="./faster_rcnn_inception_v2_coco_2018_01_28/config.pbtxt"):
        # for tf log messages
        """
            0 = all messages are logged (default behavior)
            1 = INFO messages are not printed
            2 = INFO and WARNING messages are not printed
            3 = INFO, WARNING, and ERROR messages are not printed
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
        mask_weight_path = "./mask_rcnn/frozen_inference_graph.pb"
        mask_config_path = "./mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        self.mask_net = cv2.dnn.readNetFromTensorflow(mask_weight_path, mask_config_path)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def segment(self, image): #learned from --> https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
        image = image.copy()
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.mask_net.setInput(blob)
        start = time.time()
        (boxes, masks) = self.mask_net.forward(["detection_out_final", "detection_masks"])
        end = time.time()
        print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
        print("[INFO] boxes shape: {}".format(boxes.shape))
        print("[INFO] masks shape: {}".format(masks.shape))
        postprocess(image, boxes, masks)
        return image

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def renderBoxes(self, image, threshold=0.7):
        clone = image.copy()
        boxes, scores, class_nums, num_detected = self.processFrame(image)
        i=0
        color = (0,255,0)
        while( i < len(class_nums)):
            box = boxes[i]; score = scores[i]; class_num = class_nums[i];
            if score <= threshold: 
                i+=1
                continue
            x0, x1, y0, y1 = (box[1],box[0], box[3],box[2])
            cv2.rectangle(clone,(x0, x1),(y0, y1),(255,0,0),2)
            text = "{}: {:.4f}".format(LABELS[class_num-1], score)
            cv2.putText(clone, text, (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
            i+=1
        return clone
    def close(self):
        self.sess.close()
        self.default_graph.close()

    def get_human_count(self, frame, threshold=0.7):
        odapi = self
        img = cv2.resize(frame, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
        human_count = 0
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                human_count += 1
        
        return human_count

    @staticmethod 
    def get_human_count(frame, threshold=0.7):
        odapi = DetectorAPI()
        img = cv2.resize(frame, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
        human_count = 0
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                human_count += 1
        
        return human_count
    # @staticmethod 
    # def get_objects(frame, threshold=0.7):
    #     odapi = DetectorAPI()
    #     img = cv2.resize(frame, (1280, 720))

    #     boxes, scores, classes, num = odapi.processFrame(img)
    #     human_count = 0
    #     for i in range(len(boxes)):
    #         # Class 1 represents human
    #         if classes[i] == 1 and scores[i] > threshold:
    #             human_count += 1
        
    #     return human_count

# d = DetectorAPI()
# img = cv2.imread("./temp.jpg")
# # other_img = d.renderBoxes(img)
# segmented = d.segment(img)

# while True:
#     cv2.imshow("preview", img)
#     cv2.imshow("preview alt", segmented)
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord('q'):
#         break


# cv2.destroyAllWindows()