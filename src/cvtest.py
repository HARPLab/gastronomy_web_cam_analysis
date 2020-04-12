import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter
import six
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

# CODE COPIED FROM https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('../models/research/object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


#model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, image_np):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  #image_np = np.array(Image.open(image_path))
  #image_np = image_np[150:(150+125),160:(160+200),:]
  #image_np = image_np[200:(210 + 40), 220:(220 + 50), :]
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=3)
  result = Image.fromarray(image_np)
  display(Image.fromarray(image_np))
  result.save("test.jpg")

def retrieveclasses(model, image_np):
    output_dict = run_inference_for_single_image(model, image_np)
    classes = output_dict['detection_classes']
    class_freq = defaultdict(lambda: 0)
    for i in range(0, min(20,output_dict['detection_boxes'].shape[0])):
        if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]['name']
        else:
            class_name = 'N/A'
        class_freq[class_name] += 1
    return class_freq

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#for image_path in TEST_IMAGE_PATHS:
#  show_inference(detection_model, image_path)
cap = cv2.VideoCapture('../videos/9-10-18_cropped.mp4')
#cap.set(cv2.CAP_PROP_POS_FRAMES,28800)
#ret, frame = cap.read()
vertices = np.array([[[140, 170], [160,220], [280, 280],[360,200],[350, 170],[220,120]]], dtype=np.int32)
#cv2.imshow("frame", frame)
#cv2.waitKey(0)
#maskedimage = region_of_interest(frame,vertices)
#cv2.imshow("masked", maskedimage)
#cv2.waitKey(0)
#show_inference(detection_model, maskedimage[120:280, 140:360])

#cv2.imshow('mask', maskedimage)
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
overall_class_freq = defaultdict(lambda : 0)
ret = True
i = 0
while ret:
    ret, frame = cap.read()
    #cv2.imshow('frame',frame)
    if i % 5 == 0:
        maskedimage = frame # region_of_interest(frame,vertices)
        detectedclasses = retrieveclasses(detection_model, maskedimage)#[120:200, 140:360])
        print(detectedclasses)
        for key in detectedclasses.keys():
            overall_class_freq[key] += detectedclasses[key]
    i +=1
    #show_inference(detection_model, maskedimage[120:280, 140:360])
    #cv2.imshow('mask', maskedimage)
    #cv2.waitKey(0)

# changes
print(overall_class_freq)
