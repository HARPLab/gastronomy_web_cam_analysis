import cv2
import numpy as np
#cap = cv2.VideoCapture(0)


#ret, frame = cap.read()
#mask = np.zeros_like(frame)
#if len(frame.shape) > 2:
#    channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
#    ignore_mask_color = (255,) * channel_count
#else:
#    ignore_mask_color = 255

#vertices = np.array([[[100,100],[200,200],[300,100]]], dtype=np.int32)
# filling pixels inside the polygon defined by "vertices" with the fill color
#cv2.fillPoly(mask, vertices.astype('int32'), ignore_mask_color)

# returning the image only where mask pixels are nonzero
#masked_image = cv2.bitwise_and(frame, mask)
#cv2.imshow("mask", masked_image)
#cv2.waitKey(0)
# testing s
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

cap = cv2.VideoCapture('9-10-18_cropped.mp4')
vertices = np.array([[[140, 170], [160,220], [280, 280],[360,200],[350, 170],[220,120]]], dtype=np.int32)
ret, frame = cap.read()
#cv2.imshow('frame',frame)
maskedimage = region_of_interest(frame,vertices)
#show_inference(detection_model, maskedimage)
cv2.imshow('mask', maskedimage)
cv2.waitKey(0)