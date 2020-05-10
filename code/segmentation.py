import glob, io, cv2, numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy, copy
from skimage.filters import threshold_local
from skimage import measure
import os

'''Prepare image by making grayscale, bigger and thresholded for better consistency.'''
def clean_image(img):
    h, w, c = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_img = cv2.resize(gray_img, None, fx=12.0, fy=12.0, interpolation=cv2.INTER_CUBIC)

    resized_img = cv2.GaussianBlur(resized_img,(5,5),0)

    equalized_img = cv2.equalizeHist(resized_img)


    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(reduced, 140, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, kernel, iterations = 1)

    return mask

''' Helper for image consistency
'''
def reduce_colors(img, n):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

'''Use area proportion calculations of each contour over the whole image itself to determine what are license plate characters or not.
    Use bounding boxes to extract these characters and return in an array.'''
def findCharacterContour(img):
    img = clean_image(img)
    height, width = img.shape
    img_area = height * width

    plate_characters = []
    bw_image = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        center = (x + w/2, y + h/2)

        #Extract characters based on expected area ratios
        if (float(area/img_area) > 0.008) and (float(area/img_area) < 0.032) and h > w:
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        try:
            char_image = cv2.resize(char_image, (50, 100))
            plate_characters.append(char_image)
        except:
            continue

    return np.array(plate_characters)



