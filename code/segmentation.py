import glob, io, cv2, numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy, copy
from imutils import perspective
import imutils
from skimage.filters import threshold_local
from skimage import measure
import os


def cropCharacter(img, dimensions):
    [x, y, w, h] = dimensions
    character = deepcopy(img)
    character = deepcopy(character[y:y + h, x:x + w])
    return character

#OLD SEGMENTATION
# def findCharacterContour(img):
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     plate_characters = []
#     gray_plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_plate = cv2.GaussianBlur(gray_plate, (3, 3), 0)
#
#     _, threshold = cv2.threshold(gray_plate, 140, 255, 0)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
#
#     w, h, x, y = 0, 0, 0, 0
#
#     print("%s contours found.", str(len(contours)))
#     for contour in sorted_ctrs:
#         area = cv2.contourArea(contour)
#
#         # rough range of areas of a plate number
#         if area > 120 and area < 2000:
#             [x, y, w, h] = cv2.boundingRect(contour)
#
#         # rough dimensions of a character
#         if h > 20 and h < 90 and w > 10 and w < 50:
#             character = cropCharacter(img, [x, y, w, h])
#             plate_characters.append(character)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#             cv2.imshow('image', character)
#             cv2.waitKey(0)
#
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     print("%s plate characters found", str(len(plate_characters)))


def clean_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #PLAY AROUND WITH THESE RESIZING AND CLEANING
    resized_img = cv2.resize(gray_img
        , None
        , fx=4.0
        , fy=2.0
        , interpolation=cv2.INTER_CUBIC)

    resized_img = cv2.GaussianBlur(resized_img,(5,5),0)

    equalized_img = cv2.equalizeHist(resized_img)


    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)


    ret, mask = cv2.threshold(reduced, 140, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, kernel, iterations = 1)

    return mask

def findCharacterContour(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    img = clean_image(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    plate_characters = []
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        center = (x + w/2, y + h/2)
        print(area)
        # PLAY AROUND WITH THESE CHECKS
        if (area > 2000) and (area < 15000) and w < h:
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        cv2.imshow('image', char_image)
        cv2.waitKey(0)
        plate_characters.append(char_image)

    return plate_characters

#Copied
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

for filename in os.listdir('data_license_only/crop_h1/'):
    if filename.endswith(".png"):
        print(filename)
        findCharacterContour(cv2.imread('data_license_only/crop_h1/' + filename, 1))
#findCharacterContour(cv2.imread("data_license_only/crop_h1/I00000.png", 1))
