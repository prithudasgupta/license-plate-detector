import glob, io, cv2, numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy, copy


def cropCharacter(img, dimensions):
		[x,y,w,h] = dimensions
		character = deepcopy(img)
		character = deepcopy(character[y:y+h,x:x+w])
		return character

def findCharacterContour(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    plate_characters = []
    gray_plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_plate = cv2.GaussianBlur(gray_plate, (3,3), 0)

    _,threshold = cv2.threshold(gray_plate, 140, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    w,h,x,y = 0,0,0,0

    print("%s contours found.", str(len(contours)))
    for contour in sorted_ctrs:
        area = cv2.contourArea(contour)

        # rough range of areas of a plate number
        if area > 120 and area < 2000:
            [x,y,w,h] = cv2.boundingRect(contour)

        # rough dimensions of a character
        if h > 20 and h < 90 and w > 10 and w < 50:
            character = cropCharacter(img, [x,y,w,h])
            plate_characters.append(character)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.imshow('image',character)
            cv2.waitKey(0)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    print("%s plate characters found", str(len(plate_characters)))
		
findCharacterContour(cv2.imread("data_license_only/crop_h1/I00000.png", 1))