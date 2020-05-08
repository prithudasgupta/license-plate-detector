import glob, io, cv2, numpy as np
from csv import reader
from segmentation import findCharacterContour
import tensorflow as tf

vals = open('data_license_only/trainVal.csv', 'r')
lines = reader(vals)
count = 0
for row in lines:
    count += 1
    if count < 461:
        continue

    file_path = row[1]
    plate_number = row[2]
    file_path = file_path.replace("./crop", "./data_license_only/crop")

    img = cv2.imread(file_path, 1)
    character_contours = findCharacterContour(img)

    for i in range(min(len(character_contours), len(plate_number))):
        file_name = 'data_segmented/' + str(count) + '_' + str(i) + '_' + plate_number[i]  + '.jpg'
        cv2.imwrite(file_name, character_contours[i, :, :])
        print("Wrote to file " + str(file_name))
