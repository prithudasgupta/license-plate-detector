vals = open('data_license_only/trainVal.csv', 'r')
lines = reader(vals)
count = 0
for row in lines:
    count += 1
    if count == 1:
        continue

    file_path = row[1]
    plate_number = row[2]
    file_path = file_path.replace("./crop", "./data_license_only/crop")

    img = cv2.imread(file_path, 1)
    character_contours = findCharacterContour(img)

    for i in range(min(len(character_contours), len(plate_number))):
        cv2.imwrite('data_segmented/' + str(count) + '_' + plate_number[i]  + '.jpg', character_contours[i, :, :])