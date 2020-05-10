import cv2
import numpy as np
import math

from segmentation import findCharacterContour

def validate_contour(contour, img, aspect_ratio_range, area_range):
    rect = cv2.minAreaRect(contour)
    img_width = img.shape[1]
    img_height = img.shape[0]
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    X = rect[0][0]
    Y = rect[0][1]
    angle = rect[2]
    width = rect[1][0]
    height = rect[1][1]

    angle = (angle + 180) if width < height else (angle + 90)

    output = False

    if (width > 0 and height > 0) and ((width < img_width/2.0) and (height < img_width/2.0)):
        aspect_ratio = float(width)/height if width > height else float(height)/width
        if (aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]):
            if((height*width > area_range[0]) and (height*width < area_range[1])):

                box_copy = list(box)
                point = box_copy[0]
                del(box_copy[0])
                dists = [((p[0]-point[0])**2 + (p[1]-point[1])**2) for p in box_copy]
                sorted_dists = sorted(dists)
                opposite_point = box_copy[dists.index(sorted_dists[1])]
                tmp_angle = 90

                if abs(point[0]-opposite_point[0]) > 0:
                    tmp_angle = abs(float(point[1]-opposite_point[1]))/abs(point[0]-opposite_point[0])
                    tmp_angle = (180 / np.pi) * (math.atan(tmp_angle))

                if tmp_angle <= 45:
                    output = True
    return output

def getSubImage(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out

def get_bounding_box(img):
    '''
    Given image of a potential car,
    returns potential bounding boxes of potenital license plate.
    '''
    # Could add probability in addition to just box.
    results = []

    output_image = np.copy(img)

    # 1. Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply Guassian blur
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    # 3. Vertical sobel
    img_sobel = cv2.Sobel(img_blur, -1, 1, 0)
    h,sobel = cv2.threshold(img_sobel,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_RECT,(16,4))
    morph = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)

    ed_img = np.copy(morph)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best_image = None
    best_contrast = -1
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        aspect_ratio_range = (1, 12)
        area_range = (500, 18000)
        test_contour = validate_contour(contour, morph, aspect_ratio_range, area_range)
        if test_contour:
            xs = [i[0] for i in box]
            ys = [i[1] for i in box]
            x1 = min(xs)
            x2 = max(xs)
            y1 = min(ys)
            y2 = max(ys)

            angle = rect[2]
            if angle < -45:
                angle += 90

            w = rect[1][0]
            h = rect[1][1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = float(w) / h if w > h else float(h) / w

            center = ((x1+x2)/2,(y1+y2)/2)
            size = (x2-x1, y2-y1)

            m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
            tmp = cv2.getRectSubPix(ed_img, size, center)
            tmp = cv2.warpAffine(tmp, m, size)

            tmpw = h if h > w else w
            tmph = h if h < w else w
            tmp = cv2.getRectSubPix(tmp, (int(tmpw), int(tmph)), (size[0] / 2, size[1] / 2))
            _, tmp = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            white_pixels = 0
            if tmp is None:
                continue
            for x in range(tmp.shape[0]):
                for y in range(tmp.shape[1]):
                    if tmp[x][y] == 255:
                        white_pixels += 1

            edge_density = float(white_pixels) / (tmp.shape[0] * tmp.shape[1])

            # separate out false positives
            if edge_density > 0.45:
                img_crop = getSubImage(rect, img)
                h, w, c = img_crop.shape
                if h > w:
                    img_crop = np.rot90(img_crop)
                plate_chars = findCharacterContour(img_crop)
                right_chars = plate_chars.shape[0] >= 3 and plate_chars.shape[0] <= 8
                if right_chars:
                    cv2.drawContours(output_image, [box.astype(int)], 0, (127,0,255),2)
                    best_image = img_crop

    cv2.imshow('image',output_image)
    cv2.waitKey(0)
    cv2.imshow('image',best_image)
    cv2.waitKey(0)
    return best_image