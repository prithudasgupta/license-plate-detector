import cv2
import numpy as np
import math

def get_bounding_box(img):
    '''
    Given image of a potential car,
    returns top left and bottom right pixel of potenital license plate.
    '''
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
    
    for contour in contours:
        rect = cv2.minAreaRect(contour)  
        box = cv2.boxPoints(rect)

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

        tmp = cv2.getRectSubPix(img, size, center)
        tmp = cv2.warpAffine(tmp, m, size)
        tmpw = h if h > w else w
        tmph = h if h < w else w
        tmp = cv2.getRectSubPix(tmp, (int(tmpw),int(tmph)), (size[0] / 2, size[1] / 2))

        if edge_density > 0.5:
            cv2.drawContours(output_image, [box.astype(int)], 0, (127,0,255),2)

    cv2.imshow('image',output_image)
    cv2.waitKey(0)


def main():
    img = cv2.imread('testImage.jpg',1)
    res = get_bounding_box(img)

main()


