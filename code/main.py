import cv2
import numpy as np
import math

from detector import (validate_contour, get_bounding_box)
from model import Model

def train(model, datasets, checkpoint_path):
    print("todo")

def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def main():
    img = cv2.imread('testImage.jpg',1)
    img = cv2.imread('0b86cecf-67d1-4fc0-87c9-b36b0ee228bb.jpg', 1)
    img = cv2.imread('12c6cb72-3ea3-49e7-b381-e0cdfc5e8960.jpg', 1)
    img = cv2.imread('12c6cb72-3ea3-49e7-b381-e0cdfc5e8960.jpg', 1)
    img = cv2.imread('1e241dc8-8f18-4955-8988-03a0ab49f813.jpg', 1)
    res = get_bounding_box(img)

    model = Model()

if __name__ == "__main__":
    main()
