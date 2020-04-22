import cv2
import numpy as np
import math
import preprocess
from detector import (validate_contour, get_bounding_box)
from model import Model

DATA_DIR = 'data/'
TRAIN_TEST_RATIO = 0.10

def train(model, datasets, checkpoint_path):
    print("todo")

def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def main():
    img = cv2.imread('testImage.jpg',1)
        # img = cv2.imread('0b86cecf-67d1-4fc0-87c9-b36b0ee228bb.jpg', 1)
        # img = cv2.imread('12c6cb72-3ea3-49e7-b381-e0cdfc5e8960.jpg', 1)
        # img = cv2.imread('12c6cb72-3ea3-49e7-b381-e0cdfc5e8960.jpg', 1)
        # img = cv2.imread('1e241dc8-8f18-4955-8988-03a0ab49f813.jpg', 1)
        # res = get_bounding_box(img)

    # Take note of how txt file in data directory must be formatted for every jpg file included in dataset!! Also, as of rn, must be jpg file but not hard to incorporate other file types.
    train_images, train_labels, test_images, test_labels = preprocess.parse_images_and_labels(DATA_DIR, TRAIN_TEST_RATIO)
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    model = Model()

if __name__ == "__main__":
    main()
