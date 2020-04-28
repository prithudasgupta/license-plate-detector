import cv2
import numpy as np
import math
import preprocess
import preprocess_plates
import glob
#from detector import (validate_contour, get_bounding_box)
from model import Model
import hyperparameters as hp
import tensorflow as tf

DATA_DIR = 'data/'
TRAIN_TEST_RATIO = 0.10

# Precalculated on current data set
SEQ_LEN = 8
VOCAB_SIZE = 37

def train(model, train_inputs, train_labels):
    for batch_num in range(0, len(train_inputs), model.batch_size):
        with tf.GradientTape() as tape:
            logits = model.call(train_inputs[batch_num : batch_num + model.batch_size])
            loss = model.loss(logits, train_labels[batch_num : batch_num + model.batch_size])
            #print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    p = model.call(test_inputs)
    return model.accuracy(p, test_labels)

def main():
    # Use this to train on license plate data only
    #train_images, train_labels, test_images, test_labels = preprocess_plates.parse_images_and_labels('data_license_only/trainVal.csv', TRAIN_TEST_RATIO)

    # Take note of how txt file in data directory must be formatted for every jpg file included in dataset!! Also, as of rn, must be jpg file but not hard to incorporate other file types.
    train_images, train_labels, test_images, test_labels = preprocess.parse_images_and_labels(DATA_DIR, TRAIN_TEST_RATIO)
    # print(train_images.shape)
    # print(train_labels.shape)
    # print(test_images.shape)
    # print(test_labels.shape)

    model = Model(SEQ_LEN, VOCAB_SIZE)
    train(model, train_images, train_labels)

    acc = test(model, test_images, test_labels)
    print(acc)

if __name__ == "__main__":
    main()
