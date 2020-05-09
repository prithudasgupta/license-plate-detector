import cv2
import numpy as np
import math
import preprocess
import glob
#from detector import (validate_contour, get_bounding_box)
from model import Model
import hyperparameters as hp
import tensorflow as tf
from character_model import CharacterModel

DATA_DIR_BACKGROUNDS = 'data_backgrounds/'
DATA_DIR_LICENSE_ONLY = 'data_license_only/trainVal.csv'
DATA_DIR_SEGMENTED = 'data_segmented'
WEIGHTS_DIRECTORY = 'weights/weights'
TRAIN_TEST_RATIO = 0.10
VOCAB_SIZE = 36

def train(model, train_inputs, train_labels, test_inputs, test_labels):
    for e in range(hp.epochs):
        for batch_num in range(0, len(train_inputs), model.batch_size):
            with tf.GradientTape() as tape:
                logits = model.call(train_inputs[batch_num : batch_num + model.batch_size])
                loss = model.loss(logits, train_labels[batch_num : batch_num + model.batch_size])
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        p = model.call(test_inputs)
        acc = model.accuracy(p, test_labels)
        print("EPOCH " + str(e) + " | ACCURACY: " + str(acc))
    
    # model.save_weights(WEIGHTS_DIRECTORY)


def test(model, test_inputs, test_labels):
    p = model.call(test_inputs)
    return model.accuracy(p, test_labels)

def main():
    train_images, train_labels, test_images, test_labels = preprocess.parse_images_and_labels(DATA_DIR_SEGMENTED, TRAIN_TEST_RATIO)
    model = CharacterModel()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    train(model, train_images, train_labels, test_images, test_labels)
    # model.load_weights(WEIGHTS_DIRECTORY)
    # acc = test(model, test_images, test_labels)
    # print(acc)



if __name__ == "__main__":
    main()
