import cv2
import numpy as np
import math
import argparse
import preprocess
import glob
from detector import (validate_contour, get_bounding_box)
from model import Model
import os
import shutil
import hyperparameters as hp
import tensorflow as tf
from character_model import CharacterModel
from segmentation import findCharacterContour

DATA_DIR_BACKGROUNDS = 'data_backgrounds/'
DATA_DIR_LICENSE_ONLY = 'data_license_only/trainVal.csv'
DATA_DIR_SEGMENTED = 'data_segmented'
WEIGHTS_DIRECTORY = 'weights/weights'
SAVED_WEIGHTS_DIR = 'saved_weights/'
TRAIN_TEST_RATIO = 0.10
VOCAB_SIZE = 36

def parse_args():
    parser = argparse.ArgumentParser(
        description="License Plate Detector")
    parser.add_argument(
        '--generate-weights',
        action='store_true',
        help='''Use this flag to train model from scratch and generate/save new weights''')
    parser.add_argument(
        '--load-weight',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. Load weights to test on custom image''')
    parser.add_argument(
        '--test-uploaded-image',
        default=None,
        help='''Path to image of license plate to read license plate with uploaded trained model weights.'''
    )
    return parser.parse_args()

def train(model, train_inputs, train_labels, test_inputs, test_labels):
    for e in range(hp.epochs):
        for batch_num in range(0, len(train_inputs), model.batch_size):
            with tf.GradientTape() as tape:
                logits = model.call(train_inputs[batch_num : batch_num + model.batch_size])
                loss = model.loss(logits, train_labels[batch_num : batch_num + model.batch_size])
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        p = model.call(test_inputs)
        acc = float(model.accuracy(p, test_labels))
        model.save_weights(SAVED_WEIGHTS_DIR + 'epoch' + str(e) + '-acc' + str(round(acc, 4)) + '.h5')
        print("EPOCH " + str(e) + " | ACCURACY: " + str(acc))
    
    # model.save_weights(WEIGHTS_DIRECTORY)


def test(model, test_inputs, test_labels):
    p = model.call(test_inputs)
    return model.accuracy(p, test_labels)

def main(ARGS):

    if ARGS.generate_weights:
        train_images, train_labels, test_images, test_labels = preprocess.parse_images_and_labels(DATA_DIR_SEGMENTED, TRAIN_TEST_RATIO)

        if os.path.exists(SAVED_WEIGHTS_DIR):
            shutil.rmtree(SAVED_WEIGHTS_DIR)

        os.mkdir(SAVED_WEIGHTS_DIR)
        model = CharacterModel()
        train(model, train_images, train_labels, test_images, test_labels)
    elif ARGS.load_weight is not None and ARGS.test_uploaded_image is not None and os.path.exists(ARGS.load_weight) and os.path.exists(ARGS.test_uploaded_image):

        img = cv2.imread(ARGS.test_uploaded_image, 1)
        plate = get_bounding_box(img)
        #
        # cv2.imshow('image',plate)
        # cv2.waitKey(0)


        characters = findCharacterContour(plate)
        characters = np.float32(characters)

        # for c in characters:
        #     cv2.imshow('image',c)
        #     cv2.waitKey(0)

        characters = np.reshape(characters, (characters.shape[0], characters.shape[1], characters.shape[2], 1))
        characters = characters / 255.0

        print(characters.shape)
        model = CharacterModel()
        model(tf.keras.Input(shape=(100, 50, 1)))
        model.load_weights(ARGS.load_weight)
        logits = model.call(characters)
        final_plate = find_license_strings(logits)
        print("Plate number found:", final_plate)

    else:
        print("ERROR: Ensure both trained weights path and an uploaded image path is provided and are both valid paths!")


def find_license_strings(logits):
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    final_license = ""

    for pred in predictions:
        final_license += preprocess.get_id_from_char(int(pred))

    return final_license



if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
