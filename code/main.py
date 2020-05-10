import cv2
import numpy as np
import math
import argparse
import preprocess
import glob
import matplotlib.pyplot as plt
from detector import (validate_contour, get_bounding_box)
from model import Model
import os
import shutil
import hyperparameters as hp
import tensorflow as tf
from character_model import CharacterModel
from segmentation import findCharacterContour
from scipy.interpolate import make_interp_spline

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
    parser.add_argument(
        '--needs-detector',
        action='store_true',
        help='''Use this flag if license plate in the image needs to be detected first, but do not use this flag if the picture is simply
        a plate already extracted itself.'''
    )
    return parser.parse_args()

def train(model, train_inputs, train_labels, test_inputs, test_labels):
    loss_values = []
    for e in range(hp.epochs):
        for batch_num in range(0, len(train_inputs), model.batch_size):
            with tf.GradientTape() as tape:
                #Classic train for model, gets logits and runs optimizer with loss results per epoch and batch.
                logits = model.call(train_inputs[batch_num : batch_num + model.batch_size])
                loss = model.loss(logits, train_labels[batch_num : batch_num + model.batch_size])
                loss_values.append(float(loss))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        p = model.call(test_inputs)
        acc = float(model.accuracy(p, test_labels))

        #Save weights and print epoch accuracy
        model.save_weights(SAVED_WEIGHTS_DIR + 'epoch' + str(e) + '-acc' + str(round(acc, 4)) + '.h5')
        print("EPOCH " + str(e) + " | ACCURACY: " + str(acc))
    return loss_values


def test(model, test_inputs, test_labels):
    p = model.call(test_inputs)
    return model.accuracy(p, test_labels)

def main(ARGS):

    if ARGS.generate_weights:

        #If training model from scratch, use preprocess pre-segmented license plates (found in data_segmented folder) data and feed train and test sets into model.
        train_images, train_labels, test_images, test_labels = preprocess.parse_images_and_labels(DATA_DIR_SEGMENTED, TRAIN_TEST_RATIO)

        if os.path.exists(SAVED_WEIGHTS_DIR):
            shutil.rmtree(SAVED_WEIGHTS_DIR)

        os.mkdir(SAVED_WEIGHTS_DIR)
        model = CharacterModel()
        loss_values = train(model, train_images, train_labels, test_images, test_labels)

        # Plotting code for loss graph
        timesteps = list(range(len(loss_values)))
        a_BSpline = make_interp_spline(timesteps, loss_values)
        y_smooth = a_BSpline(timesteps)
        plt.plot(y_smooth)
        plt.ylabel('loss')
        plt.xlabel('batch number')
        plt.show()

    elif ARGS.load_weight is not None and ARGS.test_uploaded_image is not None and ARGS.needs_detector is not None and os.path.exists(ARGS.load_weight) and os.path.exists(ARGS.test_uploaded_image):

        img = cv2.imread(ARGS.test_uploaded_image, 1)

        #If needs detector is specified, use detector to find and extract license plate.
        if ARGS.needs_detector:
            img = get_bounding_box(img)

        #Use segmentation to retrieve individual characters in plate
        characters = findCharacterContour(img)
        characters = np.float32(characters)
        characters = np.reshape(characters, (characters.shape[0], characters.shape[1], characters.shape[2], 1))
        characters = characters / 255.0

        #Load model and feed image into model to calculate logits and determine the plate string.
        model = CharacterModel()
        model(tf.keras.Input(shape=(100, 50, 1)))
        model.load_weights(ARGS.load_weight)
        logits = model.call(characters)
        final_plate = find_license_strings(logits)
        print("Plate number found:", final_plate)

    else:
        print("ERROR: Ensure both trained weights path and an uploaded image path is provided and are both valid paths! Also, ensure needs_detector flag is there if needed.")


def find_license_strings(logits):

    #Find argmax of indices to find which character model believes is each segmented image in plate.
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    final_license = ""
    for pred in predictions:
        final_license += preprocess.get_id_from_char(int(pred))
    return final_license


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
