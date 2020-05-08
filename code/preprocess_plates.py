import glob, io, cv2, numpy as np
from csv import reader
from segmentation import findCharacterContour
import tensorflow as tf

def parse_images_and_labels(directory, train_test_ratio):

    image_filepaths = []
    images = []
    labels = []

    word2id = {}
    vocab_size = 0

    vals = open(directory, 'r')
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
            print(plate_number[i])
            cv2.imshow('image', character_contours[i, :, :])
            cv2.waitKey(0)
            images.append(tf.cast(np.reshape(character_contours[i, :, :], (50, 100, 1)), tf.float32))
            curr_char = plate_number[i]
            if not word2id.get(curr_char):
                word2id[curr_char] = vocab_size
                vocab_size = vocab_size + 1
            labels.append(word2id[curr_char])

    images = np.array(images)
    labels = np.array(labels)
    print(images.shape)
    print(labels.shape)
    assert len(images) == len(labels)

    # Split into train and test sets
    train_images, train_labels, test_images, test_labels = split_train_test(images, labels, train_test_ratio)

    assert len(train_images) == len(train_labels)
    assert len(test_images) == len(test_labels)

    return train_images, train_labels, test_images, test_labels


# Shuffles Images and Labels and Splits into Train and test sets based on ratio
def split_train_test(images, labels, ratio):
    num_test = int(images.shape[0] * ratio)
    s = np.arange(images.shape[0])
    np.random.shuffle(s)
    images = images[s]
    labels = labels[s]
    return images[num_test:], labels[num_test:], images[:num_test], labels[:num_test]
