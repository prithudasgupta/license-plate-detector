import glob, io, cv2, numpy as np
from csv import reader

# Should change so this is calculated on the fly
SEQ_LEN = 8

def parse_images_and_labels(directory, train_test_ratio):

    image_filepaths = []
    images = []
    labels = []

    word2id = {}
    vocab_size = 1

    vals = open(directory, 'r')
    lines = reader(vals)
    count = 0
    for row in lines:
        count += 1
        if count == 1:
            continue
        file_path = row[1]
        file_path = file_path.replace("./crop", "./data_license_only/crop")

        img = cv2.imread(file_path, 1)
        img = np.float32(img)

        # TODO: Detector, parsing, and resizing should go here
        img = cv2.resize(img, (100, 100))

        images.append(img)

        image_filepaths.append(file_path)

        plate_label = row[2]

        # Covert license plate string to an array of seq_len unique ids based on characters
        spliced_characters = list(plate_label)
        for i in range(len(spliced_characters)):
            if not word2id.get(spliced_characters[i]):
                word2id[spliced_characters[i]] = vocab_size
                vocab_size = vocab_size + 1
            spliced_characters[i] = word2id[spliced_characters[i]]
        while len(spliced_characters) < SEQ_LEN:
            # 0 is padding token
            spliced_characters.append(0)

        labels.append(spliced_characters)

    images = np.array(images)
    labels = np.array(labels)
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
