import glob, io, cv2, os, numpy as np

word2id = {}
id2word = {}

def build_dictionaries():
    for i in range(10):
        word2id[str(i)] = i
        id2word[i] = str(i)
    for i in range(65, 91):
        char = chr(i)
        id = i - 55
        word2id[char] = id
        id2word[id] = char

def get_char_from_id(id):
    if len(word2id) == 0:
        build_dictionaries()
    return word2id[id]

def get_id_from_char(char):
    if len(id2word) == 0:
        build_dictionaries()
    return id2word[char]

def parse_images_and_labels(directory, train_test_ratio):

    build_dictionaries()

    images = []
    labels = []

    # Parse image arrays and get associated label
    for filename in os.listdir(directory):
        # Get image as grayscale
        img = cv2.imread(directory + "/" + filename, 0)
        img = np.float32(img)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img = img / 255.0
        images.append(img)

        # Get label
        character = filename[-5]
        labels.append(word2id[character])

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
