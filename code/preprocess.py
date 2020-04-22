import glob, io, cv2, numpy as np

def parse_images_and_labels(directory, train_test_ratio):

    image_filepaths = []
    images = []
    labels = []

    # Parse image arrays and store filenames to associate with txt label files
    for file_path in sorted(glob.glob(directory + '*.jpg')):
        #print("Current File Being Processed is: " + file)

        img = cv2.imread(file_path, 1)
        images.append(img)

        # Remove .jpg from end of filename
        file_path = file_path[:-4]
        image_filepaths.append(file_path)

    # Find txt label files and parse labels for every image
    for file_path in image_filepaths:
        #print("Current File Being Processed is: " + file)

        # Add .txt to end of filename for check that correct label is being read (sanity check in case typo in filenames affecting sorting, label txt file DNE, etc)
        file_path = file_path + ".txt"
        try:
            with io.open(file_path, mode="r", encoding="utf-8") as f:
                plate_label = f.readline().split()[5]
                labels.append(plate_label)
        except:
            print("Associated txt file", file_path, "was not found with its jpg image. Ensure proper guidelines were followed for associating image with label.")

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


