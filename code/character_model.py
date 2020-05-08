import tensorflow as tf
from tensorflow.keras import Model
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

class CharacterModel(tf.keras.Model):
    """
    NN model for license plate detection on a single character
    """
    def __init__(self):
        super(Model, self).__init__()

        model = tf.keras.Sequential()
        model.add(Conv2D(120, 5, strides=1, padding="same", activation="relu"))
        model.add(MaxPool2D(4, 4))
        model.add(Conv2D(384, 2, strides=1, padding="same", activation="relu"))
        model.add(MaxPool2D(2, 2))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(37, activation="softmax"))

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.batch_size = 32

    @tf.function
    def call(self, img):
        return self.model(img)


    @tf.function
    def loss(self, logits, labels):
        """
        Loss function for the model.

        Labels are batch_size
        Predictions are batch_size by 37
        """
        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=False)


    @tf.function
    def accuracy(self, logits, labels):
        count = 0.0
        max = tf.argmax(logits, 1)
        for num in range(0,len(labels)):
            print(max[num])
            print(labels[num])
            if max[num] == labels[num]:
                count = count + 1.0
        res = count/len(labels)
        return res
