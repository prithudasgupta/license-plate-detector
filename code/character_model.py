import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

class CharacterModel(tf.keras.Model):
    """
    NN model for license plate detection on a single character
    """
    def __init__(self):
        super(CharacterModel, self).__init__()

        model = tf.keras.Sequential()
        model.add(Conv2D(120, 5, strides=1, padding="same", activation="relu"))
        model.add(MaxPool2D(4, 4))
        model.add(Conv2D(384, 2, strides=1, padding="same", activation="relu"))
        model.add(MaxPool2D(2, 2))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(36, activation="softmax"))
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
        Predictions are batch_size by 36
        """
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=False))


    @tf.function
    def accuracy(self, logits, labels):
        tf.print(logits, summarize=-1)
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        tf.print(predictions, summarize=-1)
        tf.print(labels, summarize=-1)
        actual = tf.cast(labels, tf.int32)
        correct = tf.equal(predictions, actual)
        return tf.reduce_mean(tf.cast(correct, tf.float32))
