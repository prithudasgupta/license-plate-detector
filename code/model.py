import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense



class Model(tf.keras.Model):
    """ NN model for license plate detection """
    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.architecture = [
            Conv2D(96, 11, strides=4, padding="same", activation="relu"),
            Conv2D(256, 5, strides=2, padding="same", activation="relu"),
            MaxPool2D(2),
            Flatten(),
            Dropout(0.3),
            Dense(1024, activation="relu"),
            Dense(1000, activation="softmax")
        ]

    def call(self, img):
        for layer in self.architecture:
            img = layer(img)
        return img


    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
