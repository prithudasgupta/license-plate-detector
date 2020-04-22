import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense, Reshape, Softmax

voc_size = 36

class Model(tf.keras.Model):
    """ 
    NN model for license plate detection 
    
    seq_len: Number of characters in license plate (~6/7)
    vocab_size: Number of possible characters (~(26 + 10))


    """
    def __init__(self, seq_len, vocab_size):
        super(Model, self).__init__()

        # logits should be batch_size by seq_len by vocab_size
        # labels will be batch_size by seq_len
        # flatten both for cross entropy loss
        # last dense layer should be seq_len * vocab_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        voc_size = vocab_size

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
            Dense(self.seq_len * self.vocab_size),
            Reshape((self.seq_len, self.vocab_size)),
            Softmax(axis=-1)
        ]

    def call(self, img):
        for layer in self.architecture:
            img = layer(img)
        return img


    @staticmethod
    def loss_fn(labels, predictions):
        """ 
        Loss function for the model. 
        
        Labels are batch_size by seq_len
        Predictions are batch_size by seq_len by vocab_size
        """
        flattened_labels = tf.reshape(labels, [-1])
        # Probably a better way to write this but this should work for now
        flattened_predictions = tf.reshape(predictions, [-1, voc_size])
        return tf.keras.losses.sparse_categorical_crossentropy(
            flattened_labels, flattened_predictions, from_logits=False)
