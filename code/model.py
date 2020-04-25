import tensorflow as tf
from tensorflow.keras import Model
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense, Reshape, Softmax

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

        model = tf.keras.Sequential()
        model.add(Conv2D(96, 11, strides=4, padding="same", activation="relu"))
        model.add(Conv2D(256, 5, strides=2, padding="same", activation="relu"))
        model.add(MaxPool2D(2))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(self.seq_len * self.vocab_size))
        model.add(Reshape((self.seq_len, self.vocab_size)))
        model.add(Softmax(axis=-1))

        self.model = model
        self.optimizer =tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.batch_size = 32

    @tf.function
    def call(self, img):
        return self.model(img)


    @tf.function
    def loss(self, logits, labels):
        """
        Loss function for the model.

        Labels are batch_size by seq_len
        Predictions are batch_size by seq_len by vocab_size
        """
        flattened_labels = tf.reshape(labels, [-1])
        flattened_predictions = tf.reshape(logits, [-1, self.vocab_size])
        return tf.keras.losses.sparse_categorical_crossentropy(
            flattened_labels, flattened_predictions, from_logits=False)


    @tf.function
    def accuracy(self, logits, labels):
        count = 0.0
        max = tf.argmax(logits, 2)
        # IF CHARACTER ACCURACY USE THIS hp.by_plate:
        if not hp.by_plate:
            equals = tf.equal(max, labels)
            res = tf.reduce_mean(tf.cast(equals, tf.float32))

        # PLATE ACCURACY USE THIS:
        else:
            for num in range(0,len(labels)):
                equals = tf.equal(max[num], labels[num])
                equals = tf.reduce_sum(tf.cast(equals, tf.float32))
                if equals == len(labels[num]):
                    count = count + 1.0
            res = count/len(labels)
        return res
