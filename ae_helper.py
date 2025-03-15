
import tensorflow as tf
from tensorflow.keras import layers, models, Model


def get_auto():
    encoder = tf.keras.Sequential([
        layers.Input(shape=(552, 2)),
        layers.Conv1D(filters=32, kernel_size=20, activation='relu', strides=2, padding="same"),
        layers.Conv1D(filters=64, kernel_size=20, activation='relu', strides=2, padding="same"),
        layers.MaxPool1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(60, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(5, activation='linear'),
    ])

    decoder = tf.keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(30, activation='relu'),
        layers.Dense(4416, activation='relu'),
        layers.Reshape((69, 64)),
        layers.UpSampling1D(2),
        layers.Conv1DTranspose(filters=64, kernel_size=20, activation='relu', strides=2, padding="same", output_padding=2),
        layers.Conv1DTranspose(filters=32, kernel_size=20, activation='relu', strides=2, padding="same"),
        layers.Conv1DTranspose(filters=2, kernel_size=20, activation='linear', padding="same"),
    ])

    return encoder, decoder


class AE(Model):
    def __init__(self, encoder: tf.keras.Sequential, decoder: tf.keras.Sequential):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def set_trainable(self, trainable):
        self.trainable = trainable
        self.encoder.trainable = trainable
        self.decoder.trainable = trainable