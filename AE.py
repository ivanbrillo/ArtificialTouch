import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class AE(Model):
    def __init__(self, encoder: tf.keras.Sequential, decoder: tf.keras.Sequential):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_auto():
    # Encoder
    encoder = tf.keras.Sequential([
        layers.Input(shape=(702, 2)),
        layers.Conv1D(filters=32, kernel_size=32, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),

        layers.Conv1D(filters=64, kernel_size=16, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='linear'),
    ])

    # Decoder
    decoder = tf.keras.Sequential([
        layers.Input(shape=(16,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(87 * 128, activation='relu'),
        layers.Reshape((87, 128)),

        layers.UpSampling1D(2),  # 320
        layers.Conv1DTranspose(filters=128, kernel_size=8, activation='relu', padding='same'),

        layers.UpSampling1D(2),  # 640
        layers.Conv1DTranspose(filters=64, kernel_size=16, activation='relu', padding='same'),
        layers.ZeroPadding1D(padding=(0, 3)),  # Pad to reach 1286 from 1280

        layers.UpSampling1D(2),  # 1280
        layers.Conv1DTranspose(filters=32, kernel_size=32, activation='relu', padding='same'),
        layers.Conv1DTranspose(filters=2, kernel_size=32, activation='linear', padding='same'),
    ])

    return encoder, decoder
