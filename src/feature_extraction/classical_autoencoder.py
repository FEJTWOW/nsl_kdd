import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as k_layers
from src.feature_extraction import AbstractAutoencoder


class ClassicalAutoencoder(AbstractAutoencoder):

    def __init__(self, n_bottleneck, root='.', name='ClassicalAutoencoder', input_len=38, model_path=None):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param root: path to root directory
        :param name: Autoencoder name
        :param input_len: Length of base feature vectors
        :param model_path: Path to save model weights and loss history. If None, will be set to
        f'model_weights/autoencoder/{n_bottleneck}'
        """
        AbstractAutoencoder.__init__(self, n_bottleneck, root, name, input_len, model_path)

    @staticmethod
    def _retrieve_output(X):
        return X

    def _compile(self, lr):
        self.autoencoder.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

    def _build_model(self):
        inputs = k_layers.Input(self.input_len)

        x = k_layers.Dense(30, activation='relu')(inputs)
        encoded_outputs = k_layers.Dense(self.n_bottleneck)(x)
        x = k_layers.Dense(30, activation='relu')(encoded_outputs)
        outputs = k_layers.Dense(self.input_len)(x)

        encoder = tf.keras.Model(inputs, encoded_outputs)
        autoencoder = tf.keras.Model(inputs, outputs)

        return encoder, autoencoder
