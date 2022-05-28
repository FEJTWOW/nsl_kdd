import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as k_layers
from src.feature_extraction import AbstractAutoencoder


class CategoricalAutoencoder(AbstractAutoencoder):

    def __init__(self, n_bottleneck, root='.', name='CategoricalAutoencoder', input_len=122, model_path=None):
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
        return [X[:, :38], X[:, 38:41], X[:, 41:111], X[:, 111:122]]

    def _compile(self, lr):
        self.autoencoder.compile(
            tf.keras.optimizers.Adam(learning_rate=lr),
            loss=['mse', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
            loss_weights=[1, 1, 1, 1]
        )

    def _build_model(self):
        inputs = k_layers.Input(self.input_len)

        x = k_layers.Dense(120, activation='relu')(inputs)
        x = k_layers.Dense(60, activation='relu')(x)
        x = k_layers.Dense(30, activation='relu')(x)
        encoded_outputs = k_layers.Dense(self.n_bottleneck)(x)
        x = k_layers.Dense(30, activation='relu')(encoded_outputs)
        x = k_layers.Dense(60, activation='relu')(x)
        x = k_layers.Dense(120, activation='relu')(x)

        output_1 = k_layers.Dense(38)(x)
        output_2 = k_layers.Dense(3, activation='softmax')(x)
        output_3 = k_layers.Dense(70, activation='softmax')(x)
        output_4 = k_layers.Dense(11, activation='softmax')(x)

        encoder = tf.keras.Model(inputs=inputs, outputs=encoded_outputs)
        autoencoder = tf.keras.Model(inputs=inputs, outputs=[output_1, output_2, output_3, output_4])

        return encoder, autoencoder
