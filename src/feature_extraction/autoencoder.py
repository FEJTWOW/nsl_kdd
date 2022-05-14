import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as k_layers


class Autoencoder:
    """
    Class implementing an autoencoder for nsl kdd feature extraction
    """

    def __init__(self, n_bottleneck, root='.', name='default', input_len=38, model_path=None):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param root: path to root directory
        :param name: Autoencoder name
        :param input_len: Length of base feature vectors
        :param model_path: Path to save model weights and loss history. If None, will be set to
        f'model_weights/autoencoder/{n_bottleneck}'
        """
        self.n_bottleneck = n_bottleneck
        self.name = name
        self.input_len = input_len
        self.model_path = model_path
        if model_path is None:
            self.model_path = f"{root}/model_weights/{name}/autoencoder_{n_bottleneck}"

        self.encoder, self.autoencoder = self._build_model()
        self.history = None

    def compile_and_train(self, X, batch_size=8192, n_epochs=400, lr=0.002, lr_patience=20, verbose=1, save=True):
        """
        Compile and train the autoencoder
        :param X: An array of shape (n x m) with n vectors, each of length m
        :param batch_size: batch size
        :param n_epochs: number of epochs
        :param lr: learning rate.
        :param lr_patience: patience for the ReduceLROnPlateau callback. If None, learning rate is constant
        :param verbose: verbose parameter for training the model
        :param save: True: save the weights and history to self.model_path; False: do not save
        :return: self
        """
        self.autoencoder.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

        callbacks = []
        if lr_patience is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=lr_patience))

        self.history = self.autoencoder.fit(X, X, batch_size=batch_size, epochs=n_epochs, verbose=verbose,
                                            callbacks=callbacks).history
        if save:
            self.autoencoder.save_weights(f"{self.model_path}/weights")
            pd.DataFrame(self.history).to_csv(f"{self.model_path}/loss_log.csv")

        return self

    def load_weights(self, path=None, load_history=False):
        """
        Load weights and history from self.model_path
        :return: self
        """
        if path is None:
            path = self.model_path

        if load_history:
            self.history = pd.read_csv(f"{path}/loss_log.csv")
        self.autoencoder.load_weights(f"{path}/weights").expect_partial()

        return self

    def encode(self, X):
        """
        Extract features
        :param X: An array of shape (n x m) with n vectors, each of length m
        :return: Extracted features
        """
        return self.encoder.predict(X)

    def predict(self, X):
        """
        Autoencodes the data
        :param X: An array of shape (n x m) with n vectors, each of length
        :return: Autoencoded data
        """
        return self.autoencoder.predict(X)

    def compile_and_evaluate(self, X):
        """
        Compile and evaluate model on X
        :param X: An array of shape (n x m) with n vectors, each of length
        :return: Loss value
        """
        self.autoencoder.compile("adam", loss="mse")
        return self.autoencoder.evaluate(X, X)

    def summary(self):
        """
        :return: Summary of the full model
        """
        return self.autoencoder.summary()

    def _build_model(self):
        inputs = k_layers.Input(self.input_len)

        x = k_layers.Dense(30, activation='relu')(inputs)
        encoded_outputs = k_layers.Dense(self.n_bottleneck)(x)
        x = k_layers.Dense(30, activation='relu')(encoded_outputs)
        outputs = k_layers.Dense(self.input_len)(x)

        encoder = tf.keras.Model(inputs, encoded_outputs)
        autoencoder = tf.keras.Model(inputs, outputs)

        return encoder, autoencoder


if __name__ == "__main__":
    autoencoder = Autoencoder(4)
    print(autoencoder.summary())
    print("==============================================================================")
    print("==============================================================================")
    print("==============================================================================")
    print(autoencoder.encoder.summary())

    X = np.random.random((100, 38))
    autoencoder.compile_and_train(X, save=False, n_epochs=500)

    print(autoencoder.encode(X[:5]))

    print("==============================================================================")
    print(X[3])
    print(autoencoder.predict(X)[3])
