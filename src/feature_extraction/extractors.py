from sklearn.preprocessing import StandardScaler

from src.feature_extraction import AbstractAutoencoder
from src.feature_extraction import CategoricalAutoencoder
from src.feature_extraction import ClassicalAutoencoder


class AutoencoderExtractor:
    """
    Enables easy access to autoencoders and feature extraction
    """

    def __init__(self, n_bottleneck, input_len, name='default', root=None):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param name: Autoencoder name
        :param root: Root directory of the repo
        :param input_len: Length of base feature vectors
        """
        if root is None:
            root = "../.."
        model_path = f"{root}/src/feature_extraction/model_weights/{name}/autoencoder_{n_bottleneck}/"
        self.autoencoder = None
        self._set_autoencoder(n_bottleneck, input_len, model_path)

    def extract_features(self, X, scale=True):
        """
        Use the autoencoder to extract features from the vector
        :param X: An array of shape (n x m) with n vectors, each of length m
        :param scale: If True, features will be transformed with StandardScaler
        :return: Extracted features
        """
        Y = self.autoencoder.encode(X)
        if scale:
            Y = StandardScaler().fit_transform(Y)
        return Y

    def evaluate(self, X):
        """
        Evaluates the autoencoder
        :param X: An array of shape (n x m) with n vectors, each of length m
        :return: Loss value
        """
        return self.autoencoder.compile_and_evaluate(X)

    def _set_autoencoder(self, n_bottleneck, input_len, model_path):
        raise "Method not implemented"


class CategoricalExtractor(AutoencoderExtractor):

    def __init__(self, n_bottleneck, input_len=122, name='CategoricalAutoencoder', root=None):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param name: Autoencoder name
        :param root: Root directory of the repo
        :param input_len: Length of base feature vectors
        """
        AutoencoderExtractor.__init__(self, n_bottleneck, input_len, name, root)

    def _set_autoencoder(self, n_bottleneck, input_len, model_path):
        self.autoencoder = CategoricalAutoencoder(n_bottleneck, input_len=input_len, model_path=model_path)
        self.autoencoder.load_weights()


class ClassicalExtractor(AutoencoderExtractor):

    def __init__(self, n_bottleneck, input_len=38, name='ClassicalAutoencoder', root=None):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param name: Autoencoder name
        :param root: Root directory of the repo
        :param input_len: Length of base feature vectors
        """
        AutoencoderExtractor.__init__(self, n_bottleneck, input_len, name, root)

    def _set_autoencoder(self, n_bottleneck, input_len, model_path):
        self.autoencoder = ClassicalAutoencoder(n_bottleneck, input_len=input_len, model_path=model_path)
        self.autoencoder.load_weights()
