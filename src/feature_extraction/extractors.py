from src.feature_extraction import Autoencoder


class AutoencoderExtractor:
    """
    Enables easy access to autoencoders and feature extraction
    """

    def __init__(self, n_bottleneck, name='default', root=None, input_len=38):
        """
        :param n_bottleneck: Number of neurons in the bottleneck
        :param name: Autoencoder name
        :param root: Root directory of the repo
        :param input_len: Length of base feature vectors
        """
        if root is None:
            root = "../.."
        model_path = f"{root}/src/feature_extraction/model_weights/{name}/autoencoder_{n_bottleneck}/"
        self.autoencoder = Autoencoder(n_bottleneck, input_len=input_len, model_path=model_path)
        self.autoencoder.load_weights()

    def extract_features(self, X):
        """
        Use the autoencoder to extract features from the vector
        :param X: An array of shape (n x m) with n vectors, each of length m
        :return: Extracted features
        """
        return self.autoencoder.encode(X)

    def evaluate(self, X):
        """
        Evaluates the autoencoder
        :param X: An array of shape (n x m) with n vectors, each of length m
        :return: Loss value
        """
        return self.autoencoder.compile_and_evaluate(X)
