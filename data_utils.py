import numpy as np
import pandas as pd
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from tensorflow.keras import layers as k_layers

ATTACK_LABELS_TYPES = {
    'normal': ['normal'],
    'DoS': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm',
            'worm'],
    'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
    'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'],
    'R2L': ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
            'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
}

CATEGORICAL_FEATURES = [1, 2, 3]


def _preprocess_Y(Y_train_base, Y_test_base, return_classes=True):
    attack_labels_dict = {v: k for k, v_list in ATTACK_LABELS_TYPES.items() for v in v_list}
    Y_train_grouped = np.array(list(map(lambda x: attack_labels_dict[x], Y_train_base)))
    Y_test_grouped = np.array(list(map(lambda x: attack_labels_dict[x], Y_test_base)))
    attack_labels_base = list(set(np.unique(Y_train_grouped)) | set(np.unique(Y_test_grouped)))

    label_binarizer = LabelBinarizer()
    label_binarizer = label_binarizer.fit(attack_labels_base)

    Y_train = label_binarizer.transform(Y_train_grouped)
    Y_test = label_binarizer.transform(Y_test_grouped)

    if return_classes:
        return (Y_train, Y_test), label_binarizer.classes_
    else:
        return Y_train, Y_test


def _preprocess_X(X_train_base, X_test_base):
    numerical_features = list(set(range(X_train_base.shape[1])) - set(CATEGORICAL_FEATURES))

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(np.vstack([X_train_base[:, CATEGORICAL_FEATURES], X_test_base[:, CATEGORICAL_FEATURES]]))

    X_train = np.hstack([X_train_base[:, numerical_features],
                         one_hot_encoder.transform(X_train_base[:, CATEGORICAL_FEATURES]).toarray()])
    X_test = np.hstack(
        [X_test_base[:, numerical_features], one_hot_encoder.transform(X_test_base[:, CATEGORICAL_FEATURES]).toarray()])

    return X_train, X_test


def load_train_test_data(pwd='.', return_classes=True):
    df_train = pd.read_csv(f'{pwd}/NSL-KDD/KDDTrain+.txt', header=None)
    df_test = pd.read_csv(f'{pwd}/NSL-KDD/KDDTest+.txt', header=None)
    X_train_base, Y_train_base = df_train.iloc[:, :41].values, df_train.iloc[:, 41].values
    X_test_base, Y_test_base = df_test.iloc[:, :41].values, df_test.iloc[:, 41].values

    (Y_train, Y_test), attack_classes = _preprocess_Y(Y_train_base, Y_test_base, return_classes=True)

    X_train, X_test = _preprocess_X(X_train_base, X_test_base)

    if return_classes:
        return (X_train, X_test, Y_train, Y_test), attack_classes
    else:
        return X_train, X_test, Y_train, Y_test


def run_and_measure(fun, *args, **kwargs):
    start = time.time()
    retval = fun(*args, **kwargs)
    end = time.time()
    print(f'{(end - start):0.2f} s')
    return retval


def plot_2d(X_list, y, subtitles, title, figsize=(24, 7), labels=None, bbox_shift=1.15, s=15, rows=1, cols=3):
    plt.figure(figsize=figsize)
    for i, X in enumerate(X_list):
        plt.subplot(rows, cols, i + 1)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=s)
        plt.title(subtitles[i])

    handles, classes = scatter.legend_elements()
    if labels is not None:
        classes = labels
    plt.legend(handles, classes, bbox_to_anchor=(bbox_shift, 1.02), loc='upper right')

    plt.suptitle(title)


def write_to_file(images, path):
    f = open(path, "w")
    f.write(f"{images.shape[0]} {images.shape[1]}\n")
    pd.DataFrame(images).to_csv(path, sep=" ", header=False, index=False, mode='a')
