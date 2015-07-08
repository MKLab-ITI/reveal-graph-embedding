__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
from scipy.sparse import issparse

from reveal_graph_embedding.common import load_pickle, store_pickle


def read_features(method_name, path):

    sparse_feature_method_set = set()
    sparse_feature_method_set.update(["lapple",
                                      "apple",
                                      "arcte",
                                      "basecomm",
                                      "mroc",
                                      "edgecluster",
                                      "louvain",
                                      "oslom",
                                      "bigclam"])

    continuous_feature_method_set = set()
    continuous_feature_method_set.update(["lapeig",
                                          "deepwalk",
                                          "rwmodmax",
                                          "repeig"])

    method_name_lowercase = method_name.lower()
    if method_name_lowercase in sparse_feature_method_set:
        features = read_sparse_features(path)
    elif method_name_lowercase in continuous_feature_method_set:
        features = read_continuous_features(path)
    else:
        print("Invalid method name.")
        raise RuntimeError

    return features


def read_continuous_features(path):

    features = np.load(path + ".npy")

    return features


def read_sparse_features(path):

    features = load_pickle(path + ".pkl")
    features = features.tocsr()

    return features


def write_features(path, data):

    if issparse(data):
        write_sparse_data(path, data)
    else:
        write_continuous_features(path, data)


def write_continuous_features(path, data):

    np.save(path, data)


def write_sparse_data(path, data):

    store_pickle(path + ".pkl", data)
