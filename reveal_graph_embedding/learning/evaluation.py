__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as sparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score


def form_node_label_prediction_matrix(y_pred, y_test):
    """
    Given the discriminator distances, this function forms the node-label prediction matrix.

    It is assumed that the number of true labels is known.

    Inputs:  - y_pred: A NumPy array that contains the distance from the discriminator for each label for each user.
             - y_test: The node-label ground truth for the test set in a SciPy sparse CSR matrix format.

    Outputs: - y_pred: The node-label prediction for the test set in a SciPy sparse CSR matrix format.
    """
    number_of_test_nodes = y_pred.shape[0]

    # We calculate the number of true labels for each node.
    true_number_of_labels = np.squeeze(y_test.sum(axis=1))

    # We sort the prediction array for each node.
    index = np.argsort(y_pred, axis=1)

    row = np.empty(y_test.getnnz(), dtype=np.int64)
    col = np.empty(y_test.getnnz(), dtype=np.int64)
    start = 0
    for n in np.arange(number_of_test_nodes):
        end = start + true_number_of_labels[0, n]
        row[start:end] = n
        col[start:end] = index[n, -1:-true_number_of_labels[0, n]-1:-1]
        start = end
    data = np.ones_like(row, dtype=np.int8)

    y_pred = sparse.coo_matrix((data, (row, col)), shape=y_test.shape)

    return y_pred


def calculate_measures(y_pred, y_test):
    """
    Calculates the F-scores and F-score averages given a classification result and a ground truth.

    Inputs:  - y_pred: The node-label prediction for the test set in a SciPy sparse CSR matrix format.
             - y_test: The node-label ground truth for the test set in a SciPy sparse CSR matrix format.

    Outputs: - measures: A number of NumPy arrays containing evaluation scores for the experiment.
    """
    y_pred = y_pred.toarray()
    y_test = y_test.toarray()

    macro_precision, macro_recall, macro_F1, macro_support = precision_recall_fscore_support(y_test,
                                                                                             y_pred,
                                                                                             beta=1.0,
                                                                                             average="macro")

    micro_precision, micro_recall, micro_F1, micro_support = precision_recall_fscore_support(y_test,
                                                                                             y_pred,
                                                                                             beta=1.0,
                                                                                             average="micro")

    F1 = f1_score(y_test,
                  y_pred,
                  average=None)

    measures = [macro_recall, micro_recall, macro_precision, micro_precision, macro_F1, micro_F1, F1]
    return measures
