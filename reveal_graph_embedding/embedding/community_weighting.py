__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer


def chi2_contingency_matrix(X_train, y_train):
    X = X_train.copy()
    X.data = np.ones_like(X.data)
    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y_train)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features

    feature_count = check_array(X.sum(axis=0))
    class_prob = check_array(Y.mean(axis=0))
    expected = np.dot(class_prob.T, feature_count)

    observed = np.asarray(observed, dtype=np.float64)

    k = len(observed)
    # Reuse f_obs for chi-squared statistics
    contingency_matrix = observed
    contingency_matrix -= expected
    contingency_matrix **= 2
    contingency_matrix /= expected  # TODO: Invalid value encountered in true divide.
    # weights = contingency_matrix.max(axis=0)

    return contingency_matrix


def peak_snr_weight_aggregation(contingency_matrix):
    contingency_matrix[np.isnan(contingency_matrix)] = 0.0

    variance = np.zeros(contingency_matrix.shape[0], dtype=np.float64)
    for k in range(contingency_matrix.shape[0]):
        within_class_signal = contingency_matrix[k, :]
        variance[k] = np.var(within_class_signal)
    variance = np.sqrt(np.mean(variance))

    weights = np.zeros(contingency_matrix.shape[1], dtype=np.float64)
    for f in range(contingency_matrix.shape[1]):
        distribution = contingency_matrix[:, f]
        distribution = distribution[distribution > 0.0]
        if distribution.size > 1:
            signal_to_noise_ratio = (np.max(distribution) - np.min(distribution))/variance
        elif distribution.size == 1:
            signal_to_noise_ratio = np.max(distribution)/variance
        else:
            signal_to_noise_ratio = 0.0
        weights[f] = signal_to_noise_ratio

    # weights = np.zeros(chisq.shape[1], dtype=np.float64)
    # for f in range(chisq.shape[1]):
    #     distribution = chisq[:, f]
    #     distribution = distribution[distribution > 0.0]
    #     if distribution.size > 1:
    #         max_el = np.sum(distribution)
    #         distribution = distribution/np.sum(chisq[:, f])
    #         signal_to_noise_ratio = max_el*((- np.log(1/chisq.shape[0]))/(- np.sum(np.multiply(distribution, np.log(distribution)))))
    #     elif distribution.size == 1:
    #         signal_to_noise_ratio = np.max(distribution)
    #     else:
    #         signal_to_noise_ratio = 0.0
    #     # print(signal_to_noise_ratio)
    #     weights[f] = signal_to_noise_ratio

    return weights


def community_weighting(X_train, X_test, community_weights):
    if issparse(X_train):
        # chi2score = chi2_contingency_matrix(X_train, y_train)
        # chi2score[np.isnan(chi2score)] = 0.0

        X_train = X_train.tocsc()
        X_test = X_test.tocsc()
        for j in range(X_train.shape[1]):
            document_frequency = X_train.getcol(j).data.size
            if document_frequency > 1:
                if community_weights[j] == 0.0:
                    reinforcement = 0.0
                else:
                    reinforcement = np.log(1.0 + community_weights[j])

                X_train.data[X_train.indptr[j]: X_train.indptr[j + 1]] =\
                    X_train.data[X_train.indptr[j]: X_train.indptr[j + 1]]*reinforcement

            document_frequency = X_test.getcol(j).data.size
            if document_frequency > 1:
                if community_weights[j] == 0.0:
                    reinforcement = 0.0
                else:
                    reinforcement = np.log(1.0 + community_weights[j])
                X_test.data[X_test.indptr[j]: X_test.indptr[j + 1]] =\
                    X_test.data[X_test.indptr[j]: X_test.indptr[j + 1]]*reinforcement

        X_train = X_train.tocsr()
        X_test = X_test.tocsr()

        X_train.eliminate_zeros()
        X_test.eliminate_zeros()

        X_train = normalize(X_train, norm="l2")
        X_test = normalize(X_test, norm="l2")
    else:
        pass

    return X_train, X_test


def chi2_psnr_community_weighting(X_train, X_test, y_train):
    if issparse(X_train):
        contingency_matrix = chi2_contingency_matrix(X_train, y_train)
        community_weights = peak_snr_weight_aggregation(contingency_matrix)
        X_train, X_test = community_weighting(X_train, X_test, community_weights)
    else:
        pass

    return X_train, X_test
