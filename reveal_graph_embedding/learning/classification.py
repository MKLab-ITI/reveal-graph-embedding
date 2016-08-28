__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import scipy.sparse as spsp
from scipy.sparse import issparse


def model_fit(X_train, y_train, svm_hardness, fit_intercept, number_of_threads, classifier_type="LinearSVC"):
    """
    Fits a Linear Support Vector Classifier to the labelled graph-based features using the LIBLINEAR library.

    One-vs-All: http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    LinearSVC:  http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    Inputs:  - feature_matrix: The graph based-features in either NumPy or SciPy sparse array format.
             - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - svm_hardness: Penalty of the error term.
             - fit_intercept: Data centering as per scikit-learn.
             - number_of_threads: The number of threads to use for training the multi-label scheme.
             - classifier_type: A string to be chosen among: * LinearSVC (LibLinear)
                                                             * LogisticRegression (LibLinear)
                                                             * RandomForest

    Output:  - model: A trained scikit-learn One-vs-All multi-label scheme of linear SVC models.
    """
    if classifier_type == "LinearSVC":
        if X_train.shape[0] > X_train.shape[1]:
            dual = False
        else:
            dual = True

        model = OneVsRestClassifier(LinearSVC(C=svm_hardness, random_state=0, dual=dual,
                                              fit_intercept=fit_intercept),
                                    n_jobs=number_of_threads)
        model.fit(X_train, y_train)
    elif classifier_type == "LogisticRegression":
        if X_train.shape[0] > X_train.shape[1]:
            dual = False
        else:
            dual = True

        model = OneVsRestClassifier(LogisticRegression(C=svm_hardness, random_state=0, dual=dual,
                                                       fit_intercept=fit_intercept),
                                    n_jobs=number_of_threads)
        model.fit(X_train, y_train)
    elif classifier_type == "RandomForest":
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000, criterion="gini",
                                                           n_jobs=number_of_threads, random_state=0))
        if issparse(X_train):
            model.fit(X_train.tocsc(), y_train.toarray())
        else:
            model.fit(X_train, y_train.toarray())
    else:
        print("Invalid classifier type.")
        raise RuntimeError

    return model


def meta_model_fit(X_train, y_train, svm_hardness, fit_intercept, number_of_threads, regressor_type="LinearSVR"):
    """
    Trains meta-labeler for predicting number of labels for each user.

    Based on: Tang, L., Rajan, S., & Narayanan, V. K. (2009, April).
              Large scale multi-label classification via metalabeler.
              In Proceedings of the 18th international conference on World wide web (pp. 211-220). ACM.
    """
    if regressor_type == "LinearSVR":
        if X_train.shape[0] > X_train.shape[1]:
            dual = False
        else:
            dual = True

        model = LinearSVR(C=svm_hardness, random_state=0, dual=dual,
                          fit_intercept=fit_intercept)
        y_train_meta = y_train.sum(axis=1)
        model.fit(X_train, y_train_meta)
    else:
        print("Invalid regressor type.")
        raise RuntimeError

    return model


def weigh_users(X_test, model, classifier_type="LinearSVC"):
    """
    Uses a trained model and the unlabelled features to produce a user-to-label distance matrix.

    Inputs:  - feature_matrix: The graph based-features in either NumPy or SciPy sparse array format.
             - model: A trained scikit-learn One-vs-All multi-label scheme of linear SVC models.
             - classifier_type: A string to be chosen among: * LinearSVC (LibLinear)
                                                             * LogisticRegression (LibLinear)
                                                             * RandomForest

    Output:  - decision_weights: A NumPy array containing the distance of each user from each label discriminator.
    """
    if classifier_type == "LinearSVC":
        decision_weights = model.decision_function(X_test)
    elif classifier_type == "LogisticRegression":
        decision_weights = model.predict_proba(X_test)
    elif classifier_type == "RandomForest":
        # if issparse(X_test):
        #     decision_weights = np.hstack(a[:, 1].reshape(X_test.shape[0], 1) for a in model.predict_proba(X_test.tocsr()))
        # else:
        #     decision_weights = np.hstack(a[:, 1].reshape(X_test.shape[0], 1) for a in model.predict_proba(X_test))
        if issparse(X_test):
            decision_weights = model.predict_proba(X_test.tocsr())
        else:
            decision_weights = model.predict_proba(X_test)
    else:
        print("Invalid classifier type.")
        raise RuntimeError

    return decision_weights


def classify_users(X_test, model, classifier_type, meta_model, upper_cutoff):
    """
    Uses a trained model and the unlabelled features to associate users with labels.

    The decision is done as per scikit-learn:
    http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.predict

    Inputs:  - feature_matrix: The graph based-features in either NumPy or SciPy sparse array format.
             - model: A trained scikit-learn One-vs-All multi-label scheme of linear SVC models.

    Output:  - decision_weights: A NumPy array containing the distance of each user from each label discriminator.
    """
    if classifier_type == "LinearSVC":
        prediction = model.decision_function(X_test)
        # prediction = penalize_large_classes(prediction)

        meta_prediction = meta_model.predict(X_test)
        meta_prediction = np.rint(meta_prediction)
        meta_prediction[meta_prediction > upper_cutoff] = upper_cutoff

        prediction_indices = np.argsort(prediction, axis=1)

        prediction_row = np.empty(int(np.sum(meta_prediction)), dtype=np.int32)
        prediction_col = np.empty(int(np.sum(meta_prediction)), dtype=np.int32)
        prediction_data = np.empty(int(np.sum(meta_prediction)), dtype=np.float64)

        nnz_counter = 0
        for i in range(X_test.shape[0]):
            jj = prediction_indices[i, -int(meta_prediction[i]):]
            dd = prediction[i, jj]

            prediction_row[nnz_counter:nnz_counter+int(meta_prediction[i])] = i
            prediction_col[nnz_counter:nnz_counter+int(meta_prediction[i])] = jj
            prediction_data[nnz_counter:nnz_counter+int(meta_prediction[i])] = dd

            nnz_counter += int(meta_prediction[i])

        prediction = spsp.coo_matrix((prediction_data,
                                      (prediction_row,
                                       prediction_col)),
                                     shape=prediction.shape)

        prediction = normalize(prediction, norm="l2", axis=0)
    elif classifier_type == "LogisticRegression":
        prediction = model.predict_proba(X_test)
        # prediction = penalize_large_classes(prediction)

        meta_prediction = meta_model.predict(X_test)
        meta_prediction = np.rint(meta_prediction)
        meta_prediction[meta_prediction > upper_cutoff] = upper_cutoff
        meta_prediction[meta_prediction < 1] = 1

        prediction_indices = np.argsort(prediction, axis=1)

        prediction_row = np.empty(int(np.sum(meta_prediction)), dtype=np.int32)
        prediction_col = np.empty(int(np.sum(meta_prediction)), dtype=np.int32)
        prediction_data = np.empty(int(np.sum(meta_prediction)), dtype=np.float64)

        nnz_counter = 0
        for i in range(X_test.shape[0]):
            jj = prediction_indices[i, -int(meta_prediction[i]):]
            dd = prediction[i, jj]

            prediction_row[nnz_counter:nnz_counter+int(meta_prediction[i])] = i
            prediction_col[nnz_counter:nnz_counter+int(meta_prediction[i])] = jj
            prediction_data[nnz_counter:nnz_counter+int(meta_prediction[i])] = dd

            nnz_counter += int(meta_prediction[i])

        prediction = spsp.coo_matrix((prediction_data,
                                      (prediction_row,
                                       prediction_col)),
                                     shape=prediction.shape)
    elif classifier_type == "RandomForest":
        if issparse(X_test):
            prediction = model.predict_proba(X_test.tocsr())
        else:
            prediction = model.predict_proba(X_test)
        # prediction = penalize_large_classes(prediction)

        meta_prediction = meta_model.predict(X_test)
        meta_prediction = np.rint(meta_prediction)
        meta_prediction[meta_prediction > upper_cutoff] = upper_cutoff

        prediction_indices = np.argsort(prediction, axis=1)

        prediction_row = np.empty(int(np.sum(meta_prediction)), dtype=np.int32)
        prediction_col = np.empty(int(np.sum(meta_prediction)), dtype=np.int32)
        prediction_data = np.empty(int(np.sum(meta_prediction)), dtype=np.float64)

        nnz_counter = 0
        for i in range(X_test.shape[0]):
            jj = prediction_indices[i, -int(meta_prediction[i]):]
            dd = prediction[i, jj]

            prediction_row[nnz_counter:nnz_counter+int(meta_prediction[i])] = i
            prediction_col[nnz_counter:nnz_counter+int(meta_prediction[i])] = jj
            prediction_data[nnz_counter:nnz_counter+int(meta_prediction[i])] = dd

            nnz_counter += int(meta_prediction[i])

        prediction = spsp.coo_matrix((prediction_data,
                                      (prediction_row,
                                       prediction_col)),
                                     shape=prediction.shape)
    else:
        print("Invalid classifier type.")
        raise RuntimeError

    return prediction


def penalize_large_classes(prediction):

    for j in range(prediction.shape[1]):
        prediction[:, j] = np.sum(prediction[:, j])

    for i in range(prediction.shape[0]):
        prediction[i, :] = np.sum(prediction[i, :])

    return prediction
