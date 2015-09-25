__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
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
        model = OneVsRestClassifier(LinearSVC(C=svm_hardness, random_state=0, dual=False,
                                              fit_intercept=fit_intercept),
                                    n_jobs=number_of_threads)
        model.fit(X_train, y_train)
    elif classifier_type == "LogisticRegression":
        model = OneVsRestClassifier(LogisticRegression(C=svm_hardness, random_state=0, dual=False,
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
        decision_weights = model.decision_function(X_test)
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


def classify_users(X_test, model, classifier_type="LinearSVC"):
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

        prediction[prediction <= 0.0] = 0.0
        prediction = spsp.coo_matrix(prediction)

        prediction = normalize(prediction, norm="l2", axis=0)
    elif classifier_type == "LogisticRegression":
        prediction = model.predict_proba(X_test)

        prediction[prediction <= 0.1] = 0.0
        prediction = spsp.coo_matrix(prediction)
    elif classifier_type == "RandomForest":
        if issparse(X_test):
            prediction = model.predict_proba(X_test.tocsr())
        else:
            prediction = model.predict_proba(X_test)

        prediction[prediction <= 0.1] = 0.0
        prediction = spsp.coo_matrix(prediction)
    else:
        print("Invalid classifier type.")
        raise RuntimeError

    return prediction
