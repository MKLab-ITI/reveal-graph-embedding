__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import time

import numpy as np
from scipy.sparse import issparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.preprocessing import normalize

from reveal_graph_embedding.datautil.snow_datautil import snow_read_data
from reveal_graph_embedding.datautil.asu_datautil import asu_read_data
from reveal_graph_embedding.datautil.insight_datautil import insight_read_data
from reveal_graph_embedding.embedding.arcte.arcte import arcte
from reveal_graph_embedding.embedding.competing_methods import laplacian_eigenmaps, replicator_eigenmaps, louvain,\
    mroc, base_communities
from reveal_graph_embedding.embedding.common import normalize_columns
from reveal_graph_embedding.learning.holdout import generate_folds
from reveal_graph_embedding.embedding.community_weighting import chi2_contingency_matrix,\
    peak_snr_weight_aggregation, community_weighting
from reveal_graph_embedding.learning import evaluation


def run_experiment(dataset_name,
                   dataset_folder,
                   feature_extraction_method_name,
                   percentages,
                   trial_num,
                   thread_num,
                   feature_extraction_parameters,
                   classifier_parameters):
    if dataset_name == "snow2014":
        adjacency_matrix,\
        node_label_matrix,\
        labelled_node_indices,\
        number_of_categories = read_snow2014graph_data(dataset_folder)
    elif dataset_name == "flickr":
        adjacency_matrix,\
        node_label_matrix,\
        labelled_node_indices,\
        number_of_categories = read_asu_data(dataset_folder)
    elif dataset_name == "youtube":
        adjacency_matrix,\
        node_label_matrix,\
        labelled_node_indices,\
        number_of_categories = read_asu_data(dataset_folder)
    elif dataset_name == "politicsuk":
        adjacency_matrix,\
        node_label_matrix,\
        labelled_node_indices,\
        number_of_categories = read_insight_data(dataset_folder)
    else:
        print("Invalid dataset name.")
        raise RuntimeError
    print("Graphs and labels read.")

    feature_matrix,\
    feature_extraction_elapsed_time = feature_extraction(adjacency_matrix,
                                                         feature_extraction_method_name,
                                                         thread_num,
                                                         feature_extraction_parameters)
    print("Feature extraction elapsed time: ", feature_extraction_elapsed_time)
    if feature_extraction_parameters["community_weighting"] is None:
        pass
    elif feature_extraction_parameters["community_weighting"] == "chi2":
        feature_matrix = normalize_columns(feature_matrix)
    elif feature_extraction_parameters["community_weighting"] == "ivf":
        feature_matrix = normalize_columns(feature_matrix)
    else:
        print("Invalid community weighting selection.")
        raise RuntimeError

    C = classifier_parameters["C"]
    fit_intercept = classifier_parameters["fit_intercept"]

    for p in np.arange(percentages.size):
        percentage = percentages[p]

        # Initialize the metric storage arrays to zero
        macro_F1 = np.zeros(trial_num, dtype=np.float)
        micro_F1 = np.zeros(trial_num, dtype=np.float)

        folds = generate_folds(node_label_matrix,
                               labelled_node_indices,
                               number_of_categories,
                               percentage,
                               trial_num)

        for trial in np.arange(trial_num):
            train, test = next(folds)
            ########################################################################################################
            # Separate train and test sets
            ########################################################################################################
            X_train, X_test, y_train, y_test = feature_matrix[train, :],\
                                               feature_matrix[test, :],\
                                               node_label_matrix[train, :],\
                                               node_label_matrix[test, :]

            if issparse(feature_matrix):
                if feature_extraction_parameters["community_weighting"] == "chi2":
                    contingency_matrix = chi2_contingency_matrix(X_train, y_train)
                    community_weights = peak_snr_weight_aggregation(contingency_matrix)

                    X_train, X_test = community_weighting(X_train, X_test, community_weights)
                else:
                    X_train = normalize(X_train, norm="l2")
                    X_test = normalize(X_test, norm="l2")

            ############################################################################################################
            # Train model
            ############################################################################################################
            # Train classifier.
            start_time = time.time()
            model = OneVsRestClassifier(svm.LinearSVC(C=C,
                                                      random_state=None,
                                                      dual=False,
                                                      fit_intercept=fit_intercept),
                                        n_jobs=thread_num)

            model.fit(X_train, y_train)
            hypothesis_training_time = time.time() - start_time
            print('Model fitting time: ', hypothesis_training_time)

            ############################################################################################################
            # Make predictions
            ############################################################################################################
            start_time = time.time()
            y_pred = model.decision_function(X_test)
            prediction_time = time.time() - start_time
            print('Prediction time: ', prediction_time)

            ############################################################################################################
            # Calculate measures
            ############################################################################################################
            y_pred = evaluation.form_node_label_prediction_matrix(y_pred, y_test)

            measures = evaluation.calculate_measures(y_pred, y_test)

            macro_F1[trial] = measures[4]
            micro_F1[trial] = measures[5]

            # print('Trial ', trial+1, ':')
            # print(' Macro-F1:        ', macro_F1[trial])
            # print(' Micro-F1:        ', micro_F1[trial])
            # print('\n')

        ################################################################################################################
        # Experiment results
        ################################################################################################################
        print(percentage)
        print('\n')
        print('Macro F1        average: ', np.mean(macro_F1))
        print('Micro F1        average: ', np.mean(micro_F1))
        print('Macro F1            std: ', np.std(macro_F1))
        print('Micro F1            std: ', np.std(micro_F1))


def read_snow2014graph_data(dataset_folder):
    adjacency_matrix = snow_read_data.read_adjacency_matrix(file_path=dataset_folder + "/men_ret_graph.tsv",
                                                            separator="\t")
    node_label_matrix,\
    labelled_node_indices,\
    number_of_categories = snow_read_data.read_node_label_matrix(file_path=dataset_folder + "/user_label_matrix.tsv",
                                                                 separator="\t")
    return adjacency_matrix,\
           node_label_matrix,\
           labelled_node_indices,\
           number_of_categories


def read_asu_data(dataset_folder):
    adjacency_matrix = asu_read_data.read_adjacency_matrix(file_path=dataset_folder + "/edges.csv",
                                                           separator=",")
    node_label_matrix,\
    labelled_node_indices,\
    number_of_categories = asu_read_data.read_node_label_matrix(file_path=dataset_folder + "/group-edges.csv",
                                                                separator=",",
                                                                number_of_nodes=adjacency_matrix.shape[0])
    return adjacency_matrix,\
           node_label_matrix,\
           labelled_node_indices,\
           number_of_categories


def read_insight_data(dataset_folder):
    adjacency_matrix = insight_read_data.read_adjacency_matrix(file_path=dataset_folder + "/men_ret_graph.tsv",
                                                               separator="\t")
    node_label_matrix,\
    labelled_node_indices,\
    number_of_categories = insight_read_data.read_node_label_matrix(file_path=dataset_folder + "/user_label_matrix.tsv",
                                                                    separator="\t")
    return adjacency_matrix,\
           node_label_matrix,\
           labelled_node_indices,\
           number_of_categories


def feature_extraction(adjacency_matrix,
                       feature_extraction_method_name,
                       thread_num,
                       feature_extraction_parameters):
    start_time = time.time()
    if feature_extraction_method_name == "arcte":
        epsilon = feature_extraction_parameters["epsilon"]
        rho = feature_extraction_parameters["rho"]

        feature_matrix = arcte(adjacency_matrix, rho, epsilon, thread_num)
    elif feature_extraction_method_name == "mroc":
        alpha = feature_extraction_parameters["alpha"]
        feature_matrix = mroc(adjacency_matrix, alpha)
    elif feature_extraction_method_name == "louvain":
        feature_matrix = louvain(adjacency_matrix)
    elif feature_extraction_method_name == "basecomm":
        feature_matrix = base_communities(adjacency_matrix)
    elif feature_extraction_method_name == "lapeig":
        dimensionality = feature_extraction_parameters["dimensionality"]

        feature_matrix = laplacian_eigenmaps(adjacency_matrix, dimensionality)
    elif feature_extraction_method_name == "repeig":
        dimensionality = feature_extraction_parameters["dimensionality"]

        feature_matrix = replicator_eigenmaps(adjacency_matrix, dimensionality)
    else:
        print("Invalid feature extraction name.")
        raise RuntimeError
    elapsed_time = time.time() - start_time

    return feature_matrix, elapsed_time
