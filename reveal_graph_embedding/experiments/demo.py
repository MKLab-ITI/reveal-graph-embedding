__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_graph_embedding.common import get_threads_number
from reveal_graph_embedding.experiments.utility import run_experiment

########################################################################################################################
# Configure experiments by setting values for the capital letter variables and also for the parameters.
########################################################################################################################
DATASET_NAME = "youtube"  # snow2014, flickr, youtube, politicsuk
DATASET_FOLDER = "/path/to/dataset/folder"

FEATURE_EXTRACTION_METHOD_NAME = "arcte"  # acte, lapeig, repeig, louvain, mroc, basecomm


def get_feature_extraction_parameters(feature_extraction_method_name):
    feature_extraction_parameters = dict()
    if feature_extraction_method_name == "arcte":
        feature_extraction_parameters["epsilon"] = 0.00001
        feature_extraction_parameters["rho"] = 0.1
        feature_extraction_parameters["community_weighting"] = "chi2"  # chi2, ivf, None
    elif feature_extraction_method_name == "mroc":
        feature_extraction_parameters["community_weighting"] = "chi2"  # chi2, ivf, None
        feature_extraction_parameters["alpha"] = 1000
    elif feature_extraction_method_name == "louvain":
        feature_extraction_parameters["community_weighting"] = "chi2"  # chi2, ivf, None
    elif feature_extraction_method_name == "basecomm":
        feature_extraction_parameters["community_weighting"] = "chi2"  # chi2, ivf, None
    elif feature_extraction_method_name == "lapeig":
        feature_extraction_parameters["dimensionality"] = 50
    elif feature_extraction_method_name == "repeig":
        feature_extraction_parameters["dimensionality"] = 50
    else:
        print("Invalid method name.")
        raise RuntimeError

    return feature_extraction_parameters


def get_classifier_parameters(feature_extraction_method_name):
    classifier_parameters = dict()
    if feature_extraction_method_name == "arcte":
        classifier_parameters["C"] = 1.0
        classifier_parameters["fit_intercept"] = True
    elif feature_extraction_method_name == "mroc":
        classifier_parameters["C"] = 1.0
        classifier_parameters["fit_intercept"] = True
    elif feature_extraction_method_name == "louvain":
        classifier_parameters["C"] = 1.0
        classifier_parameters["fit_intercept"] = True
    elif feature_extraction_method_name == "basecomm":
        classifier_parameters["C"] = 1.0
        classifier_parameters["fit_intercept"] = True
    elif feature_extraction_method_name == "lapeig":
        classifier_parameters["C"] = 50.0
        classifier_parameters["fit_intercept"] = False
    elif feature_extraction_method_name == "repeig":
        classifier_parameters["C"] = 50.0
        classifier_parameters["fit_intercept"] = False
    else:
        print("Invalid method name.")
        raise RuntimeError

    return classifier_parameters

PERCENTAGES = np.arange(1, 11)  # [1, 10]
TRIAL_NUM = 10
THREAD_NUM = get_threads_number()

########################################################################################################################
# Experiment execution.
########################################################################################################################

feature_extraction_parameters = get_feature_extraction_parameters(FEATURE_EXTRACTION_METHOD_NAME)
classifier_parameters = get_classifier_parameters(FEATURE_EXTRACTION_METHOD_NAME)

run_experiment(DATASET_NAME,
               DATASET_FOLDER,
               FEATURE_EXTRACTION_METHOD_NAME,
               PERCENTAGES,
               TRIAL_NUM,
               THREAD_NUM,
               feature_extraction_parameters,
               classifier_parameters)
