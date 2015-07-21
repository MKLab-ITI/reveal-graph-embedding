__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse

import scipy.sparse as spsp

from reveal_graph_embedding.common import get_threads_number
from reveal_graph_embedding.datautil.datarw import read_adjacency_matrix, write_features
from reveal_graph_embedding.embedding.arcte.arcte import arcte


def main():
    ####################################################################################################################
    # Parse arguments.
    ####################################################################################################################
    parser = argparse.ArgumentParser()

    # File paths.
    parser.add_argument("-i", "--input", dest="input_edge_list_path",
                        help="This is the file path of the graph in edge list format.",
                        type=str, required=True)
    parser.add_argument("-o", "--output", dest="output_feature_path",
                        help="This is the file path of the graph in edge list format.",
                        type=str, required=True)

    # Edge list parsing configuration.
    parser.add_argument("-s", "--separator", dest="separator",
                        help="The character(s) separating the values in the edge list (default is tab: \"\t\").",
                        type=str, required=False, default="\t")
    parser.add_argument("-u", "--undirected", dest="undirected",
                        help="Also create the reciprocal edge for each edge in edge list.",
                        type=bool, required=False, default=False)

    # Algorithm configuration.
    parser.add_argument("-r", "--rho", dest="restart_probability",
                        help="The restart probability for the vertex-centric PageRank calculation.",
                        type=float, required=False, default=0.1)
    parser.add_argument("-e", "--epsilon", dest="epsilon_threshold",
                        help="The tolerance for calculating vertex-centric PageRank values.",
                        type=float, required=False, default=1.0e-05)
    parser.add_argument("-nt", "--tasks", dest="number_of_tasks",
                        help="The number of parallel tasks to create.",
                        type=int, required=False, default=None)

    args = parser.parse_args()

    input_edge_list_path = args.input_edge_list_path
    output_feature_path = args.output_feature_path

    separator = args.separator
    undirected = args.undirected

    restart_probability = args.restart_probability
    epsilon_threshold = args.epsilon_threshold
    number_of_tasks = args.number_of_tasks

    if number_of_tasks is None:
        number_of_tasks = get_threads_number()

    ####################################################################################################################
    # Perform algorithm.
    ####################################################################################################################
    # Read the adjacency matrix.
    adjacency_matrix,\
    node_to_id = read_adjacency_matrix(file_path=input_edge_list_path,
                                       separator=separator,
                                       undirected=undirected)

    # Make sure we are dealing with a symmetric adjacency matrix.
    adjacency_matrix = spsp.csr_matrix(adjacency_matrix)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose())/2

    # Perform ARCTE.
    features = arcte(adjacency_matrix=adjacency_matrix,
                     rho=restart_probability,
                     epsilon=epsilon_threshold,
                     number_of_threads=number_of_tasks)
    features = spsp.csr_matrix(features)

    # Write features to output file.
    write_features(file_path=output_feature_path,
                   features=features,
                   separator=separator,
                   node_to_id=node_to_id)
