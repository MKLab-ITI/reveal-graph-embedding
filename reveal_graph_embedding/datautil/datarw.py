__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import scipy.sparse as spsp


def get_file_row_generator(file_path, separator, encoding=None):
    """
    Reads an separated value file row by row.

    Inputs: - file_path: The path of the separated value format file.
            - separator: The delimiter among values (e.g. ",", "\t", " ")
            - encoding: The encoding used in the stored text.

    Yields: - words: A list of strings corresponding to each of the file's rows.
    """
    with open(file_path, encoding=encoding) as file_object:
        for line in file_object:
            words = line.strip().split(separator)
            yield words


def store_pickle(file_path, data):
    """
    Pickle some data to a given path.

    Inputs: - file_path: Target file path.
            - data: The python object to be serialized via pickle.
    """
    pkl_file = open(file_path, 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()


def load_pickle(file_path):
    """
    Unpickle some data from a given path.

    Input:  - file_path: Target file path.

    Output: - data: The python object that was serialized and stored in disk.
    """
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def read_adjacency_matrix(file_path, separator, undirected):
    """
    Reads an edge list in csv format and returns the adjacency matrix in SciPy Sparse COOrdinate format.

    Inputs:  - file_path: The path where the adjacency matrix is stored.
             - separator: The delimiter among values (e.g. ",", "\t", " ")
             - undirected: If True, create the reciprocal edge for each edge in edge list.

    Outputs: - adjacency_matrix: The adjacency matrix in SciPy Sparse COOrdinate format.
             - node_to_id: A dictionary that maps anonymized node ids to the original node ids.
    """
    # Open file
    file_row_generator = get_file_row_generator(file_path, separator)

    # Initialize lists for row and column sparse matrix arguments.
    row = list()
    col = list()
    data = list()
    append_row = row.append
    append_col = col.append
    append_data = data.append

    # Initialize node anonymizer.
    id_to_node = dict()

    # Read all file rows.
    for file_row in file_row_generator:
        if file_row[0][0] == "#":
            continue

        source_node_id = int(file_row[0])
        target_node_id = int(file_row[1])

        number_of_nodes = len(id_to_node)
        source_node = id_to_node.setdefault(source_node_id,
                                            number_of_nodes)
        number_of_nodes = len(id_to_node)
        target_node = id_to_node.setdefault(target_node_id,
                                            number_of_nodes)

        edge_weight = float(file_row[2])

        # Add edge.
        append_row(source_node)
        append_col(target_node)
        append_data(edge_weight)

        # Since this is an undirected network also add the reciprocal edge.
        if undirected:
            if source_node != target_node:
                append_row(target_node)
                append_col(source_node)
                append_data(edge_weight)

    number_of_nodes = len(id_to_node)
    node_to_id = dict(zip(list(id_to_node.values()),
                          list(id_to_node.keys())))

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.array(data, dtype=np.float64)

    # Form sparse adjacency matrix.
    adjacency_matrix = spsp.coo_matrix((data, (row, col)), shape=(number_of_nodes,
                                                                  number_of_nodes))

    return adjacency_matrix, node_to_id


def write_features(file_path,
                   features,
                   separator,
                   node_to_id):
    features = spsp.coo_matrix(features)

    row = features.row
    col = features.col
    data = features.data

    with open(file_path, "w") as f:
        for element in range(row.size):
            node = row[element]
            node_id = node_to_id[node]

            community_id = col[element]

            value = int(data[element])

            file_row = str(node_id) + separator + str(community_id) + separator + str(value) + "\n"
            f.write(file_row)
