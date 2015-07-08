__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as spsp
from collections import defaultdict

from reveal_graph_embedding.common import get_file_row_generator


def read_adjacency_matrix(file_path, separator, numbering="matlab"):
    """
    Reads an edge list in csv format and returns the adjacency matrix in SciPy Sparse COOrdinate format.

    Inputs:  - file_path: The path where the adjacency matrix is stored.
             - separator: The delimiter among values (e.g. ",", "\t", " ")
             - numbering: Array numbering style: * "matlab"
                                                 * "c"

    Outputs: - adjacency_matrix: The adjacency matrix in SciPy Sparse COOrdinate format.
    """
    # Open file
    file_row_generator = get_file_row_generator(file_path, separator)

    file_row = next(file_row_generator)
    number_of_rows = file_row[1]
    number_of_columns = file_row[3]
    directed = file_row[7]
    if directed == "True":
        directed = True
    elif directed == "False":
        directed = False
    else:
        print("Invalid metadata.")
        raise RuntimeError

    # Initialize lists for row and column sparse matrix arguments
    row = list()
    col = list()
    data = list()
    append_row = row.append
    append_col = col.append
    append_data = data.append

    # Read all file rows
    for file_row in file_row_generator:
        source_node = np.int64(file_row[0])
        target_node = np.int64(file_row[1])
        edge_weight = np.float64(file_row[2])

        # Add edge
        append_row(source_node)
        append_col(target_node)
        append_data(edge_weight)

        # Since this is an undirected network also add the reciprocal edge
        if not directed:
            if source_node != target_node:
                append_row(target_node)
                append_col(source_node)
                append_data(edge_weight)

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.array(data, dtype=np.float64)

    if numbering == "matlab":
        row -= 1
        col -= 1
    elif numbering == "c":
        pass
    else:
        print("Invalid numbering style.")
        raise RuntimeError

    # Form sparse adjacency matrix
    adjacency_matrix = spsp.coo_matrix((data, (row, col)), shape=(number_of_rows, number_of_columns))

    return adjacency_matrix


def read_node_label_matrix(file_path, separator, numbering="matlab"):
    """
    Reads node-label pairs in csv format and returns a list of tuples and a node-label matrix.

    Inputs:  - file_path: The path where the node-label matrix is stored.
             - separator: The delimiter among values (e.g. ",", "\t", " ")
             - number_of_nodes: The number of nodes of the full graph. It is possible that not all nodes are labelled.
             - numbering: Array numbering style: * "matlab"
                                                 * "c"

    Outputs: - node_label_matrix: The node-label associations in a NumPy array of tuples format.
             - number_of_categories: The number of categories/classes the nodes may belong to.
             - labelled_node_indices: A NumPy array containing the labelled node indices.
    """
    # Open file
    file_row_generator = get_file_row_generator(file_path, separator)

    file_row = next(file_row_generator)
    number_of_rows = file_row[1]
    number_of_categories = int(file_row[3])

    # Initialize lists for row and column sparse matrix arguments
    row = list()
    col = list()
    append_row = row.append
    append_col = col.append

    # Populate the arrays
    for file_row in file_row_generator:
        node = np.int64(file_row[0])
        label = np.int64(file_row[1])

        # Add label
        append_row(node)
        append_col(label)

    labelled_node_indices = np.array(list(set(row)))

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)

    if numbering == "matlab":
        row -= 1
        col -= 1
        labelled_node_indices -= 1
    elif numbering == "c":
        pass
    else:
        print("Invalid numbering style.")
        raise RuntimeError

    # Form sparse adjacency matrix
    node_label_matrix = spsp.coo_matrix((data, (row, col)), shape=(number_of_rows, number_of_categories))
    node_label_matrix = node_label_matrix.tocsr()

    return node_label_matrix, number_of_categories, labelled_node_indices


def scipy_sparse_to_csv(filepath, matrix, separator=",", directed=False, numbering="matlab"):
    """
    Writes sparse matrix in separated value format.
    """
    matrix = spsp.coo_matrix(matrix)

    shape = matrix.shape
    nnz = matrix.getnnz()

    if numbering == "matlab":
        row = matrix.row + 1
        col = matrix.col + 1
        data = matrix.data
    elif numbering == "c":
        row = matrix.row
        col = matrix.col
        data = matrix.data
    else:
        print("Invalid numbering style.")
        raise RuntimeError

    with open(filepath, "w") as f:
        # Write metadata.
        file_row = "n_rows:" + separator + str(shape[0]) + separator +\
                   "n_cols:" + separator + str(shape[1]) + separator +\
                   "nnz:" + separator + str(nnz) + separator +\
                   "directed:" + separator + str(directed) +\
                   "\n"
        f.write(file_row)

        for edge in range(row.size):
            if directed is False:
                if col[edge] < row[edge]:
                    continue
            file_row = str(row[edge]) + separator + str(col[edge]) + separator + str(data[edge]) + "\n"
            f.write(file_row)


def write_screen_name_to_topics(filepath, user_label_matrix, node_to_id, id_to_name, label_to_lemma, lemma_to_keyword, separator=","):
    """
    Writes a user name and associated topic names per row.
    """
    user_label_matrix = spsp.coo_matrix(user_label_matrix)

    shape = user_label_matrix.shape
    nnz = user_label_matrix.getnnz()

    row = user_label_matrix.row
    col = user_label_matrix.col
    data = user_label_matrix.data

    name_to_topic_set = defaultdict(set)

    for edge in range(row.size):
        node = row[edge]
        user_twitter_id = node_to_id[node]
        name = id_to_name[user_twitter_id]

        label = col[edge]
        lemma = label_to_lemma[label]
        # topic = lemma_to_keyword[lemma]

        name_to_topic_set[name].add(lemma)

    with open(filepath, "w") as f:
        # Write metadata.
        file_row = "n_rows:" + separator + str(shape[0]) + separator +\
                   "nnz:" + separator + str(nnz) + separator +\
                   "\n"
        f.write(file_row)

        for name, topic_set in name_to_topic_set.items():
            file_row = list()
            file_row.append(name)
            file_row.extend(topic_set)
            file_row = separator.join(file_row) + "\n"
            f.write(file_row)
