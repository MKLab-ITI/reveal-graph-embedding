__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as sparse

from reveal_graph_embedding.common import get_file_row_generator


def read_adjacency_matrix(file_path, separator):
    """
    Reads an edge list in csv format and returns the adjacency matrix in SciPy Sparse COOrdinate format.

    Inputs:  - file_path: The path where the adjacency matrix is stored.
             - separator: The delimiter among values (e.g. ",", "\t", " ")

    Outputs: - adjacency_matrix: The adjacency matrix in SciPy Sparse COOrdinate format.
    """
    # Open file
    file_row_generator = get_file_row_generator(file_path, separator)

    # Initialize lists for row and column sparse matrix arguments
    row = list()
    col = list()
    append_row = row.append
    append_col = col.append

    # Read all file rows
    for file_row in file_row_generator:
        source_node = np.int64(file_row[0])
        target_node = np.int64(file_row[1])

        # Add edge
        append_row(source_node)
        append_col(target_node)

        # Since this is an undirected network also add the reciprocal edge
        append_row(target_node)
        append_col(source_node)

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)

    number_of_nodes = np.max(row)  # I assume that there are no missing nodes at the end.

    # Array count should start from 0.
    row -= 1
    col -= 1

    # Form sparse adjacency matrix
    adjacency_matrix = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_nodes))

    return adjacency_matrix


def read_node_label_matrix(file_path, separator, number_of_nodes):
    """
    Reads node-label pairs in csv format and returns a list of tuples and a node-label matrix.

    Inputs:  - file_path: The path where the node-label matrix is stored.
             - separator: The delimiter among values (e.g. ",", "\t", " ")
             - number_of_nodes: The number of nodes of the full graph. It is possible that not all nodes are labelled.

    Outputs: - node_label_matrix: The node-label associations in a NumPy array of tuples format.
             - number_of_categories: The number of categories/classes the nodes may belong to.
             - labelled_node_indices: A NumPy array containing the labelled node indices.
    """
    # Open file
    file_row_generator = get_file_row_generator(file_path, separator)

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

    number_of_categories = len(set(col))  # I assume that there are no missing labels. There may be missing nodes.
    labelled_node_indices = np.array(list(set(row)))

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)

    # Array count should start from 0.
    row -= 1
    col -= 1
    labelled_node_indices -= 1

    # Form sparse adjacency matrix
    node_label_matrix = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_categories))
    node_label_matrix = node_label_matrix.tocsr()

    return node_label_matrix, number_of_categories, labelled_node_indices

