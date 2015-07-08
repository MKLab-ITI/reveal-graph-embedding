__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def conductance(adjacency_matrix, node_array):
    number_of_nodes = adjacency_matrix.shape[0]

    node_array_bar = np.setdiff1d(np.arange(number_of_nodes), node_array)

    submatrix = adjacency_matrix[np.ix_(node_array, node_array)]
    submatrix_bar = adjacency_matrix[np.ix_(node_array_bar, node_array_bar)]

    submatrix_volume = submatrix.getnnz()  # TODO: If empty?
    submatrix_bar_volume = submatrix_bar.getnnz()  # TODO: If empty?
    matrix_volume = adjacency_matrix.getnnz()
    cut_volume = (matrix_volume - submatrix_volume - submatrix_bar_volume)/2

    try:
        cut_conductance = cut_volume/min(submatrix_volume, submatrix_bar_volume)
    except ZeroDivisionError:
        cut_conductance = np.Inf

    return cut_conductance


def conductance_and_clustering_coefficient(adjacency_matrix, node_array, seed_node):
    number_of_nodes = adjacency_matrix.shape[0]

    node_array_bar = np.setdiff1d(np.arange(number_of_nodes), node_array)

    submatrix = adjacency_matrix[np.ix_(node_array, node_array)]
    submatrix_bar = adjacency_matrix[np.ix_(node_array_bar, node_array_bar)]

    submatrix_volume = submatrix.getnnz()  # TODO: If empty?
    submatrix_bar_volume = submatrix_bar.getnnz()  # TODO: If empty?
    matrix_volume = adjacency_matrix.getnnz()
    cut_volume = (matrix_volume - submatrix_volume - submatrix_bar_volume)/2

    cut_conductance = cut_volume/min(submatrix_volume, submatrix_bar_volume)

    new_node_array = np.setdiff1d(node_array, seed_node)
    clustering_coefficient = adjacency_matrix[np.ix_(new_node_array, new_node_array)]
    clustering_coefficient = clustering_coefficient.getnnz()/(new_node_array.size*new_node_array.size)

    return cut_conductance, clustering_coefficient


def fast_conductance(array_of_arrays, node_array, matrix_volume):
    submatrix_volume = 0
    cut_volume = 0

    for node in node_array:
        neighbors = array_of_arrays[node]
        degree = neighbors.size

        common = np.intersect1d(node_array, neighbors).size
        submatrix_volume += common
        cut_volume += degree - common

    submatrix_bar_volume = matrix_volume - submatrix_volume - 2*cut_volume

    try:
        cut_conductance = cut_volume/min(submatrix_volume, submatrix_bar_volume)
    except ZeroDivisionError:
        cut_conductance = np.Inf

    return cut_conductance, cut_volume, submatrix_volume


def incremental_conductance(array_of_arrays, node_array, new_node, cut_volume, submatrix_volume, matrix_volume):
    # TODO: What if I have ones in the diagonal?
    neighbors = array_of_arrays[new_node]
    degree = neighbors.size

    common = np.intersect1d(node_array, neighbors).size
    submatrix_volume += common
    cut_volume += degree - common

    submatrix_bar_volume = matrix_volume - submatrix_volume - 2*cut_volume

    try:
        cut_conductance = cut_volume/min(submatrix_volume, submatrix_bar_volume)
    except ZeroDivisionError:
        cut_conductance = np.Inf

    return cut_conductance, cut_volume, submatrix_volume


def decremental_conductance(array_of_arrays, node_array, new_node, cut_volume, submatrix_volume, matrix_volume):
    # TODO: What if I have ones in the diagonal?
    neighbors = array_of_arrays[new_node]
    degree = neighbors.size

    common = np.intersect1d(node_array, neighbors).size
    submatrix_volume -= common
    cut_volume += common - degree

    submatrix_bar_volume = matrix_volume - submatrix_volume - 2*cut_volume

    try:
        cut_conductance = cut_volume/min(submatrix_volume, submatrix_bar_volume)
    except ZeroDivisionError:
        cut_conductance = np.Inf

    return cut_conductance, cut_volume, submatrix_volume
