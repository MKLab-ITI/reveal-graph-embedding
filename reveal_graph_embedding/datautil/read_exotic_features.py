__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as spsp

from reveal_user_annotation.common.datarw import get_file_row_generator


def read_oslom_features(oslom_folder, number_of_nodes):

    oslom_path = oslom_folder + "/tp"

    number_of_levels = 0

    while True:
        try:
            with open(oslom_path + str(number_of_levels + 1)):
                number_of_levels += 1
        except EnvironmentError:
            number_of_levels += 1
            print('OSLOM hierarchy level: ', number_of_levels)
            break

    number_of_communities = 0
    with open(oslom_path) as fp:
        lines = fp.readlines()
    for line in lines:
        if line[0] == '#':
            continue
        else:
            number_of_communities += 1

    for i in np.arange(1, number_of_levels):
        with open(oslom_path + str(i)) as fp:
            lines = fp.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            else:
                number_of_communities += 1

    features = spsp.dok_matrix((number_of_nodes, number_of_communities), dtype=np.float64)

    j = 0
    with open(oslom_path) as fp:
        lines = fp.readlines()
    for line in lines:
        if line[0] == '#':
            continue
        else:
            words = line.strip().split(' ')
            for word in words:
                features[int(word) - 1, j] = 1
            j += 1

    for i in np.arange(1, number_of_levels):
        with open(oslom_path + str(i)) as fp:
            lines = fp.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            else:
                words = line.strip().split(' ')
                for word in words:
                    features[int(word) - 1, j] = 1
                j += 1

    features = features.tocoo()

    return features


def read_bigclam_features(path, number_of_nodes):
    with open(path, "r") as fp:
        lines_crisp = fp.readlines()
    number_of_communities = len(lines_crisp)

    # Calculate number of nonzero elements
    nnz = 0
    for c in np.arange(number_of_communities):
        words = lines_crisp[c].strip().split('\t')
        nnz += len(words)

    # Form feature matrices
    data_crisp = np.zeros(nnz, dtype=np.float64)
    row_crisp = np.zeros(nnz, dtype=np.int32)
    col_crisp = np.zeros(nnz, dtype=np.int32)

    nnz = 0
    for c in np.arange(number_of_communities):
        words_crisp = lines_crisp[c].strip().split('\t')

        for i in np.arange(len(words_crisp)):
            data_crisp[nnz] = 1.0
            row_crisp[nnz] = int(words_crisp[i])
            col_crisp[nnz] = c

            nnz += 1

    features = spsp.coo_matrix((data_crisp, (row_crisp, col_crisp)),
                               shape=(number_of_nodes, number_of_communities))

    features = features.tocoo()

    return features


def read_matlab_features(array_paths, number_of_nodes, dimensionality):
    """
    Returns a sparse feature matrix as calculated by a Matlab routine.
    """
    # Read the data array
    file_row_gen = get_file_row_generator(array_paths[0], "\t")
    data = list()
    append_data = data.append
    for file_row in file_row_gen:
        append_data(float(file_row[0]))

    # Read the row array
    file_row_gen = get_file_row_generator(array_paths[1], "\t")
    row = list()
    append_row = row.append
    for file_row in file_row_gen:
        append_row(int(float(file_row[0])))

    # Read the data array
    file_row_gen = get_file_row_generator(array_paths[2], "\t")
    col = list()
    append_col = col.append
    for file_row in file_row_gen:
        append_col(int(float(file_row[0])))

    data = np.array(data).astype(np.float64)
    row = np.array(row).astype(np.int64) - 1  # Due to Matlab numbering
    col = np.array(col).astype(np.int64) - 1  # Due to Matlab numbering

    print(np.max(row), np.min(row))
    print(np.max(col), np.min(col))

    # centroids_new = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes + 1, k))
    features = spsp.coo_matrix((data, (row, col)), shape=(number_of_nodes, dimensionality))

    return features


def read_deepwalk_features(deepwalk_folder, number_of_nodes=None):
    file_row_gen = get_file_row_generator(deepwalk_folder + "/deepwalk.txt", " ")

    first_row = next(file_row_gen)

    if number_of_nodes is not None:
        features = np.zeros((number_of_nodes, int(first_row[1])), dtype=np.float64)
    else:
        features = np.zeros((int(first_row[0]), int(first_row[1])), dtype=np.float64)

    for file_row in file_row_gen:
        node = int(file_row[0]) - 1
        features[node, :] = np.array([np.float64(coordinate) for coordinate in file_row[1:]])

    return features


def read_dense_separated_value_file(file_path, number_of_nodes, separator=","):

    file_row_gen = get_file_row_generator(file_path=file_path, separator=separator)

    first_file_row = next(file_row_gen)
    number_of_dimensions = len(first_file_row)

    features = np.empty((number_of_nodes, number_of_dimensions), dtype=np.float64)

    file_row_counter = 0
    features[file_row_counter, :] = np.array(first_file_row)

    for file_row in file_row_gen:
        file_row_counter += 1
        features[file_row_counter, :] = np.array(file_row)

    return features
