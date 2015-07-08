__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as spsp

from reveal_graph_embedding.embedding.implicit import get_adjacency_matrix_via_combinatorial_laplacian,\
    get_adjacency_matrix_via_directed_laplacian


def get_unnormalized_laplacian(adjacency_matrix):
    # Calculate diagonal matrix of node degrees.
    degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
    degree = degree.tocsr()

    # Calculate sparse graph Laplacian.
    laplacian = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

    return laplacian


def get_normalized_laplacian(adjacency_matrix):
    # Calculate diagonal matrix of node degrees.
    degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
    degree = degree.tocsr()

    # Calculate sparse graph Laplacian.
    adjacency_matrix = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

    # Calculate inverse square root of diagonal matrix of node degrees.
    degree.data = np.real(1/np.sqrt(degree.data))

    # Calculate sparse normalized graph Laplacian.
    normalized_laplacian = degree*adjacency_matrix*degree

    return normalized_laplacian


def get_random_walk_laplacian(adjacency_matrix):
    # Calculate diagonal matrix of node degrees.
    degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
    degree = degree.tocsr()

    # Calculate sparse graph Laplacian.
    adjacency_matrix = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

    # Calculate inverse of diagonal matrix of node degrees.
    degree.data = np.real(1/degree.data)

    # Calculate sparse normalized graph Laplacian.
    random_walk_laplacian = degree*adjacency_matrix

    return random_walk_laplacian


def get_directed_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    effective_adjacency_matrix, rw_distribution = get_adjacency_matrix_via_directed_laplacian(adjacency_matrix, rho)

    I = spsp.spdiags(rw_distribution, [0], number_of_nodes, number_of_nodes)
    theta_matrix = I - effective_adjacency_matrix

    return theta_matrix


def get_combinatorial_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    effective_adjacency_matrix, rw_distribution = get_adjacency_matrix_via_combinatorial_laplacian(adjacency_matrix, rho)

    I = spsp.spdiags(rw_distribution, [0], number_of_nodes, number_of_nodes)
    theta_matrix = I - effective_adjacency_matrix

    return theta_matrix
