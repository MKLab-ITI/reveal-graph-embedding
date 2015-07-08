__author__ = 'Georgios Rizos (georgerizos@iti.gr)'
# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sparse

from reveal_graph_embedding.eps_randomwalk.cython_opt.transition import get_natural_random_walk_matrix
from reveal_graph_embedding.eps_randomwalk.cython_opt.similarity import fast_approximate_cumulative_pagerank_difference
from reveal_graph_embedding.embedding.common import normalize_community_features

FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t

INT64 = np.int64
ctypedef np.int64_t INT64_t


def arcte(adjacency_matrix, double rho, double epsilon):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate natural random walk transition probability matrix
    cdef np.ndarray[FLOAT64_t, ndim=1] out_degree
    cdef np.ndarray[FLOAT64_t, ndim=1] in_degree
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    cdef np.ndarray adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    cdef np.ndarray base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    # Calculate local communities for all nodes
    row_list = list()
    col_list = list()
    extend_row = row_list.extend
    extend_col = col_list.extend

    cdef long number_of_local_communities = 0

    cdef np.ndarray [FLOAT64_t, ndim=1] s = np.zeros(number_of_nodes, dtype=FLOAT64)
    cdef np.ndarray [FLOAT64_t, ndim=1] r = np.zeros(number_of_nodes, dtype=FLOAT64)

    cdef np.ndarray[INT64_t, ndim=1] iterate_nodes = np.where(out_degree != 0)[0]
    cdef long seed_node
    for seed_node in iterate_nodes:
        print(seed_node)
        s[:] = 0.0
        r[:] = 0.0

        # Calculate similarity matrix slice
        nop = fast_approximate_cumulative_pagerank_difference(s,
                                                              r,
                                                              base_transitions,
                                                              adjacent_nodes,
                                                              out_degree,
                                                              in_degree,
                                                              seed_node,
                                                              rho,
                                                              epsilon)

        s_sparse = sparse.csr_matrix(s, shape=(1, number_of_nodes))

        # Perform degree normalization of approximate similarity matrix slice
        relevant_degrees = in_degree[s_sparse.indices]
        s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

        # Sort the degree normalized approximate similarity matrix slice
        sorted_indices = np.argsort(s_sparse.data, axis=0)
        s_sparse.data = s_sparse.data[sorted_indices]
        s_sparse.indices = s_sparse.indices[sorted_indices]

        # Iterate over the support of the distribution to detect local community
        base_community = set(adjacent_nodes[seed_node])
        base_community.add(seed_node)
        base_community_size = len(base_community)

        base_community_count = 0
        most_unlikely_index = 0
        for i in np.arange(1, s_sparse.data.size + 1):
            if s_sparse.indices[-i] in base_community:
                base_community_count += 1
                if base_community_count == base_community_size:
                    most_unlikely_index = i
                    break

        # Save feature matrix coordinates
        if most_unlikely_index > base_community_count:
            new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
            extend_row(new_rows)
            extend_col(number_of_local_communities*np.ones_like(new_rows))
            number_of_local_communities += 1

    # Form local community feature matrix
    row = np.array(row_list, dtype=np.int64)
    col = np.array(col_list, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))

    # Form base community feature matrix
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    base_community_features = identity_matrix + adjacency_matrix

    # Stack horizontally matrices to form feature matrix
    features = sparse.hstack([base_community_features, features]).tocoo()

    features = normalize_community_features(features)

    return features


def arcte_and_centrality(adjacency_matrix, double rho, double epsilon):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
             - centrality in R^(nx1): A vector containing the RCT measure of centrality.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate natural random walk transition probability matrix
    cdef np.ndarray[FLOAT64_t, ndim=1] out_degree
    cdef np.ndarray[FLOAT64_t, ndim=1] in_degree
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    cdef np.ndarray adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    cdef np.ndarray base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    # Calculate local communities for all nodes
    row_list = list()
    col_list = list()
    extend_row = row_list.extend
    extend_col = col_list.extend

    cdef long number_of_local_communities = 0

    cdef np.ndarray [FLOAT64_t, ndim=1] s = np.zeros(number_of_nodes, dtype=FLOAT64)
    cdef np.ndarray [FLOAT64_t, ndim=1] r = np.zeros(number_of_nodes, dtype=FLOAT64)
    cdef np.ndarray centrality = np.zeros(number_of_nodes, dtype=FLOAT64)

    cdef np.ndarray[INT64_t, ndim=1] iterate_nodes = np.where(out_degree != 0)[0]
    cdef long seed_node
    for seed_node in iterate_nodes:
        # print(seed_node)

        for node in range(number_of_nodes):
            s[node] = 0.0
            r[node] = 0.0

        # Calculate similarity matrix slice
        nop = fast_approximate_cumulative_pagerank_difference(s,
                                                              r,
                                                              base_transitions,
                                                              adjacent_nodes,
                                                              out_degree,
                                                              in_degree,
                                                              seed_node,
                                                              rho,
                                                              epsilon)

        s_sparse = sparse.csr_matrix(s, shape=(1, number_of_nodes))

        # Perform degree normalization of approximate similarity matrix slice
        relevant_degrees = in_degree[s_sparse.indices]
        s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

        # Adjust centrality
        centrality += s_sparse

        # Sort the degree normalized approximate similarity matrix slice
        sorted_indices = np.argsort(s_sparse.data, axis=0)
        s_sparse.data = s_sparse.data[sorted_indices]
        s_sparse.indices = s_sparse.indices[sorted_indices]

        # Iterate over the support of the distribution to detect local community
        base_community = set(adjacent_nodes[seed_node])
        base_community.add(seed_node)
        base_community_size = len(base_community)

        base_community_count = 0
        most_unlikely_index = 0
        for i in np.arange(1, s_sparse.data.size + 1):
            if s_sparse.indices[-i] in base_community:
                base_community_count += 1
                if base_community_count == base_community_size:
                    most_unlikely_index = i
                    break

        # Save feature matrix coordinates
        if most_unlikely_index > base_community_count:
            new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
            extend_row(new_rows)
            extend_col(number_of_local_communities*np.ones_like(new_rows))
            number_of_local_communities += 1

        s_sparse = None

    centrality[np.setdiff1d(np.arange(number_of_nodes), iterate_nodes)] = 1.0

    # Form local community feature matrix
    row = np.array(row_list, dtype=np.int64)
    col = np.array(col_list, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))

    # Form base community feature matrix
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    base_community_features = identity_matrix + adjacency_matrix

    # Stack horizontally matrices to form feature matrix
    try:
        features = sparse.hstack([base_community_features, features]).tocoo()
    except ValueError as e:
       features = base_community_features

    features = normalize_community_features(features)

    return features, centrality
