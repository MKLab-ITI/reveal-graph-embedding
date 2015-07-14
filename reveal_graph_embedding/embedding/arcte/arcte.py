__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import multiprocessing as mp
import itertools
import numpy as np
import scipy.sparse as sparse

from reveal_graph_embedding.common import get_threads_number
from reveal_graph_embedding.eps_randomwalk.transition import get_natural_random_walk_matrix
from reveal_graph_embedding.eps_randomwalk.similarity import fast_approximate_cumulative_pagerank_difference,\
    fast_approximate_personalized_pagerank, lazy_approximate_personalized_pagerank


def parallel_chunks(l, n):
    for thread_id in range(n):
        yield roundrobin_chunks(l, n, thread_id)


def roundrobin_chunks(l, n, id):
    l_c = iter(l)
    x = list(itertools.islice(l_c, id, None, n))
    if len(x):
        return x


def calculate_epsilon_effective(rho, epsilon, seed_degree, neighbor_degrees, mean_degree):
    """
    Semi-automatic effective epsilon threshold calculation.
    """
    # Calculate a weighted neighborhood degree average.
    # neighborhood_degree = rho*seed_degree + (1-rho)*neighbor_degrees.mean()
    neighborhood_degree = neighbor_degrees.mean()

    # Calculate the seed neighborhood normalized effective epsilon.
    epsilon_effective = (epsilon*np.log(1 + seed_degree))/np.log(1 + neighborhood_degree)

    # Calculate the maximum epsilon for at least one push on a neighboring node.
    # Also the minimum epsilon for a push on all the neighboring nodes.
    epsilon_effective_maximum = np.max(1/(seed_degree*neighbor_degrees))
    epsilon_effective_minimum = np.min(1/(seed_degree*neighbor_degrees))

    # print(epsilon_effective, epsilon_effective_maximum, epsilon_effective_minimum)

    # The maximum epsilon is absolute, whereas we regularize for the minimum.
    if epsilon_effective > epsilon_effective_maximum:
        epsilon_effective = epsilon_effective_maximum
    elif epsilon_effective < epsilon_effective_minimum:
        epsilon_effective = (epsilon_effective_minimum + epsilon_effective)/2

    return epsilon_effective


def arcte_with_lazy_pagerank_worker(iterate_nodes,
                                    indices_c,
                                    indptr_c,
                                    data_c,
                                    out_degree,
                                    in_degree,
                                    rho,
                                    epsilon):

    iterate_nodes = np.array(iterate_nodes, dtype=np.int64)
    number_of_nodes = out_degree.size

    mean_degree = np.mean(out_degree)

    rw_transition = sparse.csr_matrix((data_c, indices_c, indptr_c))

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in range(number_of_nodes):
        adjacent_nodes[n] = rw_transition.indices[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]
        base_transitions[n] = rw_transition.data[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]

    # Separate nodes to iterate in smaller chunks in order to compress the feature batches.
    features = sparse.dok_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)
    features = sparse.csr_matrix(features)

    if iterate_nodes.size > 2000:
        node_chunks = list(parallel_chunks(iterate_nodes, iterate_nodes.size//2000))
    else:
        node_chunks = list()
        node_chunks.append(iterate_nodes)

    for node_chunk in node_chunks:
        node_chunk = np.array(node_chunk, dtype=np.int64)

        # Calculate local communities for all nodes.
        row_list = list()
        col_list = list()
        extend_row = row_list.extend
        extend_col = col_list.extend

        # number_of_local_communities = 0

        s = np.zeros(number_of_nodes, dtype=np.float64)  #TODO: What if it is only one?
        r = np.zeros(number_of_nodes, dtype=np.float64)
        for n_index in range(node_chunk.size):
            print(n_index)
            n = node_chunk[n_index]

            # Calculate similarity matrix slice.
            s[:] = 0.0
            r[:] = 0.0

            epsilon_eff = calculate_epsilon_effective(rho, epsilon, out_degree[n], out_degree[adjacent_nodes[n]], mean_degree)

            lazy_rho = (rho*(0.5))/(1-(0.5*rho))

            nop = lazy_approximate_personalized_pagerank(s,
                                                         r,
                                                         base_transitions[:],
                                                         adjacent_nodes[:],
                                                         out_degree,
                                                         in_degree,
                                                         n,
                                                         lazy_rho,
                                                         epsilon_eff)

            s_sparse = sparse.csr_matrix(s)

            # Perform degree normalization of approximate similarity matrix slice.
            relevant_degrees = in_degree[s_sparse.indices]
            s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

            base_community = np.append(adjacent_nodes[n], n)

            # If base community is not strictly non zero, then we break.
            intersection = np.intersect1d(base_community, s_sparse.indices)

            if intersection.size < base_community.size:
                continue

            base_community_rankings = np.searchsorted(s_sparse.indices, base_community)
            min_similarity = np.min(s_sparse.data[base_community_rankings])

            # Sort the degree normalized approximate similarity matrix slice.
            sorted_indices = np.argsort(s_sparse.data, axis=0)
            s_sparse.data = s_sparse.data[sorted_indices]
            s_sparse.indices = s_sparse.indices[sorted_indices]

            most_unlikely_index = s_sparse.indices.size - np.searchsorted(s_sparse.data, min_similarity)

            # Save feature matrix coordinates.
            if most_unlikely_index > base_community.size:
                # print(n_index, out_degree[n], epsilon_eff)
                new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
                extend_row(new_rows)
                # extend_col(number_of_local_communities*np.ones_like(new_rows))
                # number_of_local_communities += 1
                extend_col(n*np.ones_like(new_rows))

        # Form local community feature matrix.
        row = np.array(row_list, dtype=np.int64)
        col = np.array(col_list, dtype=np.int64)
        data = np.ones_like(row, dtype=np.float64)
        # features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))
        chunk_features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_nodes))
        chunk_features = sparse.csr_matrix(chunk_features)

        features += chunk_features

    return features


def arcte_with_pagerank_worker(iterate_nodes,
                               indices_c,
                               indptr_c,
                               data_c,
                               out_degree,
                               in_degree,
                               rho,
                               epsilon):

    iterate_nodes = np.array(iterate_nodes, dtype=np.int64)
    number_of_nodes = out_degree.size

    mean_degree = np.mean(out_degree)

    rw_transition = sparse.csr_matrix((data_c, indices_c, indptr_c))

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in range(number_of_nodes):
        adjacent_nodes[n] = rw_transition.indices[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]
        base_transitions[n] = rw_transition.data[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]

    # Separate nodes to iterate in smaller chunks in order to compress the feature batches.
    features = sparse.dok_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)
    features = sparse.csr_matrix(features)

    if iterate_nodes.size > 2000:
        node_chunks = list(parallel_chunks(iterate_nodes, iterate_nodes.size//2000))
    else:
        node_chunks = list()
        node_chunks.append(iterate_nodes)

    for node_chunk in node_chunks:
        node_chunk = np.array(node_chunk, dtype=np.int64)

        # Calculate local communities for all nodes.
        row_list = list()
        col_list = list()
        extend_row = row_list.extend
        extend_col = col_list.extend

        # number_of_local_communities = 0

        s = np.zeros(number_of_nodes, dtype=np.float64)  #TODO: What if it is only one?
        r = np.zeros(number_of_nodes, dtype=np.float64)
        for n_index in range(node_chunk.size):
            print(n_index)
            n = node_chunk[n_index]

            # Calculate similarity matrix slice.
            s[:] = 0.0
            r[:] = 0.0

            epsilon_eff = calculate_epsilon_effective(rho, epsilon, out_degree[n], out_degree[adjacent_nodes[n]], mean_degree)

            nop = fast_approximate_personalized_pagerank(s,
                                                         r,
                                                         base_transitions[:],
                                                         adjacent_nodes[:],
                                                         out_degree,
                                                         in_degree,
                                                         n,
                                                         rho,
                                                         epsilon_eff)

            s_sparse = sparse.csr_matrix(s)

            # Perform degree normalization of approximate similarity matrix slice.
            relevant_degrees = in_degree[s_sparse.indices]
            s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

            base_community = np.append(adjacent_nodes[n], n)

            # If base community is not strictly non zero, then we break.
            intersection = np.intersect1d(base_community, s_sparse.indices)

            if intersection.size < base_community.size:
                continue

            base_community_rankings = np.searchsorted(s_sparse.indices, base_community)
            min_similarity = np.min(s_sparse.data[base_community_rankings])

            # Sort the degree normalized approximate similarity matrix slice.
            sorted_indices = np.argsort(s_sparse.data, axis=0)
            s_sparse.data = s_sparse.data[sorted_indices]
            s_sparse.indices = s_sparse.indices[sorted_indices]

            most_unlikely_index = s_sparse.indices.size - np.searchsorted(s_sparse.data, min_similarity)

            # Save feature matrix coordinates.
            if most_unlikely_index > base_community.size:
                # print(n_index, out_degree[n], epsilon_eff)
                new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
                extend_row(new_rows)
                # extend_col(number_of_local_communities*np.ones_like(new_rows))
                # number_of_local_communities += 1
                extend_col(n*np.ones_like(new_rows))

        # Form local community feature matrix.
        row = np.array(row_list, dtype=np.int64)
        col = np.array(col_list, dtype=np.int64)
        data = np.ones_like(row, dtype=np.float64)
        # features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))
        chunk_features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_nodes))
        chunk_features = sparse.csr_matrix(chunk_features)

        features += chunk_features

    return features


def arcte_worker(iterate_nodes,
                 indices_c,
                 indptr_c,
                 data_c,
                 out_degree,
                 in_degree,
                 rho,
                 epsilon):

    iterate_nodes = np.array(iterate_nodes, dtype=np.int64)
    number_of_nodes = out_degree.size

    mean_degree = np.mean(out_degree)

    rw_transition = sparse.csr_matrix((data_c, indices_c, indptr_c))

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in range(number_of_nodes):
        adjacent_nodes[n] = rw_transition.indices[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]
        base_transitions[n] = rw_transition.data[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]

    # Separate nodes to iterate in smaller chunks in order to compress the feature batches.
    features = sparse.dok_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)
    features = sparse.csr_matrix(features)

    if iterate_nodes.size > 2000:
        node_chunks = list(parallel_chunks(iterate_nodes, iterate_nodes.size//2000))
    else:
        node_chunks = list()
        node_chunks.append(iterate_nodes)

    for node_chunk in node_chunks:
        node_chunk = np.array(node_chunk, dtype=np.int64)

        # Calculate local communities for all nodes.
        row_list = list()
        col_list = list()
        extend_row = row_list.extend
        extend_col = col_list.extend

        # number_of_local_communities = 0

        s = np.zeros(number_of_nodes, dtype=np.float64)  #TODO: What if it is only one?
        r = np.zeros(number_of_nodes, dtype=np.float64)
        for n_index in range(node_chunk.size):
            # print(n_index)
            n = node_chunk[n_index]

            # Calculate similarity matrix slice.
            s[:] = 0.0
            r[:] = 0.0

            epsilon_eff = calculate_epsilon_effective(rho, epsilon, out_degree[n], out_degree[adjacent_nodes[n]], mean_degree)

            nop = fast_approximate_cumulative_pagerank_difference(s,
                                                                  r,
                                                                  base_transitions[:],
                                                                  adjacent_nodes[:],
                                                                  out_degree,
                                                                  in_degree,
                                                                  n,
                                                                  rho,
                                                                  epsilon_eff)

            s_sparse = sparse.csr_matrix(s)

            # Perform degree normalization of approximate similarity matrix slice.
            relevant_degrees = in_degree[s_sparse.indices]
            s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

            base_community = np.append(adjacent_nodes[n], n)
            base_community_rankings = np.searchsorted(s_sparse.indices, base_community)
            min_similarity = np.min(s_sparse.data[base_community_rankings])

            # Sort the degree normalized approximate similarity matrix slice.
            sorted_indices = np.argsort(s_sparse.data, axis=0)
            s_sparse.data = s_sparse.data[sorted_indices]
            s_sparse.indices = s_sparse.indices[sorted_indices]

            most_unlikely_index = s_sparse.indices.size - np.searchsorted(s_sparse.data, min_similarity)

            # Save feature matrix coordinates.
            if most_unlikely_index > base_community.size:
                # print(n_index, out_degree[n], epsilon_eff)
                new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
                extend_row(new_rows)
                # extend_col(number_of_local_communities*np.ones_like(new_rows))
                # number_of_local_communities += 1
                extend_col(n*np.ones_like(new_rows))

        # Form local community feature matrix.
        row = np.array(row_list, dtype=np.int64)
        col = np.array(col_list, dtype=np.int64)
        data = np.ones_like(row, dtype=np.float64)
        # features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))
        chunk_features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_nodes))
        chunk_features = sparse.csr_matrix(chunk_features)

        features += chunk_features

    return features


def arcte_with_lazy_pagerank(adjacency_matrix, rho, epsilon, number_of_threads=None):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    if number_of_threads is None:
        number_of_threads = get_threads_number()
    if number_of_threads == 1:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)

        a = adjacency_matrix.copy()
        a.data = np.ones_like(a.data)
        edge_count_vector = np.squeeze(np.asarray(a.sum(axis=0), dtype=np.int64))

        iterate_nodes = np.where(edge_count_vector != 0)[0]
        argsort_indices = np.argsort(edge_count_vector[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(edge_count_vector[iterate_nodes] > 1.0)[0]]

        # iterate_nodes = np.where(out_degree != 0)[0]
        # argsort_indices = np.argsort(out_degree[iterate_nodes])
        # iterate_nodes = iterate_nodes[argsort_indices][::-1]
        # iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        local_features = arcte_with_lazy_pagerank_worker(iterate_nodes,
                                                         rw_transition.indices,
                                                         rw_transition.indptr,
                                                         rw_transition.data,
                                                         out_degree,
                                                         in_degree,
                                                         rho,
                                                         epsilon)
    else:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=True)

        a = adjacency_matrix.copy()
        a.data = np.ones_like(a.data)
        edge_count_vector = np.squeeze(np.asarray(a.sum(axis=0), dtype=np.int64))

        iterate_nodes = np.where(edge_count_vector != 0)[0]
        argsort_indices = np.argsort(edge_count_vector[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(edge_count_vector[iterate_nodes] > 1.0)[0]]

        # iterate_nodes = np.where(out_degree != 0)[0]
        # argsort_indices = np.argsort(out_degree[iterate_nodes])
        # iterate_nodes = iterate_nodes[argsort_indices][::-1]
        # iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        pool = mp.Pool(number_of_threads)
        node_chunks = list(parallel_chunks(iterate_nodes, number_of_threads))
        node_count = 0
        for chunk in node_chunks:
            node_count += len(list(chunk))
        results = list()
        for chunk_no in range(len(pool._pool)):
            pool.apply_async(arcte_with_lazy_pagerank_worker,
                             args=(node_chunks[chunk_no],
                                   rw_transition.indices,
                                   rw_transition.indptr,
                                   rw_transition.data,
                                   out_degree,
                                   in_degree,
                                   rho,
                                   epsilon),
                             callback=results.append)
        pool.close()
        pool.join()
        # local_features = sparse.hstack(results)
        local_features = results[0]
        for additive_features in results[1:]:
            local_features += additive_features
        local_features = sparse.csr_matrix(local_features)

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    adjacency_matrix_ones = adjacency_matrix
    adjacency_matrix_ones.data = np.ones_like(adjacency_matrix.data)
    base_community_features = identity_matrix + adjacency_matrix_ones

    # Stack horizontally matrices to form feature matrix.
    try:
        features = sparse.hstack([base_community_features, local_features]).tocsr()
    except ValueError as e:
        print("Failure with horizontal feature stacking.")
        features = base_community_features

    return features


def arcte_with_pagerank(adjacency_matrix, rho, epsilon, number_of_threads=None):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    if number_of_threads is None:
        number_of_threads = get_threads_number()
    if number_of_threads == 1:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)

        a = adjacency_matrix.copy()
        a.data = np.ones_like(a.data)
        edge_count_vector = np.squeeze(np.asarray(a.sum(axis=0), dtype=np.int64))

        iterate_nodes = np.where(edge_count_vector != 0)[0]
        argsort_indices = np.argsort(edge_count_vector[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(edge_count_vector[iterate_nodes] > 1.0)[0]]

        # iterate_nodes = np.where(out_degree != 0)[0]
        # argsort_indices = np.argsort(out_degree[iterate_nodes])
        # iterate_nodes = iterate_nodes[argsort_indices][::-1]
        # iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        local_features = arcte_with_pagerank_worker(iterate_nodes,
                                                    rw_transition.indices,
                                                    rw_transition.indptr,
                                                    rw_transition.data,
                                                    out_degree,
                                                    in_degree,
                                                    rho,
                                                    epsilon)
    else:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=True)

        a = adjacency_matrix.copy()
        a.data = np.ones_like(a.data)
        edge_count_vector = np.squeeze(np.asarray(a.sum(axis=0), dtype=np.int64))

        iterate_nodes = np.where(edge_count_vector != 0)[0]
        argsort_indices = np.argsort(edge_count_vector[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(edge_count_vector[iterate_nodes] > 1.0)[0]]

        # iterate_nodes = np.where(out_degree != 0)[0]
        # argsort_indices = np.argsort(out_degree[iterate_nodes])
        # iterate_nodes = iterate_nodes[argsort_indices][::-1]
        # iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        pool = mp.Pool(number_of_threads)
        node_chunks = list(parallel_chunks(iterate_nodes, number_of_threads))
        node_count = 0
        for chunk in node_chunks:
            node_count += len(list(chunk))
        results = list()
        for chunk_no in range(len(pool._pool)):
            pool.apply_async(arcte_with_pagerank_worker,
                             args=(node_chunks[chunk_no],
                                   rw_transition.indices,
                                   rw_transition.indptr,
                                   rw_transition.data,
                                   out_degree,
                                   in_degree,
                                   rho,
                                   epsilon),
                             callback=results.append)
        pool.close()
        pool.join()
        # local_features = sparse.hstack(results)
        local_features = results[0]
        for additive_features in results[1:]:
            local_features += additive_features
        local_features = sparse.csr_matrix(local_features)

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    adjacency_matrix_ones = adjacency_matrix
    adjacency_matrix_ones.data = np.ones_like(adjacency_matrix.data)
    base_community_features = identity_matrix + adjacency_matrix_ones

    # Stack horizontally matrices to form feature matrix.
    try:
        features = sparse.hstack([base_community_features, local_features]).tocsr()
    except ValueError as e:
        print("Failure with horizontal feature stacking.")
        features = base_community_features

    return features


def arcte(adjacency_matrix, rho, epsilon, number_of_threads=None):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    if number_of_threads is None:
        number_of_threads = get_threads_number()
    if number_of_threads == 1:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)

        a = adjacency_matrix.copy()
        a.data = np.ones_like(a.data)
        edge_count_vector = np.squeeze(np.asarray(a.sum(axis=0), dtype=np.int64))

        iterate_nodes = np.where(edge_count_vector != 0)[0]
        argsort_indices = np.argsort(edge_count_vector[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(edge_count_vector[iterate_nodes] > 1.0)[0]]

        # iterate_nodes = np.where(out_degree != 0)[0]
        # argsort_indices = np.argsort(out_degree[iterate_nodes])
        # iterate_nodes = iterate_nodes[argsort_indices][::-1]
        # iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        local_features = arcte_worker(iterate_nodes,
                                      rw_transition.indices,
                                      rw_transition.indptr,
                                      rw_transition.data,
                                      out_degree,
                                      in_degree,
                                      rho,
                                      epsilon)
    else:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=True)

        a = adjacency_matrix.copy()
        a.data = np.ones_like(a.data)
        edge_count_vector = np.squeeze(np.asarray(a.sum(axis=0), dtype=np.int64))

        iterate_nodes = np.where(edge_count_vector != 0)[0]
        argsort_indices = np.argsort(edge_count_vector[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(edge_count_vector[iterate_nodes] > 1.0)[0]]

        # iterate_nodes = np.where(out_degree != 0)[0]
        # argsort_indices = np.argsort(out_degree[iterate_nodes])
        # iterate_nodes = iterate_nodes[argsort_indices][::-1]
        # iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        pool = mp.Pool(number_of_threads)
        node_chunks = list(parallel_chunks(iterate_nodes, number_of_threads))
        node_count = 0
        for chunk in node_chunks:
            node_count += len(list(chunk))
        results = list()
        for chunk_no in range(len(pool._pool)):
            pool.apply_async(arcte_worker,
                             args=(node_chunks[chunk_no],
                                   rw_transition.indices,
                                   rw_transition.indptr,
                                   rw_transition.data,
                                   out_degree,
                                   in_degree,
                                   rho,
                                   epsilon),
                             callback=results.append)
        pool.close()
        pool.join()
        # local_features = sparse.hstack(results)
        local_features = results[0]
        for additive_features in results[1:]:
            local_features += additive_features
        local_features = sparse.csr_matrix(local_features)

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    adjacency_matrix_ones = adjacency_matrix
    adjacency_matrix_ones.data = np.ones_like(adjacency_matrix.data)
    base_community_features = identity_matrix + adjacency_matrix_ones

    # Stack horizontally matrices to form feature matrix.
    try:
        features = sparse.hstack([base_community_features, local_features]).tocsr()
    except ValueError as e:
        print("Failure with horizontal feature stacking.")
        features = base_community_features

    return features
