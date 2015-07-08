__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import copy
import networkx as nx
import community
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

from reveal_graph_embedding.embedding.laplacian import get_normalized_laplacian


def mroc(adjacency_matrix, alpha):
    """
    Extracts hierarchical community features using the MROC method.

    Introduced in: Wang, X., Tang, L., Liu, H., & Wang, L. (2013).
                   Learning with multi-resolution overlapping communities.
                   Knowledge and information systems, 36(2), 517-535.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - alpha: A maximum community size stopping threshold.

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    # Find number of nodes
    number_of_nodes = adjacency_matrix.shape[0]

    ####################################################################################################################
    # Base community calculation
    ####################################################################################################################
    # Initialize empty lists
    base_list = list()
    base_row = list()
    base_col = list()

    # Save function handles for speed
    append_base_list = base_list.append
    append_base_row = base_row.append
    append_base_col = base_col.append

    # Find base communities
    adjacency_matrix = adjacency_matrix.tocsc()
    number_of_base_communities = 0
    for i in range(number_of_nodes):
        # Calculate base community
        base_community = set(adjacency_matrix.getcol(i).indices)
        base_community.add(i)
        flag = True
        for c in base_list:
            if c == base_community:
                flag = False
                break
        if flag:
            append_base_list(base_community)

            for n in base_community:
                append_base_row(n)
                append_base_col(number_of_base_communities)

            number_of_base_communities += 1

    # Form sparse matrices
    base_row = np.array(base_row)
    base_col = np.array(base_col)
    base_data = np.ones(base_row.size, dtype=np.float64)
    features = sparse.coo_matrix((base_data, (base_row, base_col)),
                                 shape=(number_of_nodes, number_of_base_communities))

    features = features.tocsr()
    base_community_number = features.shape[1]

    print('Base communities calculated.')

    reverse_index_csr = copy.copy(features)
    reverse_index_csc = reverse_index_csr.tocsc()
    reverse_index_csr = reverse_index_csr.tocsr()

    reverse_index_rows = np.ndarray(number_of_nodes, dtype=np.ndarray)
    reverse_index_cols = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in range(number_of_nodes):
        reverse_index_row = reverse_index_csr.getrow(n)
        reverse_index_rows[n] = reverse_index_row.indices

        if n < base_community_number:
            reverse_index_col = reverse_index_csc.getcol(n)
            reverse_index_cols[n] = reverse_index_col.indices

    flag = True

    print('Start merge iterations.')

    iteration = 0

    while flag:
        level_row = list()
        level_col = list()

        append_level_row = level_row.append
        append_level_col = level_col.append

        unavailable_communities = -1*np.ones(reverse_index_csc.shape[1])
        unavailable_communities_counter = 0
        next_level_communities = list()
        append_next_level_community = next_level_communities.append
        number_of_communities = 0
        for j in range(reverse_index_csr.shape[1]):
            if j in unavailable_communities:
                continue
            must_break = reverse_index_csr.shape[1] - unavailable_communities_counter
            print(must_break)
            if must_break < 1:
                break
            unavailable_communities[unavailable_communities_counter] = j
            unavailable_communities_counter += 1
            c_j = reverse_index_cols[j]

            indices = community_neighbors(c_j, reverse_index_rows, unavailable_communities, unavailable_communities_counter)

            max_similarity = -1
            community_index = 0
            for jj in indices:
                c_jj = reverse_index_cols[jj]
                similarity = jaccard(c_j, c_jj)
                if similarity > max_similarity:
                    max_similarity = similarity
                    community_index = jj

            jj = community_index
            if max_similarity > 0:
                # Merge two communities
                c_jj = reverse_index_cols[jj]
                c_new = np.union1d(c_j, c_jj)

                flag_1 = np.setdiff1d(c_new, c_j)
                flag_2 = np.setdiff1d(c_new, c_jj)
                if (flag_1.size != 0) and (flag_2.size != 0):
                    for n in c_new:
                        append_level_row(n)
                        append_level_col(number_of_communities)

                    if c_new.size < alpha:
                        append_next_level_community(number_of_communities)

                    number_of_communities += 1
                unavailable_communities[unavailable_communities_counter] = jj
                unavailable_communities_counter += 1

        level_row = np.array(level_row)
        level_col = np.array(level_col)
        level_data = np.ones(level_row.size, dtype=np.float64)
        communities = sparse.coo_matrix((level_data, (level_row, level_col)),
                                        shape=(number_of_nodes, number_of_communities))

        if communities.getnnz() == 0:
            break

        features = sparse.hstack([features, communities])

        reverse_index_csc = copy.copy(communities)
        reverse_index_csc = reverse_index_csc.tocsc()
        reverse_index_csc = reverse_index_csc[:, np.array(next_level_communities)]
        reverse_index_csr = reverse_index_csc.tocsr()

        reverse_index_rows = np.ndarray(number_of_nodes, dtype=np.ndarray)
        reverse_index_cols = np.ndarray(len(next_level_communities), dtype=np.ndarray)
        for n in range(number_of_nodes):
            reverse_index_row = reverse_index_csr.getrow(n)
            reverse_index_rows[n] = reverse_index_row.indices

            if n < len(next_level_communities):
                reverse_index_col = reverse_index_csc.getcol(n)
                reverse_index_cols[n] = reverse_index_col.indices

        if len(next_level_communities) > 1:
            flag = True

        iteration += 1
        print('Iteration: ', iteration)
        print('List length', len(next_level_communities))

    return features


def community_neighbors(c_j, reverse_index_rows, unavailable_communities, unavailable_communities_counter):
    """
    Finds communities with shared nodes to a seed community. Called by mroc.

    Inputs:  - c_j: The seed community for which we want to find which communities overlap.
             - reverse_index_rows: A node to community indicator matrix.
             - unavailable_communities: A set of communities that have already either been merged or failed to merge.
             - unavailable_communities_counter: The number of such communities.

    Outputs: - indices: An array containing the communities that exhibit overlap with the seed community.
    """
    indices = list()
    extend = indices.extend
    for node in c_j:
        extend(reverse_index_rows[node])

    indices = np.array(indices)
    indices = np.setdiff1d(indices, unavailable_communities[:unavailable_communities_counter+1])

    return indices


def jaccard(c_1, c_2):
    """
    Calculates the Jaccard similarity between two sets of nodes. Called by mroc.

    Inputs:  - c_1: Community (set of nodes) 1.
             - c_2: Community (set of nodes) 2.

    Outputs: - jaccard_similarity: The Jaccard similarity of these two communities.
    """
    nom = np.intersect1d(c_1, c_2).size
    denom = np.union1d(c_1, c_2).size
    return nom/denom


def louvain(adjacency_matrix):
    """
    Performs community embedding using the LOUVAIN method.

    Introduced in: Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
                   Fast unfolding of communities in large networks.
                   Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    # Convert to networkx undirected graph.
    adjacency_matrix = nx.from_scipy_sparse_matrix(adjacency_matrix, create_using=nx.Graph())

    # Call LOUVAIN algorithm to calculate a hierarchy of communities.
    tree = community.generate_dendogram(adjacency_matrix, part_init=None)

    # Embed communities
    row = list()
    col = list()
    append_row = row.append
    append_col = col.append

    community_counter = 0
    for i in range(len(tree)):
        partition = community.partition_at_level(tree, i)
        for n, c in partition.items():
            append_row(n)
            append_col(community_counter + c)

        community_counter += max(partition.values()) + 1

    row = np.array(row)
    col = np.array(col)
    data = np.ones(row.size, dtype=np.float64)

    louvain_features = sparse.coo_matrix((data, (row, col)), shape=(len(partition.keys()), community_counter),
                                         dtype=np.float64)

    return louvain_features


def laplacian_eigenmaps(adjacency_matrix, k):
    """
    Performs spectral graph embedding using the graph symmetric normalized Laplacian matrix.

    Introduced in: Belkin, M., & Niyogi, P. (2003).
                   Laplacian eigenmaps for dimensionality reduction and data representation.
                   Neural computation, 15(6), 1373-1396.

    Inputs:  -   A in R^(nxn): Adjacency matrix of an network represented as a SciPy Sparse COOrdinate matrix.
             -              k: The number of eigenvectors to extract.

    Outputs: - X in R^(nxk): The latent space embedding represented as a NumPy array. We discard the first eigenvector.
    """
    # Calculate sparse graph Laplacian.
    laplacian = get_normalized_laplacian(adjacency_matrix)

    # Calculate bottom k+1 eigenvalues and eigenvectors of normalized Laplacian.
    try:
        eigenvalues, eigenvectors = spla.eigsh(laplacian,
                                               k=k,
                                               which='SM',
                                               return_eigenvectors=True)
    except spla.ArpackNoConvergence as e:
        print("ARPACK has not converged.")
        eigenvalue = e.eigenvalues
        eigenvectors = e.eigenvectors

    # Discard the eigenvector corresponding to the zero-valued eigenvalue.
    eigenvectors = eigenvectors[:, 1:]

    return eigenvectors


def replicator_eigenmaps(adjacency_matrix, k):
    """
    Performs spectral graph embedding on the centrality reweighted adjacency matrix

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a scipy.sparse.coo_matrix
             -            k: The number of social dimensions/eigenvectors to extract
             -      max_iter: The maximum number of iterations for the iterative eigensolution method

    Outputs: - S in R^(nxk): The social dimensions represented as a numpy.array matrix
    """
    number_of_nodes = adjacency_matrix.shape[0]

    max_eigenvalue = spla.eigsh(adjacency_matrix,
                                k=1,
                                which='LM',
                                return_eigenvectors=False)

    # Calculate Replicator matrix
    eye_matrix = sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64)
    eye_matrix = eye_matrix.tocsr()
    eye_matrix.data = eye_matrix.data*max_eigenvalue
    replicator = eye_matrix - adjacency_matrix

    # Calculate bottom k+1 eigenvalues and eigenvectors of normalised Laplacian
    try:
        eigenvalues, eigenvectors = spla.eigsh(replicator,
                                               k=k+1,
                                               which='SM',
                                               return_eigenvectors=True)
    except spla.ArpackNoConvergence as e:
        print("ARPACK has not converged.")
        eigenvalue = e.eigenvalues
        eigenvectors = e.eigenvectors

    eigenvectors = eigenvectors[:, 1:]

    return eigenvectors


def base_communities(adjacency_matrix):
    """
    Forms the community indicator normalized feature matrix for any graph.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    number_of_nodes = adjacency_matrix.shape[0]

    # X = A + I
    adjacency_matrix = adjacency_matrix.tocsr()
    adjacency_matrix = adjacency_matrix.transpose()
    features = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes)) + adjacency_matrix.tocsr()
    features = features.tocsr()
    features.data = np.ones_like(features.data)

    return features
