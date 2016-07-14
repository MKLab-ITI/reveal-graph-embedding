__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import networkx as nx
from networkx.algorithms.link_analysis import pagerank_scipy

from reveal_graph_embedding.eps_randomwalk.transition import get_natural_random_walk_matrix


def calculate_entropy(array, norm=False):
    array = array/array.sum()

    if norm:
        array = array - array.min()
        array = array/array.sum()

    entropy = sum(-np.multiply(np.log(array[array > 0.0]), array[array > 0.0]))
    return entropy


def get_implicit_adjacency_matrices(adjacency_matrix, rho=0.2):
    # Calculate random walk with restart and teleportation.
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)
    rw_transition = rw_transition.tocoo()
    rw_transition_t = rw_transition.T.tocsr()
    rw_transition = rw_transition.tocsr()

    stationary_distribution = get_stationary_distribution_directed(adjacency_matrix,
                                                                   rho)

    # Calculate implicit combinatorial adjacency matrix.
    implicit_combinatorial_matrix, com_phi = get_implicit_combinatorial_adjacency_matrix(stationary_distribution,
                                                                                         rw_transition,
                                                                                         rw_transition_t)

    # Calculate implicit directed adjacency matrix.
    implicit_directed_matrix, dir_phi = get_implicit_directed_adjacency_matrix(stationary_distribution,
                                                                               rw_transition)

    return implicit_combinatorial_matrix, com_phi, implicit_directed_matrix, dir_phi


def get_stationary_distribution_directed(adjacency_matrix, rho):

    graph_nx = nx.from_scipy_sparse_matrix(adjacency_matrix, create_using=nx.DiGraph())

    stationary_distribution = pagerank_scipy(graph_nx,
                                             alpha=1-rho,
                                             personalization=None,
                                             max_iter=200,
                                             tol=1.0e-7,
                                             weight="weight",
                                             dangling=None)

    stationary_distribution = np.array([stationary_distribution[k] for k in sorted(stationary_distribution.keys())])

    return stationary_distribution


def get_pagerank_with_teleportation_from_transition_matrix(rw_transition, rw_transition_t, rho):
    number_of_nodes = rw_transition.shape[0]

    # Set up the random walk with teleportation matrix.
    non_teleportation = 1-rho
    mv = lambda l, v: non_teleportation*l.dot(v) + (rho/number_of_nodes)*np.ones_like(v)
    teleport = lambda vec: mv(rw_transition_t, vec)

    rw_transition_operator = spla.LinearOperator(rw_transition.shape, matvec=teleport, dtype=np.float64)

    # Calculate stationary distribution.
    try:
        eigenvalue, stationary_distribution = spla.eigs(rw_transition_operator,
                                                        k=1,
                                                        which='LM',
                                                        return_eigenvectors=True)
    except spla.ArpackNoConvergence as e:
        print("ARPACK has not converged.")
        eigenvalue = e.eigenvalues
        stationary_distribution = e.eigenvectors

    stationary_distribution = stationary_distribution.flatten().real/stationary_distribution.sum()

    return stationary_distribution


def get_implicit_combinatorial_adjacency_matrix(stationary_distribution, rw_transition, rw_transition_t):
    number_of_nodes = rw_transition.shape[0]

    pi_matrix = spsp.spdiags(stationary_distribution, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (pi_matrix.dot(rw_transition) + rw_transition_t.dot(pi_matrix))/2.0

    effective_adjacency_matrix = spsp.coo_matrix(spsp.csr_matrix(effective_adjacency_matrix))
    effective_adjacency_matrix.data = np.real(effective_adjacency_matrix.data)

    effective_adjacency_matrix = spsp.csr_matrix(effective_adjacency_matrix)

    return effective_adjacency_matrix, stationary_distribution


def get_implicit_directed_adjacency_matrix(stationary_distribution, rw_transition):
    number_of_nodes = rw_transition.shape[0]

    sqrtp = sp.sqrt(stationary_distribution)
    Q = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes) * rw_transition * spsp.spdiags(1.0/sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (Q + Q.T) /2.0

    effective_adjacency_matrix = spsp.coo_matrix(spsp.csr_matrix(effective_adjacency_matrix))
    effective_adjacency_matrix.data = np.real(effective_adjacency_matrix.data)

    effective_adjacency_matrix = spsp.csr_matrix(effective_adjacency_matrix)

    return effective_adjacency_matrix, np.ones(number_of_nodes, dtype=np.float64)


def safe_convex_weight_calculation(transition_matrix_list, out_degree_list, weights):
    number_of_views = len(transition_matrix_list)
    number_of_nodes = transition_matrix_list[0].shape[0]

    # Initialize non-dangling nodes with one.
    # TODO: This can be done in a smarter way; no need to give out_degree_list as an argument.
    actual_weights = np.empty((number_of_views, number_of_nodes), dtype=np.float64)
    for v in range(number_of_views):
        # print(calculate_entropy(out_degree_list[v]/out_degree_list[v].sum()))

        actual_weights[v, :] = out_degree_list[v]

    actual_weights[actual_weights > 0.0] = 1.0

    # Filter out dangling nodes in corresponding views.
    for n in range(number_of_nodes):
        actual_weights[:, n] = np.multiply(actual_weights[:, n], weights)
        row_sum = np.sum(actual_weights[:, n])
        if row_sum > 0.0:
            actual_weights[:, n] = actual_weights[:, n]/row_sum

    return actual_weights


def entropy_view_weight_calculation(adjacency_matrix_list, transition_matrix_list, out_degree_list):
    number_of_views = len(transition_matrix_list)
    number_of_nodes = transition_matrix_list[0].shape[0]

    actual_weights = np.empty((number_of_views, number_of_nodes), dtype=np.float64)
    for v in range(number_of_views):
        actual_weights[v, :] = out_degree_list[v]

    actual_weights[actual_weights > 0.0] = 1.0

    for n in range(number_of_nodes):
        row_nnz_ind = np.where(actual_weights > 0.0)[0]
        if row_nnz_ind.size > 0:
            for v in row_nnz_ind:
                transition_matrix_list


def graph_fusion(adjacency_matrix_list, weights=None, method="zhou"):
    # Get number of matrices.
    number_of_views = len(adjacency_matrix_list)
    if number_of_views < 1:
        print("Empty adjacency matrix list.")
        raise RuntimeError

    # Make sure number of weights is equal to number of matrices.
    if method == "zhou":
        if weights is None:
            weights = (1/number_of_views) * np.ones(number_of_views, dtype=np.float64)
        else:
            if len(weights) != number_of_views:
                print("Number of adjacency matrices not equal to number of weights.")
                raise RuntimeError
            else:
                weights /= npla.norm(weights, "fro")

    # Make sure all matrices are in csr format.
    adjacency_matrix_list = (spsp.csr_matrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list)

    # Get natural random walk transition matrices.
    transition_tuple_list = [get_natural_random_walk_matrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list]
    transition_matrix_list = [t[0] for t in transition_tuple_list]
    out_degree_list = [t[1] for t in transition_tuple_list]
    in_degree_list = [t[2] for t in transition_tuple_list]

    # Calculate actual weights for matrices.
    if method == "zhou":
        actual_weights = safe_convex_weight_calculation(transition_matrix_list, out_degree_list, weights)
    elif method == "entropy":
        actual_weights = entropy_view_weight_calculation(adjacency_matrix_list, transition_matrix_list, out_degree_list)
    else:
        print("Invalid view weighting method selected.")
        raise RuntimeError

    # Calculate the multiview implicit transition matrix.
    number_of_nodes = transition_matrix_list[0].shape[0]
    weight_diagonal_matrix = spsp.csr_matrix(spsp.spdiags(actual_weights[0], [0], number_of_nodes, number_of_nodes))
    multiview_implicit_transition_matrix = weight_diagonal_matrix.dot(transition_matrix_list[0])

    for v in range(1, number_of_views):
        weight_diagonal_matrix = spsp.csr_matrix(spsp.spdiags(actual_weights[v], [0], number_of_nodes, number_of_nodes))
        multiview_implicit_transition_matrix += weight_diagonal_matrix.dot(transition_matrix_list[v])

    return multiview_implicit_transition_matrix


def graph_fusion_directed(adjacency_matrix_list, weights, fusion_type, laplacian_type):
    number_of_nodes = adjacency_matrix_list[0].shape[0]

    # Get number of views.
    number_of_views = len(adjacency_matrix_list)
    if number_of_views < 1:
        print("Empty adjacency matrix list.")
        raise RuntimeError

    # Make sure number of weights is equal to number of matrices.
    if weights is None:
        weights = (1/number_of_views) * np.ones(number_of_views, dtype=np.float64)
    else:
        if len(weights) != number_of_views:
            print("Number of adjacency matrices not equal to number of weights.")
            raise RuntimeError
        else:
            weights /= np.sum(weights)

    # Make sure all matrices are in csr format.
    adjacency_matrix_list = [spsp.csr_matrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list]

    # Get natural random walk transition matrices.
    transition_tuple_list = [get_natural_random_walk_matrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list]
    transition_matrix_list = [t[0] for t in transition_tuple_list]
    out_degree_list = [t[1] for t in transition_tuple_list]
    in_degree_list = [t[2] for t in transition_tuple_list]

    # Calculate actual weights for matrices.
    if fusion_type == "zhou":
        actual_weights = safe_convex_weight_calculation(transition_matrix_list, out_degree_list, weights)

        stationary_distribution_list = [get_stationary_distribution_directed(spsp.csr_matrix(adjacency_matrix),
                                                                             0.15) for adjacency_matrix in adjacency_matrix_list]

        multiview_implicit_stationary_distribution = fuse_stationary_distributions(stationary_distribution_list,
                                                                                   actual_weights)

        multiview_implicit_transition_matrix = fuse_transition_matrices(transition_matrix_list,
                                                                        stationary_distribution_list,
                                                                        actual_weights,
                                                                        multiview_implicit_stationary_distribution)

        # Calculate the multiview implicit transition matrix.
        if laplacian_type == "combinatorial":
            multiview_implicit_adjacency_matrix,\
            diagonal = get_implicit_combinatorial_adjacency_matrix(multiview_implicit_stationary_distribution,
                                                                   multiview_implicit_transition_matrix,
                                                                   spsp.csr_matrix(multiview_implicit_transition_matrix.transpose()))
        elif laplacian_type == "directed":
            multiview_implicit_adjacency_matrix,\
            diagonal = get_implicit_directed_adjacency_matrix(multiview_implicit_stationary_distribution,
                                                              multiview_implicit_transition_matrix)
        else:
            print("Invalid laplacian type.")
            raise RuntimeError

        diagonal_matrix = spsp.spdiags(diagonal, [0], number_of_nodes, number_of_nodes)

        multiview_implicit_laplacian_matrix = diagonal_matrix - multiview_implicit_adjacency_matrix
    elif fusion_type == "addition":
        actual_weights = safe_convex_weight_calculation(transition_matrix_list, out_degree_list, weights)

        multiview_implicit_adjacency_matrix = simple_adjacency_matrix_addition(adjacency_matrix_list,
                                                                                actual_weights)

        degree = spsp.dia_matrix((multiview_implicit_adjacency_matrix.sum(axis=0), np.array([0])), shape=multiview_implicit_adjacency_matrix.shape)
        degree = degree.tocsr()

        # Calculate sparse graph Laplacian.
        multiview_implicit_laplacian_matrix = spsp.csr_matrix(-multiview_implicit_adjacency_matrix + degree, dtype=np.float64)
    elif fusion_type == "entropy":
        actual_weights = safe_convex_weight_calculation(transition_matrix_list, out_degree_list, weights)

        stationary_distribution_list = [get_stationary_distribution_directed(spsp.csr_matrix(adjacency_matrix),
                                                                             0.15) for adjacency_matrix in adjacency_matrix_list]

        multiview_implicit_stationary_distribution = fuse_stationary_distributions(stationary_distribution_list,
                                                                                   actual_weights)

        multiview_implicit_transition_matrix = fuse_transition_matrices(transition_matrix_list,
                                                                        stationary_distribution_list,
                                                                        actual_weights,
                                                                        multiview_implicit_stationary_distribution)

        degree = spsp.dia_matrix((multiview_implicit_adjacency_matrix.sum(axis=0), np.array([0])), shape=multiview_implicit_adjacency_matrix.shape)
        degree = degree.tocsr()

        # Calculate sparse graph Laplacian.
        multiview_implicit_laplacian_matrix = spsp.csr_matrix(-multiview_implicit_adjacency_matrix + degree, dtype=np.float64)
    else:
        print("Invalid fusion type.")
        raise RuntimeError

    multiview_implicit_adjacency_matrix = spsp.csr_matrix(multiview_implicit_adjacency_matrix)
    multiview_implicit_adjacency_matrix.eliminate_zeros()

    multiview_implicit_laplacian_matrix = spsp.csr_matrix(multiview_implicit_laplacian_matrix)
    multiview_implicit_laplacian_matrix.eliminate_zeros()

    return multiview_implicit_adjacency_matrix, multiview_implicit_laplacian_matrix


def fuse_stationary_distributions(stationary_distribution_list,
                                  actual_weights):
    number_of_views = len(stationary_distribution_list)

    multiview_implicit_stationary_distribution = np.multiply(stationary_distribution_list[0], actual_weights[0, :])
    # print(calculate_entropy(np.multiply(stationary_distribution_list[0], actual_weights[0, :])))

    for view_counter in range(1, number_of_views):
        multiview_implicit_stationary_distribution += np.multiply(stationary_distribution_list[view_counter], actual_weights[view_counter, :])
        # print(calculate_entropy(np.multiply(stationary_distribution_list[view_counter], actual_weights[view_counter, :])))

    multiview_implicit_stationary_distribution[multiview_implicit_stationary_distribution == 0.0] = np.min(multiview_implicit_stationary_distribution[multiview_implicit_stationary_distribution > 0.0])/2

    return multiview_implicit_stationary_distribution


def fuse_transition_matrices(transition_matrix_list,
                             stationary_distribution_list,
                             actual_weights,
                             multiview_implicit_stationary_distribution):
    number_of_views = len(transition_matrix_list)
    number_of_nodes = transition_matrix_list[0].shape[0]

    # print(np.any(np.isinf(multiview_implicit_stationary_distribution)))
    # print(np.any(np.isnan(multiview_implicit_stationary_distribution)))
    # print(np.any(multiview_implicit_stationary_distribution == 0.0))

    # Calculate convex combination weights.
    convex_combination_weights = list()
    for view_counter in range(number_of_views):
        convex_combination_weights.append(np.divide(np.multiply(stationary_distribution_list[view_counter], actual_weights[view_counter, :]),
                                                    multiview_implicit_stationary_distribution))

    # Convert convex combination weights to csr sparse matrices.
    convex_combination_weights = [spsp.spdiags(weight_vector, [0], number_of_nodes, number_of_nodes) for weight_vector in convex_combination_weights]

    # Fuse matrices.
    multiview_implicit_transition_matrix = convex_combination_weights[0].dot(transition_matrix_list[0])
    for view_counter in range(1, number_of_views):
        multiview_implicit_transition_matrix = multiview_implicit_transition_matrix + convex_combination_weights[view_counter].dot(transition_matrix_list[view_counter])

    return multiview_implicit_transition_matrix


def simple_adjacency_matrix_addition(adjacency_matrix_list,
                                     actual_weights):
    number_of_views = len(adjacency_matrix_list)
    number_of_nodes = adjacency_matrix_list[0].shape[0]

    actual_weights_csr = [spsp.spdiags(actual_weights[view_counter, :], [0], number_of_nodes, number_of_nodes) for view_counter in range(number_of_views)]

    temp = actual_weights_csr[0].dot(adjacency_matrix_list[0])
    multiview_implicit_transition_matrix = 0.5*temp + 0.5*temp.transpose()
    for view_counter in range(1, number_of_views):
        temp = actual_weights_csr[view_counter].dot(adjacency_matrix_list[view_counter])
        multiview_implicit_transition_matrix = multiview_implicit_transition_matrix + 0.5*temp + 0.5*temp.transpose()

    return multiview_implicit_transition_matrix
