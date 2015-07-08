__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spla

from reveal_graph_embedding.eps_randomwalk.transition import get_natural_random_walk_matrix


def get_implicit_adjacency_matrices(rw_transition, rho=0.2):
    # Calculate random walk with restart and teleportation.
    rw_transition = rw_transition.tocoo()
    rw_transition_t = rw_transition.T.tocsr()
    rw_transition = rw_transition.tocsr()

    stationary_distribution = get_pagerank_with_teleportation(rw_transition,
                                                              rw_transition_t,
                                                              rho)

    # Calculate implicit combinatorial adjacency matrix.
    implicit_combinatorial_matrix, com_phi = get_implicit_combinatorial_adjacency_matrix(stationary_distribution,
                                                                                         rw_transition,
                                                                                         rw_transition_t)

    # Calculate implicit directed adjacency matrix.
    implicit_directed_matrix, dir_phi = get_implicit_directed_adjacency_matrix(stationary_distribution,
                                                                               rw_transition,
                                                                               rho)

    return implicit_combinatorial_matrix, com_phi, implicit_directed_matrix, dir_phi


def get_pagerank_with_teleportation(rw_transition, rw_transition_t, rho=0.2):
    number_of_nodes = rw_transition.shape[0]

    non_teleportation = 1-rho
    mv = lambda l, v: non_teleportation*l.dot(v) + (rho/number_of_nodes)*np.ones_like(v)
    teleport = lambda vec: mv(rw_transition_t, vec)

    rw_transition_operator = spla.LinearOperator(rw_transition.shape, matvec=teleport, dtype=np.float64)

    ####################################################################################################################
    # Form theta matrix
    ####################################################################################################################
    # Form stationary distribution diagonal matrix
    try:
        eigenvalue, stationary_distribution = spla.eigs(rw_transition_operator,
                                                        k=1,
                                                        which='LM',
                                                        return_eigenvectors=True)
    except spla.ArpackNoConvergence as e:
        print("ARPACK has not converged.")
        eigenvalue = e.eigenvalues
        stationary_distribution = e.eigenvectors

    return stationary_distribution


def get_implicit_combinatorial_adjacency_matrix(stationary_distribution, rw_transition, rw_transition_t):
    number_of_nodes = rw_transition.shape[0]

    sqrtp = stationary_distribution.flatten().real/stationary_distribution.sum()

    sqrtp = sp.sqrt(sqrtp)
    pi_matrix = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (pi_matrix.dot(rw_transition) + rw_transition_t.dot(pi_matrix))/2.0

    effective_adjacency_matrix = spsp.coo_matrix(spsp.csr_matrix(effective_adjacency_matrix))
    effective_adjacency_matrix.data = np.real(effective_adjacency_matrix.data)

    effective_adjacency_matrix = spsp.csr_matrix(effective_adjacency_matrix)

    return effective_adjacency_matrix, sqrtp


def get_implicit_directed_adjacency_matrix(stationary_distribution, rw_transition, rho):
    number_of_nodes = rw_transition.shape[0]

    sqrtp = stationary_distribution.flatten().real/stationary_distribution.sum()

    sqrtp = sp.sqrt(sqrtp)
    Q = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes) * rw_transition * spsp.spdiags(1.0/sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (1-rho)*(Q + Q.T) /2.0

    effective_adjacency_matrix = spsp.coo_matrix(spsp.csr_matrix(effective_adjacency_matrix))
    effective_adjacency_matrix.data = np.real(effective_adjacency_matrix.data)

    effective_adjacency_matrix = spsp.csr_matrix(effective_adjacency_matrix)

    return effective_adjacency_matrix, np.ones(number_of_nodes, dtype=np.float64)


def get_adjacency_matrix_via_combinatorial_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    ####################################################################################################################
    # Form random walk probability transition matrix
    ####################################################################################################################
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)
    rw_transition = rw_transition.tocoo()
    rw_transition_t = rw_transition.T.tocsr()
    rw_transition = rw_transition.tocsr()

    non_teleportation = 1-rho
    mv = lambda l, v: non_teleportation*l.dot(v) + (rho/number_of_nodes)*np.ones_like(v)
    teleport = lambda vec: mv(rw_transition_t, vec)

    rw_transition_operator = spla.LinearOperator(rw_transition.shape, matvec=teleport, dtype=np.float64)

    ####################################################################################################################
    # Form theta matrix
    ####################################################################################################################
    # Form stationary distribution diagonal matrix
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

    sqrtp = sp.sqrt(stationary_distribution)
    pi_matrix = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (pi_matrix.dot(rw_transition) + rw_transition_t.dot(pi_matrix))/2.0

    effective_adjacency_matrix = spsp.coo_matrix(spsp.csr_matrix(effective_adjacency_matrix))
    effective_adjacency_matrix.data = np.real(effective_adjacency_matrix.data)

    return effective_adjacency_matrix, sqrtp


def get_adjacency_matrix_via_directed_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    ####################################################################################################################
    # Form random walk probability transition matrix
    ####################################################################################################################
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)
    rw_transition = rw_transition.tocoo()
    rw_transition_t = rw_transition.T.tocsr()
    rw_transition = rw_transition.tocsr()

    non_teleportation = 1-rho
    mv = lambda l, v: non_teleportation*l.dot(v) + (rho/number_of_nodes)*np.ones_like(v)
    teleport = lambda vec: mv(rw_transition_t, vec)

    rw_transition_operator = spla.LinearOperator(rw_transition.shape, matvec=teleport, dtype=np.float64)

    ####################################################################################################################
    # Form theta matrix
    ####################################################################################################################
    # Form stationary distribution diagonal matrix
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

    sqrtp = sp.sqrt(stationary_distribution)
    Q = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes) * rw_transition * spsp.spdiags(1.0/sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (1-rho)*(Q + Q.T) /2.0

    effective_adjacency_matrix = spsp.coo_matrix(spsp.csr_matrix(effective_adjacency_matrix))
    effective_adjacency_matrix.data = np.real(effective_adjacency_matrix.data)

    return effective_adjacency_matrix, np.ones(number_of_nodes, dtype=np.float64)


def zhou_view_weight_calculation(transition_matrix_list, out_degree_list, weights):
    number_of_views = len(transition_matrix_list)
    number_of_nodes = transition_matrix_list[0].shape[0]

    actual_weights = np.empty((number_of_views, number_of_nodes), dtype=np.float64)
    for v in range(number_of_views):
        actual_weights[v, :] = out_degree_list[v]

    actual_weights[actual_weights > 0.0] = 1.0

    for n in range(number_of_nodes):
        row_sum = np.sum(actual_weights[:, n])
        if row_sum > 0.0:
            actual_weights[:, n] = actual_weights[:, n]/row_sum

    for v in range(number_of_views):
        actual_weights[v, :] = actual_weights[v, :]*weights[v]

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


def get_multiview_transition_matrix(adjacency_matrix_list, weights=None, method="zhou"):
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
            if weights.size != number_of_views:
                print("Number of adjacency matrices not equal to number of weights.")
                raise RuntimeError
            else:
                weights = weights/npla.norm(weights, "fro")

    # Make sure all matrices are in csr format.
    adjacency_matrix_list = (spsp.csr_matrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list)

    # Get natural random walk transition matrices.
    transition_tuple_list = [get_natural_random_walk_matrix(adjacency_matrix) for adjacency_matrix in adjacency_matrix_list]
    transition_matrix_list = [t[0] for t in transition_tuple_list]
    out_degree_list = [t[1] for t in transition_tuple_list]
    in_degree_list = [t[2] for t in transition_tuple_list]

    # Calculate actual weights for matrices.
    if method == "zhou":
        actual_weights = zhou_view_weight_calculation(transition_matrix_list, out_degree_list, weights)
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
