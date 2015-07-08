__author__ = 'Georgios Rizos (georgerizos@iti.gr)'
# cython: profile=True

from collections import deque
import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sparse

from reveal_graph_embedding.eps_randomwalk.cython_opt.push import pagerank_limit_push
from reveal_graph_embedding.eps_randomwalk.cython_opt.push import pagerank_lazy_push
from reveal_graph_embedding.eps_randomwalk.cython_opt.push import cumulative_pagerank_difference_limit_push

FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t


def fast_approximate_personalized_pagerank(np.ndarray[FLOAT64_t, ndim=1] s,
                                           np.ndarray[FLOAT64_t, ndim=1] r,
                                           np.ndarray w_i,
                                           np.ndarray a_i,
                                           np.ndarray[FLOAT64_t, ndim=1] out_degree,
                                           np.ndarray[FLOAT64_t, ndim=1] in_degree,
                                           long seed_node,
                                           double rho,
                                           double epsilon):
    """
    Calculates the approximate personalized PageRank starting from a seed node without self-loops.
    """
    number_of_nodes = a_i.size

    # Initialize the similarity matrix slice and the residual distribution
    # cdef np.ndarray[FLOAT64_t, ndim=1] s = np.zeros(number_of_nodes, dtype=FLOAT64)
    # cdef np.ndarray[FLOAT64_t, ndim=1] r = np.zeros(number_of_nodes, dtype=FLOAT64)
    r[seed_node] = 1.0

    # Initialize queue of nodes to be pushed
    pushable = deque()
    pushable.append(seed_node)

    # Do one push anyway
    push_node = pushable.popleft()

    pagerank_limit_push(s,
                               r,
                               w_i[push_node],
                               a_i[push_node],
                               push_node,
                               rho)
    cdef long number_of_push_operations = 1

    i = np.where(np.divide(r[a_i[push_node]], in_degree[a_i[push_node]]) >= epsilon)[0]
    if i.size > 0:
        pushable.extend(a_i[push_node][i])

    while len(pushable) > 0:
        # While there are nodes with large residual probabilities, push
        push_node = pushable.popleft()
        if r[push_node]/in_degree[push_node] >= epsilon:
            pagerank_limit_push(s,
                                       r,
                                       w_i[push_node],
                                       a_i[push_node],
                                       push_node,
                                       rho)
            number_of_push_operations += 1

            i = np.where(np.divide(r[a_i[push_node]], in_degree[a_i[push_node]]) >= epsilon)[0]
            if i.size > 0:
                pushable.extend(a_i[push_node][i])

    # Sparsify and return.
    # s_sparse = sparse.csr_matrix(s, shape=(1, number_of_nodes))
    # r_sparse = sparse.csr_matrix(r, shape=(1, number_of_nodes))

    return number_of_push_operations


def lazy_approximate_personalized_pagerank(np.ndarray[FLOAT64_t, ndim=1] s,
                                           np.ndarray[FLOAT64_t, ndim=1] r,
                                           np.ndarray w_i,
                                         np.ndarray a_i,
                                         np.ndarray[FLOAT64_t, ndim=1] out_degree,
                                         np.ndarray[FLOAT64_t, ndim=1] in_degree,
                                         long seed_node,
                                         double rho,
                                         double epsilon,
                                         double laziness_factor=0.5):
    """
    Calculates the approximate personalized PageRank starting from a seed node with self-loops.

    Introduced in: Andersen, R., Chung, F., & Lang, K. (2006, October).
                   Local graph partitioning using pagerank vectors.
                   In Foundations of Computer Science, 2006. FOCS'06. 47th Annual IEEE Symposium on (pp. 475-486). IEEE.
    """
    number_of_nodes = a_i.size

    # Initialize the similarity matrix slice and the residual distribution
    # cdef np.ndarray[FLOAT64_t, ndim=1] s = np.zeros(number_of_nodes, dtype=FLOAT64)
    # cdef np.ndarray[FLOAT64_t, ndim=1] r = np.zeros(number_of_nodes, dtype=FLOAT64)
    r[seed_node] = 1.0

    # Initialize queue of nodes to be pushed
    pushable = deque()
    pushable.append(seed_node)

    # Do one push anyway
    push_node = pushable.popleft()

    pagerank_lazy_push(s,
                              r,
                              w_i[push_node],
                              a_i[push_node],
                              push_node,
                              rho,
                              laziness_factor)
    cdef long number_of_push_operations = 1

    i = np.where(np.divide(r[a_i[push_node]], in_degree[a_i[push_node]]) >= epsilon)[0]
    if i.size > 0:
        pushable.extend(a_i[push_node][i])

    while r[push_node]/in_degree[push_node] >= epsilon:
        pagerank_lazy_push(s,
                                  r,
                                  w_i[push_node],
                                  a_i[push_node],
                                  push_node,
                                  rho,
                                  laziness_factor)
        number_of_push_operations += 1

    # While there are nodes with large residual probabilities, push
    while len(pushable) > 0:
        push_node = pushable.popleft()

        if r[push_node]/in_degree[push_node] >= epsilon:
            pagerank_lazy_push(s,
                                      r,
                                      w_i[push_node],
                                      a_i[push_node],
                                      push_node,
                                      rho,
                                      laziness_factor)
            number_of_push_operations += 1

            i = np.where(np.divide(r[a_i[push_node]], in_degree[a_i[push_node]]) >= epsilon)[0]
            if i.size > 0:
                pushable.extend(a_i[push_node][i])

        while r[push_node]/in_degree[push_node] >= epsilon:
            pagerank_lazy_push(s,
                                      r,
                                      w_i[push_node],
                                      a_i[push_node],
                                      push_node,
                                      rho,
                                      laziness_factor)
            number_of_push_operations += 1

    # Sparsify and return.
    # s_sparse = sparse.csr_matrix(s, shape=(1, number_of_nodes))
    # r_sparse = sparse.csr_matrix(r, shape=(1, number_of_nodes))

    return number_of_push_operations


def fast_approximate_cumulative_pagerank_difference(np.ndarray[FLOAT64_t, ndim=1] s,
                                                    np.ndarray[FLOAT64_t, ndim=1] r,
                                                    np.ndarray w_i,
                                                    np.ndarray a_i,
                                                    np.ndarray[FLOAT64_t, ndim=1] out_degree,
                                                    np.ndarray[FLOAT64_t, ndim=1] in_degree,
                                                    long seed_node,
                                                    double rho,
                                                    double epsilon):
    """
    Calculates cumulative PageRank difference probability starting from a seed node without self-loops.

    Inputs:  - w_i: A NumPy array of arrays of probability transition weights from the seed nodes to its adjacent nodes.
             - a_i: A NumPy array of arrays of the nodes adjacent to the seed node.
             - out_degree: A NumPy array of node out_degrees.
             - in_degree: A NumPy array of node in_degrees.
             - seed_node: The seed for the node-centric personalized PageRank.
             - rho: The restart probability. Usually set in [0.1, 0.2].
             - epsilon: The error threshold.

    Outputs: - s in 1xn: A sparse vector that contains the approximate absorbing random walk cumulative probabilities.
             - r in 1xn: A sparse vector that contains the residual probability distribution.
             - nop: The number of limit probability push operations performed.
    """
    number_of_nodes = a_i.size

    # Initialize the similarity matrix slice and the residual distribution
    # cdef np.ndarray[FLOAT64_t, ndim=1] s = np.zeros(number_of_nodes, dtype=FLOAT64)
    # cdef np.ndarray[FLOAT64_t, ndim=1] r = np.zeros(number_of_nodes, dtype=FLOAT64)
    s[seed_node] = 1.0
    r[seed_node] = 1.0

    # Initialize double-ended queue of nodes to be pushed
    pushable = deque()
    pushable.append(seed_node)

    # Do one push anyway
    push_node = pushable.popleft()

    cumulative_pagerank_difference_limit_push(s,
                                              r,
                                              w_i[push_node],
                                              a_i[push_node],
                                              push_node,
                                              rho)
    cdef long number_of_push_operations = 1

    i = np.where(np.divide(r[a_i[push_node]], in_degree[a_i[push_node]]) >= epsilon)[0]
    if i.size > 0:
        pushable.extend(a_i[push_node][i])

    # While there are nodes with large residual probabilities, push
    while len(pushable) > 0:
        push_node = pushable.popleft()

        # If the threshold is not satisfied, perform a push operation
        # Both this and the later check are needed, since the pushable queue may contain duplicates.
        if r[push_node]/in_degree[push_node] >= epsilon:
            cumulative_pagerank_difference_limit_push(s,
                                                      r,
                                                      w_i[push_node],
                                                      a_i[push_node],
                                                      push_node,
                                                      rho)
            number_of_push_operations += 1

            # Update pushable double-ended queue
            i = np.where(np.divide(r[a_i[push_node]], in_degree[a_i[push_node]]) >= epsilon)[0]
            if i.size > 0:
                pushable.extend(a_i[push_node][i])

    # Sparsify and return.
    # s_sparse = sparse.csr_matrix(s, shape=(1, number_of_nodes))
    # r_sparse = sparse.csr_matrix(r, shape=(1, number_of_nodes))

    return number_of_push_operations
