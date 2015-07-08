__author__ = 'Georgios Rizos (georgerizos@iti.gr)'
# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t

INT64 = np.int64
ctypedef np.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pagerank_limit_push(np.ndarray[FLOAT64_t, ndim=1] s,
                        np.ndarray[FLOAT64_t, ndim=1] r,
                        np.ndarray[FLOAT64_t, ndim=1] w_i,
                        np.ndarray[INT64_t, ndim=1] a_i,
                        long push_node,
                        double rho):
    """
    Performs a random step without a self-loop.
    """
    # Calculate the A and B quantities to infinity
    cdef double A_inf = rho*r[push_node]
    cdef double B_inf = (1-rho)*r[push_node]

    # Update approximate Pagerank and residual vectors
    s[push_node] += A_inf
    r[push_node] = 0.0

    cdef long i = 0
    cdef long out_degree = a_i.size

    # Update residual vector at push node's adjacent nodes
    for i in range(out_degree):
        r[a_i[i]] += B_inf * w_i[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pagerank_lazy_push(np.ndarray[FLOAT64_t, ndim=1] s,
                       np.ndarray[FLOAT64_t, ndim=1] r,
                       np.ndarray[FLOAT64_t, ndim=1] w_i,
                       np.ndarray[INT64_t, ndim=1] a_i,
                       long push_node,
                       double rho,
                       double laziness_factor):
    """
    Performs a random step with a self-loop.

    Introduced in: Andersen, R., Chung, F., & Lang, K. (2006, October).
                   Local graph partitioning using pagerank vectors.
                   In Foundations of Computer Science, 2006. FOCS'06. 47th Annual IEEE Symposium on (pp. 475-486). IEEE.
    """
    # Calculate the A, B and C quantities
    cdef double A = rho*r[push_node]
    cdef double B = (1-rho)*(1 - laziness_factor)*r[push_node]
    cdef double C = (1-rho)*laziness_factor*(r[push_node])

    # Update approximate Pagerank and residual vectors
    s[push_node] += A
    r[push_node] = C

    cdef long i = 0
    cdef long out_degree = a_i.size

    # Update residual vector at push node's adjacent nodes
    for i in range(out_degree):
        r[a_i[i]] += B * w_i[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cumulative_pagerank_difference_limit_push(np.ndarray[FLOAT64_t, ndim=1] s, np.ndarray[FLOAT64_t, ndim=1] r, np.ndarray[FLOAT64_t, ndim=1] w_i, np.ndarray[INT64_t, ndim=1] a_i, long push_node, double rho):
    """
    Performs a random step without a self-loop.
    """
    # Calculate the B quantity to infinity
    cdef double B_inf = (1-rho)*r[push_node]

    # Update approximate regularized commute and residual vectors
    r[push_node] = 0.0

    cdef long i = 0
    cdef long out_degree = a_i.size

    cdef double commute_probability

    # Update residual vector at push node's adjacent nodes
    for i in range(out_degree):
        commute_probability = B_inf * w_i[i]
        s[a_i[i]] += commute_probability
        r[a_i[i]] += commute_probability
