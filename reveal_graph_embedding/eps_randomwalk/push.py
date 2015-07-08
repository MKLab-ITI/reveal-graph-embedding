__author__ = 'Georgios Rizos (georgerizos@iti.gr)'


def pagerank_limit_push(s, r, w_i, a_i, push_node, rho):
    """
    Performs a random step without a self-loop.
    """
    # Calculate the A and B quantities to infinity
    A_inf = rho*r[push_node]
    B_inf = (1-rho)*r[push_node]

    # Update approximate Pagerank and residual vectors
    s[push_node] += A_inf
    r[push_node] = 0.0

    # Update residual vector at push node's adjacent nodes
    r[a_i] += B_inf * w_i


def pagerank_lazy_push(s, r, w_i, a_i, push_node, rho, lazy):
    """
    Performs a random step with a self-loop.

    Introduced in: Andersen, R., Chung, F., & Lang, K. (2006, October).
                   Local graph partitioning using pagerank vectors.
                   In Foundations of Computer Science, 2006. FOCS'06. 47th Annual IEEE Symposium on (pp. 475-486). IEEE.
    """
    # Calculate the A, B and C quantities
    A = rho*r[push_node]
    B = (1-rho)*(1 - lazy)*r[push_node]
    C = (1-rho)*lazy*(r[push_node])

    # Update approximate Pagerank and residual vectors
    s[push_node] += A
    r[push_node] = C

    # Update residual vector at push node's adjacent nodes
    r[a_i] += B * w_i


def cumulative_pagerank_difference_limit_push(s, r, w_i, a_i, push_node, rho):
    """
    Performs a random step without a self-loop.

    Inputs:  - s: A NumPy array that contains the approximate absorbing random walk cumulative probabilities.
             - r: A NumPy array that contains the residual probability distribution.
             - w_i: A NumPy array of probability transition weights from the seed nodes to its adjacent nodes.
             - a_i: A NumPy array of the nodes adjacent to the push node.
             - push_node: The node from which the residual probability is pushed to its adjacent nodes.
             - rho: The restart probability.

    Outputs: - s in 1xn: A NumPy array that contains the approximate absorbing random walk cumulative probabilities.
             - r in 1xn: A NumPy array that contains the residual probability distribution.
    """
    # Calculate the commute quantity
    commute = (1-rho)*r[push_node]

    # Update approximate regularized commute and residual vectors
    r[push_node] = 0.0

    # Update residual vector at push node's adjacent nodes
    commute_probabilities = commute * w_i
    s[a_i] += commute_probabilities
    r[a_i] += commute_probabilities
