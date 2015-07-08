__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def calculate_entropy(p):
    """
    Calculates entropy in bits of a probability distribution.

    Input:  - p: A probability distribution in numpy array format.

    Output: - entropy: The statistical entropy of distribution p.
    """
    entropy = - np.sum(np.multiply(p, np.log2(p)))

    return entropy


def calculate_cross_entropy(p, q):
    """
    Calculates cross-entropy in bits of an approximate probability distribution q from p.

    Input:  - p: The source probability distribution in numpy array format.
            - q: An approximate probability distribution in numpy array format.

    Output: - cross_entropy: The cross-entropy of distribution q from p.
    """
    cross_entropy = - np.sum(np.multiply(p, np.log2(q)))

    return cross_entropy


def calculate_kl_divergence(p, q):
    """
    Calculates KL-divergence in bits of an approximate probability distribution q from p.

    Input:  - p: The source probability distribution in numpy array format.
            - q: An approximate probability distribution in numpy array format.

    Output: - p_q_kl_divergence: The KL-divergence of distribution q from p.
    """
    # Normalize p distribution.
    p /= p.sum()

    p_entropy = calculate_entropy(p)

    # Normalize q distribution.
    q /= q.sum()

    p_q_cross_entropy = calculate_cross_entropy(p, q)

    p_q_kl_divergence = p_q_cross_entropy - p_entropy

    return p_q_kl_divergence


def calculate_kl_divergence_stream(p, q_gen):
    # Normalize p distribution.
    p /= p.sum()

    p_entropy = calculate_entropy(p)

    for q in q_gen:
        # Normalize q distribution.
        q /= q.sum()

        p_q_cross_entropy = calculate_cross_entropy(p, q)

        p_q_kl_divergence = p_q_cross_entropy - p_entropy

        yield p_q_kl_divergence



