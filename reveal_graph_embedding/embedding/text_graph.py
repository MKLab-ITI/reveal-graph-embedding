__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as spsp
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex


def make_text_graph(user_lemma_matrix, dimensionality, metric, number_of_estimators, number_of_neighbors):
    user_lemma_matrix_tfidf = augmented_tf_idf(user_lemma_matrix)
    # print(user_lemma_matrix_tfidf.shape)
    if (user_lemma_matrix_tfidf.shape[0] <= dimensionality) or (user_lemma_matrix_tfidf.shape[1] <= dimensionality):
        X_svd = user_lemma_matrix_tfidf.toarray()
    else:
        X_svd = TruncatedSVD(n_components=dimensionality).fit_transform(user_lemma_matrix_tfidf)

    annoy_index = AnnoyIndex(X_svd.shape[1], metric=metric)

    for q in range(X_svd.shape[0]):
        annoy_index.add_item(q, X_svd[q, :])

    annoy_index.build(number_of_estimators)

    row = list()
    col = list()
    data = list()
    for q in range(X_svd.shape[0]):
        neighbors, distances = annoy_index.get_nns_by_item(q, number_of_neighbors, include_distances=True)

        row.extend([q] * number_of_neighbors)
        col.extend(neighbors)
        data.extend(distances)

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.array(data, dtype=np.float64)

    text_graph = spsp.coo_matrix((data,
                                  (row,
                                   col)),
                                 shape=(X_svd.shape[0],
                                        X_svd.shape[0]))
    text_graph = spsp.csr_matrix(text_graph)

    return text_graph


def augmented_tf_idf(attribute_matrix):
    """
    Performs augmented TF-IDF normalization on a bag-of-words vector representation of data.

    Augmented TF-IDF introduced in: Manning, C. D., Raghavan, P., & Schütze, H. (2008).
                                    Introduction to information retrieval (Vol. 1, p. 6).
                                    Cambridge: Cambridge university press.

    Input:  - attribute_matrix: A bag-of-words vector representation in SciPy sparse matrix format.

    Output: - attribute_matrix: The same matrix after augmented tf-idf normalization.
    """
    number_of_documents = attribute_matrix.shape[0]

    max_term_frequencies = np.ones(number_of_documents, dtype=np.float64)
    idf_array = np.ones(attribute_matrix.shape[1], dtype=np.float64)

    # Calculate inverse document frequency
    attribute_matrix = attribute_matrix.tocsc()
    for j in range(attribute_matrix.shape[1]):
        document_frequency = attribute_matrix.getcol(j).data.size
        if document_frequency > 1:
            idf_array[j] = np.log(number_of_documents/document_frequency)

    # Calculate maximum term frequencies for a user
    attribute_matrix = attribute_matrix.tocsr()
    for i in range(attribute_matrix.shape[0]):
        max_term_frequency = attribute_matrix.getrow(i).data
        if max_term_frequency.size > 0:
            max_term_frequency = max_term_frequency.max()
            if max_term_frequency > 0.0:
                max_term_frequencies[i] = max_term_frequency

    # Do augmented tf-idf normalization
    attribute_matrix = attribute_matrix.tocoo()
    attribute_matrix.data = 0.5 + np.divide(0.5*attribute_matrix.data, np.multiply((max_term_frequencies[attribute_matrix.row]), (idf_array[attribute_matrix.col])))
    attribute_matrix = attribute_matrix.tocsr()

    return attribute_matrix
