__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import math
import numpy as np
import numpy.random as rand

from reveal_graph_embedding.common import get_file_row_generator


def get_folds_generator(node_label_matrix,
                        labelled_node_indices,
                        number_of_categories,
                        dataset_memory_folder,
                        percentage,
                        number_of_folds=10):
    """
    Read or form and store the seed nodes for training and testing.

    Inputs: - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
            - labelled_node_indices: A NumPy array containing the labelled node indices.
            - number_of_categories: The number of categories/classes in the learning.
            - memory_path: The folder where the results are stored.
            - percentage: The percentage of labelled samples that will be used for training.

    Output: - folds: A generator containing train/test set folds.
    """
    number_of_labeled_nodes = labelled_node_indices.size
    training_set_size = int(np.ceil(percentage*number_of_labeled_nodes/100))

    ####################################################################################################################
    # Read or generate folds
    ####################################################################################################################
    fold_file_path = dataset_memory_folder + "/folds/" + str(percentage) + "_folds.txt"
    train_list = list()
    test_list = list()
    if not os.path.exists(fold_file_path):
        with open(fold_file_path, "w") as fp:
            for trial in np.arange(number_of_folds):
                train, test = valid_train_test(node_label_matrix[labelled_node_indices, :],
                                               training_set_size,
                                               number_of_categories,
                                               trial)
                train = labelled_node_indices[train]
                test = labelled_node_indices[test]

                # Write test nodes
                row = [str(node) for node in test]
                row = "\t".join(row) + "\n"
                fp.write(row)

                # Write train nodes
                row = [str(node) for node in train]
                row = "\t".join(row) + "\n"
                fp.write(row)

                train_list.append(train)
                test_list.append(test)
    else:
        file_row_gen = get_file_row_generator(fold_file_path, "\t")

        for trial in np.arange(number_of_folds):
            # Read test nodes
            test = next(file_row_gen)
            test = [int(node) for node in test]
            test = np.array(test)

            # Read train nodes
            train = next(file_row_gen)
            train = [int(node) for node in train]
            train = np.array(train)

            train_list.append(train)
            test_list.append(test)

    folds = ((train, test) for train, test in zip(train_list, test_list))
    return folds


def generate_folds(node_label_matrix, labelled_node_indices, number_of_categories, percentage, number_of_folds=10):
    """
    Form the seed nodes for training and testing.

    Inputs:  - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - labelled_node_indices: A NumPy array containing the labelled node indices.
             - number_of_categories: The number of categories/classes in the learning.
             - percentage: The percentage of labelled samples that will be used for training.

    Output:  - folds: A generator containing train/test set folds.
    """
    number_of_labeled_nodes = labelled_node_indices.size
    training_set_size = int(np.ceil(percentage*number_of_labeled_nodes/100))

    ####################################################################################################################
    # Generate folds
    ####################################################################################################################
    train_list = list()
    test_list = list()
    for trial in np.arange(number_of_folds):
        train, test = valid_train_test(node_label_matrix[labelled_node_indices, :],
                                       training_set_size,
                                       number_of_categories,
                                       trial)
        train = labelled_node_indices[train]
        test = labelled_node_indices[test]

        train_list.append(train)
        test_list.append(test)

    folds = ((train, test) for train, test in zip(train_list, test_list))
    return folds


def valid_train_test(node_label_matrix, training_set_size, number_of_categories, random_seed=0):
    """
    Partitions the labelled node set into training and testing set, making sure one category exists in both sets.

    Inputs:  - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - training_set_size: The minimum required size for the training set.
             - number_of_categories: The number of categories/classes in the learning.
             - random_seed: A seed for numpy random.

    Outputs: - train_set: A NumPy array containing the training set node ids.
             - test_set: A NumPy array containing the testing set node ids.

    TODO: This function might benefit from some cleaning.
    """
    number_of_labelled_nodes = node_label_matrix.shape[0]

    # Randomize process
    np.random.seed(random_seed)
    perm = rand.permutation(number_of_labelled_nodes)
    node_label_matrix = node_label_matrix[perm, :]

    remaining_nodes = set(list(np.arange(number_of_labelled_nodes)))

    # Choose at least one user for any category for the training set
    train_ids = list()
    for c in np.arange(number_of_categories):
        not_found = True
        for t in remaining_nodes:
            indices = node_label_matrix.getrow(t).indices
            if c in indices:
                train_ids.append(t)
                not_found = False
                break
        if not_found:
            # This should never be reached
            print("Not found enough training data for training set.")
            raise RuntimeError

    # This is done to remove duplicates via the set() structure
    train_ids = set(train_ids)

    remaining_nodes.difference_update(train_ids)

    # This is done for speed
    train_ids = np.array(list(train_ids))

    # Choose at least one user for any category for the testing set
    test_ids = list()
    for c in np.arange(number_of_categories):
        not_found = True
        for t in remaining_nodes:
            indices = node_label_matrix.getrow(t).indices
            if c in indices:
                test_ids.append(t)
                not_found = False
                break
        if not_found:
            # This should never be reached
            print("Not found enough testing data for testing set.")
            raise RuntimeError

    # This is done to remove duplicates via the set() structure
    test_ids = set(test_ids)

    remaining_nodes.difference_update(test_ids)

    # Meet the training set size quota by adding new nodes
    if train_ids.size < training_set_size:
        # Calculate how many more nodes are needed
        remainder = training_set_size - train_ids.size

        # Find the nodes not currently in the training set
        more_train_ids = np.array(list(remaining_nodes))

        # Choose randomly among the nodes
        perm2 = rand.permutation(more_train_ids.size)
        more_train_ids = list(more_train_ids[perm2[:remainder]])
        train_ids = np.array(list(set(list(train_ids) + more_train_ids)))

    remaining_nodes.difference_update(set(list(train_ids)))

    # Form the test set
    test_ids.update(remaining_nodes)
    test_ids = np.array(list(set(list(test_ids))))

    return perm[train_ids], perm[test_ids]


def iterative_stratification(node_label_matrix, training_set_size, number_of_categories, random_seed=0):
    """
    Iterative data fold stratification/balancing for two folds.

    Based on: Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011).
              On the stratification of multi-label data.
              In Machine Learning and Knowledge Discovery in Databases (pp. 145-158).
              Springer Berlin Heidelberg.

    Inputs:  - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - training_set_size: The minimum required size for the training set.
             - number_of_categories: The number of categories/classes in the learning.
             - random_seed: A seed for numpy random.

    Outputs: - train_set: A NumPy array containing the training set node ids.
             - test_set: A NumPy array containing the testing set node ids.
    """
    number_of_labelled_nodes = node_label_matrix.shape[0]
    testing_set_size = number_of_labelled_nodes - training_set_size
    training_set_proportion = training_set_size/number_of_labelled_nodes
    testing_set_proportion = testing_set_size/number_of_labelled_nodes

    # Calculate the desired number of examples of each label at each subset.
    desired_label_number = np.zeros((2, number_of_categories), dtype=np.int64)
    node_label_matrix = node_label_matrix.tocsc()
    for j in range(number_of_categories):
        category_label_number = node_label_matrix.getcol(j).indices.size
        desired_label_number[0, j] = math.ceil(category_label_number*training_set_proportion)
        desired_label_number[1, j] = category_label_number - desired_label_number[0, j]

    train_ids = list()
    test_ids = list()

    append_train_id = train_ids.append
    append_test_id = test_ids.append

    # Randomize process
    np.random.seed(random_seed)
    while True:
        if len(train_ids) + len(test_ids) >= number_of_labelled_nodes:
            break
        # Find the label with the fewest (but at least one) remaining examples, breaking the ties randomly
        remaining_label_distribution = desired_label_number.sum(axis=0)
        min_label = np.min(remaining_label_distribution[np.where(remaining_label_distribution > 0)[0]])
        label_indices = np.where(remaining_label_distribution == min_label)[0]
        chosen_label = int(np.random.choice(label_indices, 1)[0])

        # Find the subset with the largest number of desired examples for this label,
        # breaking ties by considering the largest number of desired examples, breaking further ties randomly.
        fold_max_remaining_labels = np.max(desired_label_number[:, chosen_label])
        fold_indices = np.where(desired_label_number[:, chosen_label] == fold_max_remaining_labels)[0]
        chosen_fold = int(np.random.choice(fold_indices, 1)[0])

        # Choose a random example for the selected label.
        relevant_nodes = node_label_matrix.getcol(chosen_label).indices
        chosen_node = int(np.random.choice(np.setdiff1d(relevant_nodes,
                                                        np.union1d(np.array(train_ids),
                                                                   np.array(test_ids))),
                                           1)[0])
        if chosen_fold == 0:
            append_train_id(chosen_node)
            desired_label_number[0, node_label_matrix.getrow(chosen_node).indices] -= 1
        elif chosen_fold == 1:
            append_test_id(chosen_node)
            desired_label_number[1, node_label_matrix.getrow(chosen_node).indices] -= 1
        else:
            raise RuntimeError

    return np.array(train_ids), np.array(test_ids)
