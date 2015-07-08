__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import numpy as np

from reveal_graph_embedding.common import get_file_row_generator


def write_results(performance_measures, target_file_path):
    with open(target_file_path, "w") as fp:
        first_row = "*** Percentages:" + "\n"
        fp.write(first_row)

        second_row = "1\t2\t3\t4\t5\t6\t7\t8\t9\t10" + "\n"
        fp.write(second_row)

        fp.write("\n\n")
        write_average_score_row(fp, "Macro F1", performance_measures[4])

        fp.write("\n\n")
        write_average_score_row(fp, "Micro F1", performance_measures[5])


def store_performace_measures(performance_measures, memory_path, experiment_string):
    # Unpack performance measures
    # mean_macro_precision = performance_measures[0][0]
    # std_macro_precision = performance_measures[0][1]
    # mean_micro_precision = performance_measures[1][0]
    # std_micro_precision = performance_measures[1][1]
    # mean_macro_recall = performance_measures[2][0]
    # std_macro_recall = performance_measures[2][1]
    # mean_micro_recall = performance_measures[3][0]
    # std_micro_recall = performance_measures[3][1]
    # mean_macro_F1 = performance_measures[4][0]
    # std_macro_F1 = performance_measures[4][1]
    # mean_micro_F1 = performance_measures[5][0]
    # std_micro_F1 = performance_measures[5][1]
    F1 = performance_measures[6]

    number_of_categories = F1.shape[1]

    # Store average scores
    path = memory_path + "/scores/" + experiment_string + "_average_scores.txt"
    if not os.path.exists(path):
        with open(path, "w") as fp:
            write_average_score_row(fp, "Macro Precision", performance_measures[0])
            fp.write("\n\n")

            write_average_score_row(fp, "Micro Precision", performance_measures[1])
            fp.write("\n\n")

            write_average_score_row(fp, "Macro Recall", performance_measures[2])
            fp.write("\n\n")

            write_average_score_row(fp, "Micro Recall", performance_measures[3])
            fp.write("\n\n")

            write_average_score_row(fp, "Macro F1", performance_measures[4])
            fp.write("\n\n")

            write_average_score_row(fp, "Micro F1", performance_measures[5])

    # Store category-specific F scores
    path = memory_path + "/scores/" + experiment_string + "_F_scores.txt"
    if not os.path.exists(path):
        with open(path, "w") as fp:
            for c in np.arange(number_of_categories):
                row = list(F1[:, c])
                row = [str(score) for score in row]
                row = "\t".join(row) + "\n"
                fp.write(row)


def write_average_score_row(fp, score_name, scores):
    """
    Simple utility function that writes an average score row in a file designated by a file pointer.

    Inputs:  - fp: A file pointer.
             - score_name: What it says on the tin.
             - scores: An array of average score values corresponding to each of the training set percentages.
    """
    row = "--" + score_name + "--"
    fp.write(row)
    for vector in scores:
        row = list(vector)
        row = [str(score) for score in row]
        row = "\n" + "\t".join(row)
        fp.write(row)


def read_performance_measures(file_path, number=10):
    file_row_gen = get_file_row_generator(file_path, "\t")

    F1_macro_mean = np.zeros(number, dtype=np.float64)
    F1_macro_std = np.zeros(number, dtype=np.float64)
    F1_micro_mean = np.zeros(number, dtype=np.float64)
    F1_micro_std = np.zeros(number, dtype=np.float64)

    for r in range(18):
        file_row = next(file_row_gen)

    file_row = [float(score) for score in file_row]
    F1_macro_mean[:] = file_row

    file_row = next(file_row_gen)
    file_row = [float(score) for score in file_row]
    F1_macro_std[:] = file_row

    for r in range(3):
        file_row = next(file_row_gen)

    file_row = [float(score) for score in file_row]
    F1_micro_mean[:] = file_row

    file_row = next(file_row_gen)
    file_row = [float(score) for score in file_row]
    F1_micro_std[:] = file_row

    return F1_macro_mean, F1_macro_std, F1_micro_mean, F1_micro_std
