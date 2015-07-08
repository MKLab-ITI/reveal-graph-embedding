__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import cProfile as profile
import pstats
import gc
import os
import operator
import numpy as np
from reveal_user_classification.eps_randomwalk import similarity as similarity
from reveal_user_classification.eps_randomwalk.transition import get_natural_random_walk_matrix


# from reveal_user_classification.eps_randomwalk.cython_opt import similarity as csimilarity
# from reveal_user_classification.eps_randomwalk.cython_opt.similarity import *
# from reveal_user_classification.eps_randomwalk.cython_opt.push import *
from reveal_user_classification.embedding.arcte.arcte import calculate_epsilon_effective


def similarity_slice_benchmark(number_of_trials,
                               adjacency_matrix,
                               rho_effective,
                               epsilon,
                               laziness_factor,
                               profile_individual_stat_folder):
    """
    Compares the efficiency of approaches calculating similarity matrix slices.
    """
    adjacency_matrix = adjacency_matrix.tocsr()
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate random walk transition probability matrix.
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Calculate base communities(ego excluded) and out-degrees.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    # Calculate restart probability in the case of lazy PageRank.
    rho = (rho_effective * (1 - laziness_factor))/(1 - laziness_factor * rho_effective)

    s = np.zeros(number_of_nodes, dtype=np.float64)
    r = np.zeros(number_of_nodes, dtype=np.float64)

    iterate_nodes = np.where(out_degree != 0)[0]

    for trial in range(number_of_trials):
        gc.collect()
        print("Trial #", trial+1)
        for n, node in enumerate(iterate_nodes):

            if n % 500 == 0:
                gc.collect()

            epsilon_effective = calculate_epsilon_effective(None,
                                                            epsilon,
                                                            out_degree[node],
                                                            out_degree[adjacent_nodes[node]],
                                                            None)

            # Perform fast PageRank.
            s[:] = 0.0
            r[:] = 0.0
            filename = profile_individual_stat_folder + "/fpr_vertex_" + str(node) + "_trial_" + str(trial) + ".stats"
            profile.runctx("""similarity.fast_approximate_personalized_pagerank(s,
                                                                                r,
                                                                                base_transitions,
                                                                                adjacent_nodes,
                                                                                out_degree,
                                                                                in_degree,
                                                                                node,
                                                                                rho_effective,
                                                                                epsilon)""",
                           globals(),
                           {"s": s,
                            "r": r,
                            "base_transitions": base_transitions,
                            "adjacent_nodes": adjacent_nodes,
                            "out_degree": out_degree,
                            "in_degree": in_degree,
                            "node": node,
                            "rho_effective": rho_effective,
                            "epsilon": epsilon_effective},
                           filename=filename)

            # Perform lazy PageRank
            s[:] = 0.0
            r[:] = 0.0
            filename = profile_individual_stat_folder + "/lpr_vertex_" + str(node) + "_trial_" + str(trial) + ".stats"
            profile.runctx("""similarity.lazy_approximate_personalized_pagerank(s,
                                                                                r,
                                                                                base_transitions,
                                                                                adjacent_nodes,
                                                                                out_degree,
                                                                                in_degree,
                                                                                node,
                                                                                rho,
                                                                                epsilon)""",
                           globals(),
                           {"s": s,
                            "r": r,
                            "base_transitions": base_transitions,
                            "adjacent_nodes": adjacent_nodes,
                            "out_degree": out_degree,
                            "in_degree": in_degree,
                            "node": node,
                            "rho": rho,
                            "epsilon": epsilon_effective},
                           filename=filename)

            # Perform Regularized Commute-Time
            s[:] = 0.0
            r[:] = 0.0
            filename = profile_individual_stat_folder + "/fcprd_vertex_" + str(node) + "_trial_" + str(trial) + ".stats"
            profile.runctx("""similarity.fast_approximate_regularized_commute(s,
                                                                              r,
                                                                              base_transitions,
                                                                              adjacent_nodes,
                                                                              out_degree,
                                                                              in_degree,
                                                                              node,
                                                                              rho_effective,
                                                                              epsilon)""",
                           globals(),
                           {"s": s,
                            "r": r,
                            "base_transitions": base_transitions,
                            "adjacent_nodes": adjacent_nodes,
                            "out_degree": out_degree,
                            "in_degree": in_degree,
                            "node": node,
                            "rho_effective": rho_effective,
                            "epsilon": epsilon_effective},
                           filename=filename)


def aggregate_measurements(number_of_trials,
                           adjacency_matrix,
                           profile_individual_stat_folder,
                           profile_aggregate_stat_folder):
    adjacency_matrix = adjacency_matrix.tocsr()
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate random walk transition probability matrix.
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    iterate_nodes = np.where(out_degree != 0)[0]

    fpr_stats = pstats.Stats()
    lpr_stats = pstats.Stats()
    fcprd_stats = pstats.Stats()

    for trial in range(number_of_trials):
        for n in range(iterate_nodes.size):
            node = iterate_nodes[n]

            # Fast PR filename.
            filename = profile_individual_stat_folder + "/fpr_vertex_" + str(node) + "_trial_" + str(trial) + ".stats"
            fpr_stats.add(filename)
            # os.remove(filename)

            # Lazy PR filename.
            filename = profile_individual_stat_folder + "/lpr_vertex_" + str(node) + "_trial_" + str(trial) + ".stats"
            lpr_stats.add(filename)
            # os.remove(filename)

            # Fast CPRD filename.
            filename = profile_individual_stat_folder + "/fcprd_vertex_" + str(node) + "_trial_" + str(trial) + ".stats"
            fcprd_stats.add(filename)
            # os.remove(filename)

    fpr_stats.dump_stats(profile_aggregate_stat_folder + "/fpr_vertex.stats")
    lpr_stats.dump_stats(profile_aggregate_stat_folder + "/lpr_vertex.stats")
    fcprd_stats.dump_stats(profile_aggregate_stat_folder + "/fcprd_vertex.stats")


def print_stats(number_of_trials,
                profile_aggregate_stat_folder,
                profile_output_stat_folder):
    lpr_stats = pstats.Stats(profile_aggregate_stat_folder + "/lpr_vertex.stats")
    lpr_output_stats = extract_output_stats(number_of_trials, lpr_stats, ["pagerank_lazy_push", "lazy_approximate_personalized_pagerank"], None)

    fpr_stats = pstats.Stats(profile_aggregate_stat_folder + "/fpr_vertex.stats")
    fcprd_stats = pstats.Stats(profile_aggregate_stat_folder + "/fcprd_vertex.stats")

    fpr_output_stats = extract_output_stats(number_of_trials, fpr_stats, ["pagerank_limit_push", "fast_approximate_personalized_pagerank"], lpr_output_stats)
    fcprd_output_stats = extract_output_stats(number_of_trials, fcprd_stats, ["regularized_limit_commute", "fast_approximate_regularized_commute"], lpr_output_stats)

    target_file = profile_output_stat_folder + "/output_stats.txt"

    write_output_stats(target_file, lpr_output_stats, fpr_output_stats, fcprd_output_stats)


def extract_output_stats(number_of_trials, aggregate_stats_object, function_list, base_stats):
    output_stats = dict()

    aggregate_stats_object.strip_dirs().sort_stats(-1).print_stats()
    s = aggregate_stats_object.stats

    push_key = [tup for tup in s.keys() if operator.getitem(tup, 2) == function_list[0]][0]
    similarity_key = [tup for tup in s.keys() if operator.getitem(tup, 2) == function_list[1]][0]

    push_value = s[push_key]
    similarity_value = s[similarity_key]

    number_of_nodes = similarity_value[1]/number_of_trials

    output_stats["total_number_of_operations"] = push_value[1]/number_of_trials
    output_stats["total_time"] = push_value[3]/number_of_trials
    output_stats["per_node_number_of_operations"] = output_stats["total_number_of_operations"]/number_of_nodes
    output_stats["per_node_time"] = output_stats["total_time"]/number_of_nodes

    if base_stats is not None:
        output_stats["speedup_number_of_operations"] = percentage_increase(output_stats["total_number_of_operations"],
                                                                           base_stats["total_number_of_operations"])
        output_stats["speedup_time"] = percentage_increase(output_stats["total_time"],
                                                           base_stats["total_time"])

    print(output_stats)

    return output_stats


def percentage_increase(fast, lazy):
    return (np.abs(fast - lazy)/lazy)*100


def write_output_stats(target_file, lpr_output_stats, fpr_output_stats, fcprd_output_stats):
    with open(target_file, "w") as fp:
        row = "*** Lazy PageRank ***" + "\n"
        fp.write(row)

        row = "total nop:" + "\t" + str(lpr_output_stats["total_number_of_operations"]) + "\n"
        fp.write(row)
        row = "total time:" + "\t" + str(lpr_output_stats["total_time"]) + "\n"
        fp.write(row)
        row = "per node nop:" + "\t" + str(lpr_output_stats["per_node_number_of_operations"]) + "\n"
        fp.write(row)
        row = "per node time:" + "\t" + str(lpr_output_stats["per_node_time"]) + "\n"
        fp.write(row)

        row = "\n"
        fp.write(row)

        row = "*** Fast PageRank ***" + "\n"
        fp.write(row)

        row = "total nop:" + "\t" + str(fpr_output_stats["total_number_of_operations"]) + "\n"
        fp.write(row)
        row = "total time:" + "\t" + str(fpr_output_stats["total_time"]) + "\n"
        fp.write(row)
        row = "per node nop:" + "\t" + str(fpr_output_stats["per_node_number_of_operations"]) + "\n"
        fp.write(row)
        row = "per node time:" + "\t" + str(fpr_output_stats["per_node_time"]) + "\n"
        fp.write(row)
        row = "speedup nop:" + "\t" + str(fpr_output_stats["speedup_number_of_operations"]) + "\n"
        fp.write(row)
        row = "speedup time:" + "\t" + str(fpr_output_stats["speedup_time"]) + "\n"
        fp.write(row)

        row = "\n"
        fp.write(row)

        row = "*** Fast Cumulative PageRank Differences ***" + "\n"
        fp.write(row)

        row = "total nop:" + "\t" + str(fcprd_output_stats["total_number_of_operations"]) + "\n"
        fp.write(row)
        row = "total time:" + "\t" + str(fcprd_output_stats["total_time"]) + "\n"
        fp.write(row)
        row = "per node nop:" + "\t" + str(fcprd_output_stats["per_node_number_of_operations"]) + "\n"
        fp.write(row)
        row = "per node time:" + "\t" + str(fcprd_output_stats["per_node_time"]) + "\n"
        fp.write(row)
        row = "speedup nop:" + "\t" + str(fcprd_output_stats["speedup_number_of_operations"]) + "\n"
        fp.write(row)
        row = "speedup time:" + "\t" + str(fcprd_output_stats["speedup_time"]) + "\n"
        fp.write(row)

        row = "\n"
        fp.write(row)


# def similarity_slice_benchmark(adjacency_matrix, alpha_effective, epsilon, laziness_factor):
#     """
#     Compares the efficiency of approaches calculating similarity matrix slices.
#     """
#     adjacency_matrix = adjacency_matrix.tocsr()
#     number_of_nodes = adjacency_matrix.shape[0]
#
#     # Calculate random walk transition probability matrix
#     rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)
#
#     # Calculate base communities(ego excluded) and out-degrees
#     adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
#     base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
#     for n in np.arange(number_of_nodes):
#         rw_transition_row = rw_transition.getrow(n)
#         adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
#         base_transitions[n] = rw_transition_row.data
#
#     # Calculate restart probability in the case of lazy PageRank
#     alpha = (alpha_effective * (1 - laziness_factor))/(1 - laziness_factor * alpha_effective)
#
#     number_of_operations_fast = np.zeros(number_of_nodes, dtype=np.int32)
#     execution_time_fast = np.zeros(number_of_nodes, dtype=np.float64)
#
#     number_of_operations_lazy = np.zeros(number_of_nodes, dtype=np.int32)
#     execution_time_lazy = np.zeros(number_of_nodes, dtype=np.float64)
#
#     number_of_operations_rct = np.zeros(number_of_nodes, dtype=np.int32)
#     execution_time_rct = np.zeros(number_of_nodes, dtype=np.float64)
#     s = np.zeros(number_of_nodes, dtype=np.float64)
#     r = np.zeros(number_of_nodes, dtype=np.float64)
#     for n in np.arange(number_of_nodes):
#         print(n)
#
#         # Perform fast PageRank
#         s[:] = 0.0
#         r[:] = 0.0
#         start_time = time.process_time()
#         nop = csimilarity.fast_approximate_personalized_pagerank(s,
#                                                                  r,
#                                                                  base_transitions,
#                                                                  adjacent_nodes,
#                                                                  out_degree,
#                                                                  in_degree,
#                                                                  n,
#                                                                  alpha_effective,
#                                                                  epsilon)
#         elapsed_time = time.process_time() - start_time
#
#         number_of_operations_fast[n] = nop
#         execution_time_fast[n] = elapsed_time
#
#         # Perform lazy PageRank
#         s[:] = 0.0
#         r[:] = 0.0
#         start_time = time.process_time()
#         nop = csimilarity.lazy_approximate_personalized_pagerank(s,
#                                                                  r,
#                                                                  base_transitions,
#                                                                  adjacent_nodes,
#                                                                  out_degree,
#                                                                  in_degree,
#                                                                  n,
#                                                                  alpha,
#                                                                  epsilon)
#         elapsed_time = time.process_time() - start_time
#
#         number_of_operations_lazy[n] = nop
#         execution_time_lazy[n] = elapsed_time
#
#         # Perform Regularized Commute-Time
#         s[:] = 0.0
#         r[:] = 0.0
#         start_time = time.process_time()
#         nop = csimilarity.fast_approximate_regularized_commute(s,
#                                                                r,
#                                                                base_transitions,
#                                                                adjacent_nodes,
#                                                                out_degree,
#                                                                in_degree,
#                                                                n,
#                                                                alpha_effective,
#                                                                epsilon)
#         elapsed_time = time.process_time() - start_time
#
#         number_of_operations_rct[n] = nop
#         execution_time_rct[n] = elapsed_time
#
#     return number_of_operations_fast, execution_time_fast, number_of_operations_lazy, execution_time_lazy, number_of_operations_rct, execution_time_rct
#
#
# def cython_optimization_benchmark(adjacency_matrix, alpha, epsilon):
#     """
#     Compares the efficiency of approaches calculating similarity matrix slices.
#     """
#     adjacency_matrix = adjacency_matrix.tocsr()
#     number_of_nodes = adjacency_matrix.shape[0]
#
#     # Calculate random walk transition probability matrix
#     rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)
#
#     # Calculate base communities(ego excluded) and out-degrees
#     adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
#     base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
#     for n in np.arange(number_of_nodes):
#         rw_transition_row = rw_transition.getrow(n)
#         adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
#         base_transitions[n] = rw_transition_row.data
#
#     execution_time_non_opt = np.zeros(number_of_nodes, dtype=np.float64)
#
#     execution_time_opt = np.zeros(number_of_nodes, dtype=np.float64)
#
#     s = np.zeros(number_of_nodes, dtype=np.float64)
#     r = np.zeros(number_of_nodes, dtype=np.float64)
#     for n in np.arange(number_of_nodes):
#         print(n)
#         # Perform cython-optimized absorbing cumulative random walk probability.
#         s[:] = 0.0
#         r[:] = 0.0
#         start_time = time.process_time()
#         nop = csimilarity.fast_approximate_regularized_commute(s,
#                                                                r,
#                                                                base_transitions,
#                                                                      adjacent_nodes,
#                                                                      out_degree,
#                                                                      in_degree,
#                                                                      n,
#                                                                      alpha,
#                                                                      epsilon)
#         elapsed_time = time.process_time() - start_time
#
#         execution_time_opt[n] = elapsed_time
#
#         # Perform unoptimized absorbing cumulative random walk probability.
#         s[:] = 0.0
#         r[:] = 0.0
#         start_time = time.process_time()
#         nop = similarity.fast_approximate_regularized_commute(s,
#                                                               r,
#                                                               base_transitions,
#                                                                     adjacent_nodes,
#                                                                     out_degree,
#                                                                     in_degree,
#                                                                     n,
#                                                                     alpha,
#                                                                     epsilon)
#         elapsed_time = time.process_time() - start_time
#
#         execution_time_non_opt[n] = elapsed_time
#
#     return execution_time_non_opt, execution_time_opt
