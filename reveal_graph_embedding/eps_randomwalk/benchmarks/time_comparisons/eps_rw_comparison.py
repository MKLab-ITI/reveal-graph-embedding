__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_user_classification.datautil.asu_read_data import read_adjacency_matrix
# from reveal_user_classification.datautil.snow_datautil.snow_read_data import read_adjacency_matrix
# from reveal_user_classification.datautil.insight_datautil.insight_read_data import read_adjacency_matrix
from reveal_user_classification.eps_randomwalk.benchmarks.benchmark_design import similarity_slice_benchmark,\
    aggregate_measurements, print_stats


########################################################################################################################
# Configure experiment.
########################################################################################################################
# Select dataset
DATASET = "YouTube_cc"  # Choices are: BlogCatalog, Flickr, YouTube_cc.

# Define approximate PageRank method parameters.
RHO_EFFECTIVE = 0.1
EPSILON = 0.00001

# Define laziness factor for the lazy PageRank.
LAZINESS = 0.5

NUMBER_OF_TRIALS = 2

########################################################################################################################
# Read data.
########################################################################################################################
# Define data path.
EDGE_LIST_PATH = get_raw_datasets_path() + "/ASU/" + DATASET + "/edges.csv"
adjacency_matrix = read_adjacency_matrix(EDGE_LIST_PATH, ',')
number_of_nodes = adjacency_matrix.shape[0]

profile_individual_stat_folder = get_memory_path() + "/ASU/" + DATASET + "/profile/individual"
profile_aggregate_stat_folder = get_memory_path() + "/ASU/" + DATASET + "/profile/aggregate"
profile_output_stat_folder = get_memory_path() + "/ASU/" + DATASET + "/profile/output"

########################################################################################################################
# Perform experiment.
########################################################################################################################
# profiler_stats = similarity_slice_benchmark(number_of_trials=NUMBER_OF_TRIALS,
#                                             adjacency_matrix=adjacency_matrix,
#                                             rho_effective=RHO_EFFECTIVE,
#                                             epsilon=EPSILON,
#                                             laziness_factor=LAZINESS,
#                                             profile_individual_stat_folder=profile_individual_stat_folder)

# aggregate_measurements(number_of_trials=NUMBER_OF_TRIALS,
#                        adjacency_matrix=adjacency_matrix,
#                        profile_individual_stat_folder=profile_individual_stat_folder,
#                        profile_aggregate_stat_folder=profile_aggregate_stat_folder)

print_stats(number_of_trials=NUMBER_OF_TRIALS,
            profile_aggregate_stat_folder=profile_aggregate_stat_folder,
            profile_output_stat_folder=profile_output_stat_folder)
