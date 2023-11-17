###############################################################################
#   _________________________________________________
# ((                                                 ))
#  ))     LP's CDC Optimization Toolbox             ((
# ((                                                 ))
#   -------------------------------------------------
#
# Doing unconstrained optimization right now
#
# After running Script_CDCOptimization_FinerGrid_AdditionalReps.py,
#   this script does analysis on aggregated datasets to get unconstrained
#   lowest_cost and other performance statistics such as average days in
#   each stage and ICU patient days violation. Also generates figures.
#
# TODO: still reforming this script file so others can
#   run it on their own set of policies and simulation data
#   and so that a set of policies is not hardcoded
# See subsection labeled "TOGGLES AND OPTIONS" to configure for
#   different computers -- can specify folder names and file names in
#   this subsection -- must have .csv files for policies'
#   performance measures of days in stage 2, days in stage 3,
#   and ICU violation patient days across 4 peaks
#
# Note on "all peaks" vs "across peaks" -- tried to create a distinction
#   but hopefully this does not add additional confusion --
#   "all peaks" refers to dataframes that have 1200 rows -- 300 reps x 4 peaks
#   and just have the data from all peaks concatenated together --
#   "across peaks" refers to dataframes that take into account all 4 peaks.
# feasibility_across_peaks has 300 rows and a row is True if for each
#   of the 4 peaks, the ith replication was above ICU capacity
# full_df_across_peaks.csv and the corresponding dataframe has performance
#   measures that are AVERAGED across all 4 peaks -- except for feasibility,
#   which is computed as
#       proportion of sample paths i such that replication i was feasible
#       (below ICU capacity) for all 4 peaks

###############################################################################

#######################################
############## IMPORTS ################
#######################################

import pandas as pd
import numpy as np
import itertools
import matplotlib

from pathlib import Path

base_path = Path(__file__).parent

###############################################################################

###################################################
############## TOGGLES AND OPTIONS ################
###################################################

# Change this to the name of a folder with aggregated files to parse
aggregated_files_folder_name = "3000I"

# Change this to prefix of the aggregated files to parse
# Files to parse are
#   aggregated_files_prefix + str(peak) + str(suffix)
# where peak values are 0, 1, 2, 3
# and where suffix values are "_stage2_days.csv", "_stage3_days.csv",
#   or "_ICU_violation_patient_days.csv"
# For each peak and for each performance measure (suffix),
#   each aggregated .csv file has N rows and K columns,
#   where N is the number of simulation replications and
#   K is the number of policies simulated
# Each column name is the ID of a policy -- these columns are not
#   in order -- ID X corresponds to policy X -- the mapping
#   of IDs to actual policies occurs in the simulation file
#   and is recreated here as well -- make sure to preserve the
#   mapping order so that IDs are matched appropriately to
#   the actual policy
aggregated_files_prefix = "8000reps_aggregated_peak"

full_df_files_prefix = "8000reps_"

# Toggle for whether or not to create reorganized dfs
#   -- these dfs contain columns for thresholds values
#   as well as summary statistics -- more intuitive way of analyzing
#   simulation data and also amenable to simple spreadsheet analysis
# If full_df_peak0.csv, full_df_peak1.csv, full_df_peak2.csv,
#   full_df_peak3.csv, full_df_across_peaks.csv already exist,
#   can set this to False
# TODO: There is some bug that KN does not run unless we
#   create_reorganized_dfs? Maybe it has to do with the column names
#   being strings vs integers...
create_reorganized_dfs = True

# Set to True if have dataframe for subset of 7452 policies
subset_of_policies = True

# Toggle whether or not to run Rinott
run_Rinott = False

# Toggle whether or not to run KN
run_KN = True

reps_offset = 0
num_reps_per_peak = 8000

# Switch to True and see policy generation under "if coordinate_subset:"
# Basically when doing coordinates, the indexing gets messed up, so this
#   was my fix...
coordinate_subset = False

###############################################################################

# Setting NA color in heatmaps

matplotlib.cm.get_cmap("magma_r").set_bad("white")
matplotlib.cm.get_cmap("cool").set_bad("white")

###############################################################################

# This is an identical copy of the method from Tools_Optimization.py
#   so that this file can run standalone without importing simulation modules
def thresholds_generator(stage2_info, stage3_info, stage4_info, stage5_info):
    """
    Creates a list of 5-tuples, where each 5-tuple has the form
        (-1, t2, t3, t4, t5) with 0 <= t2 <= t3 <= t4 <= t5 < inf.
    The possible values t2, t3, t4, and t5 can take come from
        the grid generated by stage2_info, stage3_info, stage4_info,
        and stage5_info respectively.
    Stage 1 threshold is always fixed to -1 (no social distancing).

    :param stage2_info: [3-tuple] with elements corresponding to
        start point, end point, and step size
        for candidate values for stage 2
    :param stage3_info: same as above but for stage 3
    :param stage4_info: same as above but for stage 4
    :param stage5_info: same as above but for stage 5
    :return: [array] of 5-tuples
    """

    # Create an array (grid) of potential thresholds for each stage
    stage2_options = np.arange(stage2_info[0], stage2_info[1], stage2_info[2])
    stage3_options = np.arange(stage3_info[0], stage3_info[1], stage3_info[2])
    stage4_options = np.arange(stage4_info[0], stage4_info[1], stage4_info[2])
    stage5_options = np.arange(stage5_info[0], stage5_info[1], stage5_info[2])

    # Using Cartesian products, create a list of 5-tuple combos
    stage_options = [stage2_options, stage3_options, stage4_options, stage5_options]
    candidate_feasible_combos = []
    for combo in itertools.product(*stage_options):
        candidate_feasible_combos.append((-1,) + combo)

    # Eliminate 5-tuples that do not satisfy monotonicity constraint
    # However, ties in thresholds are allowed
    feasible_combos = []
    for combo in candidate_feasible_combos:
        if np.all(np.diff(combo) >= 0):
            feasible_combos.append(combo)

    return feasible_combos


def write_non_surge_CDC_policy_ID_mapping_csv(list_of_tuples_for_policies,
                                              filename):
    '''
    Important note: this is for non-surge policies only
        (e.g. policies in which the case threshold (which triggers a "surge")
        is inf, so we do not consider the "surge" thresholds and only
        consider the "non-surge" thresholds)

    From list of tuples (with each tuple containing non-surge
        information characterizing a CDC-style policy),
        create a .csv file with 6 columns:
            "ID" (policy ID)
            "hosp1" (hospital admits blue-to-yellow non-surge threshold)
            "hosp2" (hospital admits yellow-to-red non-surge threshold)
            "beds1" (staffed beds blue-to-yellow non-surge threshold)
            "beds2" (staffed beds yellow-to-red non-surge threshold)

    i.e. creates a .csv file with a mapping from an integer ID
        to the information characterizing that policy -- therefore
        we can refer to a policy simply as "policy X" instead of
        needing to explicitly note all of its threshold values

    Does not return anything -- simply writes .csv file to
        working directory

    :param list_of_tuples_for_policies [list of tuples] -- list of
        3-tuples with the following structure:
        1st element in tuple is a scalar corresponding to
            case thresholds
        2nd element in tuple is dictionary with keys
            "non_surge" and "surge" -- respective values
            are 3-tuples -- first element in tuple is always
            -1, second element is a nonnegative integer
            corresponding to blue-to-yellow threshold for
            hospital admits, third element is a nonnegative integer
            corresponding to yellow-to-red threshold for hospital
            admits trigger
        3rd element in tuple is analagous to 2nd element in tuple but
            for percent staffed beds trigger

        e.g. example of element (3-tuple) in list
        (inf,
        {'non_surge': (-1, 0, 5), 'surge': (-1, -1, 0)},
        {'non_surge': (inf, inf, inf), 'surge': (-1, -1, inf)})
        corresponds to a CDC-style policy with no case threshold trigger,
            a hospital-admits blue-to-yellow threshold of 0 and
            yellow-to-red threshold of 5, and no staffed beds trigger

    :param filename [str ending in ".csv"] -- name of .csv file
        into which we write mapping information
    '''

    non_surge_hosp_adm_first_thresholds = []
    non_surge_staffed_first_thresholds = []

    non_surge_hosp_adm_second_thresholds = []
    non_surge_staffed_second_thresholds = []

    for i in range(len(list_of_tuples_for_policies)):
        non_surge_hosp_adm_first_thresholds.append(list_of_tuples_for_policies[i][1]["non_surge"][1])
        non_surge_staffed_first_thresholds.append(list_of_tuples_for_policies[i][2]["non_surge"][1])

        non_surge_hosp_adm_second_thresholds.append(list_of_tuples_for_policies[i][1]["non_surge"][2])
        non_surge_staffed_second_thresholds.append(list_of_tuples_for_policies[i][2]["non_surge"][2])

    mapping_df = pd.DataFrame({"ID": np.arange(len(list_of_tuples_for_policies)),
                               "hosp1": non_surge_hosp_adm_first_thresholds,
                               "hosp2": non_surge_hosp_adm_second_thresholds,
                               "beds1": non_surge_staffed_first_thresholds,
                               "beds2": non_surge_staffed_second_thresholds})

    mapping_df.to_csv(filename)

###############################################################################

# Helper functions for computing unconstrained objective function
#   and finding lowest_cost wrt unconstrained objective function

def make_weighted_sum(df1, df2, df3, df4, weight1, weight2, weight3, weight4):
    '''
    Return a DataFrame that is the weighted sum of DataFrames
        df1, df2, df3, df4 with respective weights weight1, weight2, weight3, weight4

    Adds dataframes element-wise -- df1, df2, df3, df4 should all be the same size
        and have the same columns

    We use this subroutine to build the cost / objective function that sums
        the weighted cost of days in each stage and the cost (penalized) of
        ICU patient days over capacity -- in this context, the each column
        df1, df2, df3, and df4 is a string of an integer corresponding to a policy ID
    '''
    weighted_df1 = df1 * weight1
    weighted_df2 = df2 * weight2
    weighted_df3 = df3 * weight3
    weighted_df4 = df4 * weight4
    summed_df = weighted_df1.add(weighted_df2, axis=1).add(weighted_df3, axis=1).add(weighted_df4, axis=1)
    return summed_df


def find_lowest_cost_weighted_sum(summed_df):
    '''
    Returns the lowest average value (cost) from summed_df
        as well as the policy ID corresponding to this value
    '''
    min_val = summed_df.mean().iloc[summed_df.mean().argmin()]
    min_ix = int(summed_df.mean().index[summed_df.mean().argmin()])
    return min_val, min_ix

###############################################################################

# Creates list of policies (not policy objects like in the
#   simulation, simply list of tuples containing info on thresholds
# Each item in list is 3-tuple with case thresholds,
#   hospital admissions thresholds, staffed beds thresholds
# Order of list: single-indicator policies (hospital admits only
#   then staffed beds only), two-indicator policies, CDC policy last

# This example creates 7452 policies

# Turn off case_threshold!
case_threshold = np.inf

policies = []

non_surge_hosp_adm_thresholds_array = thresholds_generator((-1, 0, 1),
                                                           (-1, 0, 1),
                                                           (0, 11, 1),
                                                           (5, 21, 1))

non_surge_staffed_thresholds_array = thresholds_generator((-1, 0, 1),
                                                          (-1, 0, 1),
                                                          (0, .05, .01),
                                                          (0, .11, .01))

for non_surge_hosp_adm_thresholds in non_surge_hosp_adm_thresholds_array:
    hosp_adm_thresholds = {"non_surge": (non_surge_hosp_adm_thresholds[2],
                                         non_surge_hosp_adm_thresholds[3],
                                         non_surge_hosp_adm_thresholds[4]),
                           "surge": (-1,
                                     -1,
                                     non_surge_hosp_adm_thresholds[3])}
    staffed_thresholds = {"non_surge": (np.inf,
                                        np.inf,
                                        np.inf),
                          "surge": (-1,
                                    -1,
                                    np.inf)}
    policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:
    hosp_adm_thresholds = {"non_surge": (np.inf,
                                         np.inf,
                                         np.inf),
                           "surge": (-1,
                                     -1,
                                     np.inf)}
    staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                        non_surge_staffed_thresholds[3],
                                        non_surge_staffed_thresholds[4]),
                          "surge": (-1,
                                    -1,
                                    non_surge_staffed_thresholds[3])}
    policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

# 2-indicator policies

for non_surge_hosp_adm_thresholds in non_surge_hosp_adm_thresholds_array:

    hosp_adm_thresholds = {"non_surge": (non_surge_hosp_adm_thresholds[2],
                                         non_surge_hosp_adm_thresholds[3],
                                         non_surge_hosp_adm_thresholds[4]),
                           "surge": (-1,
                                     -1,
                                     non_surge_hosp_adm_thresholds[3])}

    for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:
        staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                            non_surge_staffed_thresholds[3],
                                            non_surge_staffed_thresholds[4]),
                              "surge": (-1,
                                        -1,
                                        non_surge_staffed_thresholds[3])}
        policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

# Adding CDC policy!

hosp_adm_thresholds = {"non_surge": (-1,
                                     10,
                                     20),
                       "surge": (-1,
                                 -1,
                                 10)}

staffed_thresholds = {"non_surge": (-1,
                                    0.1,
                                    0.15),
                      "surge": (-1,
                                -1,
                                0.15)}

policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

if coordinate_subset:
    policies = []

    case_threshold = np.inf

    staffed_thresholds = {"non_surge": (-1,
                                        np.inf,
                                        np.inf),
                          "surge": (-1,
                                    -1,
                                    np.inf)}

    for first_threshold in np.arange(15):
        hosp_adm_thresholds = {"non_surge": (-1,
                                             first_threshold,
                                             14),
                               "surge": (-1,
                                         -1,
                                         first_threshold)}

        policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

    for second_threshold in np.arange(1, 21):
        hosp_adm_thresholds = {"non_surge": (-1,
                                             1,
                                             second_threshold),
                               "surge": (-1,
                                         -1,
                                         1)}

        policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

write_non_surge_CDC_policy_ID_mapping_csv(policies, "CDC_optimization_mapping.csv")

###############################################################################




###############################################################################

# Read data from all 4 peaks and store into dictionaries
# Also create across-peak dataframes

peaks_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]

stage1_days_dict = {}
stage2_days_dict = {}
stage3_days_dict = {}
ICU_violation_patient_days_dict = {}

ICU_violation_patient_days_per_peak = []
stage1_days_dfs_per_peak = []
stage2_days_dfs_per_peak = []
stage3_days_dfs_per_peak = []

for peak in np.arange(3):
    stage2_days_df = pd.read_csv(
        base_path / aggregated_files_folder_name /
        (aggregated_files_prefix + str(peak) + "_stage2_days.csv"),
        index_col=0)
    stage2_days_dict[str(peak)] = stage2_days_df[reps_offset:reps_offset + num_reps_per_peak]
    stage2_days_dfs_per_peak.append(stage2_days_df[reps_offset:reps_offset + num_reps_per_peak])

    stage3_days_df = pd.read_csv(
        base_path / aggregated_files_folder_name /
        (aggregated_files_prefix + str(peak) + "_stage3_days.csv"),
        index_col=0)
    stage3_days_dict[str(peak)] = stage3_days_df[reps_offset:reps_offset + num_reps_per_peak]
    stage3_days_dfs_per_peak.append(stage3_days_df[reps_offset:reps_offset + num_reps_per_peak])

    ICU_violation_patient_days_df = pd.read_csv(
        base_path / aggregated_files_folder_name /
        (aggregated_files_prefix + str(peak) + "_ICU_violation_patient_days.csv"),
        index_col=0)
    ICU_violation_patient_days_dict[str(peak)] = ICU_violation_patient_days_df[
                                                 reps_offset:reps_offset + num_reps_per_peak]
    ICU_violation_patient_days_per_peak.append(
        ICU_violation_patient_days_df[reps_offset:reps_offset + num_reps_per_peak])

    # Also compute number of days in stage 1 from number of days in stages 2 and 3
    stage1_days_df = peaks_end_times[peak] - peaks_start_times[peak] - stage2_days_df.add(stage3_days_df, axis=1)
    stage1_days_dict[str(peak)] = stage1_days_df[reps_offset:reps_offset + num_reps_per_peak]
    stage1_days_dfs_per_peak.append(stage1_days_df[reps_offset:reps_offset + num_reps_per_peak])

stage1_days_all_peaks = pd.concat(stage1_days_dfs_per_peak)
stage2_days_all_peaks = pd.concat(stage2_days_dfs_per_peak)
stage3_days_all_peaks = pd.concat(stage3_days_dfs_per_peak)
ICU_violation_patient_days_all_peaks = pd.concat(ICU_violation_patient_days_per_peak)

# If doing 4 peaks, need to add 4th peak
feasibility_across_peaks = ((ICU_violation_patient_days_all_peaks == 0)[:num_reps_per_peak] &
                            (ICU_violation_patient_days_all_peaks == 0)[num_reps_per_peak:2 * num_reps_per_peak] &
                            (ICU_violation_patient_days_all_peaks == 0)[2 * num_reps_per_peak:3 * num_reps_per_peak])

feasibility_across_peaks_mean = feasibility_across_peaks.mean()

###############################################################################

# Find unconstrained optimization lowest_cost across peaks
#   for a given ICU violation patient days penalty w
# Test different values of w -- choose w so that the unconstrained
#   lowest_cost roughly has 95% feasibility

# Even if not searching for a lowest_cost penalty, need to
#   run the following to generate cost_df_all_peaks
#   (with lowest_cost penalty) for further analysis

for w in [503]:
    cost_dfs_per_peak = []

    for peak in np.arange(3):
        cost_df = make_weighted_sum(stage1_days_dict[str(peak)],
                                    stage2_days_dict[str(peak)],
                                    stage3_days_dict[str(peak)],
                                    ICU_violation_patient_days_dict[str(peak)],
                                    1, 10, 100, w)
        cost_dfs_per_peak.append(cost_df)

    cost_df_all_peaks = pd.concat(cost_dfs_per_peak)
    lowest_cost_val, lowest_cost_ix = find_lowest_cost_weighted_sum(cost_df_all_peaks)
    lowest_cost_policy = policies[lowest_cost_ix]

    print(w, lowest_cost_policy[1]["non_surge"], lowest_cost_policy[2]["non_surge"],
          ICU_violation_patient_days_all_peaks.mean()[str(lowest_cost_ix)],
          feasibility_across_peaks_mean[str(lowest_cost_ix)])

# 502 (-1, 1, 15) (inf, inf, inf) 0.44666666666666666 0.9
# 503 (-1, 0, 14) (inf, inf, inf) 0.06666666666666667 0.97

breakpoint()

###############################################################################

# GENERATE DATAFRAME WHERE COLUMNS ARE SORTED IN ORDER OF POLICY ID
# AND ALSO WE HAVE COLUMNS FOR HOSP1, HOSP2, BEDS1, BEDS2

non_surge_hosp_adm_first_thresholds = []
non_surge_staffed_first_thresholds = []

for i in range(len(policies)):
    non_surge_hosp_adm_first_thresholds.append(policies[i][1]["non_surge"][1])
    non_surge_staffed_first_thresholds.append(policies[i][2]["non_surge"][1])

non_surge_hosp_adm_second_thresholds = []
non_surge_staffed_second_thresholds = []

for i in range(len(policies)):
    non_surge_hosp_adm_second_thresholds.append(policies[i][1]["non_surge"][2])
    non_surge_staffed_second_thresholds.append(policies[i][2]["non_surge"][2])

if subset_of_policies:
    subset_policies_ix = np.sort(np.array(cost_df_all_peaks.columns.astype("int")))
    non_surge_hosp_adm_first_thresholds = np.array(non_surge_hosp_adm_first_thresholds)[subset_policies_ix]
    non_surge_staffed_first_thresholds = np.array(non_surge_staffed_first_thresholds)[subset_policies_ix]
    non_surge_hosp_adm_second_thresholds = np.array(non_surge_hosp_adm_second_thresholds)[subset_policies_ix]
    non_surge_staffed_second_thresholds = np.array(non_surge_staffed_second_thresholds)[subset_policies_ix]

# There's some wonky business with the column names
# This is not a problem for the single-indicator ordering because that is from 0 to 2652
# And also not a problem for computing the unconstrained lowest_cost because we use indexing
# But it does present a problem for the 2-indicator heatmaps
# For the 2-indicator heatmaps we want the columns in numerical order 0, 1, 2, 3, ...
#   so that it matches up with the order of policies
# But cost_df_all_peaks columns are lexsorted as strings "0", "1", "10", etc...
# So here we reorder the columns of cost_df_all_peaks

# Need to adjust feasibility for across-peak stuff

def create_full_df(cost_df,
                   stage1_days_df,
                   stage2_days_df,
                   stage3_days_df,
                   ICU_violation_patient_days_df,
                   feasibility_df,
                   filename):
    cost_df.columns = cost_df.columns.astype("int")
    cost_array_index_corrected = np.array(cost_df[cost_df.columns.sort_values()].mean())
    cost_array_index_corrected_standarderror = np.array(cost_df[cost_df.columns.sort_values()].sem())

    stage1_days_df.columns = stage1_days_df.columns.astype("int")
    stage1_days_index_corrected = np.array(stage1_days_df[stage1_days_df.columns.sort_values()].mean())
    stage1_days_index_corrected_standarderror = np.array(stage1_days_df[stage1_days_df.columns.sort_values()].sem())

    stage2_days_df.columns = stage2_days_df.columns.astype("int")
    stage2_days_index_corrected = np.array(stage2_days_df[stage2_days_df.columns.sort_values()].mean())
    stage2_days_index_corrected_standarderror = np.array(stage2_days_df[stage2_days_df.columns.sort_values()].sem())

    stage3_days_df.columns = stage3_days_df.columns.astype("int")
    stage3_days_index_corrected = np.array(stage3_days_df[stage3_days_df.columns.sort_values()].mean())
    stage3_days_index_corrected_standarderror = np.array(stage3_days_df[stage2_days_df.columns.sort_values()].sem())

    ICU_violation_patient_days_df.columns = ICU_violation_patient_days_df.columns.astype("int")
    ICU_violation_patient_days_index_corrected = np.array(
        ICU_violation_patient_days_df[ICU_violation_patient_days_df.columns.sort_values()].mean())
    ICU_violation_patient_days_index_corrected_standard_error = np.array(
        ICU_violation_patient_days_df[ICU_violation_patient_days_df.columns.sort_values()].sem())

    feasibility_df.columns = feasibility_df.columns.astype("int")
    feasibility_df_index_corrected = np.array(feasibility_df[feasibility_df.columns.sort_values()].mean())
    feasibility_df_index_corrected_standard_error = np.array(feasibility_df[feasibility_df.columns.sort_values()].sem())

    full_df = pd.DataFrame({"hosp1": non_surge_hosp_adm_first_thresholds,
                            "hosp2": non_surge_hosp_adm_second_thresholds,
                            "beds1": non_surge_staffed_first_thresholds,
                            "beds2": non_surge_staffed_second_thresholds,
                            "cost": cost_array_index_corrected,
                            "cost_stderror": cost_array_index_corrected_standarderror,
                            "stage1": stage1_days_index_corrected,
                            "stage1_stderror": stage1_days_index_corrected_standarderror,
                            "stage2": stage2_days_index_corrected,
                            "stage2_stderror": stage2_days_index_corrected_standarderror,
                            "stage3": stage3_days_index_corrected,
                            "stage3_stderror": stage3_days_index_corrected_standarderror,
                            "violation": ICU_violation_patient_days_index_corrected,
                            "violation_stderror": ICU_violation_patient_days_index_corrected_standard_error,
                            "feasibility": feasibility_df_index_corrected,
                            "feasibility_stderror": feasibility_df_index_corrected_standard_error})

    full_df.to_csv(base_path / aggregated_files_folder_name / filename)

    return full_df

# cost_dfs_per_peak is a list -- has cost_df according to the last w simulated
#   (see above code), so be mindful of the weights that are used for the cost!

# TODO: also may want to have sums rather than averages for across-peak dataframe
#   -- right now we have averages across the 4 peaks

# Also important note: for across-peak dataframe feasibility,
#   the feasibility is not the average of yes/no below ICU capacity based on 1200 sample paths
#   -- the feasibility is the average of yes/no below ICU cap ~across 4 peaks~
#   based on 300 sample paths
# Note that the latter does not necessarily give a lower percentage!

full_df_dict = {}

if create_reorganized_dfs:
    for peak in np.arange(3):
        feasibility_df = (ICU_violation_patient_days_dict[str(peak)] < 1e-3)

        full_df_dict[str(peak)] = create_full_df(cost_dfs_per_peak[peak],
                                                 stage1_days_dict[str(peak)],
                                                 stage2_days_dict[str(peak)],
                                                 stage3_days_dict[str(peak)],
                                                 ICU_violation_patient_days_dict[str(peak)],
                                                 feasibility_df,
                                                 full_df_files_prefix + "full_df_peak" + str(peak) + ".csv")

    full_df_across_peaks = create_full_df(cost_df_all_peaks,
                                          stage1_days_all_peaks,
                                          stage2_days_all_peaks,
                                          stage3_days_all_peaks,
                                          ICU_violation_patient_days_all_peaks,
                                          feasibility_across_peaks,
                                          full_df_files_prefix + "full_df_across_peaks.csv")
else:
    for peak in np.arange(3):
        full_df_dict[str(peak)] = pd.read_csv(base_path / aggregated_files_folder_name /
                                              (full_df_files_prefix + "full_df_peak" + str(peak) + ".csv"), index_col=0)
    full_df_across_peaks = pd.read_csv(
        base_path / aggregated_files_folder_name / (full_df_files_prefix + "full_df_across_peaks.csv"),
        index_col=0)

###############################################################################

if subset_of_policies:
    for peak in np.arange(3):
        # Reset index so index matches the actual policy ID (wrt original 12172+1 policies)
        full_df_dict[str(peak)].index = subset_policies_ix
    full_df_across_peaks.index = subset_policies_ix

# Additional analysis (currently unorganized)

###############################################################################

breakpoint()

# KN for every peak and also for across-peak
# Then get additional simulation output for these systems
# Maybe run bi-PASS with unconstrained objective function?
#   ^ But maybe not if the number of systems is small

# Using original k and sample variance based off of 300 replications

c = 1
alpha = 0.05
n0 = 100

# Here r is the number of replications and it happens
#   to be the same as n0 for our case
r = n0
k = 7452
eta = (1 / 2) * (((2 * alpha) / (k - 1)) ** (-2 / (n0 - 1)) - 1)
hsquared = 2 * c * eta * (n0 - 1)

# 1 day in red? or 1 patient day violation? think about...
# iz_param = 540

iz_param = 503

# Use the same k and same variances based on initial round of KN for subsequent iterations of KN
#   to get desired statistical guarantee
# num_feasible_policies_100_reps = [7037, 7399, 6104, 6002]
# round 3
num_feasible_policies_100_reps = [20, 3, 17, 7]

breakpoint()

reps_needed = []

# peak 3 is across-peak!
# for peak in np.arange(4):
for peak in np.arange(4):

    # Peak-specific k, eta, hsquared
    k = num_feasible_policies_100_reps[peak]

    Z = np.random.normal(size=(int(1e5), k - 1))
    Y = np.random.chisquare(df=(n0 - 1), size=(int(1e5), k - 1))
    C = np.random.chisquare(df=(n0 - 1), size=int(1e5))
    C = np.reshape(C, (len(C), 1))
    Cmat = np.repeat(C, k - 1, axis=1)
    denom = np.sqrt((n0 - 1) * (1 / Y + 1 / Cmat))
    H = np.sort(np.max(Z * denom, axis=1))
    rinott_constant = np.quantile(H, 1 - 0.05 / 2.0)

    subset_policies_ix = pd.read_csv("w503_5000reps_non_eliminated_feasible_policies_peak" + str(peak) + ".csv",
                                     header=None)
    subset_policies_ix = np.array(subset_policies_ix, dtype="int")

    for ix in subset_policies_ix:
        # breakpoint()
        if peak <= 2:
            var = np.sum((cost_dfs_per_peak[peak][ix][:n0] - np.average(cost_dfs_per_peak[peak][ix][:n0])) ** 2) / (
                        n0 - 1)
        elif peak == 3:
            current_ix_costs = (cost_dfs_per_peak[0][ix][:n0] +
                                cost_dfs_per_peak[1][ix][:n0] +
                                cost_dfs_per_peak[2][ix][:n0]) / 3
            var = np.sum((current_ix_costs - np.average(current_ix_costs[:n0])) ** 2) / (n0 - 1)

        if peak == 3:
            reps_needed.append(rinott_constant ** 2 * var / iz_param ** 2)

breakpoint()

# Also, idea is to do the pairwise comparisons more intelligently for KN
# Compare policies to best sample mean policy and then eliminate policies
# Then compare surviving policies to 2nd best sample mean policy and eliminate
#   and so on...
# Right now doing KN where top 100 policies are compared to other policies
#   rather than do O(10k^2) comparisons

# full_df_across_peaks[full_df_across_peaks["feasibility"] > 0.95].sort_values("cost")

# This is correct -- just confusing because we use both full_df and
#   cost_dfs_per_peak -- and their formatting is different -- and former uses
#   integers for indexing and latter has columns that are strings
# ^ TODO: want to re-write / comment more clearly

if run_KN:

    # peak 3 is across-peak!
    # for peak in np.arange(4):
    for peak in np.arange(4):

        if peak <= 2:
            feasible_policies_df_current_peak = full_df_dict[str(peak)][full_df_dict[str(peak)]["feasibility"] > 0.95]
        elif peak == 3:
            feasible_policies_df_current_peak = full_df_across_peaks[full_df_across_peaks["feasibility"] > 0.95]

        # Peak-specific k, eta, hsquared
        k = num_feasible_policies_100_reps[peak]
        eta = (1 / 2) * (((2 * alpha) / (k - 1)) ** (-2 / (n0 - 1)) - 1)
        hsquared = 2 * c * eta * (n0 - 1)

        eliminated_policies_ix = set([])

        for i in range(len(feasible_policies_df_current_peak)):
            # for i in range(100):

            eliminated_policies_ix = set(eliminated_policies_ix)
            eliminated_policies_ix = list(eliminated_policies_ix)

            # Policy we are using as "reference" (or "basis" for pairwise comparisons),
            #   based on cost -- we want to use smaller cost policies to knock out other policies
            reference_ix_smaller_cost_peak = feasible_policies_df_current_peak.sort_values("cost").index[i]
            comparison_mean = feasible_policies_df_current_peak.sort_values("cost").iloc[i]["cost"]

            if peak <= 2:
                reference_ix_costs = cost_dfs_per_peak[peak][reference_ix_smaller_cost_peak][:n0]
            elif peak == 3:
                # Average costs on 3 peaks
                reference_ix_costs = (cost_dfs_per_peak[0][reference_ix_smaller_cost_peak][:n0] +
                                      cost_dfs_per_peak[1][reference_ix_smaller_cost_peak][:n0] +
                                      cost_dfs_per_peak[2][reference_ix_smaller_cost_peak][:n0]) / 3

            # breakpoint()

            var_of_diff_dict = {}

            for ix in feasible_policies_df_current_peak.index:
                # breakpoint()
                if peak <= 2:
                    var_of_diff = np.sum((cost_dfs_per_peak[peak][ix][:n0] - reference_ix_costs -
                                          (feasible_policies_df_current_peak["cost"].loc[
                                               ix] - comparison_mean)) ** 2) / (n0 - 1)
                elif peak == 3:
                    current_ix_costs = (cost_dfs_per_peak[0][ix][:n0] +
                                        cost_dfs_per_peak[1][ix][:n0] +
                                        cost_dfs_per_peak[2][ix][:n0]) / 3
                    var_of_diff = np.sum((current_ix_costs - reference_ix_costs -
                                          (feasible_policies_df_current_peak["cost"].loc[
                                               ix] - comparison_mean)) ** 2) / (n0 - 1)

                var_of_diff_dict[ix] = var_of_diff

                print(var_of_diff * hsquared / iz_param ** 2)

            # counter = 0

            for ix in feasible_policies_df_current_peak.index:
                # counter += 1
                # print(counter)
                if ix == reference_ix_smaller_cost_peak:
                    continue
                elif ix in eliminated_policies_ix:
                    continue
                else:
                    wiggle_room = max(0,
                                      (iz_param / (2 * c * r)) * (hsquared * var_of_diff_dict[ix] / iz_param ** 2 - r))
                    if feasible_policies_df_current_peak["cost"].loc[ix] > comparison_mean + wiggle_room:
                        eliminated_policies_ix.append(ix)

        # print(len(eliminated_policies_ix))

        # The way I set this up is that we simulate the same set of policies per peak, so
        #   we are just grabbing the columns from the 0th (1st) peak
        non_eliminated_policies = set(stage3_days_dict[str(0)].columns.astype(int)).difference(
            set(eliminated_policies_ix))
        feasible_policies = set(feasible_policies_df_current_peak.index)
        non_eliminated_feasible_policies = list(non_eliminated_policies.intersection(feasible_policies))

        # breakpoint()

        np.savetxt("non_eliminated_feasible_policies_peak" + str(peak) + ".csv",
                   np.array(non_eliminated_feasible_policies).astype("int"))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#  / \-----------------------------------------------------,
#  \_,|                                                    |
#     |                                                    |
#     |   Code graveyard                                   |
#     |  ,--------------------------------------------------
#     \_/__________________________________________________/
#               \\     =o)
#               (o>    /\\
#               _(()_  _\_V_
#               //     \\
#                       \\
###############################################################################
###############################################################################
###############################################################################
###############################################################################
