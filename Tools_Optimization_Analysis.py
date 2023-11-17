###############################################################################
#
#  / \-----------------------------------------------------,
#  \_,|                                                    |
#     |  Tools_Optimization_Analysis.py                    |
#     |  Linda's toolkit for large-scale analysis          |
#     |     and parsing for optimization after generating  |
#     |     simulation replications                        |
#     |  ,--------------------------------------------------
#     \_/__________________________________________________/
#               \\     =o)
#               (o>    /\\
#               _(()_  _\_V_
#               //     \\
#                       \\
#
# Doing unconstrained optimization right now
#
# After obtaining simulation replications and aggregating them,
#   these functions do analysis on aggregated datasets to get unconstrained
#   lowest_cost and other performance statistics such as average days in
#   each stage and ICU patient days violation
#
# TODO: deal with subset of policies legacy code?
#
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

def create_performance_measure_dict_peaks_to_output(aggregated_folder_path,
                                                    aggregated_folder_name,
                                                    aggregated_files_prefix,
                                                    num_peaks,
                                                    aggregated_files_suffix,
                                                    reps_offset=0,
                                                    reps_cutoff=np.inf):
    '''
    :param aggregated_folder_path [instance of pathlib.Path] -- location of folder
        containing .csv files to parse
    :param aggregated_folder_name [str] -- folder name containing
        .csv files to parse
    :param aggregated_files_prefix [str] -- prefix for .csv filename
    :param num_peaks [int] -- number of peaks
    :param aggregated_files_suffix [str] -- suffix for .csv filename
        -- should end in ".csv"
    :reps_offset [int] -- discard any replications before this number --
        used if only want to select a subset of replications from dataframe
    :reps_cutoff [int] -- discard any replications after this number --
        used if only want to select a subset of replications from dataframe

    Assume that aggregated_files_suffix contains information about
        the performance measure name, and that the aggregated .csv
        files to combine have the form
    aggregated_files_prefix + str(peak) + aggregated_files_suffix
        where peak takes a value in np.arange(num_peaks)

    e.g.
    "aggregated_peak_0_stage2_days.csv"

    Using the default values of reps_offset and reps_cutoff, entire dataframes
        (including all replications) are imported into the dictionary
    '''

    peaks_dictionary = {}

    for peak in np.arange(num_peaks):
        df = pd.read_csv(aggregated_folder_path / aggregated_folder_name /
                         (aggregated_files_prefix + str(peak) + aggregated_files_suffix), index_col=0)
        peaks_dictionary[peak] = df

    return peaks_dictionary


def compute_feasibility_across_peaks(ICU_violation_dict, num_peaks):
    '''
    :param ICU_violation_dict [dict] -- dictionary with
        integers corresponding to specific peaks as keys -- each value
        is a dataframe with ICU violation per replication for that particular peak
    :param num_peaks [int] -- number of peaks

    e.g. a dictionary generated by create_performance_measure_dict_peaks_to_output()
        on ICU violation patient days performance measure

    IMPORTANT NOTE
    feasibility_across_peaks is NOT simply the average
    feasibility_across_peaks for row i is True if for EACH of the peaks,
        the ith replication was above ICU capacity
    so feasibility_across_peaks.mean() is the proportion of sample paths i
        such that replication i was feasible (below ICU capacity) for ALL peaks
    '''

    feasibility_across_peaks = (ICU_violation_dict[0] == 0)

    for peak in np.arange(1, num_peaks):
        feasibility_across_peaks = ((feasibility_across_peaks == 1) & (ICU_violation_dict[peak] == 0))

    return feasibility_across_peaks


###############################################################################

# Find unconstrained optimization lowest_cost across peaks
#   for a given ICU violation patient days penalty w
# Test different values of w -- choose w so that the unconstrained
#   lowest_cost roughly has 95% feasibility

def compute_optimal_given_ICU_penalty(stage1_cost,
                                      stage2_cost,
                                      stage3_cost,
                                      ICU_penalty,
                                      stage1_dict,
                                      stage2_dict,
                                      stage3_dict,
                                      ICU_violation_dict,
                                      feasibility_across_peaks,
                                      num_peaks,
                                      mapping_df):
    '''
    :param stage1_cost [int] -- nonnegative integer corresponding to
        cost per day of being in stage 1
    :param stage2_cost [int] -- analogous to above, but for stage 2
    :param stage3_cost [int] -- analogous to above, but for stage 3
    :param ICU_penalty [int] -- nonnegative integer corresponding to
        penalty for ICU violation patient days (total number of patient days
        above ICU capacity)
    :param stage1_dict [dict] -- dictionary with
        integers corresponding to specific peaks as keys -- each value
        is a dataframe with number of days in stage 1 for that particular peak
    :param stage2_dict [dict] -- analogous to above, but for number of
        days in stage 2
    :param stage3_dict [dict] -- analogous to above, but for number of
        days in stage 3
    :param ICU_violation_dict [dict] -- analogous to above, but for
        ICU violation patient days
    :param feasibility_across_peaks [DataFrame] -- each element is
        0/1 indicating whether policy (column) was feasible across ALL peaks
        for that replication (row)
    :param num_peaks [int] -- nonnegative integer for total number of peaks
    :param mapping_df [DataFrame] -- DataFrame with columns
            "ID" (policy ID)
            "hosp1" (hospital admits blue-to-yellow non-surge threshold)
            "hosp2" (hospital admits yellow-to-red non-surge threshold)
            "beds1" (staffed beds blue-to-yellow non-surge threshold)
            "beds2" (staffed beds yellow-to-red non-surge threshold)
            e.g. created by write_non_surge_CDC_policy_ID_mapping_csv()

    e.g. stage1_dict, stage2_dict, stage3_dict, ICU_violation_dict
        generated from create_performance_measure_dict_peaks_to_output() function

    Calls make_weighted_sum() as subroutine to compute cost for each replication:
        stage1_cost * [number of days in stage 1] +
        stage2_cost * [number of days in stage 2] +
        stage3_cost * [number of days in stage 3] +
        ICU_penalty * [number of ICU violation patient days]
        where values in brackets are taken from corresponding dictionaries

        and obtains the policy that has the minimum average cost (mean taken
        across all replications in dataframes)

    Prints ICU_penalty, string representation of lowest-cost policy from mapping_df,
        and across-peak feasibility
    '''

    cost_dfs_per_peak = []

    for peak in np.arange(num_peaks):
        cost_df = make_weighted_sum(stage1_dict[peak],
                                    stage2_dict[peak],
                                    stage3_dict[peak],
                                    ICU_violation_dict[peak],
                                    stage1_cost, stage2_cost, stage3_cost, ICU_penalty)
        cost_dfs_per_peak.append(cost_df)

    cost_df_all_peaks = pd.concat(cost_dfs_per_peak)
    lowest_cost_val, lowest_cost_ix = find_lowest_cost_weighted_sum(cost_df_all_peaks)

    print(ICU_penalty, mapping_df[mapping_df["ID"] == lowest_cost_ix][["hosp1", "hosp2", "beds1", "beds2"]],
          feasibility_across_peaks.mean()[str(lowest_cost_ix)])


# 502 (-1, 1, 15) (inf, inf, inf) 0.44666666666666666 0.9
# 503 (-1, 0, 14) (inf, inf, inf) 0.06666666666666667 0.97

###############################################################################

def create_full_df(cost_df,
                   stage1_days_df,
                   stage2_days_df,
                   stage3_days_df,
                   ICU_violation_patient_days_df,
                   feasibility_df,
                   mapping_df,
                   full_df_folder_path,
                   full_df_folder_name,
                   filename):
    '''
    :param cost_df [DataFrame] -- columns are policy IDs, rows correspond to cost of that replication
        e.g. element[i,j] corresponds to scalar cost value from replication i of policy j
        e.g. can be created using make_weighted_sum() function
    :param stage1_days_df [DataFrame] -- analogous to above, but for number of days in stage 1
    :param stage2_days_df [DataFrame] -- analogous to above, but for number of days in stage 2
    :param stage3_days_df [DataFrame] -- analogous to above, but for number of days in stage 3
    :param ICU_violation_patient_days_df [DataFrame] -- analogous to above, but for number of ICU
        violation patient days
    :param feasibility_df [DataFrame] -- analogous to above, but for 0/1 feasibility
    :param mapping_df [DataFrame] -- DataFrame with columns
            "ID" (policy ID)
            "hosp1" (hospital admits blue-to-yellow non-surge threshold)
            "hosp2" (hospital admits yellow-to-red non-surge threshold)
            "beds1" (staffed beds blue-to-yellow non-surge threshold)
            "beds2" (staffed beds yellow-to-red non-surge threshold)
            e.g. created by write_non_surge_CDC_policy_ID_mapping_csv()
    :param full_df_folder_path [pathlib.Path instance] -- path to full_df_folder_name
    :param full_df_folder_name [str] -- name of folder in which to write data
    :param filename [str] -- name of file in which to write data, ending in ".csv"
    '''

    # IMPORTANT NOTE
    # For the case where columns are lexsorted as strings "0", "1", "10", etc...
    #   here we reorder the columns of cost_df_all_peaks

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

    # Note: the mask [mapping_df["ID"].isin(np.array(feasibility_df.columns))]
    #   is in case the aggregated files only have a subset of the total/original number of policies
    #   specified in mapping_df -- e.g. if policies are sequentially eliminated

    full_df = pd.DataFrame({"hosp1": mapping_df["hosp1"][mapping_df["ID"].isin(np.array(feasibility_df.columns))],
                            "hosp2": mapping_df["hosp2"][mapping_df["ID"].isin(np.array(feasibility_df.columns))],
                            "beds1": mapping_df["beds1"][mapping_df["ID"].isin(np.array(feasibility_df.columns))],
                            "beds2": mapping_df["beds2"][mapping_df["ID"].isin(np.array(feasibility_df.columns))],
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

    full_df.to_csv(full_df_folder_path / full_df_folder_name / filename)

    return full_df



