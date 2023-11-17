###############################################################################
#   _________________________________________________
# ((                                                 ))
#  ))     Script_CDCOptimizationFileParsing.py       ((
# ((                                                 ))
#   -------------------------------------------------
#
# Working script (also a demo) using functions from
#   Tools_Optimization_Analysis to organize large-scale
#   simulation output (scattered across many files)
#   for analysis and optimization
# Specifically created for LP's CDC optimization (no wastewater)
#   -- so need to modify (particularly filenames) for other
#   purposes -- here, we have 3 peaks and performance measures
#   of days in stage 2, days in stage 3, and ICU violation patient
#   days for each peak
#
# IN A NUTSHELL: this code takes aggregated .csv files
#   (one per peak, one per performance measure)
#   with structure A and converts them into so-called "full"
#   .csv files (one per peak, containing all performance measures
#   and an additional "cost" measure, plus an additional file of
#   across-peak statistics) with structure B
# Structure A: columns are policy ID, rows are per-replication
#   output for that particular peak and performance measure
# Structure B: columns are
#   hosp1
#   hosp2
#   beds1
#   beds2
#   cost
#   cost_stderror
#   stage1
#   stage1_stderror
#   stage2
#   stage2_stderror
#   stage3
#   stage3_stderror
#   violation
#   violation_stderror
#   feasibility
#   feasibility_stderror
#   for that particular peak (and also for across-peak)
#   i.e. all performance measures are aggregated, and the policy ID
#   is forgotten and simply converted into 4 values that distinguish
#   the policy (hosp1, hosp2, beds1, beds2)
#   Note "violation" refers to ICU patient days violation and
#   "feasibility" refers to proportion of sample paths in which
#   ICU capacity was respected -- but for the across-peak full
#   DataFrame, "feasibility" refers to the proportion of sample paths
#   in which ICU capacity was respected on ALL peaks for that sample path
#
# See subsection labeled "TOGGLES AND OPTIONS" to configure for
#   different computers -- can specify folder names and file names in
#   this subsection -- must have .csv files for policies'
#   performance measures
#
###############################################################################

#######################################
############## IMPORTS ################
#######################################

import pandas as pd
import numpy as np
import itertools
import matplotlib

from Tools_Optimization_Analysis import *

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

# Toggle for whether to create reorganized dfs
#   -- these dfs contain columns for thresholds values
#   as well as summary statistics -- more intuitive way of analyzing
#   simulation data and also amenable to simple spreadsheet analysis
# If False, assumes full_df_peak0.csv, full_df_peak1.csv, full_df_peak2.csv,
#   full_df_peak3.csv, full_df_across_peaks.csv already exist
create_reorganized_dfs = True

# Toggle for whether to creating mapping csv file
#   with columns
#       "policy ID" (policy ID)
#       "hosp1" (hospital admits blue-to-yellow non-surge threshold)
#       "hosp2" (hospital admits yellow-to-red non-surge threshold)
#       "beds1" (staffed beds blue-to-yellow non-surge threshold)
#       "beds2" (staffed beds yellow-to-red non-surge threshold)
# If False, assumes CDC_optimization_mapping.csv already exists
create_mapping = True

###############################################################################

##############################################################
############## PEAKS AND POLICIES DEFINITIONS ################
##############################################################

peaks_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]

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

# if coordinate_subset:
#     policies = []
#
#     case_threshold = np.inf
#
#     staffed_thresholds = {"non_surge": (-1,
#                                         np.inf,
#                                         np.inf),
#                           "surge": (-1,
#                                     -1,
#                                     np.inf)}
#
#     for first_threshold in np.arange(15):
#         hosp_adm_thresholds = {"non_surge": (-1,
#                                              first_threshold,
#                                              14),
#                                "surge": (-1,
#                                          -1,
#                                          first_threshold)}
#
#         policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))
#
#     for second_threshold in np.arange(1, 21):
#         hosp_adm_thresholds = {"non_surge": (-1,
#                                              1,
#                                              second_threshold),
#                                "surge": (-1,
#                                          -1,
#                                          1)}
#
#         policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

###############################################################################

# Creating mapping csv file
if create_mapping:
    write_non_surge_CDC_policy_ID_mapping_csv(policies, "CDC_optimization_mapping.csv")
mapping = pd.read_csv("CDC_optimization_mapping.csv", header=0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Create dictionaries for each performance measure
#   Each key is a peak number
#   Each value is a DataFrame of output for that
#       performance measure for that peak
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

stage2_days_dict = create_performance_measure_dict_peaks_to_output(base_path,
                                                                   aggregated_files_folder_name,
                                                                   aggregated_files_prefix,
                                                                   3, "_stage2_days.csv")
stage3_days_dict = create_performance_measure_dict_peaks_to_output(base_path,
                                                                   aggregated_files_folder_name,
                                                                   aggregated_files_prefix,
                                                                   3, "_stage3_days.csv")
ICU_violation_patient_days_dict = create_performance_measure_dict_peaks_to_output(base_path,
                                                                                  aggregated_files_folder_name,
                                                                                  aggregated_files_prefix,
                                                                                  3, "_ICU_violation_patient_days.csv")

stage1_days_dict = {}

for peak in np.arange(3):
    # Also compute number of days in stage 1 based on number of days in stages 2 and 3
    # We save I/O by not creating a .csv file for number of days in stage 1 since it
    #   can be computed from other performance measures
    stage1_days_df = peaks_end_times[peak] - \
                     peaks_start_times[peak] - \
                     stage2_days_dict[peak].add(stage3_days_dict[peak], axis=1)
    stage1_days_dict[peak] = stage1_days_df[reps_offset:reps_offset + num_reps_per_peak]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Also compute the feasibility across peaks
# See compute_feasibility_across_peaks() function documentation
# It is NOT the average feasibility -- for a replication i to be feasible,
#   replication i must be feasible for ALL peaks
# Feasibility is defined as respecting ICU capacity for the entire simulation
#   time period
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

feasibility_across_peaks = compute_feasibility_across_peaks(ICU_violation_patient_days_dict, 3)

feasibility_across_peaks_mean = feasibility_across_peaks.mean()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Compute the optimal policy with respect to "cost" given an ICU penalty
# NOTE: here, the "cost" is the weighted sum of days in each stage
#   plus penalized ICU violation patient days, with respective weights
#   1/10/100 for blue/yellow/red days and an ICU penalty of 503
# IMPORTANT NOTE: here, 503 is hardcoded -- it is the "optimal" ICU penalty
#   for THIS PARTICULAR PROBLEM -- for your own problem, you will want to
#   run compute_optimal_given_ICU_penalty() with different weights to
#   choose properly calibrated weights -- here, 503 is the smallest ICU penalty
#   such that the across-peaks optimal policy has >95% feasibility
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

compute_optimal_given_ICU_penalty(1,
                                  10,
                                  100,
                                  503,
                                  stage1_days_dict,
                                  stage2_days_dict,
                                  stage3_days_dict,
                                  ICU_violation_patient_days_dict,
                                  feasibility_across_peaks,
                                  3,
                                  mapping)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Create dictionary with "cost" value for each of the peaks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

cost_df_dict = {}

for peak in np.arange(3):
    cost_df_dict[peak] = make_weighted_sum(stage1_days_dict[peak],
                                           stage2_days_dict[peak],
                                           stage3_days_dict[peak],
                                           ICU_violation_patient_days_dict[peak],
                                           1, 10, 100, 503)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# For across-peak analysis, for each performance measure, create
#   a DataFrame that combines all DataFrames for each peak for that
#   performance measure (i.e. simple appending of the rows)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

cost_df_all_peaks = pd.concat(cost_df_dict.values(), axis=0)
stage1_days_all_peaks = pd.concat(stage1_days_dict.values(), axis=0)
stage2_days_all_peaks = pd.concat(stage2_days_dict.values(), axis=0)
stage3_days_all_peaks = pd.concat(stage3_days_dict.values(), axis=0)
ICU_violation_patient_days_all_peaks = pd.concat(ICU_violation_patient_days_dict.values(), axis=0)

full_df_dict = {}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# For across-peak analysis, for each performance measure, create
#   a DataFrame that combines all DataFrames for each peak for that
#   performance measure (i.e. simple appending of the rows)
# See comments at beginning of document for structure of such files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if create_reorganized_dfs:
    for peak in np.arange(3):
        feasibility_df = (ICU_violation_patient_days_dict[peak] < 1e-3)

        full_df_dict[peak] = create_full_df(cost_df_dict[peak],
                                            stage1_days_dict[peak],
                                            stage2_days_dict[peak],
                                            stage3_days_dict[peak],
                                            ICU_violation_patient_days_dict[peak],
                                            feasibility_df,
                                            mapping,
                                            base_path,
                                            aggregated_files_folder_name,
                                            full_df_files_prefix + "full_df_peak" + str(peak) + ".csv")

    full_df_across_peaks = create_full_df(cost_df_all_peaks,
                                          stage1_days_all_peaks,
                                          stage2_days_all_peaks,
                                          stage3_days_all_peaks,
                                          ICU_violation_patient_days_all_peaks,
                                          feasibility_across_peaks,
                                          mapping,
                                          base_path,
                                          aggregated_files_folder_name,
                                          full_df_files_prefix + "full_df_across_peaks.csv")
else:
    for peak in np.arange(3):
        full_df_dict[peak] = pd.read_csv(base_path / aggregated_files_folder_name /
                                         (full_df_files_prefix + "full_df_peak" + str(peak) + ".csv"), index_col=0)
    full_df_across_peaks = pd.read_csv(
        base_path / aggregated_files_folder_name / (full_df_files_prefix + "full_df_across_peaks.csv"),
        index_col=0)
