###############################################################################

# Misc parsing stuff :O

# Misc routines run on local computer to combine/reorganize files
#   created from different experiments

###############################################################################

import copy
import pandas as pd

from pathlib import Path

# Import other Python packages
import numpy as np
import glob

base_path = Path(__file__).parent

###############################################################################

# Combining first 300 replications from all policies with next 700 replications
#   from surviving policies

folder_name = "Results_09292023_NoCaseThreshold_12172Policies_BetterSubset_1000Reps"

prefix1 = "first300_"
prefix2 = "second700_"

for peak in np.arange(3):

    stage2_days_df_1 = pd.read_csv(
        base_path / folder_name / (prefix1 + "aggregated_peak" + str(peak) + "_stage2_days.csv"),
        index_col=0)
    stage3_days_df_1 = pd.read_csv(
        base_path / folder_name / (prefix1 + "aggregated_peak" + str(peak) + "_stage3_days.csv"),
        index_col=0)
    ICU_violation_patient_days_df_1 = pd.read_csv(base_path / folder_name /
                                                (prefix1 + "aggregated_peak" + str(peak) +
                                                 "_ICU_violation_patient_days.csv"),
                                                index_col=0)

    stage2_days_df_2 = pd.read_csv(
        base_path / folder_name / (prefix2 + "aggregated_peak" + str(peak) + "_stage2_days.csv"),
        index_col=0)
    stage3_days_df_2 = pd.read_csv(
        base_path / folder_name / (prefix2 + "aggregated_peak" + str(peak) + "_stage3_days.csv"),
        index_col=0)
    ICU_violation_patient_days_df_2 = pd.read_csv(base_path / folder_name /
                                                (prefix2 + "aggregated_peak" + str(peak) +
                                                 "_ICU_violation_patient_days.csv"),
                                                index_col=0)

    stage2_days_df = pd.concat([stage2_days_df_1[stage2_days_df_2.columns], stage2_days_df_2], axis=0)
    stage3_days_df = pd.concat([stage3_days_df_1[stage3_days_df_2.columns], stage3_days_df_2])
    ICU_violation_patient_days_df = pd.concat([ICU_violation_patient_days_df_1[ICU_violation_patient_days_df_2.columns],
                                               ICU_violation_patient_days_df_2])

    # print(stage2_days_df[:300].mean() - stage2_days_df[300:].mean())
    # print(stage3_days_df[:300].mean() - stage3_days_df[300:].mean())
    # print(ICU_violation_patient_days_df[:300].mean() - ICU_violation_patient_days_df[300:].mean())

    stage2_days_df.reset_index(drop=True, inplace=True)
    stage3_days_df.reset_index(drop=True, inplace=True)
    ICU_violation_patient_days_df.reset_index(drop=True, inplace=True)

    stage2_days_df.to_csv("all1000_aggregated_peak" + str(peak) + "_stage2_days.csv")
    stage3_days_df.to_csv("all1000_aggregated_peak" + str(peak) + "_stage3_days.csv")
    ICU_violation_patient_days_df.to_csv("all1000_aggregated_peak" + str(peak) + "_ICU_violation_patient_days.csv")

breakpoint()

###############################################################################

# Combine the individual CDC implementation files
# Want to preserve order so that reps1-300 are in order for "CRN" / common sample path stuff

# FYI have to move finished files to relevant folder used in next code chunk

ICU_violation_patient_days_dict = {}
stage2_days_dict = {}
stage3_days_dict = {}

for peak in np.arange(2,4):

    ICU_violation_patient_days_data = []
    stage2_days_data = []
    stage3_days_data = []

    for rank in np.arange(30):
        ICU_violation_patient_days_data.append(pd.read_csv("Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" +
                                                           "/" + "rank" + str(rank) + "_peak" + str(peak) +
                                                           "_policyCDCimplementation" +
                                                           "_ICU_violation_patient_days.csv", header=None))
        stage2_days_data.append(pd.read_csv("Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" +
                                            "/" + "rank" + str(rank) + "_peak" + str(peak) +
                                            "_policyCDCimplementation" +
                                            "_stage2_days.csv", header=None))
        stage3_days_data.append(pd.read_csv("Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" +
                                            "/" + "rank" + str(rank) + "_peak" + str(peak) +
                                            "_policyCDCimplementation" +
                                            "_stage3_days.csv", header=None))

    ICU_violation_patient_days_dict[peak] = pd.concat(ICU_violation_patient_days_data, axis=0).reset_index()[0]
    stage2_days_dict[peak] = pd.concat(stage2_days_data, axis=0).reset_index()[0]
    stage3_days_dict[peak] = pd.concat(stage3_days_data, axis=0).reset_index()[0]

for peak in np.arange(2,4):
    ICU_violation_patient_days_dict[peak].to_csv("peak" + str(peak) + "_policy" + str(12172) + "_ICU_violation_patient_days.csv")
    stage2_days_dict[peak].to_csv("peak" + str(peak) + "_policy" + str(12172) + "_stage2_days.csv")
    stage3_days_dict[peak].to_csv("peak" + str(peak) + "_policy" + str(12172) + "_stage3_days.csv")

breakpoint()

###############################################################################

# This operates on combined CDC implementation files
# Adds them (CDC implemented policy is policy #12172)

for peak in np.arange(2,4):
    stage2_days_df = pd.read_csv(
        base_path / "Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" / ("aggregated_peak" + str(peak) + "_stage2_days.csv"),
        index_col=0)
    stage3_days_df = pd.read_csv(
        base_path / "Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" / ("aggregated_peak" + str(peak) + "_stage3_days.csv"),
        index_col=0)
    ICU_violation_patient_days_df = pd.read_csv(base_path / "Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" /
                                                ("aggregated_peak" + str(peak) + "_ICU_violation_patient_days.csv"),
                                                index_col=0)

    CDC_implementation_stage2_days_df = pd.read_csv(
        base_path / "Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" / (
                    "peak" + str(peak) + "_policy12172_stage2_days.csv"),
        index_col=0)
    CDC_implementation_stage3_days_df = pd.read_csv(
        base_path / "Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" / (
                    "peak" + str(peak) + "_policy12172_stage3_days.csv"),
        index_col=0)
    CDC_implementation_ICU_violation_patient_days_df = pd.read_csv(
        base_path / "Results_08292023_NoCaseThreshold_12172Policies_SingleAndDoubleIndicators_300Reps" /
        ("peak" + str(peak) + "_policy12172_ICU_violation_patient_days.csv"),
        index_col=0)

    CDC_implementation_stage2_days_df.rename(columns={"0": "12172"}, inplace=True)
    CDC_implementation_stage3_days_df.rename(columns={"0": "12172"}, inplace=True)
    CDC_implementation_ICU_violation_patient_days_df.rename(columns={"0": "12172"}, inplace=True)

    # CDC_implementation_stage2_days_df.reset_index(inplace=True)
    # CDC_implementation_stage3_days_df.reset_index(inplace=True)
    # CDC_implementation_ICU_violation_patient_days_df.reset_index(inplace=True)

    new_stage2_days_df = pd.concat((stage2_days_df, CDC_implementation_stage2_days_df["12172"]), axis=1)
    new_stage3_days_df = pd.concat((stage3_days_df, CDC_implementation_stage3_days_df["12172"]), axis=1)
    new_ICU_violation_patient_days_df = pd.concat((ICU_violation_patient_days_df, CDC_implementation_ICU_violation_patient_days_df["12172"]), axis=1)

    new_stage2_days_df.to_csv(
        "aggregated_peak" +
        str(peak) + "_stage2_days.csv")
    new_stage3_days_df.to_csv(
        "aggregated_peak" +
        str(peak) + "_stage3_days.csv")
    new_ICU_violation_patient_days_df.to_csv(
        "aggregated_peak" +
        str(peak) + "_ICU_violation_patient_days.csv")

breakpoint()

###############################################################################

# Just the parsing stuff from the CDCoptimization scripts

# Step 6: parsing

# Create 2 sets of dataframes
# Set A: 1 dataframe for each policy -- columns are cost, feasibility,
#   ICU patient-days violation, stage1 days, stage2 days, stage3 days
#   (each of the 6 performance measures), rows are replications
# Set B: 1 dataframe for each performance measure, -- columns are
#   policies, rows are replications

# Edit: kept the functionality to generate Set A, but
#   it's quite slow, and Set B is more useful. Plus Set A can be
#   generated from Set B, so it's unnecessary.

split_peaks_amongst_processors = False
rank = 0

need_parse = True

need_set_A = False

if need_parse:
    for peak in [0]:

        if split_peaks_amongst_processors:
            if rank >= 4:
                break
            else:
                peak = rank

        # Set A
        # policy_dict is a dictionary of dictionaries to contain
        #   performance measures for each policy
        if need_set_A:
            policy_dict = {}

        # Set B
        ICU_violation_patient_days_dict = {}
        stage2_days_dict = {}
        stage3_days_dict = {}

        performance_measures_dicts = [ICU_violation_patient_days_dict, stage2_days_dict, stage3_days_dict]

        ICU_violation_patient_days_filenames = glob.glob("peak" + str(peak) + "*ICU_violation_patient_days.csv")
        stage2_days_filenames = glob.glob("peak" + str(peak) + "*stage2_days.csv")
        stage3_days_filenames = glob.glob("peak" + str(peak) + "*stage3_days.csv")

        num_performance_measures = len(performance_measures_dicts)

        performance_measures_filenames = [ICU_violation_patient_days_filenames, stage2_days_filenames,
                                          stage3_days_filenames]

        # These become the column names for dataframes in Set A
        performance_measures_strs = ["icu_violation_patient_days", "stage2_days", "stage3_days"]

        # Open each .csv file and store its contents in various dataframes
        for performance_measures_id in range(num_performance_measures):
            for filename in performance_measures_filenames[performance_measures_id]:
                df = pd.read_csv(filename, header=None)
                policy_id = int(filename.split("peak" + str(peak) + "_policy")[-1].split("_")[0])

                if need_set_A:
                    if policy_id in policy_dict.keys():
                        policy_dict[policy_id][performance_measures_strs[performance_measures_id]] = np.asarray(df[0])
                    else:
                        policy_dict[policy_id] = {}
                        policy_dict[policy_id][performance_measures_strs[performance_measures_id]] = np.asarray(df[0])

                performance_measures_dicts[performance_measures_id][policy_id] = np.asarray(df[0])

        # Generate and export Set A dataframes
        if need_set_A:
            for key in policy_dict.keys():
                df = pd.DataFrame(policy_dict[key])
                df.to_csv("aggregated_peak" + str(peak) + "_policy" + str(key) + ".csv")

        # Generate and export Set B dataframes
        for performance_measures_id in range(num_performance_measures):
            df = pd.DataFrame(performance_measures_dicts[performance_measures_id])
            df.to_csv(
                "aggregated_peak" + str(peak) + "_" + str(performance_measures_strs[performance_measures_id]) + ".csv")
