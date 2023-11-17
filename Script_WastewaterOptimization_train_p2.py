###############################################################################
# Wastewater based policy
# This script contains beginning-to-end sample path generation
#   and evaluation of CDC policies. User can specify which CDC policies
#   they would like to evaluate.
# Can split up sample path generation and policy evaluation on
#   parallel processors using ''mpi4py.''
# The number of sample paths generated (and number of replications
#   that each policy is evaluated on) is
#       num_processors_evaluation x sample_paths_generated_per_processor
#           (num_processors_evaluation) is inferred from mpi call
#           (sample_paths_generated_per_processor is a variable that is
#       specified in the code)

###############################################################################

import copy
from Engine_SimObjects_Wastewater import MultiTierPolicy_Wastewater
from Engine_DataObjects_Wastewater import City, TierInfo, Vaccine
from Engine_SimModel_Wastewater import SimReplication
import Tools_InputOutput_Wastewater
import Tools_Optimization_Wastewater

import datetime as dt
import pandas as pd

# Import other Python packages
import numpy as np
import glob # Sonny's notes: packages for finding files recursively
import os

from mpi4py import MPI
from pathlib import Path

base_path = Path(__file__).parent

comm = MPI.COMM_WORLD
num_processors_evaluation = comm.Get_size()
rank = comm.Get_rank()

output_dir = "policy_evaluation"

###############################################################################

# OPTIONS
# Toggle True/False or specify values for customization

# Change to False if sample paths have already been generated
need_sample_paths = False

# Different than num_processors_evaluation because
#   num_processors_sample_path is used for naming/distinguishing
#   states .json files
num_processors_sample_paths = 1
sample_paths_generated_per_processor = 2

# Change to False if evaluation is already done
need_evaluation = True

# If only interested in evaluating on subset of reps
#num_reps_evaluated_per_policy = 2

# Reps offset
# Rep number to start on
reps_offset = 0

# If True, only test 2 policies
using_test_set_only = False

# Change to True if also want to automatically parse files
need_parse = True

# Assume that the number of processors >= 4
# When True, for parsing, will use 4 processors and give
#   1 peak to each processor
split_peaks_amongst_processors = False

###############################################################################

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

need_set_A = False

num_peaks = 4

if need_parse:
    for peak in np.arange(num_peaks):

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
        cost_dict = {}
        feasibility_dict = {}
        ICU_violation_patient_days_dict = {}
        stage1_days_dict = {}
        stage2_days_dict = {}
        stage3_days_dict = {}
        stage4_days_dict = {}
        stage5_days_dict = {}
        surge_days_dict = {}

        performance_measures_dicts = [cost_dict, feasibility_dict, ICU_violation_patient_days_dict,
                                      stage1_days_dict, stage2_days_dict, stage3_days_dict, stage4_days_dict, stage5_days_dict, surge_days_dict]

        cost_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*cost.csv"))
        feasibility_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*feasibility.csv"))
        ICU_violation_patient_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*ICU_violation_patient_days.csv"))
        stage1_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*stage1_days.csv"))
        stage2_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*stage2_days.csv"))
        stage3_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*stage3_days.csv"))
        stage4_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*stage4_days.csv"))
        stage5_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*stage5_days.csv"))

        surge_days_filenames = glob.glob(os.path.join(output_dir, "peak" + str(peak) + "*surge_days.csv"))

        num_performance_measures = len(performance_measures_dicts)

        performance_measures_filenames = [cost_filenames, feasibility_filenames, ICU_violation_patient_days_filenames,
                                          stage1_days_filenames, stage2_days_filenames, stage3_days_filenames,
                                          stage4_days_filenames, stage5_days_filenames, surge_days_filenames]

        # These become the column names for dataframes in Set A
        performance_measures_strs = ["cost", "feasibility", "icu_violation_patient_days",
                                     "stage1_days", "stage2_days", "stage3_days", "stage4_days", "stage5_days", "surge_days"]

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
                df.to_csv(os.path.join(output_dir, "aggregated_peak" + str(peak) + "_policy" + str(key) + ".csv"))

        # Generate and export Set B dataframes
        for performance_measures_id in range(num_performance_measures):
            df = pd.DataFrame(performance_measures_dicts[performance_measures_id])
            df.to_csv(
                os.path.join(output_dir, "aggregated_peak" + str(peak) + "_" + str(performance_measures_strs[performance_measures_id]) + ".csv"))
