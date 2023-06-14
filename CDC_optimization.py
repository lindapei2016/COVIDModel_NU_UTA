###############################################################################

# CDC_optimization.py

# This script contains beginning-to-end sample path generation
#   and evaluation of CDC policies. User can specify which CDC policies
#   they would like to evaluate.
# Can split up sample path generation and policy evaluation on
#   parallel processors using ''mpi4py.''
# The number of sample paths generated (and number of replications
#   that each policy is evaluated on) is
#       total_num_processors x sample_paths_generated_per_processor
#           (total_num_processors) is inferred from mpi call
#           (sample_paths_generated_per_processor is a variable that is
#       specified in the code)

# Linda Pei 2023

###############################################################################

import copy
from SimObjects import CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import OptTools

import pandas as pd

# Import other Python packages
import numpy as np
import glob

from mpi4py import MPI
from pathlib import Path

base_path = Path(__file__).parent

comm = MPI.COMM_WORLD
total_num_processors = comm.Get_size()
rank = comm.Get_rank()

###############################################################################

# Notice that we are using tiers_CDC.json here for the tiers
# Initialize key instances for the Austin simulation

austin = City("austin",
              "calendar.csv",
              "austin_setup.json",
              "variant.json",
              "transmission.csv",
              "austin_hospital_home_timeseries.csv",
              "variant_prevalence.csv")

pre_vaccine_tiers = TierInfo("austin", "tiers_CDC.json")
post_vaccine_tiers = TierInfo("austin", "tiers_CDC_reduced_values.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

###############################################################################

# OPTIONS
# Toggle True/False or specify values for customization

# Change to False if sample paths have already been generated
need_sample_paths = False
sample_paths_generated_per_processor = 1

# Change to False if evaluation is already done
need_evaluation = True

# If only interested in evaluating on subset of reps
num_reps_evaluated_per_policy = 100

# Reps offset
# Rep number to start on
reps_offset = 0

# Assume that the number of processors >= 4
# When True, for parsing, will use 4 processors and give
#   1 peak to each processor
split_peaks_amongst_processors = True

# Change to True if also want to automatically parse files
need_parse = True

###############################################################################

# Step 1: generate sample paths
# For each parallel processor, obtain 1 sample path for
#   each of the 4 peaks
# First timepoint of 25 is just to speed up sample path generation
#   using timeblocks method
# Timepoints corresponding to 93, 276, 502, and 641 correspond to
#   start of 4 peaks
if need_sample_paths:
    OptTools.get_sample_paths(austin,
                              vaccines,
                              0.75,
                              sample_paths_generated_per_processor,
                              timepoints=(25, 93, 276, 502, 641),
                              processor_rank=rank,
                              save_intermediate_states=True,
                              storage_folder_name="states",
                              fixed_kappa_end_date=763)

    # Force synchronization step so that all sample paths actually exist
    #   before evaluation begins
    # Otherwise, might have situation where one processor finishes
    #   sample paths earlier and begins evaluation, but other processors
    #   have not finished sample paths and thus their sample paths files do not
    #   yet exist, causing file read errors
    comm.Barrier()
    if rank == 0:
        print("Sample path generation completed.")

###############################################################################

# Step 2: create list of policy objects

# Nazli recommendations
# 100, 200, 500 and 1000 per 100k for case count indicators
# start from thresholds of 0 and 5 for the lowest stage
#   and increment from that point until 30 or 40 per 100k
# upper bound of 60% occupancy would suffice

case_threshold = 200

pre_vaccine_policies = []
post_vaccine_policies = []

# This creates 1296 policies
non_surge_staffed_thresholds_array = OptTools.thresholds_generator((-1, 0, 1),
                                                                   (-1, 0, 1),
                                                                   (0, 0.4, 0.05),
                                                                   (0, 0.4, 0.05))

non_surge_hosp_adm_thresholds_array = OptTools.thresholds_generator((-1, 0, 1),
                                                                    (-1, 0, 1),
                                                                    (0, 40, 5),
                                                                    (0, 40, 5))

for non_surge_hosp_adm_thresholds in non_surge_hosp_adm_thresholds_array:

    hosp_adm_thresholds = {"non_surge": (non_surge_hosp_adm_thresholds[2],
                                         non_surge_hosp_adm_thresholds[3],
                                         non_surge_hosp_adm_thresholds[4]),
                           "surge": (-1,
                                     non_surge_hosp_adm_thresholds[2],
                                     non_surge_hosp_adm_thresholds[2])}

    for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:
        staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                            non_surge_staffed_thresholds[3],
                                            non_surge_staffed_thresholds[4]),
                              "surge": (-1,
                                        non_surge_staffed_thresholds[2],
                                        non_surge_staffed_thresholds[2])}
        pre_vaccine_policy = CDCTierPolicy(austin,
                                           pre_vaccine_tiers,
                                           case_threshold,
                                           hosp_adm_thresholds,
                                           staffed_thresholds)

        post_vaccine_policy = CDCTierPolicy(austin,
                                            post_vaccine_tiers,
                                            case_threshold,
                                            hosp_adm_thresholds,
                                            staffed_thresholds)
        pre_vaccine_policies.append(pre_vaccine_policy)
        post_vaccine_policies.append(post_vaccine_policy)

# print(len(pre_vaccine_policies))
# breakpoint()

###############################################################################

# Step 3: create dictionary, where each entry corresponds to a peak
#   and contains list of SimReplication objects with loaded sample paths
#   for that peak
reps_per_peak_dict = {}
peaks_dates_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]

if need_evaluation:
    for peak in np.arange(4):
        reps = []
        for p in np.arange(total_num_processors):
            for sample_path_number in np.arange(sample_paths_generated_per_processor):
                prefix = str(p) + "_" + str(sample_path_number) + "_" + peaks_dates_strs[peak] + "_"

                # Create a rep with no policy attached
                # Will edit the random number generator later, so seed does not matter
                rep = SimReplication(austin, vaccines, None, 1000)
                InputOutputTools.import_rep_from_json(rep,
                                                      base_path / "states" / (prefix + "sim.json"),
                                                      base_path / "states" / (prefix + "v0.json"),
                                                      base_path / "states" / (prefix + "v1.json"),
                                                      base_path / "states" / (prefix + "v2.json"),
                                                      base_path / "states" / (prefix + "v3.json"),
                                                      None,
                                                      base_path / "states" / (prefix + "epi_params.json"))
                reps.append(rep)
            if len(reps) >= num_reps_evaluated_per_policy + reps_offset:
                break
        reps_per_peak_dict[peaks_dates_strs[peak]] = reps[reps_offset:]

###############################################################################

# Step 4: split policies amongst processors and create RNG for each processor
# Some processors have base_assignment
# Others have base_assignment + 1
num_policies = len(pre_vaccine_policies)
base_assignment = int(np.floor(num_policies / total_num_processors))
leftover = num_policies % total_num_processors

slicepoints = np.append([0],
                        np.cumsum(np.append(np.full(leftover, base_assignment + 1),
                                            np.full(total_num_processors - leftover, base_assignment))))

# Assuming that seeds 0 through total_num_processors-1 inclusively
#   were used for sample path generation, use seed total_num_processors
#   to start sampling
# Right now, use a different bit generator for every parallel processor

bit_generator = np.random.MT19937(total_num_processors + rank)

###############################################################################

# Step 5: evaluate policies
peaks_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]
policy_ids_to_evaluate = np.arange(slicepoints[rank], slicepoints[rank + 1])

if need_evaluation:
    for peak in np.arange(4):

        reps = reps_per_peak_dict[peaks_dates_strs[peak]]
        end_time = peaks_end_times[peak]

        for policy_id in policy_ids_to_evaluate:

            if peak == 0 or peak == 1:
                policy = pre_vaccine_policies[policy_id]
            else:
                policy = post_vaccine_policies[policy_id]

            cost_per_rep = []
            feasibility_per_rep = []
            stage1_days_per_rep = []
            stage2_days_per_rep = []
            stage3_days_per_rep = []
            ICU_violation_patient_days_per_rep = []

            rep_counter = 0

            for rep in reps:

                rep_counter += 1

                new_rep = copy.deepcopy(rep)

                epi_rand = copy.deepcopy(rep.epi_rand)
                epi_rand.random_params_dict = rep.epi_rand.random_params_dict
                epi_rand.setup_base_params()

                new_rep.epi_rand = epi_rand

                new_rep.policy = policy
                new_rep.rng = np.random.Generator(bit_generator)

                cost, feasibility, stage1_days, stage2_days, stage3_days, ICU_violation_patient_days \
                    = OptTools.evaluate_one_policy_one_sample_path(policy, new_rep, end_time)
                cost_per_rep.append(cost)
                feasibility_per_rep.append(feasibility)
                stage1_days_per_rep.append(stage1_days)
                stage2_days_per_rep.append(stage2_days)
                stage3_days_per_rep.append(stage3_days)
                ICU_violation_patient_days_per_rep.append(ICU_violation_patient_days)

                policy.reset()

                # Every 10 replications, save output
                if rep_counter % 10 == 0 or rep_counter == num_reps_evaluated_per_policy:
                    np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_cost.csv",
                               np.array(cost_per_rep), delimiter=",")
                    np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_feasibility.csv",
                               np.array(feasibility_per_rep), delimiter=",")
                    np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_stage1_days.csv",
                               np.array(stage1_days_per_rep), delimiter=",")
                    np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_stage2_days.csv",
                               np.array(stage2_days_per_rep), delimiter=",")
                    np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_stage3_days.csv",
                               np.array(stage3_days_per_rep), delimiter=",")
                    np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_ICU_violation_patient_days.csv",
                               np.array(ICU_violation_patient_days_per_rep), delimiter=",")
    comm.Barrier()
    if rank == 0:
        print("Evaluation completed.")

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

if need_parse:
    for peak in np.arange(4):

        if split_peaks_amongst_processors:
            if rank > 4:
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

        performance_measures_dicts = [cost_dict, feasibility_dict, ICU_violation_patient_days_dict,
                                      stage1_days_dict, stage2_days_dict, stage3_days_dict]

        cost_filenames = glob.glob("peak" + str(peak) + "*cost.csv")
        feasibility_filenames = glob.glob("peak" + str(peak) + "*feasibility.csv")
        ICU_violation_patient_days_filenames = glob.glob("peak" + str(peak) + "*ICU_violation_patient_days.csv")
        stage1_days_filenames = glob.glob("peak" + str(peak) + "*stage1_days.csv")
        stage2_days_filenames = glob.glob("peak" + str(peak) + "*stage2_days.csv")
        stage3_days_filenames = glob.glob("peak" + str(peak) + "*stage3_days.csv")

        num_performance_measures = 6

        performance_measures_filenames = [cost_filenames, feasibility_filenames, ICU_violation_patient_days_filenames,
                                          stage1_days_filenames, stage2_days_filenames, stage3_days_filenames]

        # These become the column names for dataframes in Set A
        performance_measures_strs = ["cost", "feasibility", "icu_violation_patient_days",
                                     "stage1_days", "stage2_days", "stage3_days"]

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