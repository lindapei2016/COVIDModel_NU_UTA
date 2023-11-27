###############################################################################
#   ________________________________________________________________________
# ((                                                                        ))
#  ))     Script_CDCOptimization_LargeScaleSimulationReplications.py       ((
# ((                                                                        ))
#   ------------------------------------------------------------------------
#
# Working script (also a demo) using functions from
#   Tools_Optimization to generate large-scale
#   simulation output (scattered across many files)
# Specifically created for LP's CDC optimization (no wastewater)
#   -- so need to modify (particularly filenames) for other
#   purposes -- here, we have 3 peaks and performance measures
#   of days in stage 2, days in stage 3, and ICU violation patient
#   days for each peak
#
# IMPORTANT!
#   This script splits policies amongst processors --
#       if you want to split replications amongst processors instead
#       (e.g. if there are a small number of policies you want to
#       simulate for many replications), you can modify this script
#       and loop it multiple times with different settings in each
#       loop iteration (e.g. start on a different replication number,
#       change the filenames so they do not overwrite each other, etc...)
#       -- see task_number variable under TOGGLES AND OPTIONS
#
# STRUCTURE:
# STEP I: GENERATE SAMPLE PATHS
# STEP II: CREATE POLICY OBJECTS TO SIMULATE
# STEP III: LOAD SAMPLE PATHS
# STEP IV: SPLIT POLICIES AMONGST PROCESSORS
# STEP V: EVALUATE POLICIES
# STEP VI: AGGREGATE INDIVIDUAL FILES
#
# Additional LP CDC optimization notes:
#   Selected policies are the non-eliminated policies based on
#       (partial) KN using # red days as performance measure
#       and based on feasibility
#   Not doing last peak (peak 4 aka peak 3 under Python indexing)

###############################################################################

#######################################
############## IMPORTS ################
#######################################

import copy
from Engine_SimObjects import CDCTierPolicy
from Engine_DataObjects import City, TierInfo, Vaccine
from Engine_SimModel import SimReplication
import Tools_InputOutput
import Tools_Optimization_Utilities

import pandas as pd
import numpy as np
import glob

from mpi4py import MPI
from pathlib import Path

import sys

###############################################################################

###################################################
############## TOGGLES AND OPTIONS ################
###################################################

# task_number is useful for running this script in a loop,
#   with command line arguments to distinguish iterations
#   -- for example if we want to run a very large number of
#   total replications on a very small number of policies,
#   since this script splits policies amongst processors,
#   we can run this script multiple times corresponding
#   to subsets of this large number of total replications
task_number = sys.argv[1]

# Can set this to "task" + str(task_number) + "_" if running
#   this script in a loop, and want to distinguish loop iterations
#   using task_number -- this helps distinguish files from
#   different loop iterations
# Set this to blank string "" if not used
task_prefix = "task" + str(task_number) + "_"

# Change to False if sample paths have already been generated
need_sample_paths = False

# Different than num_processors_evaluation because
#   num_processors_sample_path is used for naming/distinguishing
#   states .json files
num_processors_sample_paths = 80
sample_paths_generated_per_processor = 100

# Name of folder in which states .json files are stored
states_storage_folder_name = "states"

# If only interested in evaluating on subset of reps,
#   not on all reps in states_storage_folder_name folder
num_reps_evaluated_per_policy = 300

# Reps offset: reps number to start on
reps_offset = 5000 + num_reps_evaluated_per_policy * task_number

# Set to True if have .csv files that specify smaller subset of
#   policies to simulate
subset_of_policies = True

# If True, only test 2 policies
using_test_set_only = False

###############################################################################

########################################################################
############## INITIALIZE TOP-SHELF / ENVIRONMENT ITEMS ################
########################################################################

base_path = Path(__file__).parent

comm = MPI.COMM_WORLD
num_processors_evaluation = comm.Get_size()
rank = comm.Get_rank()

# Assign bit generator
# Assume that seeds 0 through num_processors_evaluation-1 inclusively
#   were used for sample path generation
# But for safety we start from seed 1000 to start sampling (rather than
#   start from num_processors_evaluation, because if sample path generation
#   is run using, say, 300 processors, but evaluation is run using, say,
#   100 processors, then there will be seed overlap
# Right now, use a different bit generator for every parallel processor
bit_generator = np.random.MT19937(4000 + reps_offset + rank)

###############################################################################

##############################################################
############## INITIALIZE AUSTIN AND CDC INFO ################
##############################################################

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

peaks_start_times = (93, 276, 502, 641)
peaks_end_times = (215, 397, 625, 762)

peaks_total_hosp_beds = (3026, 3791, 3841, 3537)

###############################################################################

##############################################################
############## STEP I: GENERATE SAMPLE PATHS #################
##############################################################

# For each parallel processor, obtain 1 sample path for
#   each of the 4 peaks
# First timepoint of 25 is just to speed up sample path generation
#   using timeblocks method
# Timepoints corresponding to 93, 276, 502, and 641 correspond to
#   start of 4 peaks
if need_sample_paths:
    Tools_Optimization_Utilities.get_sample_paths(austin,
                                                  vaccines,
                                                  0.75,
                                                  sample_paths_generated_per_processor,
                                                  timepoints=(25, 93, 276, 502, 641),
                                                  processor_rank=rank,
                                                  save_intermediate_states=True,
                                                  storage_folder_name=states_storage_folder_name,
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

###########################################################################
############## STEP II: CREATE POLICY OBJECTS TO SIMULATE #################
###########################################################################

case_threshold = np.inf

pre_vaccine_policies = []
post_vaccine_policies = []

# Single-indicator policies

non_surge_hosp_adm_thresholds_array = Tools_Optimization_Utilities.thresholds_generator((-1, 0, 1),
                                                                                        (-1, 0, 1),
                                                                                        (0, 11, 1),
                                                                                        (5, 21, 1))

non_surge_staffed_thresholds_array = Tools_Optimization_Utilities.thresholds_generator((-1, 0, 1),
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

# 2-indicator policies
non_surge_hosp_adm_thresholds_array = Tools_Optimization_Utilities.thresholds_generator((-1, 0, 1),
                                                                                        (-1, 0, 1),
                                                                                        (0, 11, 1),
                                                                                        (5, 21, 1))

non_surge_staffed_thresholds_array = Tools_Optimization_Utilities.thresholds_generator((-1, 0, 1),
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

    for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:

        staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                            non_surge_staffed_thresholds[3],
                                            non_surge_staffed_thresholds[4]),
                              "surge": (-1,
                                        -1,
                                        non_surge_staffed_thresholds[3])}

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

# Also adding CDC policy
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
                                0.1)}

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

if using_test_set_only:
    pre_vaccine_policies = pre_vaccine_policies[:2]
    post_vaccine_policies = post_vaccine_policies[:2]

num_original_policies = len(pre_vaccine_policies)

###############################################################################

if subset_of_policies:
    subset_policies_ix = []

    # Get union of KN-surviving policies for peak 0, 1, 2
    # Here, peak 3 is the KN-surviving policies for the ACROSS-PEAK average
    for peak in np.arange(4):
        non_eliminated_feasible_policies = pd.read_csv("w503_3000reps_non_eliminated_feasible_policies_peak" + str(peak) +
                                                       ".csv", header=None)
        subset_policies_ix.append(np.asarray(non_eliminated_feasible_policies))

    # And append CDC policy because we want to simulate it!
    subset_policies_ix = set(np.concatenate(subset_policies_ix).flatten())
    subset_policies_ix = np.asarray(list(subset_policies_ix) + [num_original_policies - 1]).astype(int)
else:
    subset_policies_ix = [i for i in range(len(pre_vaccine_policies))]

###############################################################################

############################################################
############## STEP III: LOAD SAMPLE PATHS #################
############################################################

# Create dictionary, where each entry corresponds to a peak
#   and contains list of SimReplication objects with loaded sample paths
#   for that peak
reps_per_peak_dict = {}
peaks_dates_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]

rep_counter = 0

for peak in np.arange(3):
    reps = []
    for p in np.arange(num_processors_sample_paths):
        for sample_path_number in np.arange(sample_paths_generated_per_processor):
            rep_counter += 1
            if rep_counter < reps_offset:
                continue
            else:
                prefix = str(p) + "_" + str(sample_path_number) + "_" + peaks_dates_strs[peak] + "_"

                # Create a replication with no policy attached
                # Will edit the random number generator later, so seed does not matter
                rep = SimReplication(austin, vaccines, None, 1)
                Tools_InputOutput.import_rep_from_json(rep,
                                                       base_path / states_storage_folder_name / (prefix + "sim.json"),
                                                       base_path / states_storage_folder_name / (prefix + "v0.json"),
                                                       base_path / states_storage_folder_name / (prefix + "v1.json"),
                                                       base_path / states_storage_folder_name / (prefix + "v2.json"),
                                                       base_path / states_storage_folder_name / (prefix + "v3.json"),
                                                       None,
                                                       base_path / states_storage_folder_name / (prefix + "epi_params.json"))
                reps.append(rep)
        if len(reps) >= num_reps_evaluated_per_policy:
            break
    reps_per_peak_dict[peaks_dates_strs[peak]] = reps[:num_reps_evaluated_per_policy]

###############################################################################

###########################################################################
############## STEP IV: SPLIT POLICIES AMONGST PROCESSORS #################
###########################################################################

# Some processors have base_assignment
# Others have base_assignment + 1
num_policies = len(subset_policies_ix)
base_assignment = int(np.floor(num_policies / num_processors_evaluation))
leftover = num_policies % num_processors_evaluation

slicepoints = np.append([0],
                        np.cumsum(np.append(np.full(leftover, base_assignment + 1),
                                            np.full(num_processors_evaluation - leftover, base_assignment))))



###############################################################################

###########################################################
############## STEP V: EVALUATE POLICIES ##################
###########################################################

# policy_id is the actual policy id (index with respect to whole set of N policies)
# subset_policy_id is the index with respect to subset of N original policies selected for
#   additional replications

subset_policy_ids_to_evaluate = np.arange(slicepoints[rank], slicepoints[rank + 1])

for peak in np.arange(3):

    reps = reps_per_peak_dict[peaks_dates_strs[peak]]
    end_time = peaks_end_times[peak]

    for subset_policy_id in subset_policy_ids_to_evaluate:

        # Peak 0 and 1 are pre-vaccine, peak 2 (and 3, but we exclude last peak) are post-vaccine
        if peak == 0 or peak == 1:
            policy = np.array(pre_vaccine_policies)[subset_policies_ix][subset_policy_id]
        else:
            policy = np.array(post_vaccine_policies)[subset_policies_ix][subset_policy_id]

        # IMPORTANT! Change the total hospital capacity to be the per-peak capacity
        policy.specified_total_hosp_beds = peaks_total_hosp_beds[peak]

        policy_id = np.arange(num_original_policies)[subset_policies_ix][subset_policy_id]

        stage2_days_per_rep = []
        stage3_days_per_rep = []
        ICU_violation_patient_days_per_rep = []

        rep_counter = 0

        for rep in reps:

            rep_counter += 1

            new_rep = copy.deepcopy(rep)

            # Note -- sample paths also contain macrostochastic parameters (random parameters)
            epi_rand = copy.deepcopy(rep.epi_rand)
            epi_rand.random_params_dict = rep.epi_rand.random_params_dict
            epi_rand.setup_base_params()

            # Assign deep copy of epi parameters to new_rep,
            #   assign policy, assign random number generator
            new_rep.epi_rand = epi_rand
            new_rep.policy = policy
            new_rep.rng = np.random.Generator(bit_generator)

            # Note that this function returns a lot more statistics than we actually use --
            #   here we only use/need stage2_days, stage3_days, and ICU_violation_patient_days
            cost, feasibility, stage1_days, stage2_days, stage3_days, ICU_violation_patient_days, surge_days \
                = Tools_Optimization_Utilities.evaluate_one_policy_one_sample_path(policy, new_rep, end_time + 1)

            stage2_days_per_rep.append(stage2_days)
            stage3_days_per_rep.append(stage3_days)
            ICU_violation_patient_days_per_rep.append(ICU_violation_patient_days)

            policy.reset()

            # Save output
            if rep_counter == 1 or rep_counter % 100 == 0 or rep_counter == num_reps_evaluated_per_policy:
                np.savetxt(task_prefix + "peak" + str(peak) + "_policy" + str(policy_id) + "_stage2_days.csv",
                           np.array(stage2_days_per_rep), delimiter=",")
                np.savetxt(task_prefix + "peak" + str(peak) + "_policy" + str(policy_id) + "_stage3_days.csv",
                           np.array(stage3_days_per_rep), delimiter=",")
                np.savetxt(task_prefix + "peak" + str(peak) + "_policy" + str(policy_id) + "_ICU_violation_patient_days.csv",
                           np.array(ICU_violation_patient_days_per_rep), delimiter=",")
comm.Barrier()
if rank == 0:
    print("Evaluation completed.")

###############################################################################

#####################################################################
############## STEP VI: AGGREGATE INDIVIDUAL FILES ##################
#####################################################################

# Creates 1 dataframe for each performance measure -- columns are
#   policies, rows are replications

# Processors 0, 1, 2 respectively aggregate data on peaks 0, 1, 2
#   to speed this step up
if rank < 3:
    peak = rank

    ICU_violation_patient_days_dict = {}
    stage2_days_dict = {}
    stage3_days_dict = {}

    performance_measures_dicts = [ICU_violation_patient_days_dict,
                                  stage2_days_dict,
                                  stage3_days_dict]

    ICU_violation_patient_days_filenames = glob.glob(task_prefix + "peak" + str(peak) + "*ICU_violation_patient_days.csv")
    stage2_days_filenames = glob.glob(task_prefix + "peak" + str(peak) + "*stage2_days.csv")
    stage3_days_filenames = glob.glob(task_prefix + "peak" + str(peak) + "*stage3_days.csv")

    num_performance_measures = len(performance_measures_dicts)

    performance_measures_filenames = [ICU_violation_patient_days_filenames,
                                      stage2_days_filenames,
                                      stage3_days_filenames]

    # These become the column names for dataframes
    performance_measures_strs = ["icu_violation_patient_days",
                                 "stage2_days",
                                 "stage3_days"]

    # Open each .csv file and store its contents in various dataframes
    for performance_measures_id in range(num_performance_measures):
        for filename in performance_measures_filenames[performance_measures_id]:
            df = pd.read_csv(filename, header=None)
            policy_id = int(filename.split(task_prefix + "peak" + str(peak) + "_policy")[-1].split("_")[0])
            performance_measures_dicts[performance_measures_id][policy_id] = np.asarray(df[0])

    # Generate and export dataframes
    for performance_measures_id in range(num_performance_measures):
        df = pd.DataFrame(performance_measures_dicts[performance_measures_id])
        df.to_csv(
            task_prefix + "aggregated_peak" + str(peak) + "_" + str(performance_measures_strs[performance_measures_id]) + ".csv")


