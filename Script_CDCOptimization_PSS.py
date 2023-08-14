###############################################################################

# Based off of Script_CDCOptimization_FinerGrid.py (and thus
#   Script_CDCOptimization.py)
# Comparing to "constant" (i.e. policy that has been simulated to death)
#   in an embarrassingly parallel way -- no message passing or communication
#   necessary

###############################################################################

import copy
from Engine_SimObjects import CDCTierPolicy
from Engine_DataObjects import City, TierInfo, Vaccine
from Engine_SimModel import SimReplication
import Tools_InputOutput
import Tools_Optimization

import pandas as pd

# Import other Python packages
import numpy as np
import glob

from mpi4py import MPI
from pathlib import Path

base_path = Path(__file__).parent

comm = MPI.COMM_WORLD
num_processors_evaluation = comm.Get_size()
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

# Different than num_processors_evaluation because
#   num_processors_sample_path is used for naming/distinguishing
#   states .json files
num_processors_sample_paths = 300
sample_paths_generated_per_processor = 1

# Change to False if evaluation is already done
need_evaluation = True

# Change to False if allowing 2 indicators
single_indicator_policies = False

# If only interested in evaluating on subset of reps
num_reps_evaluated_per_policy = 100

# Reps offset
# Rep number to start on
reps_offset = 0

# If True, only test 2 policies
using_test_set_only = False

###############################################################################

# Step 1: generate sample paths
# For each parallel processor, obtain 1 sample path for
#   each of the 4 peaks
# First timepoint of 25 is just to speed up sample path generation
#   using timeblocks method
# Timepoints corresponding to 93, 276, 502, and 641 correspond to
#   start of 4 peaks
if need_sample_paths:
    Tools_Optimization.get_sample_paths(austin,
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

if single_indicator_policies:
    non_surge_hosp_adm_thresholds_array = Tools_Optimization.thresholds_generator((-1, 0, 1),
                                                                                  (-1, 0, 1),
                                                                                  (0, 51, 1),
                                                                                  (0, 51, 1))
    non_surge_staffed_thresholds_array = Tools_Optimization.thresholds_generator((-1, 0, 1),
                                                                                 (-1, 0, 1),
                                                                                 (0, .51, .01),
                                                                                 (0, .51, .01))
else:
    non_surge_hosp_adm_thresholds_array = Tools_Optimization.thresholds_generator((-1, 0, 1),
                                                                                  (-1, 0, 1),
                                                                                  (0, 10, 1),
                                                                                  (17, 51, 1))
    non_surge_staffed_thresholds_array = Tools_Optimization.thresholds_generator((-1, 0, 1),
                                                                                 (-1, 0, 1),
                                                                                 (0, .2, 0.05),
                                                                                 (0.25, 0.55, 0.05))

if single_indicator_policies:
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
else:
    # both indicators but compare to single-indicator solutions that were feasible
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

if using_test_set_only:
    pre_vaccine_policies = pre_vaccine_policies[:2]
    post_vaccine_policies = post_vaccine_policies[:2]

###############################################################################

# Step 3: create dictionary, where each entry corresponds to a peak
#   and contains list of SimReplication objects with loaded sample paths
#   for that peak
reps_per_peak_dict = {}
peaks_dates_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]

if need_evaluation:
    for peak in np.arange(4):
        reps = []
        for p in np.arange(num_processors_sample_paths):
            for sample_path_number in np.arange(sample_paths_generated_per_processor):
                prefix = str(p) + "_" + str(sample_path_number) + "_" + peaks_dates_strs[peak] + "_"

                # Create a rep with no policy attached
                # Will edit the random number generator later, so seed does not matter
                rep = SimReplication(austin, vaccines, None, 1000)
                Tools_InputOutput.import_rep_from_json(rep,
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
base_assignment = int(np.floor(num_policies / num_processors_evaluation))
leftover = num_policies % num_processors_evaluation

slicepoints = np.append([0],
                        np.cumsum(np.append(np.full(leftover, base_assignment + 1),
                                            np.full(num_processors_evaluation - leftover, base_assignment))))

# Assume that seeds 0 through num_processors_evaluation-1 inclusively
#   were used for sample path generation
# But for safety we start from seed 1000 to start sampling (rather than
#   start from num_processors_evaluation, because if sample path generation
#   is run using, say, 300 processors, but evaluation is run using, say,
#   100 processors, then there will be seed overlap
# Right now, use a different bit generator for every parallel processor

bit_generator = np.random.MT19937(1000 + rank)

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

                cost, feasibility, stage1_days, stage2_days, stage3_days, ICU_violation_patient_days, surge_days \
                    = Tools_Optimization.evaluate_one_policy_one_sample_path(policy, new_rep, end_time)
                stage2_days_per_rep.append(stage2_days)
                stage3_days_per_rep.append(stage3_days)
                ICU_violation_patient_days_per_rep.append(ICU_violation_patient_days)

                policy.reset()

                # Every 10 replications, save output
                if rep_counter == 10 or rep_counter % 50 == 0 or rep_counter == num_reps_evaluated_per_policy:
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