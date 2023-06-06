###############################################################################

# CDC_optimization.py

# This script contains beginning-to-end sample path generation
#   and evaluation of CDC policies. User can specify which CDC policies
#   they would like to evaluate.
# Can split up sample path generation and policy evaluation on
#   parallel processors using mpi4py.
# The number of sample paths generated (and number of replications
#   that each policy is evaluated on) is
#       total_num_processors x sample_paths_generated_per_processor
#           (total_num_processors) is inferred from mpi call
#           (sample_paths_generated_per_processor is a variable that is
#       specified in the code)

# Linda Pei 2023

###############################################################################

import copy
from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import OptTools

# Import other Python packages
import numpy as np
import time
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

tiers = TierInfo("austin", "tiers_CDC.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

###############################################################################

# Change to False if sample paths have already been generated
#   and we just need to do evaluation
need_sample_paths = True
sample_paths_generated_per_processor = 1

# If only interested in evaluating on subset of reps
num_reps_evaluated_per_policy = 10

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
if need_sample_paths:
    comm.Barrier()

###############################################################################

# Step 2: create list of policy objects
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (10, 20, 20), "surge": (-1, 10, 10)}
# staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}

# Nazli recommendations
# 100, 200, 500 and 1000 per 100k for case count indicators
# start from thresholds of 0 and 5 for the lowest stage
#   and increment from that point until 30 or 40 per 100k
# upper bound of 60% occupancy would suffice

policies = []

# This creates 165 policies
non_surge_staffed_thresholds_array = OptTools.thresholds_generator((-1, 0, 1),
                                                                   (0.05, 0.5, 0.05),
                                                                   (0.05, 0.5, 0.05),
                                                                   (0.05, 0.5, 0.05))

for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:
    staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                        non_surge_staffed_thresholds[3],
                                        non_surge_staffed_thresholds[4]),
                          "surge": (-1,
                                    non_surge_staffed_thresholds[3],
                                    non_surge_staffed_thresholds[3])}
    policy = CDCTierPolicy(austin,
                           tiers,
                           case_threshold,
                           hosp_adm_thresholds,
                           staffed_thresholds)
    policies.append(policy)


###############################################################################

# Step 3: create dictionary, where each entry corresponds to a peak
#   and contains list of SimReplication objects with loaded sample paths
#   for that peak
reps_per_peak_dict = {}
peaks_dates_strs = ["2020-03-24", "2020-05-31", "2021-07-14", "2021-11-30"]

for peak in np.arange(4):
    reps = []
    for rank in np.arange(total_num_processors):
        for sample_path_number in np.arange(sample_paths_generated_per_processor):
            prefix = str(rank) + "_" + str(sample_path_number) + "_" + peaks_dates_strs[peak] + "_"

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
    reps_per_peak_dict[peaks_dates_strs[peak]] = reps[:num_reps_evaluated_per_policy]

###############################################################################

# Step 4: split policies amongst processors and create RNG for each processor
# Some processors have base_assignment
# Others have base_assignment + 1
num_policies = len(policies)
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
peaks_end_times = [215, 397, 625, 762]
policy_ids_to_evaluate = np.arange(slicepoints[rank],slicepoints[rank + 1])

for peak in np.arange(4):
    reps = reps_per_peak_dict[peaks_dates_strs[peak]]
    end_time = peaks_end_times[peak]

    for policy_id in policy_ids_to_evaluate:

        policy = policies[policy_id]
        cost_per_rep = []
        feasibility_per_rep = []
        for rep in reps:

            new_rep = copy.deepcopy(rep)

            epi_rand = copy.deepcopy(rep.epi_rand)
            epi_rand.random_params_dict = rep.epi_rand.random_params_dict
            epi_rand.setup_base_params()

            new_rep.epi_rand = epi_rand

            new_rep.policy = policy
            new_rep.rng = np.random.Generator(bit_generator)
            new_rep.simulate_time_period(945)

            cost, feasibility = OptTools.evaluate_one_policy_one_sample_path(policy, new_rep, end_time)
            cost_per_rep.append(cost)
            feasibility_per_rep.append(feasibility)

            policy.reset()

        np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_cost.csv",
                   np.array(cost_per_rep), delimiter=",")
        np.savetxt("peak" + str(peak) + "_policy" + str(policy_id) + "_feasibility.csv",
                   np.array(feasibility_per_rep), delimiter=",")

###############################################################################
