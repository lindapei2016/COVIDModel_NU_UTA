###############################################################################

# CDC_optimization.py
# Linda Pei 2023

###############################################################################

import copy
from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
from OptTools import thresholds_generator,\
    get_sample_paths, \
    evaluate_policies_on_sample_paths, \
    aggregate_evaluated_policies

# Import other Python packages
import numpy as np
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_total_workers = size - 1
master_rank = size - 1

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

# Step 1: generate sample paths
# For each parallel processor, obtain 50 sample paths for
#   each of the 4 peaks
# First timepoint of 25 is just to speed up sample path generation
#   using timeblocks method
# Timepoints corresponding to 93, 276, 502, and 641 correspond to
#   start of 4 peaks
get_sample_paths(austin,
                 vaccines,
                 0.75,
                 50,
                 timepoints=(25, 93, 276, 502, 641),
                 processor_rank=rank,
                 save_intermediate_states=True,
                 storage_folder_name="states",
                 fixed_kappa_end_date=763)

# Step 2:
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (10, 20, 20), "surge": (-1, 10, 10)}
# staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}


###############################################################################

# Nazli recommendations
# 100, 200, 500 and 1000 per 100k for case count indicators
# start from thresholds of 0 and 5 for the lowest stage
#   and increment from that point until 30 or 40 per 100k
# upper bound of 60% occupancy would suffice

policies = []

# This creates 66 policies
non_surge_staffed_thresholds_array = thresholds_generator((-1, 0, 1),
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


# First peak only
evaluate_policies_on_sample_paths(
        austin,
        vaccines,
        policies_array=policies,
        end_time=215,
        RNG=np.random.Generator(np.random.MT19937(100).jumped(rank)),
        num_reps=50,
        processor_rank=rank,
        processor_count_total=size,
        base_filename=str(1)+"_",
        storage_folder_name="states"
)

evaluate_policies_on_sample_paths(
        austin,
        vaccines,
        policies_array=policies,
        end_time=397,
        RNG=np.random.Generator(np.random.MT19937(100).jumped(rank)),
        num_reps=50,
        processor_rank=rank,
        processor_count_total=size,
        base_filename=str(1)+"_",
        storage_folder_name="states"
)

evaluate_policies_on_sample_paths(
        austin,
        vaccines,
        policies_array=policies,
        end_time=625,
        RNG=np.random.Generator(np.random.MT19937(100).jumped(rank)),
        num_reps=50,
        processor_rank=rank,
        processor_count_total=size,
        base_filename=str(1)+"_",
        storage_folder_name="states"
)

evaluate_policies_on_sample_paths(
        austin,
        vaccines,
        policies_array=policies,
        end_time=762,
        RNG=np.random.Generator(np.random.MT19937(100).jumped(rank)),
        num_reps=50,
        processor_rank=rank,
        processor_count_total=size,
        base_filename=str(1)+"_",
        storage_folder_name="states"
)

aggregate_evaluated_policies(50, 165)

###############################################################################