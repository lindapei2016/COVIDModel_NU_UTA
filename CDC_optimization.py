###############################################################################

# CDC_optimization.py
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

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_total_workers = size - 1
master_rank = size - 1

###############################################################################

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

# This is very similar to how we run the usual code, we just need to define the staged-alert policy with the
# CDC system method. The system has three different indicators; Case counts, Hospital admissions and  Percent hospital
# beds. Depending on the case count threshold the other two indicators take different values. I define them as
# "non_surge" and "surge" but we can change those later if we want to do more general systems.

# start = time.time()

case_threshold = 200
hosp_adm_thresholds = {"non_surge": (10, 20, 20), "surge": (-1, 10, 10)}
# staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}

# CDC threshold uses 7-day sum of hospital admission per 100k. The equivalent values if we were to use 7-day avg.
# hospital admission instead are as follows. We use equivalent thresholds to plot and evaluate the results in our
# indicator. I used the same CDC thresholds all the time but if we decide to optimize CDC threshold, we can calculate
# the equivalent values in the model and save to the policy.json.
# equivalent_thresholds = {"non_surge": (-1, -1, 28.57, 57.14, 57.14), "surge": (-1, -1, -1, 28.57, 28.57)}
# ctp = CDCTierPolicy(austin, tiers, case_threshold, hosp_adm_thresholds, staffed_thresholds)

###############################################################################

# Nazli recommendations
# 100, 200, 500 and 1000 per 100k for case count indicators
# start from thresholds of 0 and 5 for the lowest stage
#   and increment from that point until 30 or 40 per 100k
# upper bound of 60% occupancy would suffice

policies = []

# This creates 66 policies
non_surge_staffed_thresholds_array = OptTools.thresholds_generator((-1, 0, 1),
                                                   (0.05, 0.6, 0.05),
                                                   (0.05, 0.6, 0.05),
                                                   (0.05, 0.6, 0.05))

for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:
    staffed_thresholds = {}
    staffed_thresholds["non_surge"] = (non_surge_staffed_thresholds[1],
                                       non_surge_staffed_thresholds[2],
                                       non_surge_staffed_thresholds[3])
    staffed_thresholds["surge"] = (-1,
                                   non_surge_staffed_thresholds[2],
                                   non_surge_staffed_thresholds[2])
    policy = CDCTierPolicy(austin,
                           tiers,
                           case_threshold,
                           hosp_adm_thresholds,
                           staffed_thresholds)
    policies.append(policy)

# For desktop debugging / if no MPI...
# rank = 1
# size = 2

# First peak only
OptTools.evaluate_policies_on_sample_paths(
        austin,
        vaccines,
        policies_array=policies,
        end_time=215,
        RNG=np.random.Generator(np.random.MT19937(100).jumped(rank)),
        num_reps=20,
        processor_rank=rank,
        processor_count_total=size,
        base_filename=str(1)+"_",
        storage_folder_name="states"
)