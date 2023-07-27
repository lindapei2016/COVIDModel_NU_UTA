###############################################################################

# Based off of Script_CDCOptimization_FinerGrid.py (and thus
#   Script_CDCOptimization.py) -- heavily modified (lots of stuff removed)
# Simulating single-indicator optimal "to death" to use as constant
#   for PSS for optimization
# Each processor simulates the same single-indicator optimal
#   but for different sample paths using different RNGs
# Combine each processor's output after-the-fact to get a very refined
#   estimate of the single-indicator optimal sample mean

# Be aware: this is hardcoded for 20 parallel processors
#   and 8k sample paths (pre-generated) using 80 processors
#   each generating 100 sample paths

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
num_processors_sample_paths = 80
sample_paths_generated_per_processor = 100

# Change to False if evaluation is already done
need_evaluation = True

# If only interested in evaluating on subset of reps
# 8k reps total, using 20 processors, so each processor
#   simulates 400 replications
num_reps_evaluated_per_policy = 400

# Reps offset
# Rep number to start on
reps_offset = 0

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

# Step 2: create single-indicator optimal policy object
# Across-peak constrained optimal based off 100 reps of the following
#   2250 single indicator policies
# Hospital admits:
#   [1, 50), step size 1
# Staffed beds:
#   [0.01, 0.50), step size 0.01
# Using weights 1/10/100

case_threshold = 200

hosp_adm_thresholds = {"non_surge": (-1, 1, 17), "surge": (-1, -1, 1)}

staffed_thresholds = {"non_surge": (np.inf, np.inf, np.inf), "surge": (-1, -1, np.inf)}

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

###############################################################################

# Step 3: create dictionary, where each entry corresponds to a peak
#   and contains list of SimReplication objects with loaded sample paths
#   for that peak

# Each processor is assigned a peak
# This is certainly inefficient but for simplicity, right now each processor
#   reads in all 20k sample paths for its assigned peak (not sure how long
#   this will take) and then only simulates a subset of these sample paths

peaks_dates_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]

if need_evaluation:
    peak = rank % 4
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

    # Hardcoded -- 20 processors
    # 5 processors per peak
    # 8000/5 = 1600 reps per peak per processor
    reps = reps[(rank % 5) * 1600: ((rank % 5) + 1) * 1600]

###############################################################################

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

if need_evaluation:

    peak = rank % 4

    end_time = peaks_end_times[peak]

    if peak == 0 or peak == 1:
        policy = pre_vaccine_policy
    else:
        policy = post_vaccine_policy

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
        if rep_counter == 1 or rep_counter % 50 == 0 or rep_counter == num_reps_evaluated_per_policy:
            np.savetxt("peak" + str(peak) + "_singleindicatoroptimal_section" + str((rank % 5)) + "_stage2_days.csv",
                       np.array(stage2_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_singleindicatoroptimal_section" + str((rank % 5)) + "_stage3_days.csv",
                       np.array(stage3_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_singleindicatoroptimal_section" + str((rank % 5)) + "_ICU_violation_patient_days.csv",
                       np.array(ICU_violation_patient_days_per_rep), delimiter=",")

    comm.Barrier()
    if rank == 0:
        print("Evaluation completed.")

###############################################################################