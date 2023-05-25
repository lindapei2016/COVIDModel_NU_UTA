# Loading a replication state is more expensive than creating
#   a MultiTierPolicy object, so maybe provide an option for PASS to
#   loop (for each replication: for each system) in addition to the
#   current option which is to loop (for each system: for each replication).

# Remember to check boundary and adjust it for use cases

import sys
import numpy as np
from mpi4py import MPI
import pandas as pd
import time

import copy

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from InputOutputTools import import_rep_from_json, export_rep_to_json
from OptTools import get_sample_paths, thresholds_generator, \
    evaluate_policies_on_sample_paths, evaluate_single_policy_on_sample_path

from pathlib import Path

base_path = Path(__file__).parent

##################################################################################

austin = City("austin",
              "calendar.csv",
              "austin_setup.json",
              "variant.json",
              "transmission.csv",
              "austin_real_hosp_updated.csv",
              "austin_real_icu_updated.csv",
              "austin_hosp_ad_updated.csv",
              "austin_real_death_from_hosp_updated.csv",
              "austin_real_death_from_home.csv",
              "variant_prevalence.csv")

tiers = TierInfo("austin", "tiers5_opt_Final.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

##################################################################################

storage_folder_name = "states"


class PreloadedStates:
    def __init__(self):
        self.reps_dict = {}


preloaded = PreloadedStates()

policies = thresholds_generator((0, 10, 1), (0, 20, 2), (0, 50, 5), (0, 50, 10))

filename_prefixes = [str(i) + "_" + str(j) + "_" for i in range(1,100) for j in range(1,10+1)]


def simulation_model(system_id, worker_bit_generator, rep_number):

    mtp = MultiTierPolicy(austin, tiers, policies[system_id], "")

    if rep_number not in preloaded.reps_dict.keys():
        common_rep = SimReplication(austin, vaccines, None, None)
        import_rep_from_json(
            common_rep,
            base_path / storage_folder_name / (filename_prefixes[rep_number] + "sim.json"),
            base_path / storage_folder_name / (filename_prefixes[rep_number] + "v0.json"),
            base_path / storage_folder_name / (filename_prefixes[rep_number] + "v1.json"),
            base_path / storage_folder_name / (filename_prefixes[rep_number] + "v2.json"),
            base_path / storage_folder_name / (filename_prefixes[rep_number] + "v3.json"),
            None,
            base_path / storage_folder_name / (filename_prefixes[rep_number] + "epi_params.json"),
        )
        preloaded.reps_dict[rep_number] = common_rep

    # Recall that import_rep_from_json calls epi_rand.update_base_params().
    # even if we don't use import_rep_from_json again, we still need
    #   to update the epidemiological parameters that depend
    #   on the randomly sampled parameters loaded from the corresponding
    #   .json file.

    common_rep = preloaded.reps_dict[rep_number]

    rep = copy.deepcopy(common_rep)

    epi_rand = copy.deepcopy(common_rep.epi_rand)
    epi_rand.random_params_dict = common_rep.epi_rand.random_params_dict
    epi_rand.setup_base_params()

    rep.epi_rand = epi_rand

    rep.policy = mtp
    rep.rng = np.random.Generator(worker_bit_generator)
    rep.simulate_time_period(945)

    cost = rep.compute_cost()

    return -1 * cost


##################################################################################

true_standard = -np.inf

total_num = len(policies)

true_means = np.full(total_num, -np.inf)
true_variances = np.full(total_num, np.inf)


##################################################################################

def boundary_function(t):
    '''
    Boundary function
    '''
    return -1 * np.sqrt((8.6 + np.log(t + 1)) * (t + 1))


##################################################################################

def run_length(n):
    if n == 0:
        return 10
    else:
        return 10


def update_standard(running_sums, reps, contenders):
    standard = np.average(np.divide(running_sums[contenders], reps[contenders]))
    return standard


init_standard = -np.inf
known_variance = False
scaling_type = "custom"
num_cycles = np.inf
output_mode = "profile"

max_total_reps = np.full(total_num, len(filename_prefixes))

base_bit_generator_seed = 1000