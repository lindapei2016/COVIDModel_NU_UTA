###############################################################################
# Script_Washington.py
# This script does sample path generation and policy evaluation for Washington

# Karan Agrawal 2023
###############################################################################

from Engine_SimObjects import MultiTierPolicy, MultiTierPolicyWA
from Engine_DataObjects import City, TierInfo, Vaccine
from Engine_SimModel import SimReplication
from Tools_InputOutput import import_rep_from_json
from Tools_Optimization import get_sample_paths, evaluate_one_policy_one_sample_path_WA

# Import other Python packages
import datetime as dt
from pathlib import Path
import copy
import numpy as np
import glob
from mpi4py import MPI
import pandas as pd

# Let's first run a deterministic sample path with Austin's staged-alert system.
# The beginning is similar to main_det_alert_systems.py. You can skip to the plotting part.

base_path = Path(__file__).parent

comm = MPI.COMM_WORLD
# num_processors = comm.Get_size()
rank = comm.Get_rank()
###############################################################################
# Create a city object:
austin = City(
    "austin",
    "calendar.csv",
    "austin_setup.json",
    "variant.json",
    "transmission.csv",
    "austin_hospital_home_timeseries.csv",
    "variant_prevalence.csv"
)

vaccines = Vaccine(
    austin,
    "austin",
    "vaccines.json",
    "booster_allocation_fixed.csv",
    "vaccine_allocation_fixed.csv",
)

# Austin policy making
tiers_austin = TierInfo("austin", "tiers4.json")
thresholds_austin = (-1, 15, 25, 50)  # Austin's system has one indicator.
mtp = MultiTierPolicy(austin, tiers_austin, thresholds_austin, "green")

# Washington policy making
tiers_WA = TierInfo("austin", "tiers_CDC.json")
tiers_WA_reduced = TierInfo("austin", "tiers_CDC_reduced_values.json")
threshold_WA_case = (-1, 200, 350)
threshold_WA_hosp = (-1, 5, 10)
threshold_WA_ICU = (-1, 0.9)
mtpWA = MultiTierPolicyWA(austin, tiers_WA_reduced, threshold_WA_case, threshold_WA_hosp, threshold_WA_ICU)

# Set dates
history_end_time_list = [dt.datetime(2020, 5, 31), dt.datetime(2020, 11, 30), dt.datetime(2021, 7, 14),
                         dt.datetime(2021, 11, 30)]
simulation_end_time_list = [dt.datetime(2020, 10, 1), dt.datetime(2021, 4, 1), dt.datetime(2021, 11, 15),
                            dt.datetime(2022, 4, 1)]

# Inputs for path generation and policy evaluation
peaks_starts_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]
peaks_ends_strs = ["2020-10-01", "2021-4-01", "2021-11-15", "2022-04-01"]
reps_per_peak_dict = {}
peak_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]
num_processors = 2
num_paths_per_proc = 2

# Generate sample paths
get_sample_paths(austin, vaccines, 0.75, num_paths_per_proc, timepoints=(25, 93, 276, 502, 641),
                 processor_rank=rank, save_intermediate_states=True, storage_folder_name="states",
                 fixed_kappa_end_date=763)
print("Sample path generation complete")

for peak in np.arange(4):
    reps = []
    for sample_path_number in np.arange(num_paths_per_proc):
        prefix = str(rank) + "_" + str(sample_path_number) + "_" + peaks_starts_strs[peak] + "_"

        # Create a rep with no policy attached
        # Will edit the random number generator later, so seed does not matter
        rep = SimReplication(austin, vaccines, None, 1000)
        import_rep_from_json(rep,
                             base_path / "states" / (prefix + "sim.json"),
                             base_path / "states" / (prefix + "v0.json"),
                             base_path / "states" / (prefix + "v1.json"),
                             base_path / "states" / (prefix + "v2.json"),
                             base_path / "states" / (prefix + "v3.json"),
                             None,
                             base_path / "states" / (prefix + "epi_params.json"))
        reps.append(rep)
    reps_per_peak_dict[peaks_starts_strs[peak]] = reps

bit_generator = np.random.MT19937(1000 + rank)
for peak in np.arange(4):

    reps = reps_per_peak_dict[peaks_starts_strs[peak]]
    end_time = simulation_end_time_list[peak]
    end_time_index = peaks_end_times[peak]
    start_time = history_end_time_list[peak]
    start_time_index = peak_start_times[peak]

    if peak == 0 or peak == 1:
        policy = MultiTierPolicyWA(austin, tiers_WA, threshold_WA_case, threshold_WA_hosp, threshold_WA_ICU)
    else:
        policy = MultiTierPolicyWA(austin, tiers_WA_reduced, threshold_WA_case, threshold_WA_hosp, threshold_WA_ICU)

    cost_per_rep = []
    feasibility_per_rep = []
    stage1_days_per_rep = []
    stage2_days_per_rep = []
    stage3_days_per_rep = []
    ICU_violation_patient_days_per_rep = []
    ICU_peak_list = []
    IHT_peak_list = []
    triggerCase = []
    triggerHosp = []
    triggerIcu = []
    triggerCaseAndHosp = []

    print("Policy generation done")

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

        print("Policy evaluation done")

        cost, feasibility, stage1_days, stage2_days, stage3_days, ICU_violation_patient_days, ICU_peak, IHT_peak, \
            ICU_array, IHT_array, triggerCase_days, triggerHosp_days, triggerIcu_days, triggerCaseAndHosp_days \
            = evaluate_one_policy_one_sample_path_WA(policy, new_rep, end_time_index)
        cost_per_rep.append(cost)
        feasibility_per_rep.append(feasibility)
        stage1_days_per_rep.append(stage1_days)
        stage2_days_per_rep.append(stage2_days)
        stage3_days_per_rep.append(stage3_days)
        ICU_violation_patient_days_per_rep.append(ICU_violation_patient_days)
        ICU_peak_list.append(ICU_peak)
        IHT_peak_list.append(IHT_peak)
        triggerCase.append(triggerCase_days)
        triggerHosp.append(triggerHosp_days)
        triggerIcu.append(triggerIcu_days)
        triggerCaseAndHosp.append(triggerCaseAndHosp_days)
        np.savetxt("peak" + str(peak) + "_" + str(rank) + "_" + str(rep_counter) + "_policyWA_ICU", ICU_array,
                   delimiter=",")
        np.savetxt("peak" + str(peak) + "_" + str(rank) + "_" + str(rep_counter) + "_policyWA_IHT", IHT_array,
                   delimiter=",")

        print("Basic stats complete")

        policy.reset()

        # Every 10 replications, save output
        if rep_counter % 10 == 0 or rep_counter == num_paths_per_proc:
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_cost.csv",
                       np.array(cost_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_feasibility.csv",
                       np.array(feasibility_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_stage1_days.csv",
                       np.array(stage1_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_stage2_days.csv",
                       np.array(stage2_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_stage3_days.csv",
                       np.array(stage3_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_ICU_violation_patient_days.csv",
                       np.array(ICU_violation_patient_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_ICU_peak.csv",
                       np.array(ICU_peak_list), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_IHT_peak.csv",
                       np.array(IHT_peak_list), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_triggerCase.csv",
                       np.array(triggerCase), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_triggerHosp.csv",
                       np.array(triggerHosp), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_triggerIcu.csv",
                       np.array(triggerIcu), delimiter=",")
            np.savetxt("peak" + str(peak) + "_" + str(rank) + "_policyWA_0" + "_triggerCaseAndHosp.csv",
                       np.array(triggerCaseAndHosp), delimiter=",")


