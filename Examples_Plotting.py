###############################################################################
# main_det_alert_systems.py
# This script contains examples of how to run plot various plots.

# Nazlican Arslan 2023
###############################################################################

from Engine_SimObjects import MultiTierPolicy, CDCTierPolicy, MultiTierPolicyWA
from Engine_DataObjects import City, TierInfo, Vaccine
from Engine_SimModel import SimReplication
from Tools_InputOutput import export_rep_to_json, import_rep_from_json
from Tools_Plot import plot_from_file
from Tools_InputOutput import import_stoch_reps_for_reporting
from Tools_Plot import Plot, find_central_path, BarPlot
from Tools_Optimization import get_sample_paths, evaluate_one_policy_one_sample_path_WA

# Import other Python packages
import datetime as dt
from pathlib import Path
import copy
import os
import numpy as np
import glob

# Let's first run a deterministic sample path with Austin's staged-alert system.
# The beginning is similar to main_det_alert_systems.py. You can skip to the plotting part.

base_path = Path(__file__).parent
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
history_end_time_list = [dt.datetime(2020, 5, 30), dt.datetime(2020, 11, 30), dt.datetime(2021, 7, 14),
                         dt.datetime(2021, 11, 30)]
simulation_end_time_list = [dt.datetime(2020, 10, 1), dt.datetime(2021, 4, 1), dt.datetime(2021, 11, 15),
                            dt.datetime(2022, 4, 1)]

# Inputs for path generation and policy evaluation
peaks_dates_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]
reps_per_peak_dict = {}
peaks_end_times = [215, 397, 625, 762]
num_paths = 2

# Generate sample paths
get_sample_paths(austin, vaccines, 0.75, num_paths, timepoints=(25, 93, 276, 502, 641),
                 processor_rank=0, save_intermediate_states=True, storage_folder_name="states",
                 fixed_kappa_end_date=763)

for peak in np.arange(4):
    reps = []
    for sample_path_number in np.arange(num_paths):
        prefix = str(0) + "_" + str(sample_path_number) + "_" + peaks_dates_strs[peak] + "_"

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
    reps_per_peak_dict[peaks_dates_strs[peak]] = reps

bit_generator = np.random.MT19937(1000)
for peak in np.arange(4):

    reps = reps_per_peak_dict[peaks_dates_strs[peak]]
    end_time = peaks_end_times[peak]

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
                = evaluate_one_policy_one_sample_path_WA(policy, new_rep, end_time)
        cost_per_rep.append(cost)
        feasibility_per_rep.append(feasibility)
        stage1_days_per_rep.append(stage1_days)
        stage2_days_per_rep.append(stage2_days)
        stage3_days_per_rep.append(stage3_days)
        ICU_violation_patient_days_per_rep.append(ICU_violation_patient_days)

        policy.reset()

        # Every 10 replications, save output
        if rep_counter % 10 == 0 or rep_counter == num_paths:
            np.savetxt("peak" + str(peak) + "_policyWA" + "_cost.csv",
                       np.array(cost_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_policyWA" + "_feasibility.csv",
                       np.array(feasibility_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_policyWA" + "_stage1_days.csv",
                       np.array(stage1_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_policyWA" + "_stage2_days.csv",
                       np.array(stage2_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_policyWA" + "_stage3_days.csv",
                       np.array(stage3_days_per_rep), delimiter=",")
            np.savetxt("peak" + str(peak) + "_policyWA" + "_ICU_violation_patient_days.csv",
                       np.array(ICU_violation_patient_days_per_rep), delimiter=",")


###############################################################################
"""
# Now let's plot what we have simulated. Normally I use the Plotting.ph as a helper function but let's skip that
# for a simple example.
# First I read the json files that I recorded at the end of simulation.
# Read the simulation outputs:

# This is a helper function I used to plot a bunch of simulation outcomes. The import_stoch_reps_reporting
# collect multiple sample path outputs and combine them in a list. So this is again a helper function.
sim_outputs, policy_outputs = import_stoch_reps_for_reporting([seed], 1, history_end_time, austin,
                                                              str(mtpWA), "input_output_folder/WA")

# Now I have my simulation outputs I can start plotting.
# Here this example is only for the deterministic path. So there is only one sample path.
# But you can plot more than one. In the stochastic plots we highlight one representative sample path that has the
# highest rsq value. You can find the central or representative path with the following function:

# Choose the central path among 300 sample paths (you don't need this here really but just to illustrate):
central_path_id = find_central_path(sim_outputs["ICU_history"],
                                    sim_outputs["IH_history"],
                                    austin.real_IH_history,
                                    austin.cal.calendar.index(history_end_time))

# The plotting file is very flexible you can plot any compartment you keep track of.
# And you can plot in different formats. Let's plot hospitalization data in different versions:
ToIHT_hist = sim_outputs["ToIHT_history"]  # grab the hospitalization data from the dictionary.
real_ToIHT_hist = austin.real_ToIHT_history  # let's also grab the real historical data.

# Let's define a Plot object:
plot = Plot(austin, history_end_time, real_ToIHT_hist, ToIHT_hist, "ToIHT_history", str(mtpWA), central_path_id,
            color=('k', 'silver'))

# Let's first plot the thresholds with horizontal lines in the background:
tier_colors_ctp = {0: "blue", 1: "yellow", 2: "orange", 3: "red"}  # define the plot colors you would like for each tier.
tier_colors_mtp = {0: "yellow", 1: "orange", 2: "red"}

# I use horizontal_plot to plot horizontal ones:

### Commented this piece out because it was still not running
#plot.horizontal_plot(policy_outputs["lockdown_thresholds"][0],tier_colors_mtp)

# Now let's plot ICU with Dali plots:
ICU_hist = sim_outputs["ICU_history"]
real_ICU_hist = austin.real_ICU_history
plot_icu_dali = Plot(austin, history_end_time, real_ICU_hist, ICU_hist, "ICU_history", str(mtpWA), central_path_id)
plot_icu_dali.dali_plot(policy_outputs["tier_history"], tier_colors_mtp, austin.icu)

# You can instead plot ICU with vertical background:
plot_icu_vertical = Plot(austin, history_end_time, real_ICU_hist, ICU_hist, "ICU_history", str(mtpWA), central_path_id)
plot_icu_vertical.vertical_plot(policy_outputs["tier_history"], tier_colors_mtp, austin.icu)

# There are even more plotting options in the Plot class.
"""
