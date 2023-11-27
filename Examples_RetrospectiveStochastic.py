###############################################################################
# Examples_RetrospectiveStochastic.py
# This script contains examples of how to run the stochastic simulation
# for retrospective staged-alert system evaluation.

# Note: this must be run using multiple processors to work correctly,
#   since the multiprocessing package is used

# Nazlican Arslan 2023
###############################################################################

from Engine_SimObjects import MultiTierPolicy, CDCTierPolicy
from Engine_DataObjects import City, TierInfo, Vaccine
from Tools_Optimization_Utilities import evaluate_single_policy_on_sample_path, get_sample_paths
from Tools_Plot import plot_from_file, report_from_file, bar_plot_from_file

import datetime as dt
import multiprocessing as mp
import numpy as np
from Engine_SimModel import SimReplication

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
###############################################################################
# Define the staged-alert policy to run the simulation with.
# This script has Austin's and the CDC's staged-alert systems.

# tiers file contains the staged-alert levels with the corresponding transmission reductions:
# CDC system has three levels and Austin system has four levels.
tiers_CDC = TierInfo("austin", "tiers_CDC.json")
tiers_austin = TierInfo("austin", "tiers4.json")

# Define the threshold values for each indicator:
thresholds_austin = (-1, 15, 25, 50)  # Austin's system has one indicator.
mtp = MultiTierPolicy(austin, tiers_austin, thresholds_austin, None)

# the CDC's system has three indicators:
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, 10, 20), "surge": (-1, -1, 10)}
staffed_thresholds = {"non_surge": (-1, 0.1, 0.15), "surge": (-1, -1, 0.1)}

# The CDC's system uses case count as an indicator.
# We estimated real case count data as 40% of ToIY for Austin.
percentage_cases = 0.4
ctp = CDCTierPolicy(austin, tiers_CDC, case_threshold, hosp_adm_thresholds, staffed_thresholds, percentage_cases)

###############################################################################

# First: Generate plausible sample paths according to rsq values.
# This example code only run 2 parallel processors to generate 4 sample paths in total.
# Normally I run 10 parallel processors to generate 300 sample paths.
seeds = [1, 2]  # seeds for path generations
new_seeds = [3, 4]  # seeds for policy evaluations.
# seeds = np.arange(100, 160, 2)
# new_seeds = np.arange(200, 260, 2)

num_reps = 2  # 10

# Time points corresponds to the start date for each peak analysis:
time_points_rsq = [dt.datetime(2020, 5, 30),
                   dt.datetime(2020, 11, 30),
                   dt.datetime(2021, 7, 14),
                   dt.datetime(2021, 11, 30),
                   dt.datetime(2022, 4, 1)
                   ]

time_points_rsq = [austin.cal.calendar.index(date) for date in time_points_rsq]

if __name__ == '__main__':
    # Generate sample paths. Note sample path generation takes long
    for i in seeds:
        p = mp.Process(target=get_sample_paths, args=(austin, vaccines, 0.75, num_reps, i,
                                                      "input_output_folder/austin/base_files",
                                                      True, time_points_rsq[-1], time_points_rsq)
                       )
        p.start()
    for i in range(len(seeds)):
        p.join()

    ###############################################################################
    # Define the time horizon you want to run the policy on:
    # This example run the system for the first peak:
    # IMPORTANT!!:
    # The commented part is for implementing the policy:
    # I suggest either having separate scripts for sample path generation, policy evaluation and plotting.
    # or defining a class for this script.
    # Run the following commented out code after generating sample paths.

    history_end_time = dt.datetime(2020, 5, 30)  # use fixed transmission value until history en time.
    simulation_end_time = dt.datetime(2020, 10, 1)

    # for i in range(len(seeds)):
    #     base_filename = f"{seeds[i]}_"
    #     p = mp.Process(target=evaluate_single_policy_on_sample_path,
    #                    args=(austin,
    #                          vaccines,
    #                          ctp,
    #                          austin.cal.calendar.index(simulation_end_time),
    #                          austin.cal.calendar.index(history_end_time),
    #                          new_seeds[i],
    #                          num_reps,
    #                          base_filename,
    #                          "input_output_folder/austin/base_files",
    #                          "input_output_folder/austin")
    #                    )
    #
    #     p.start()
    # for i in range(len(seeds)):
    #     p.join()

    ###############################################################################
    # Read the outputs and plot the results:
    tier_colors_ctp = {0: "blue", 1: "gold", 2: "red"}
    tier_colors_mtp = {0: "blue", 1: "yellow", 2: "orange", 3: "red"}
    equivalent_thresholds = {"non_surge": (-1, 28.57, 57.14), "surge": (-1, -1, 28.57)}

    # I suggest separate script for plotting!!
    # Below is the commented out plotting codes:

    # report_template_ctp = "report_template_CDC.tex"
    # report_template_mtp = "report_template.tex"
    # plot_from_file(seeds, num_reps, austin, history_end_time, equivalent_thresholds, f"{str(ctp)}",
    #                tier_colors_ctp,
    #                "input_output_folder/austin")
    # report_from_file(seeds, num_reps, austin, history_end_time, simulation_end_time, f"{str(ctp)}",
    #                  tier_colors_ctp,
    #                  report_template_ctp, "input_output_folder/austin")

    # history_end_list = [dt.datetime(2020, 5, 30),
    #                     dt.datetime(2020, 11, 30),
    #                     dt.datetime(2021, 7, 14),
    #                     dt.datetime(2021, 11, 30)
    #                     ]
    #
    # simulation_end_list = [dt.datetime(2020, 10, 1),
    #                        dt.datetime(2021, 4, 1),
    #                        dt.datetime(2021, 11, 15),
    #                        dt.datetime(2022, 4, 1)
    #                        ]

    # bar_plot_from_file(seeds, num_reps, austin, history_end_list, simulation_end_list,
    #                    {f"{str(ctp)}": "CDC system", f"{policy_name_mtp}": "Austin system"},
    #                    {f"{str(ctp)}": tier_colors_ctp, f"{policy_name_mtp}": tier_colors_mtp},
    #                    {f"{str(ctp)}": {0: "gray"}, f"{policy_name_mtp}": {0: "gray"}},
    #                    {f"{str(ctp)}": report_template_ctp, f"{policy_name_mtp}": report_template_mtp},
    #                    "input_output_folder/austin")

