from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from OptTools import evaluate_single_policy_on_sample_path, get_sample_paths
from Plotting import plot_from_file, report_from_file

import datetime as dt
import multiprocessing as mp
import numpy as np
from SimModel import SimReplication

austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "variant.json",
    "transmission.csv",
    "austin_real_hosp_updated.csv",
    "austin_real_icu_updated.csv",
    "austin_hosp_ad_updated.csv",
    "austin_real_death_from_hosp_updated.csv",
    "austin_real_death_from_home.csv",
    "variant_prevalence.csv"
)

tiers = TierInfo("austin", "tiers4.json")
tiers_CDC = TierInfo("austin", "tiers_CDC.json")
vaccines = Vaccine(
    austin,
    "austin",
    "vaccines.json",
    "booster_allocation_fixed.csv",
    "vaccine_allocation_fixed.csv",
)

###############################################################################
history_end_time = dt.datetime(2021, 11, 30)
simulation_end_time = dt.datetime(2022, 3, 31)
seeds = np.arange(100, 160, 2)
new_seeds = np.arange(200, 260, 2)
num_reps = 10
time_points = [dt.datetime(2020, 5, 30),
               dt.datetime(2020, 11, 30),
               dt.datetime(2021, 7, 14),
               dt.datetime(2021, 11, 30),
               dt.datetime(2022, 3, 30)
               ]

time_points = [austin.cal.calendar.index(date) for date in time_points]

case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, 10, 20), "surge": (-1, -1, 10)}
staffed_thresholds = {"non_surge": (-1, 0.1, 0.15), "surge": (-1, -1, 0.1)}
percentage_cases = 0.4
ctp = CDCTierPolicy(austin, tiers_CDC, case_threshold, hosp_adm_thresholds, staffed_thresholds, percentage_cases)

thresholds = (-1, 15, 25, 50)
mtp = MultiTierPolicy(austin, tiers, thresholds, None)

if __name__ == '__main__':
    # for i in seeds:
    #     p = mp.Process(target=get_sample_paths, args=(austin, vaccines, 0.75, num_reps, i, time_points))
    #     p.start()
    # for i in range(len(seeds)):
    #     p.join()

    # for i in range(len(seeds)):
    #     base_filename = f"{seeds[i]}_"
    #     evaluate_single_policy_on_sample_path(austin,
    #                                           vaccines,
    #                                           mtp,
    #                                           austin.cal.calendar.index(simulation_end_time),
    #                                           austin.cal.calendar.index(history_end_time),
    #                                           new_seeds[i],
    #                                           num_reps,
    #                                           base_filename)
    # for i in range(len(seeds)):
    #     base_filename = f"{seeds[i]}_"
    #     p = mp.Process(target=evaluate_single_policy_on_sample_path,
    #                    args=(austin,
    #                          vaccines,
    #                          mtp,
    #                          austin.cal.calendar.index(simulation_end_time),
    #                          austin.cal.calendar.index(history_end_time),
    #                          new_seeds[i],
    #                          num_reps,
    #                          base_filename)
    #                    )
    #     p.start()
    # for i in range(len(seeds)):
    #     p.join()

    tier_colors_ctp = {0: "blue", 1: "gold", 2: "red"}
    tier_colors_mtp = {0: "blue", 1: "yellow", 2: "orange", 3: "red"}
    equivalent_thresholds = {"non_surge": (-1, 28.57, 57.14), "surge": (-1, -1, 28.57)}
    policy_name_mtp = str(thresholds)
    policy_name_ctp = f"CDC_{case_threshold}"
    report_template_ctp = "report_template_CDC.tex"
    report_template_mtp = "report_template.tex"
    # plot_from_file(seeds, num_reps, austin, history_end_time, equivalent_thresholds, policy_name_ctp, tier_colors_ctp)
    report_from_file(seeds, num_reps, austin, history_end_time, simulation_end_time, policy_name_ctp, tier_colors_ctp,
                     report_template_ctp)
