from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from OptTools import evaluate_single_policy_on_sample_path, get_sample_paths
from InputOutputTools import export_rep_to_json
from Plotting import plot_from_file

# Import other Python packages
import numpy as np
import datetime as dt
from pathlib import Path

base_path = Path(__file__).parent

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
tiers_CDC = TierInfo("austin", "tiers_CDC.json")
tiers = TierInfo("austin", "tiers5_opt_Final.json")
vaccines = Vaccine(
    austin,
    "austin",
    "vaccines.json",
    "booster_allocation_fixed.csv",
    "vaccine_allocation_fixed.csv",
)

###############################################################################
thresholds = (-1, 0, 15, 25, 50)
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
history_end_time = dt.datetime(2022, 3, 30)
simulation_end_time = dt.datetime(2022, 4, 30)
policy_name_mtp = str(thresholds)

case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, 10, 20), "surge": (-1, -1, 10)}
staffed_thresholds = {"non_surge": (-1, 0.1, 0.15), "surge": (-1, -1, 0.1)}
percentage_cases = 0.4
ctp = CDCTierPolicy(austin, tiers_CDC, case_threshold, hosp_adm_thresholds, staffed_thresholds, percentage_cases)

policy_name_ctp = f"CDC_{case_threshold}"
seed = -1
rep = SimReplication(austin, vaccines, ctp, seed)
rep.simulate_time_period(austin.cal.calendar.index(simulation_end_time), austin.cal.calendar.index(history_end_time))


base_filename = f"{base_path}/input_output_folder/austin/{seed}_1_{history_end_time.date()}_{policy_name_ctp}"
export_rep_to_json(
    rep,
    f"{base_filename}_sim_updated.json",
    f"{base_filename}_v0.json",
    f"{base_filename}_v1.json",
    f"{base_filename}_v2.json",
    f"{base_filename}_v3.json",
    f"{base_filename}_policy.json"
)
tier_colors_ctp = {0: "blue", 1: "gold", 2: "red"}
equivalent_thresholds = {"non_surge": (-1, 28.57, 57.14), "surge": (-1, -1, 28.57)}
plot_from_file([seed], 1, austin, history_end_time, equivalent_thresholds, policy_name_ctp, tier_colors_ctp, "input_output_folder/austin")
