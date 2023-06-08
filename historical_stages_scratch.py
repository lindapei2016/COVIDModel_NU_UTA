######################################################################################
# This script is for retrospective analysis. Not an actual policy evaluation.
# This if for coloring the background according to historical data and policy indicators.
# What would be the corresponding stages historically if we were to use a given policy.

# Nazlican Arslan, 2023
######################################################################################

from SimObjects import find_tier
from DataObjects import City, TierInfo
from Plot_Manager import Plot
from SimObjects import CDCTierPolicy, MultiTierPolicy

import numpy as np
import pandas as pd
import datetime as dt
import json
from pathlib import Path

base_path = Path(__file__).parent


######################################################################################
class HistoricalStages:
    """
    Decide on historical stages from given data and indicator. This is not running a new simulation.
    """

    def __init__(self,
                 instance,
                 tiers,
                 ToIHT_history,
                 IH_history,
                 ICU_history,
                 ToIY_history,
                 policy,
                 percentage_cases=1):

        self.instance = instance
        self.policy = policy
        self.percentage_cases = percentage_cases
        self.ToIHT_history = np.array(ToIHT_history)
        self.IH_history = np.array(IH_history)
        self.ToIY_history = np.array(ToIY_history)
        self.ICU_history = np.array(ICU_history)
        self.N = instance.N
        self.tier_history, self.surge_history = [], []

    def create_historic_stages(self):
        for t in range(len(self.ToIHT_history)):
            if "CDC" in str(self.policy):
                new_tier, new_surge_state = self.compute_indicators_CDC(t)
            else:
                new_tier, new_surge_state = self.compute_indicators_Austin(t)
            self.tier_history += [new_tier]
            self.surge_history += [new_surge_state]

        self.save_history()

    def compute_indicators_Austin(self, t):
        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - self.instance.moving_avg_len)
        if t > 0:
            criStat_avg = self.ToIHT_history[moving_avg_start:t].mean()
        else:
            criStat_avg = 0
        # find new tier
        new_tier = find_tier(self.policy.lockdown_thresholds, criStat_avg)
        return new_tier, None

    def compute_indicators_CDC(self, t):
        # Compute daily admissions moving sum
        moving_avg_start = np.maximum(0, t - self.instance.moving_avg_len)
        hosp_adm_sum = 100000 * self.ToIHT_history[moving_avg_start:t].sum() / self.N.sum((0, 1))

        # Compute 7-day total new cases:

        ToIY_avg = self.ToIY_history[moving_avg_start:t].sum() * 100000 / np.sum(self.N, axis=(0, 1))

        # Compute 7-day average percent of COVID beds:
        IH_total = self.IH_history + self.ICU_history
        if t > 0:
            IH_avg = IH_total[moving_avg_start:t].mean() / self.instance.hosp_beds
        else:
            IH_avg = 0
        # Decide on the active hospital admission and staffed bed thresholds depending on the estimated
        # case count level:
        if ToIY_avg * percentage_cases < case_threshold:
            hosp_adm_th = hosp_adm_thresholds["non_surge"]
            staffed_bed_th = self.policy.staffed_bed_thresholds["non_surge"]
            new_surge_state = 0
        else:
            hosp_adm_th = hosp_adm_thresholds["surge"]
            staffed_bed_th = self.policy.staffed_bed_thresholds["surge"]
            new_surge_state = 1
        # find hosp admission new tier:
        hosp_adm_tier = find_tier(hosp_adm_th, hosp_adm_sum)
        # find staffed bed new tier:
        staffed_bed_tier = find_tier(staffed_bed_th, IH_avg)
        # choose the stricter tier among tiers the two indicators suggesting:
        new_tier = max(hosp_adm_tier, staffed_bed_tier)

        return new_tier, new_surge_state

    def save_history(self):
        policy_filename = f"{base_path}/input_output_folder/austin/historic_{self.policy}.json"
        d = {"policy_type": f"{self.policy}",
             "surge_history": self.surge_history,
             "tier_history": self.tier_history}

        json.dump(d, open(policy_filename, "w"))


######################################################################################

austin = City(
    "austin",
    "calendar.csv",
    "austin_setup.json",
    "variant.json",
    "transmission.csv",
    "austin_hospital_home_timeseries.csv",
    "variant_prevalence.csv"
)
filename = 'austin_real_case.csv'
real_ToIY = pd.read_csv(
    str(austin.path_to_data / filename),
    parse_dates=["date"],
    date_parser=pd.to_datetime,
)["admits"]

real_IH = [ai - bi for (ai, bi) in zip(austin.real_IH_history, austin.real_IH_history)]
history_end_time = dt.datetime(2022, 3, 30)

tiers_CDC = TierInfo("austin", "tiers_CDC.json")
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, 10, 20), "surge": (-1, -1, 10)}
staffed_thresholds = {"non_surge": (-1, 0.1, 0.15), "surge": (-1, -1, 0.1)}

percentage_cases = 0.4
ctp = CDCTierPolicy(austin, tiers_CDC, case_threshold, hosp_adm_thresholds, staffed_thresholds, percentage_cases)
historic_stages_cdc = HistoricalStages(austin,
                                       tiers_CDC.tier,
                                       austin.real_ToIHT_history,
                                       real_IH,
                                       austin.real_ICU_history,
                                       real_ToIY,
                                       ctp)

historic_stages_cdc.create_historic_stages()

policy_name_ctp = f"CDC_{case_threshold}"
redundant_data = [np.array([np.zeros((5, 2)) for t in range(len(austin.real_ToIHT_history))])]

tier_colors_ctp = {0: "blue", 1: "gold", 2: "red"}
equivalent_thresholds = {"non_surge": (-1, 28.57, 57.14), "surge": (-1, -1, 28.57)}

policy_filename = f"{base_path}/input_output_folder/austin/historic_{ctp}.json"
with open(policy_filename) as file:
    data = json.load(file)

plot = Plot(austin, history_end_time, austin.real_ToIHT_history, redundant_data, "ToIHT_history_sum", policy_name_ctp, 0,
            color=('k', 'silver'))
# plot.vertical_plot([data["tier_history"]], tier_colors_ctp, 1100)

######################################################################################
tiers = TierInfo("austin", "tiers5_opt_Final.json")
thresholds = (-1, 0, 15, 25, 50)
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
historic_stages_austin = HistoricalStages(austin,
                                          tiers.tier,
                                          austin.real_ToIHT_history,
                                          real_IH,
                                          austin.real_ICU_history,
                                          real_ToIY,
                                          mtp)

historic_stages_austin.create_historic_stages()
redundant_data = [np.array([np.zeros((5, 2)) for t in range(len(austin.real_ToIHT_history))])]

tier_colors_mtp = {0: "green", 1: "blue", 2: "yellow", 3: "orange", 4: "red"}
equivalent_thresholds = {"non_surge": (-1, 28.57, 57.14), "surge": (-1, -1, 28.57)}

policy_filename = f"{base_path}/input_output_folder/austin/historic_{mtp}.json"
with open(policy_filename) as file:
    data = json.load(file)

plot = Plot(austin, history_end_time, austin.real_ToIHT_history, redundant_data, "ToIHT_history_sum", str(mtp), 0,
            color=('k', 'silver'))
# plot.vertical_plot([data["tier_history"]], tier_colors_mtp, 1100)

filename = f"{base_path}/instances/austin/austin_historical_stages.csv"
austin_historic_stages = pd.read_csv(
            str(filename),
            parse_dates=["date"],
            date_parser=pd.to_datetime,
        )

plot = Plot(austin, history_end_time, austin.real_ToIHT_history, redundant_data, "ToIHT_history_sum", str(mtp), 0,
            color=('k', 'silver'))
stages = [int(a) if a != "None" else None for a in austin_historic_stages["stage"].values]
plot.vertical_plot([stages], tier_colors_mtp, 1100)
