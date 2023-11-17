import numpy as np
import datetime as dt
import pandas as pd
import json
import time

# extra packages
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path(__file__).parent

start_date = dt.datetime(2020,1,2)
fixed_kappa_end_date = dt.datetime(2021, 11, 1)
simulation_end_date = dt.datetime(2022, 2, 28)
seed = -1
wastewater_thresholds = (-1, 79999999.0, 159999999.0, 319999999.0, 959999999.0)

# import tier history
tier_history = pd.read_csv(f"{base_path}/retrospective_analysis/seed{seed}/tier_history{wastewater_thresholds}.csv")
tier_history = tier_history.fillna(-1)
tier_history["tier"] = tier_history["tier"].astype(int)
tier_history["date"] = pd.to_datetime(tier_history["date"], format="mixed")

print(tier_history.info())

# import viral load
viral_load = pd.read_csv(f"{base_path}/retrospective_analysis/seed{seed}/viral_load{wastewater_thresholds}.csv")
viral_load["date"] = pd.to_datetime(viral_load["date"], format="mixed")

plt.figure()
plt.plot(viral_load["date"], viral_load["viral_load"], linewidth=1, color='g')

tier_color = {0: "tab:blue",
              1: "tab:olive",
              2: "tab:orange",
              3: "tab:red",
              4: "tab:purple"}

for idx in range(len(tier_history.index)):
    cur_tier = tier_history["tier"].iloc[idx]
    if cur_tier >= 0:
        #print(tier_color[cur_tier])
        plt.axvspan(tier_history["date"].iloc[idx], tier_history["date"].iloc[idx] + dt.timedelta(1), facecolor=tier_color[cur_tier], alpha=0.5)

plt.plot([fixed_kappa_end_date, fixed_kappa_end_date], [0, 8e15], color="k")
plt.xlim([start_date, simulation_end_date])
plt.ylim([0, 8e15])

plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Daily Viral Load")
plt.savefig(f"{base_path}/retrospective_analysis/seed{seed}/viral_load_ww_policy{wastewater_thresholds}.png", dpi=300, bbox_inches='tight')
