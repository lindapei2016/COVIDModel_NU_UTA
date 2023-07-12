# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_sample_paths_v2"

output_dir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_sample_paths_analysis_v2"

folders = None
for subdir, dirs, files in os.walk(rootdir):
    print(dirs)
    folders = dirs
    break

num_scenarios = len(folders)
print("Number of sample paths: {}".format(num_scenarios))

fig, ax1 = plt.subplots()

num = 0
color_viral = "tab:grey"
color_admission = "tab:cyan"
end_date = dt.datetime(2021, 12, 10)
average_viral_load = None
dates = None
for folder in folders:
    num += 1
    viral_load_path = rootdir + "/" + folder + "/viral_load.csv"# hospital admission
    viral_load = pd.read_csv(viral_load_path)
    viral_load["date"] = pd.to_datetime(viral_load["date"], format="mixed")
    viral_load["viral_load"] = viral_load ["viral_load"].rolling(window=7).mean()
    plt.plot(viral_load["date"], viral_load["viral_load"], marker='', color=color_viral, linewidth=1, alpha=0.1)
    if num > 1:
        viral_load_np = viral_load["viral_load"].to_numpy()
        average_viral_load = average_viral_load + viral_load_np
    else:
        average_viral_load = viral_load["viral_load"].to_numpy()
        dates = viral_load["date"]

average_viral_load = average_viral_load / num
plt.plot(dates, average_viral_load, color="tab:orange", linewidth=1, label="Average Simulated Viral Load")

real_viral_load_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/calbrienstick_viral_load_Kpower_correct.csv"
real_viral_load = pd.read_csv(real_viral_load_path)
real_viral_load["date"] = pd.to_datetime(real_viral_load["date"], format="mixed")
real_viral_load_part = real_viral_load.loc[real_viral_load["date"] <= end_date]
plt.scatter(real_viral_load_part["date"], real_viral_load_part["viral_load"], c="g", s=2, label="Corrected Viral Load\n(By Katelyn's Power-Law Model)")
plt.plot([dt.datetime(2021, 11, 20), dt.datetime(2021, 11, 20)], [0, 6e15], color="k")
plt.legend()
plt.ylim([0, 6e15])
plt.xlim([dt.datetime(2020, 1, 2), dt.datetime(2021, 12, 10)])
#plt.xlim([dt.datetime(2021, 10, 1), dt.datetime(2021, 12, 10)])
plt.xlabel("Date")
plt.ylabel("Daily Viral Load \n (7-Day Moving Average GC)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "viral_load_plot1.png"), dpi=300, bbox_inches='tight')

