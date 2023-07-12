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
average_hosp_admin = None
for folder in folders:
    num += 1
    viral_load_path = rootdir + "/" + folder + "/viral_load.csv"# hospital admission
    viral_load = pd.read_csv(viral_load_path)
    viral_load["date"] = pd.to_datetime(viral_load["date"], format="mixed")
    viral_load["viral_load"] = viral_load ["viral_load"].rolling(window=7).mean()
    #plt.plot(viral_load["date"], viral_load["viral_load"], marker='', color=color_viral, linewidth=1, alpha=0.1)
    hosp_admin_path = rootdir + "/" + folder + "/ToIHT.csv"  # hospital admission
    hosp_admin = pd.read_csv(hosp_admin_path)
    hosp_admin["date"] = pd.to_datetime(hosp_admin["date"], format="mixed")
    hosp_admin["hospitalized"] = hosp_admin["hospitalized"].rolling(window=7).mean()
    if num > 1:
        viral_load_np = viral_load["viral_load"].to_numpy()
        average_viral_load = average_viral_load + viral_load_np
        hosp_admin_np = hosp_admin["hospitalized"].to_numpy()
        average_hosp_admin = average_hosp_admin + hosp_admin_np
    else:
        average_viral_load = viral_load["viral_load"].to_numpy()
        dates = viral_load["date"]
        average_hosp_admin = hosp_admin["hospitalized"].to_numpy()

average_viral_load = average_viral_load / num
average_hosp_admin = average_hosp_admin / num

fig, ax1 = plt.subplots()

ax1.plot(dates, average_viral_load, color="tab:green", linewidth=1.5, label="Simulated Viral Load")
ax1.set_ylabel("Daily Viral Load \n (7-Day Moving Average GC)", color="tab:green")
ax1.legend()
ax1.tick_params(axis='y', labelcolor="tab:green")

ax2 = ax1.twinx()
ax2.plot(dates, average_hosp_admin, marker='', color="b", linewidth=1.5,
         label="Simulated Hospital Admissions")
ax2.set_ylabel("COVID-19 Hospital Admissions \n (7-Day Moving Average)", color="b")

real_admission_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart/instances/calbrienstick/calbrienstick_hosp_admin_est.csv"
real_admission = pd.read_csv(real_admission_path)
real_admission["date"] = pd.to_datetime(real_admission["date"], format="mixed")
real_admission_plot_part = real_admission.loc[real_admission["date"] <= end_date]
ax2.scatter(real_admission_plot_part["date"], real_admission_plot_part["hospitalized"], c="r", s=2, label="Reported Hospital Admissions")
'''
real_viral_load_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart/instances/calbrienstick/calbrienstick_viral_load_Kpower_correct.csv"
real_viral_load = pd.read_csv(real_viral_load_path)
real_viral_load["date"] = pd.to_datetime(real_viral_load["date"], format="mixed")
real_viral_load_part = real_viral_load.loc[real_viral_load["date"] <= end_date]
plt.scatter(real_viral_load_part["date"], real_viral_load_part["viral_load"], c="g", s=2)
'''
ax1.plot([dt.datetime(2021, 11, 20), dt.datetime(2021, 11, 20)], [0, 6e15], color="k")
ax1.set_ylim([0, 6e15])
#ax1.set_xlim([dt.datetime(2020, 1, 2), dt.datetime(2021, 12, 10)])
ax1.set_xlim([dt.datetime(2021, 10, 1), dt.datetime(2021, 12, 10)])
ax1.set_xlabel("Date")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax2.legend()
ax2.set_ylim([0, 500])
#plt.show()
plt.savefig(os.path.join(output_dir, "viral_load_hosp_admin_plot2.png"), dpi=300, bbox_inches='tight')

