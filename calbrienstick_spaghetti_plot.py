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
color_viral = "tab:green"
color_admission = "tab:cyan"
end_date = dt.datetime(2021, 12, 10)
average_hosp_admin = None
dates = None
for folder in folders:
    num += 1
    hosp_admin_path = rootdir + "/" + folder + "/ToIHT.csv"# hospital admission
    hosp_admin = pd.read_csv(hosp_admin_path)
    hosp_admin["date"] = pd.to_datetime(hosp_admin["date"], format="mixed")
    hosp_admin["hospitalized"] = hosp_admin["hospitalized"].rolling(window=7).mean()
    plt.plot(hosp_admin["date"], hosp_admin["hospitalized"], marker='', color=color_admission, linewidth=1, alpha=0.1)
    if num > 1:
        hosp_admin_np = hosp_admin["hospitalized"].to_numpy()
        average_hosp_admin = average_hosp_admin + hosp_admin_np
    else:
        average_hosp_admin = hosp_admin["hospitalized"].to_numpy()
        dates = hosp_admin["date"]

average_hosp_admin = average_hosp_admin / num
plt.plot(dates, average_hosp_admin, color="b", linewidth=1, label="Average Simulated Hospital Admissions")

real_admission_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/calbrienstick_hosp_admin_est.csv"
real_admission = pd.read_csv(real_admission_path)
real_admission["date"] = pd.to_datetime(real_admission["date"], format="mixed")
real_admission_plot_part = real_admission.loc[real_admission["date"] <= end_date]
#print(real_admission_plot_part.head(10))
#real_admission_plot_part["hospitalized"] = real_admission_plot_part["hospitalized"].rolling(window=7).mean()
#print(real_admission_plot_part.head(10))
plt.scatter(real_admission_plot_part["date"], real_admission_plot_part["hospitalized"], c="r", s=2, label="Reported Hospital Admissions")
plt.plot([dt.datetime(2021, 11, 20), dt.datetime(2021, 11, 20)], [0, 500], color="k")
plt.legend()
plt.ylim([0, 500])
#plt.ylim([0, 300])
plt.xlim([dt.datetime(2020, 1, 2), dt.datetime(2021, 12, 10)])
#plt.xlim([dt.datetime(2021, 10, 1), dt.datetime(2021, 12, 10)])
plt.xlabel("Date")
plt.ylabel("COVID-19 Hospital Admissions \n (7-Day Moving Average)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "hosp_admin_plot1.png"), dpi=300, bbox_inches='tight')

