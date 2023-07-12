import numpy as np
import datetime as dt
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

root_path = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_smart_v2"
#plot_title = "Jan. 02, 2020 - July 01, 2020"
#png_name = "20200102_20200701.png"

#plot_title = "July 02, 2020 - Oct. 21, 2020 "
#png_name = "20200702_20201021.png"

#plot_title = "Oct. 22, 2020 - Feb. 20, 2021"
#png_name = "20201022_20210220.png"

#plot_title = "Feb. 21, 2021 - May. 20, 2021"
#png_name = "20210221_20210520.png"

#plot_title = "May 21, 2021 - Aug. 20, 2021"
#png_name = "20210521_20210820.png"

plot_title = "Aug. 21, 2021 - Nov. 20, 2021"
png_name = "20210821_20211120.png"

png_path = os.path.join(root_path, "model_selection/{}".format(png_name))
rss_df = pd.read_csv(os.path.join(root_path, "rss_6_Kpower.csv"))

max_kappa = 0
min_kappa = 999

print(rss_df.info())
rss_dict = {}
for index, row in rss_df.iterrows():
    file_name = row["file_name"]
    num_kappa = int(file_name.split("_")[1]) - 1
    if num_kappa > max_kappa:
        max_kappa = num_kappa
    if num_kappa < min_kappa:
        min_kappa = num_kappa

    rss_dict[num_kappa] = row["rss(ToIHT_history)"]

rss_array = []
for i in range(min_kappa, max_kappa + 1):
    rss_array.append(rss_dict[i])

kappa_array = [i for i in range(min_kappa, max_kappa + 1)]

plt.plot(kappa_array, rss_array, color="r")
plt.xlabel("Number of Kappas")
plt.ylabel("RSS (Hospital Admissions)")
plt.title(plot_title)
#plt.show()
plt.savefig(png_path, dpi=300, bbox_inches='tight')