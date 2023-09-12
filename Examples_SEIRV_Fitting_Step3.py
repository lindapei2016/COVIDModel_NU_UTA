# ================================================
# This Python script is the 3rd step of the example of training SEIR-V model
# Description: Calculate Residual Sum of Squares and Plot the Curve of Residual Sum of Squares
# ================================================

from Engine_DataObjects_Wastewater import City, Vaccine
from Fitting_Wastewater import ParameterFitting
from Tools_Auto_Param_Tune_Wastewater import model_selection
from pathlib import Path
import os
# Import other Python packages
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
# PART 1: INITIALIZATION
# =============================
# set up input/output root dir
base_path = Path(__file__).parent
paramdir = os.path.join(base_path, "instances/calbrienstick/backward_pt")
output_path_rss = os.path.join(base_path, "instances/calbrienstick/model_selection/seirv_rss.csv")
# =============================

objective = "ToIHT_history"
variables = ["transmission_reduction"]

calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json", "variant.json",
                    "transmission_null.csv",
                    "IH_null.csv",  # hospitalization file name
                    "icu_null.csv",
                    "calbrienstick_hosp_admin_est_Katelyn.csv",  # ToIHT
                    "death_null.csv",
                    "home_death_null.csv",
                    "variant_prevalence.csv")

vaccines = Vaccine(
    calbrienstick,
    "calbrienstick",
    "vaccines.json",
    "calbrienstick_booster_allocation_Sonny.csv",
    "calbrienstick_vaccine_allocation_Sonny.csv",
)

# start date of residual sum of squares calculation
rss_start_date = dt.datetime(2020, 10, 21)
# end date of residual sum of squares calculation
final_end_date = dt.datetime(2020, 11, 8)

# previously calculated change dates
change_dates = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 2, 27),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 19),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 5, 21),
                dt.datetime(2020, 6, 11),
                dt.datetime(2020, 10, 8),
                dt.datetime(2020, 10, 21)]

time_frame = (calbrienstick.cal.calendar.index(rss_start_date), calbrienstick.cal.calendar.index(final_end_date))

# PART 2: CALCULATING RESIDUAL SUM OF SQUARES
param_fitting = ParameterFitting(calbrienstick,
                                 vaccines,
                                variables,
                                initial_guess = None,
                                bounds = None,
                                objective_weights = None,
                                time_frame = time_frame,
                                change_dates = change_dates,
                                transmission_reduction = [],
                                cocoon = [])


model_selection(param_fitting, objective, paramdir, output_path_rss)

# PART 3: PLOT THE CURVE OF RESIDUAL OF SUM OF SQUARES
plot_title = "Jan. 02, 2020 - Oct. 8, 2020"
png_name = "20200102_20201008.png"

png_path = os.path.join(base_path, "instances/calbrienstick/model_selection/{}".format(png_name))

rss_df = pd.read_csv(output_path_rss)
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

plt.savefig(png_path, dpi=300, bbox_inches='tight')




