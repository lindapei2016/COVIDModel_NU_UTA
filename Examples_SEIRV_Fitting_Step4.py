# ================================================
# This Python script is the 4th step of the example of training SEIR-V model
# Description: Plot Hospital Admissions and Viral Load
# ================================================
import json

from Engine_DataObjects_Wastewater import City, Vaccine
from Fitting_Wastewater import ParameterFitting
from Tools_Auto_Param_Tune_Wastewater import model_selection,model_plot
from pathlib import Path
import os
# Import other Python packages
import datetime as dt

# PART 1: INITIALIZATION
# =============================
# set up input/output root dir
base_path = Path(__file__).parent

filename = "st_11_2020-02-27_2020-11-08_transmission_lsq_estimated_data"
filename2 = "st_11_2020-02-27_2020-11-08_lsq_estimated_data"
variables = ["transmission_reduction", "viral_shedding_profile"]

hosp_admin = "ToIHT_history"
viral_load = "wastewater_viral_load"

variables = ["transmission_reduction", "viral_shedding_profile"]

input_model_paramter = os.path.join(base_path, "instances/calbrienstick/backward_pt/" + filename2 + ".json")
param_path = os.path.join(base_path, "instances/calbrienstick/backward_pt/" + filename + ".csv")

output_hosp_admin_png = os.path.join(base_path, "instances/calbrienstick/model_selection/" + hosp_admin + "_" + filename + ".png")
output_viral_load_png = os.path.join(base_path, "instances/calbrienstick/model_selection/" + viral_load + "_" + filename + ".png")

file_model = open(input_model_paramter,)
model_paramter = json.load(file_model)

calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json", "variant.json",
                    "transmission_null.csv",
                    "IH_null.csv",  # hospitalization file name
                    "icu_null.csv",
                    "calbrienstick_hosp_admin_est_Katelyn.csv",  # ToIHT
                    "death_null.csv",
                    "home_death_null.csv",
                    "variant_prevalence.csv",
                    "calbrienstick_viral_merge_dpcr_qcpr.csv",
                    "viral_shedding_profile_corr.json")

vaccines = Vaccine(
    calbrienstick,
    "calbrienstick",
    "vaccines.json",
    "calbrienstick_booster_allocation_Sonny.csv",
    "calbrienstick_vaccine_allocation_Sonny.csv",
)

rss_start_date = dt.datetime(2020, 10, 21)

# This is the time period where the entire parameter tuning ends
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

viral_shedding_param = [(model_paramter["viral_shedding_profile"][0], model_paramter["viral_shedding_profile"][1])]
viral_shedding_profile_end_date = [dt.datetime(2020, 11, 8)]

time_frame = (calbrienstick.cal.calendar.index(rss_start_date), calbrienstick.cal.calendar.index(final_end_date))
viral_start_index = calbrienstick.cal.calendar.index(dt.datetime(2020, 10, 21))

# PART 2 PLOT HOSPITAL ADMISSIONS AND VIRAL LOAD
param_fitting = ParameterFitting(calbrienstick,
                                 vaccines,
                                variables,
                                initial_guess = None,
                                bounds = None,
                                objective_weights = None,
                                time_frame = time_frame,
                                change_dates = change_dates,
                                transmission_reduction = [],
                                cocoon = [],
                                 viral_shedding_param=viral_shedding_param,
                                 viral_shedding_profile_end_date=viral_shedding_profile_end_date
                                 )


# plot hospital admissions
model_plot(param_fitting, hosp_admin, param_path, "upper right", 0, output_hosp_admin_png)

# plot viral load
model_plot(param_fitting, viral_load, param_path, "upper right", viral_start_index, output_viral_load_png)