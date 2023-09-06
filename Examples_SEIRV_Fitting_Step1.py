# ================================================
# This Python script is the 1st step of the example of training SEIR-V model
# Description: Forward Parameter Tuning
# ================================================
from Engine_DataObjects_Wastewater import City, Vaccine
from Tools_Auto_Param_Tune_Wastewater import auto_pt_foward
import os
# Import other Python packages
import datetime as dt

# PART 1: INITIALIZATION
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
# Parameters in the forward parameter tuning
step_size = 7
NUM_TIME_BLOCK_TO_OPTIMIZE = 10

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

# previously calculated transmission reduction and cocoon parameters
transmission_reduction = [0.6390371782202954, 0.5516637746067924, 0.4593874352740992, 0.5431581035156047, 0.8010305164649246, 0.8650087984865349, 0.7704266683825446, 0.7022778460719028]
cocoon = [0.6390371782202954, 0.5516637746067924, 0.4593874352740992, 0.5431581035156047, 0.8010305164649246, 0.8650087984865349, 0.7704266683825446, 0.7022778460719028]

# viral shedding parameters
viral_shedding_param = [None]
viral_shedding_profile_end_date = [final_end_date]

# weights on the objectives
objective_weights = {"ToIHT_history": 1,
                    "kappa_weight": step_size * 50,
                     "log_wastewater_viral_load": 8}

# PART 2: RUN FORWARD PARAMETER TUNING
auto_pt_foward(calbrienstick,
                   vaccines,
                   step_size,
                   objective_weights,
                   rss_start_date,
                   final_end_date,
                   change_dates,
                   transmission_reduction,
                   cocoon,
                   NUM_TIME_BLOCK_TO_OPTIMIZE,
               viral_shedding_param=viral_shedding_param,
               viral_shedding_profile_end_date=viral_shedding_profile_end_date
               )

