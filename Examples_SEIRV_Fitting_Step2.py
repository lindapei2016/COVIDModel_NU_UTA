# ================================================
# This Python script is the 2nd step of the example of training SEIR-V model
# Description: Backward Parameter Tuning
# ================================================
from Engine_DataObjects_Wastewater import City, Vaccine
from Tools_Auto_Param_Tune_Wastewater import auto_pt_backward
import os
# Import other Python packages
import datetime as dt

# PART 1: INITIALIZATION
# input file from the forward parameter tuning
solution_file = "./instances/calbrienstick/forward_pt/st_7_2020-02-27_2020-11-08_lsq_estimated_data.json"

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

objective_weights = {"ToIHT_history": 1,
                     "kappa_weight": 7 * 50,
                     "log_wastewater_viral_load": 8}



# PART 2: BACKWARD PARAMETER TUNING
auto_pt_backward(solution_file,
                     calbrienstick,
                     vaccines,
                     objective_weights,
                     transmission_reduction,
                     cocoon,
                     change_dates,
                     rss_start_date,
                     final_end_date,
                 viral_shedding_param=viral_shedding_param,
                 viral_shedding_profile_end_date=viral_shedding_profile_end_date
                 )