
from DataObjects import City, Vaccine
from utils_auto_param_tune import auto_pt_foward
import os
# Import other Python packages
import datetime as dt

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick"
part_num = 1
correct_method = "Kpower"
output_path_sol = os.path.join(rootdir, "sol_part{}_{}.txt".format(part_num, correct_method))
output_path_transmission = os.path.join(rootdir, "transmission_part{}_{}.csv".format(part_num, correct_method))
output_path_sim_hosp_admin = os.path.join(rootdir, "sim_hosp_admin_part{}_{}.csv".format(part_num, correct_method))
output_path_vsp = os.path.join(rootdir, "viral_shedding_profile_part{}_{}.csv".format(part_num, correct_method))
output_path_viral_load = os.path.join(rootdir, "viral_load_part{}_{}.csv".format(part_num, correct_method))


calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json", "variant.json",
                "transmission.csv",
                "IH_null.csv",  # hospitalization file name
                "icu_null.csv",
                "calbrienstick_hosp_admin_est.csv",  # ToIHT
                "death_null.csv",
                "home_death_null.csv",
                "variant_prevalence.csv")

'''
calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json", "variant.json",
                "transmission.csv",
                "IH_null.csv",  # hospitalization file name
                "icu_null.csv",
                "calbrienstick_hosp_admin_est.csv",  # ToIHT
                "death_null.csv",
                "home_death_null.csv",
                "variant_prevalence.csv",
                "calbrienstick_viral_load_Kpower_correct.csv",
                "viral_shedding_profile_corr.json")
'''
vaccines = Vaccine(
    calbrienstick,
    "calbrienstick",
    "vaccines.json",
    "booster_calbrienstick.csv",
    "vaccine_allocation_calbrienstick.csv",
)

step_size = 7
NUM_TIME_BLOCK_TO_OPTIMIZE = 10
# This is the time period where fixed kappa ends
# If there is no fixed kappa, then this date would need to be the same as the start of the simulation
fixed_kappa_end_date = dt.datetime(2020, 1, 2)

# This is the time period where the entire parameter tuning ends
final_end_date = dt.datetime(2020, 6, 26)

# You can change this based on your case
# if there is no time blocks you want to define, then change_dates = [dt.datetime(start date of the simulation)]
change_dates = [dt.datetime(2020, 1, 2)]

transmission_reduction = []
cocoon = []

# weights for the forward parameter tunes
# Since we are only fitting tramsission and cocoon, the only weights that matters is the hospital admission weight and kappa_weight
# kappa_weight is designed to minimize the change from kappa_{t-1} to kappa_{t}
objective_weights = {"ToIHT_history": 1,
                     "kappa_weight": step_size * 50}

auto_pt_foward(calbrienstick,
                   vaccines,
                   step_size,
                   objective_weights,
                   fixed_kappa_end_date,
                   final_end_date,
                   change_dates,
                   transmission_reduction,
                   cocoon,
                   NUM_TIME_BLOCK_TO_OPTIMIZE)



