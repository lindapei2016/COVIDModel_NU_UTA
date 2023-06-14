###############################################################################

# AutomaticTuning_Forward.py
# Guyi chen 2023

###############################################################################

from Engine_DataObjects import City, Vaccine
from AutomaticTuning_Forward import auto_pt_foward

# Import other Python packages
import datetime as dt

###############################################################################
# We need to define city, vaccine and tier object as the least square fit will
# run the deterministic simulation model.


cook = City("cook",
              "cook_test_IHT.json",
              "calendar.csv",
              "setup_data_param_fit_YHR_HICUR.json",
              "variant.json",
              "transmission_lsq_test.csv",
              "cook_total_beds_2023_idph.csv",
              "cook_icu_2023_idph.csv",
              "cook_ad_2023_idph_daily_estimate_init_zero.csv",
              "cook_deaths_from_hosp_est.csv",
              "cook_deaths_from_home_est.csv",
              "variant_prevalence.csv")

vaccines = Vaccine(
    cook,
    "cook",
    "vaccines.json",
    "booster_allocation_fixed_scaled.csv",
    "cook_vaccine_ww_weekly_zeros.csv",
)

step_size = 7
NUM_TIME_BLOCK_TO_OPTIMIZE = 10
# This is the time period where fixed kappa ends 
# If there is no fixed kappa, then this date would need to be the same as the start of the simulation
fixed_kappa_end_date = dt.datetime(2020, 5, 18)

# This is the time period where the entire parameter tuning ends
final_end_date = dt.datetime(2021, 3, 7)

# You can change this based on your case
# if there no time blocks you want to define, then change_dates = [dt.datetime(start date of the simulation)]
change_dates = [dt.datetime(2020, 2, 17),
                dt.datetime(2020, 3, 23),
                dt.datetime(2020, 4, 6), 
                dt.datetime(2020, 5, 18),]

# The fixed kappa and cocoon values corresponding to the change_dates defined above
# Note the len(transmission_reduction) == len(cocoon) == len(change_dates)
transmission_reduction = [0.11922764695638084, 0.47262082948700784,0.7989372094967291,]
cocoon = [0.11922764695638084, 0.47262082948700784,0.7989372094967291,]

# weights for the forward parameter tunes 
# Since we are only fitting tramsission and cocoon, the only weights that matters is the hospital admission weight and kappa_weight
# kappa_weight is designed to minimize the change from kappa_{t-1} to kappa_{t}
objective_weights = {"ToIHT_history": 1, 
                    "kappa_weight": step_size * 50}

auto_pt_foward(cook, vaccines, step_size, fixed_kappa_end_date, final_end_date, change_dates, transmission_reduction, cocoon, NUM_TIME_BLOCK_TO_OPTIMIZE)
    