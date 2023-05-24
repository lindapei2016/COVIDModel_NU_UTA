###############################################################################

# auto_param_tune_back.py

###############################################################################

from DataObjects import City, Vaccine
from auto_param_tune_backward import auto_pt_backward

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
              "transmission_lsq_estimated_data.csv",
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

# If there is no fixed kappa, then this date would need to be the same as the start of the simulation
fixed_kappa_end_date = dt.datetime(2020, 5, 18)

# This is the time period where the entire parameter tuning ends
end_date = dt.datetime(2021, 3, 8)

# You can change this based on your case
# if there no time blocks you want to define, then change_dates = [dt.datetime(start date of the simulation)]
change_dates = [dt.datetime(2020, 2, 17),
                dt.datetime(2020, 3, 23),
                dt.datetime(2020, 4, 6), 
                dt.datetime(2020, 5, 18),]

# The fixed kappa and cocoon values corresponding to the change_dates defined above
# Note the len(transmission_reduction) == len(cocoon) == len(change_dates)
transmission_reduction_fixed = [0.11922764695638084, 0.47262082948700784,0.7989372094967291,]
cocoon_fixed = [0.11922764695638084, 0.47262082948700784,0.7989372094967291,]

# weights for the forward parameter tunes 
# Since we are only fitting tramsission and cocoon, the only weights that matters is the hospital admission weight and kappa_weight
# kappa_weight is designed to minimize the change from kappa_{t-1} to kappa_{t}
objective_weights = {"ToIHT_history": 1, 
                    "kappa_weight": 7 * 50}

# This is the json solution file. It is the starting point of the backward parameter tuning.
solution_file = "./instances/cook/forward_pt/st_7_2020-03-23_2021-03-08_lsq_estimated_data.json"

auto_pt_backward(solution_file, cook, vaccines, transmission_reduction_fixed, cocoon_fixed, change_dates)
    