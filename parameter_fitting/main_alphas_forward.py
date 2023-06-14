###############################################################################

# auto_param_tune_forward.py
# Guyi chen 2023

###############################################################################

from DataObjects import City, Vaccine
from auto_param_alphas import auto_alphas_forward

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
              "./backward_pt/st_8_2020-03-23_2021-03-08_transmission_lsq_estimated_data.csv",
              "cook_total_beds_2023_idph.csv",
              "cook_icu_2023_idph.csv",
              "cook_ad_2023_idph_daily_estimate_init_zero.csv",
              "cook_deaths_from_hosp_0523.csv",
              "cook_deaths_from_home_0523.csv",
              "variant_prevalence.csv", 
              "other_time_varying_parameters.csv")

vaccines = Vaccine(
    cook,
    "cook",
    "vaccines.json",
    "booster_allocation_fixed_scaled.csv",
    "cook_vaccine_ww_weekly_zeros.csv",
)

step_size = 14
NUM_TIME_BLOCK_TO_OPTIMIZE = 5
# This is the time period where fixed kappa ends 
# If there is no fixed kappa, then this date would need to be the same as the start of the simulation
fixed_alpha_end_date = dt.datetime(2020, 2, 22)

# This is the time period where the entire parameter tuning ends
final_end_date = dt.datetime(2021, 3, 7)

# You can change this based on your case
# if there no time blocks you want to define, then change_dates = [dt.datetime(start date of the simulation)]
change_dates = [dt.datetime(2020, 2, 22)]

# The fixed alphas values corresponding to the change_dates defined above
alpha_ICUs = []
alpha_IHs = []
alpha_mu_ICUs = []
alpha_IYDs = []


# weights for the forward parameter tunes 

objective_weights = {"IH_history": 10,
                    "ICU_history": 15,
                    "ToICUD_history": 10,
                    "ToIYD_history": 40,}

# print(cook.cal.calendar)
auto_alphas_forward(cook, vaccines, step_size, objective_weights, fixed_alpha_end_date, final_end_date, change_dates, alpha_ICUs, alpha_IHs, alpha_mu_ICUs, alpha_IYDs, NUM_TIME_BLOCK_TO_OPTIMIZE)
    