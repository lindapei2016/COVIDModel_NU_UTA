###############################################################################

# auto_param_tune_forward.py
# Guyi chen 2023

###############################################################################

from DataObjects import City, Vaccine
from Paramfitting_Cook import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd

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

def auto_pt_foward(city, vaccines, step_size, objective_weights, fixed_kappa_end_date, final_end_date, change_dates, transmission_reduction, cocoon, NUM_TIME_BLOCK_TO_OPTIMIZE = 10):
    print("step_size", step_size)
    end_date = fixed_kappa_end_date

    variables = ["transmission_reduction"]
    solution = None

    num_fixed_time_blocks = len(change_dates) - 1
    newly_fixed_time_blocks = 0
    while(end_date < final_end_date):
        end_date += dt.timedelta(days=step_size)
        transmission_reduction.extend([None])
        cocoon.extend([None])
        if (sum(tr is None for tr in transmission_reduction) - 1) == NUM_TIME_BLOCK_TO_OPTIMIZE:
            newly_fixed_time_blocks += 1
            transmission_reduction[-NUM_TIME_BLOCK_TO_OPTIMIZE-1] = solution["transmission_reduction"][num_fixed_time_blocks + newly_fixed_time_blocks] 
            cocoon[-NUM_TIME_BLOCK_TO_OPTIMIZE-1] = solution["transmission_reduction"][num_fixed_time_blocks + newly_fixed_time_blocks] 
        if solution is None:
            initial_guess = np.array([0.5] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks))
        else:
            initial_guess = np.array(solution["transmission_reduction"][num_fixed_time_blocks + newly_fixed_time_blocks:] +  [0.5])
        # Lower and upper bound tuple:
        lower_bound = [0] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks)
        upper_bound = [1] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks)
        x_bound = (lower_bound, upper_bound)
        change_dates.append(end_date)

        print(end_date)
        print(initial_guess)
        print(x_bound)
        print(change_dates)
        print("cocoon", cocoon)
        time_frame = (city.cal.calendar.index(fixed_kappa_end_date), city.cal.calendar.index(end_date))
        param_fitting = ParameterFitting(city,
                                        vaccines,
                                        variables,
                                        initial_guess,
                                        x_bound,
                                        objective_weights,
                                        time_frame,
                                        change_dates,
                                        transmission_reduction,
                                        cocoon,
                                        is_auto=True,
                                        is_forward=True,
                                        step_size=step_size)
        solution = param_fitting.run_fit()


