###############################################################################

# auto_param_tune_back.py

###############################################################################

from Engine_DataObjects import City, TierInfo, Vaccine
from Fitting_CookCounty import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd
import json

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


# backward_step

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

def auto_pt_backward(solution_file, city, vaccines, transmission_reduction_fixed, cocoon_fixed, change_dates):
    num_fixed_kappa = len(change_dates) - 1
    for i in range(((end_date - fixed_kappa_end_date).days)//7):
        change_dates.append(fixed_kappa_end_date + dt.timedelta(days=(i+1) * 7))
    with open(solution_file) as file:
        solution = json.load(file)

    variables = ["transmission_reduction"]

    while True:
        if len(change_dates) > num_fixed_kappa:
            transmission_reduction_solution = solution["transmission_reduction"][num_fixed_kappa:]
            # find the absolute difference between kappa_{t-1} and kappa_{t} and its index
            difference_in_transmission_reduction = [abs(transmission_reduction_solution[i] - transmission_reduction_solution[i+1]) for i in range(len(transmission_reduction_solution) - 1)]
            print("transmission_non_fixed:",len(transmission_reduction_solution))
            min_change = min(difference_in_transmission_reduction)
            min_change_idx = difference_in_transmission_reduction.index(min_change)
            print(min_change, min_change_idx)
            # delete the corresponding time block from the tranmission_reduction
            del transmission_reduction_solution[min_change_idx]
            del change_dates[min_change_idx + num_fixed_kappa + 1]
            transmission_reduction = transmission_reduction_fixed + [None] * (len(change_dates) - num_fixed_kappa - 1)
            cocoon = cocoon_fixed + [None] * (len(change_dates) - num_fixed_kappa - 1)
            initial_guess = np.array(transmission_reduction_solution)
            # Lower and upper bound tuple:
            lower_bound = [0] * (len(change_dates) - num_fixed_kappa - 1)
            upper_bound = [1] * (len(change_dates) - num_fixed_kappa - 1)
            x_bound = (lower_bound, upper_bound)
            print("iteration 0")
            print(change_dates)

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
                                            is_forward=False,
                                            step_size=len(change_dates))
            solution = param_fitting.run_fit()
        else:
            break
        # transmission_reduction_solution = solution["transmission_reduction"][num_fixed_kappa:]
        # difference_in_transmission_reduction = [abs(transmission_reduction_solution[i] - transmission_reduction_solution[i+1]) for i in range(len(transmission_reduction_solution) - 1)]
        # min_change = min(difference_in_transmission_reduction)
        # min_change_idx = difference_in_transmission_reduction.index(min_change)
        # print(transmission_reduction_solution)
        # del transmission_reduction_solution[min_change_idx]
        # del change_dates[min_change_idx + num_fixed_kappa + 1]
        # transmission_reduction = transmission_reduction[:-1]
        # cocoon = cocoon[:-1]
        # initial_guess = np.array(transmission_reduction_solution)
        # # Lower and upper bound tuple:
        # lower_bound = [0] * (len(change_dates) - num_fixed_kappa - 1)
        # upper_bound = [1] * (len(change_dates) - num_fixed_kappa - 1)
        # x_bound = (lower_bound, upper_bound)
        # print(initial_guess)
        # print(x_bound)
        # print(change_dates)



