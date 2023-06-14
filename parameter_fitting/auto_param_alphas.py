###############################################################################

# main_param_fitting_example.py
# This document contains examples of how to use the least square fitting tool.
# Nazlican Arslan 2022

###############################################################################

from DataObjects import City, TierInfo, Vaccine
from Paramfitting_Cook import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd

###############################################################################
# We need to define city, vaccine and tier object as the least square fit will
# run the deterministic simulation model.


# cook = City("cook",
#               "cook_test_IHT.json",
#               "calendar.csv",
#               "setup_data_param_fit_YHR_HICUR.json",
#               "variant.json",
#               "./backward_pt/st_8_2020-03-23_2021-03-08_transmission_lsq_estimated_data.csv",
#               "cook_total_beds_2023_idph.csv",
#               "cook_icu_2023_idph.csv",
#               "cook_ad_2023_idph_daily_estimate_init_zero.csv",
#               "cook_deaths_from_hosp_0523.csv",
#               "cook_deaths_from_home_0523.csv",
#               "variant_prevalence.csv", 
#               "other_time_varying_parameters.csv",)

# vaccines = Vaccine(
#     cook,
#     "cook",
#     "vaccines.json",
#     "booster_allocation_fixed_scaled.csv",
#     "cook_vaccine_ww_weekly_zeros.csv",
# )

# step_size = 14
# print("step_size", step_size)
# start_date = dt.datetime(2020, 2, 17)

# fixed_alpha_end_date =  dt.datetime(2020, 2, 17)
# final_end_date = dt.datetime(2021, 3, 7)


# alpha_ICUs = []
# alpha_IHs = []
# alpha_mu_ICUs = []
# alpha_IYDs = []

# # # weights for the least-square fit:
# # # You can input the data you would like to use in the process and corresponding weights. Different data have different
# # # scales that's why we use weights.
# objective_weights = {"IH_history": 10,
#                     "ICU_history": 15,
#                     "ToICUD_history": 10,
#                     "ToIYD_history": 40,}

# # # We generally use the least square fit to find transmission reduction and cocooning in a population. But time to time
# # # we may need to estimate other parameters. Fitting transmission reduction is optional. In the current version of
# # # the parameter fitting you can input the name of parameter you would like to fit, and you don't need to change anything
# # # else in the source code.

# change_dates = [start_date]


def auto_alphas_forward(city, vaccines, step_size, objective_weights, fixed_alpha_end_date, final_end_date, change_dates, alpha_gamma_ICUs, alpha_IHs, alpha_mu_ICUs, alpha_IYDs, NUM_TIME_BLOCK_TO_OPTIMIZE = 10):
    print("step_size", step_size)
    end_date = fixed_alpha_end_date
    variables = ["alphas"]
    solution = None
    num_fixed_time_blocks = len(change_dates) - 1
    newly_fixed_time_blocks = 0
    while(end_date < final_end_date):
        end_date += dt.timedelta(days=step_size)
        alpha_gamma_ICUs.extend([None])
        alpha_IHs.extend([None])
        alpha_IYDs.extend([None])
        alpha_mu_ICUs.extend([None])
        if (sum(tr is None for tr in alpha_gamma_ICUs) - 1) == NUM_TIME_BLOCK_TO_OPTIMIZE:
            newly_fixed_time_blocks += 1
            alpha_gamma_ICUs[-NUM_TIME_BLOCK_TO_OPTIMIZE-1] = solution["alpha_gamma_ICU"][num_fixed_time_blocks + newly_fixed_time_blocks] 
            alpha_IHs[-NUM_TIME_BLOCK_TO_OPTIMIZE-1] = solution["alpha_IH"][num_fixed_time_blocks + newly_fixed_time_blocks]
            alpha_IYDs[-NUM_TIME_BLOCK_TO_OPTIMIZE-1] = solution["alpha_IYD"][num_fixed_time_blocks + newly_fixed_time_blocks]
            alpha_mu_ICUs[-NUM_TIME_BLOCK_TO_OPTIMIZE-1] = solution["alpha_mu_ICU"][num_fixed_time_blocks + newly_fixed_time_blocks]
        if solution is None:
            initial_guess = np.array([0] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks) * 4) 
        else:
            # initial_guess = []
            # modelo = len(solution["alphas"]) / 4
            # for i in range(len(solution["alphas"])):
            #     initial_guess.append(solution["alphas"][i])
            #     if (i + 1) % modelo == 0:
            #         initial_guess.append(0.5)
            alphas = zip(solution["alpha_gamma_ICU"][num_fixed_time_blocks + newly_fixed_time_blocks:],
                                                        solution["alpha_IH"][num_fixed_time_blocks + newly_fixed_time_blocks:], 
                                                        solution["alpha_IYD"][num_fixed_time_blocks + newly_fixed_time_blocks:],
                                                        solution["alpha_mu_ICU"][num_fixed_time_blocks + newly_fixed_time_blocks:])
            
            initial_guess_alphas = []
            for (a, b,c,d) in alphas:
                initial_guess_alphas.extend([a, b, c, d])
            initial_guess = np.array(initial_guess_alphas + [0] * 4 )
        # Lower and upper bound tuple:
        lower_bound = [0] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks) * 4
        upper_bound = [1] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks) * 4
        x_bound = (lower_bound, upper_bound)
        change_dates.append(end_date)
        print(end_date)
        print(initial_guess)
        print(x_bound)
        print(change_dates)
        time_frame = (city.cal.calendar.index(fixed_alpha_end_date), city.cal.calendar.index(end_date))
        param_fitting = ParameterFitting(city,
                                        vaccines,
                                        variables,
                                        initial_guess,
                                        x_bound,
                                        objective_weights,
                                        time_frame,
                                        change_dates,
                                        None,
                                        None,
                                        change_dates,
                                        alpha_IHs,
                                        alpha_gamma_ICUs,
                                        alpha_IYDs,
                                        alpha_mu_ICUs,
                                        is_auto=True,
                                        is_forward=True,
                                        step_size=step_size)
        solution = param_fitting.run_fit()


