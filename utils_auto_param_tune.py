###############################################################################

# auto_param_tune_forward.py
# Designed and Implemented by Guyi chen 2023
# Reorganized by Sonny
###############################################################################

from DataObjects import City, Vaccine
from ParamFittingTools_Smart import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd
import json
import os
###############################################################################

# forward
def auto_pt_foward(city,
                   vaccines,
                   step_size,
                   objective_weights,
                   fixed_kappa_end_date,
                   final_end_date,
                   change_dates,
                   transmission_reduction,
                   cocoon,
                   NUM_TIME_BLOCK_TO_OPTIMIZE = 10,
                   viral_shedding_param = None,
                   viral_shedding_profile_end_date = None,
                   output_path_sol = None,
                   csv_viral_shedding_path=None,
                   csv_transmission_reduction_path=None,
                   csv_hosp_admin=None,
                   csv_viral_load=None):
    print("step_size", step_size)
    end_date = fixed_kappa_end_date
    num_vsp = 0
    if viral_shedding_param is None:
        variables = ["transmission_reduction"]
    else:
        variables = ["viral_shedding_profile", "transmission_reduction"]
        num_vsp = count_none(viral_shedding_param)
    solution = None

    num_fixed_time_blocks = len(change_dates) - 1
    newly_fixed_time_blocks = 0
    print(end_date)
    print(final_end_date)
    while (end_date < final_end_date):
        end_date += dt.timedelta(days=step_size)
        if end_date > final_end_date: # sonny, moving end date needs to be smaller than the final end date
            end_date = final_end_date
        transmission_reduction.extend([None])
        cocoon.extend([None])
        initial_guess = []
        if viral_shedding_param is not None:
            initial_guess.extend([50, 4] * num_vsp)
        if (sum(tr is None for tr in transmission_reduction) - 1) == NUM_TIME_BLOCK_TO_OPTIMIZE:
            newly_fixed_time_blocks += 1
            transmission_reduction[-NUM_TIME_BLOCK_TO_OPTIMIZE - 1] = solution["transmission_reduction"][
                num_fixed_time_blocks + newly_fixed_time_blocks]
            cocoon[-NUM_TIME_BLOCK_TO_OPTIMIZE - 1] = solution["transmission_reduction"][
                num_fixed_time_blocks + newly_fixed_time_blocks]
        if solution is None:
            initial_guess.extend([0.5] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks))
        else:
            initial_guess.extend(
                solution["transmission_reduction"][num_fixed_time_blocks + newly_fixed_time_blocks:] + [0.5])
        # convert initial guess to numpy array
        initial_guess = np.array(initial_guess)
        # Lower and upper bound tuple:\
        lower_bound = []
        upper_bound = []
        if viral_shedding_param is not None:
            lower_bound.extend([0, 3] * num_vsp)
            upper_bound.extend([200, 5] * num_vsp)
        lower_bound.extend([0] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks))
        upper_bound.extend([1] * (len(change_dates) - num_fixed_time_blocks - newly_fixed_time_blocks))
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
                                         step_size=step_size,
                                         viral_shedding_param=viral_shedding_param,
                                         viral_shedding_profile_end_date=viral_shedding_profile_end_date)

        solution = param_fitting.run_fit(path=output_path_sol,
                                         csv_viral_shedding_path=csv_viral_shedding_path,
                                         csv_transmission_reduction_path=csv_transmission_reduction_path,
                                         csv_hosp_admin=csv_hosp_admin,
                                         csv_viral_load=csv_viral_load)


# backward
def auto_pt_backward(solution_file,
                     city,
                     vaccines,
                     objective_weights,
                     transmission_reduction_fixed,
                     cocoon_fixed,
                     change_dates,
                     fixed_kappa_end_date,
                     end_date,
                     viral_shedding_param=None,
                     viral_shedding_profile_end_date=None,
                     output_path_sol=None,
                     csv_viral_shedding_path=None,
                     csv_transmission_reduction_path=None,
                     csv_hosp_admin=None,
                     csv_viral_load=None):
    num_fixed_kappa = len(change_dates) - 1
    print("Number of fixed kappa: {}".format(num_fixed_kappa))
    #for i in range(((end_date - fixed_kappa_end_date).days) // 7):
    #    change_dates.append(fixed_kappa_end_date + dt.timedelta(days=(i + 1) * 7))
    cur_date = fixed_kappa_end_date # Sonny
    while cur_date < end_date:
        cur_date = cur_date + dt.timedelta(7)
        if cur_date > end_date:
            cur_date = end_date
        change_dates.append(cur_date)
    with open(solution_file) as file:
        solution = json.load(file)
    num_vsp = 0
    if viral_shedding_param is None:
        variables = ["transmission_reduction"]
    else:
        variables = ["viral_shedding_profile", "transmission_reduction"]
        num_vsp = count_none(viral_shedding_param)

    while True:
        if len(change_dates) > num_fixed_kappa:
            transmission_reduction_solution = solution["transmission_reduction"][num_fixed_kappa:]
            # find the absolute difference between kappa_{t-1} and kappa_{t} and its index
            difference_in_transmission_reduction = [
                abs(transmission_reduction_solution[i] - transmission_reduction_solution[i + 1]) for i in
                range(len(transmission_reduction_solution) - 1)]
            print("transmission_non_fixed:", len(transmission_reduction_solution))
            if len(difference_in_transmission_reduction) < 1:
                break
            min_change = min(difference_in_transmission_reduction)
            min_change_idx = difference_in_transmission_reduction.index(min_change)
            print(min_change, min_change_idx)
            # delete the corresponding time block from the transmission_reduction
            del transmission_reduction_solution[min_change_idx]
            del change_dates[min_change_idx + num_fixed_kappa + 1]
            transmission_reduction = transmission_reduction_fixed + [None] * (len(change_dates) - num_fixed_kappa - 1)
            cocoon = cocoon_fixed + [None] * (len(change_dates) - num_fixed_kappa - 1)
            initial_guess = []
            if viral_shedding_param is not None:
                initial_guess.extend([50, 4] * num_vsp)
            initial_guess.extend(transmission_reduction_solution)
            initial_guess = np.array(initial_guess)

            # Lower and upper bound tuple:
            lower_bound = []
            upper_bound = []
            if viral_shedding_param is not None:
                lower_bound.extend([0, 3] * num_vsp)
                upper_bound.extend([200, 5] * num_vsp)
            lower_bound.extend([0] * (len(change_dates) - num_fixed_kappa - 1))
            print("debug, dim of lower bound: {}".format(len(lower_bound)))
            upper_bound.extend([1] * (len(change_dates) - num_fixed_kappa - 1))
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
                                             step_size=len(change_dates),
                                             viral_shedding_param=viral_shedding_param,
                                             viral_shedding_profile_end_date=viral_shedding_profile_end_date)

            solution = param_fitting.run_fit(path=output_path_sol,
                                         csv_viral_shedding_path=csv_viral_shedding_path,
                                         csv_transmission_reduction_path=csv_transmission_reduction_path,
                                         csv_hosp_admin=csv_hosp_admin,
                                         csv_viral_load=csv_viral_load)
        else:
            break

# model selection, Sonny
def model_selection(param_fitting_model, key, dir_path, output_path): # root path for the transmission reduction parameters
    writeFile = open(output_path, 'w')
    writeFile.write("file_name,rss({})\n".format(key))
    min_rss = 0
    min_filename = ""
    flag_first = True
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            print("Load {}".format(file))
            param_fitting_model.load_transmission_reduction(os.path.join(dir_path, file))
            rss = param_fitting_model.compute_rss(key)
            print("Residual sum of squares: {}".format(rss))
            if flag_first is True:
                min_rss = rss
                min_filename = file
                flag_first = False
            elif min_rss > rss:
                min_rss = rss
                min_filename = file
            writeFile.write("{},{}\n".format(file, rss))

    print("Model with lowest rss: {}".format(min_filename))
    print("Lowest rss({}): {}".format(key, min_rss))
    writeFile.close()


def model_plot(param_fitting_model, key, input_path, location, viral_start_index = 0, output_path = None):
    dates = param_fitting_model.load_transmission_reduction(input_path)
    param_fitting_model.plot_rss(key, dates, location, viral_start_index = viral_start_index, output_path = output_path)


def count_none(var_list):
    res = 0
    for var in var_list:
        if var is None:
            res += 1
    return res