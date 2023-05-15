from DataObjects import City, TierInfo, Vaccine
from ParamFittingTools import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd
import os
from pathlib import Path

###############################################################################
# We need to define city, vaccine and tier object as the least square fit will
# run the deterministic simulation model.

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/feasible_sample_path_corr"

folders = None
for subdir, dirs, files in os.walk(rootdir):
    print(dirs)
    folders = dirs
    break

print("Number of sample paths: {}".format(len(folders)))
count_folder = 0
for folder in folders:
    count_folder += 1
    print("************************************")
    print("Folder: " + folder)
    print("Number of Scenarios Solved: {}/{}".format(count_folder, len(folders)))
    curdir = rootdir + "/" + folder
    austin = City("austin", "austin_test_IHT.json", "calendar.csv", "setup_data_Final.json", "variant.json",
                  "transmission.csv",
                  curdir + "/IH.csv",  # hospitalization file name
                  "austin_real_icu_updated.csv",
                  curdir + "/ToIHT.csv",
                  "austin_real_death_from_hosp_updated.csv",
                  "austin_real_death_from_home.csv",
                  "variant_prevalence.csv",
                  curdir + "/viral_load.csv",
                  "tparam_fit/viral_shedding_profile.json")

    output_folder = curdir + "/calibration_new"
    output_path_fixed_transmission = output_folder + "/fixed_transmission_reduction.csv"
    output_path_fixed_cocooning = output_folder + "/fixed_cocooning.csv"

    output_path_sol = output_folder + "/sol_output.txt"
    output_path_transmission = output_folder + "/sol_transmission.csv"
    output_path_viral = output_folder + "/sol_viral.csv"
    if os.path.exists(output_path_sol) is True:
        continue
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    vaccines = Vaccine(
        austin,
        "austin",
        "vaccines.json",
        "booster_allocation_fixed.csv",
        "vaccine_allocation_fixed.csv",
    )

    vaccines = Vaccine(
        austin,
        "austin",
        "vaccines.json",
        "booster_allocation_fixed.csv",
        "vaccine_allocation_fixed.csv",
    )

    # We need to define time blocks where the transmission reduction (or the behaviour in the population) changes:
    change_dates = [dt.date(2020, 2, 15),
                    dt.date(2020, 3, 24),
                    dt.date(2020, 5, 21),
                    dt.date(2020, 6, 26),
                    dt.date(2020, 8, 20),
                    dt.date(2020, 10, 29),
                    dt.date(2020, 11, 30)]
    # the change_dates has one more entry than transmission reduction

    # We don't fit all the transmission reduction values from scratch as the least square fit cannot handle too many
    # decision variables. We used the existing fitted values for the earlier fit data. In transmission_reduction and
    # cocoon lists you can input the already existing values and for the new dates you need to estimate just input None.
    transmission_reduction = [0.052257,
                              0.787752,
                              None,
                              None,
                              None,
                              None]
    # for the high risk groups uses cocoon instead of contact reduction
    cocoon = np.array([0,
                       0.787752,
                       None,
                       None,
                       None,
                       None])

    end_date = []
    for idx in range(len(change_dates[1:])):
        end_date.append(str(change_dates[1:][idx] - dt.timedelta(days=1)))

    #
    table = pd.DataFrame(
        {
            "start_date": change_dates[:-1],
            "end_date": end_date,
        }
    )

    # The initial guess of the variables to estimate:
    initial_guess = np.array([50, 5, 0.75, 0.75, 0.75, 0.75])

    # Lower and upper bound tuple:
    x_bound = ([0, 0, 0, 0, 0, 0], [100, 10, 1, 1, 1, 1])

    # Austin weights for the least-square fit:
    # You can input the data you would like to use in the process and corresponding weights. Different data have different
    # scales that's why we use weights.
    # objective_weights = {"ToIHT_history": 1,
    #                    "wastewater_viral_load": 0.26}
    objective_weights = {"ToIHT_history": 1,
                         "wastewater_viral_load": 26.75}
    variables = ["viral_shedding_profile", "transmission_reduction"]

    # We can define the time frame we would like to use data from as follows:
    time_frame = (
    austin.cal.calendar.index(dt.datetime(2020, 3, 1)), austin.cal.calendar.index(dt.datetime(2020, 11, 30)))

    param_fitting = ParameterFitting(austin,
                                     vaccines,
                                     variables,
                                     initial_guess,
                                     x_bound,
                                     objective_weights,
                                     time_frame,
                                     change_dates,
                                     transmission_reduction,
                                     cocoon)

    # output solutions

    solution = param_fitting.run_fit(output_path_sol, output_path_viral, output_path_transmission)

    param_fitting.rep.instance.cal.output_fixed_transmission_reduction(output_path_fixed_transmission)
    param_fitting.rep.instance.cal.output_fixed_cocooning(output_path_fixed_cocooning)

