from DataObjects import City, Vaccine
from ParamFittingTools_Smart import ParameterFitting
from utils_auto_param_tune import model_selection
import os
# Import other Python packages
import datetime as dt

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_smart_v2"
paramdir = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt"
part_num = 6
correct_method = "Kpower"
output_path_rss = os.path.join(rootdir, "rss_{}_{}.csv".format(part_num, correct_method))

objective = "ToIHT_history"
variables = ["transmission_reduction"]

calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json", "variant.json",
                "transmission.csv",
                "IH_null.csv",  # hospitalization file name
                "icu_null.csv",
                "calbrienstick_hosp_admin_est.csv",  # ToIHT
                "death_null.csv",
                "home_death_null.csv",
                "variant_prevalence.csv")

vaccines = Vaccine(
    calbrienstick,
    "calbrienstick",
    "vaccines.json",
    "booster_calbrienstick.csv",
    "vaccine_allocation_calbrienstick.csv",
)

'''
#rss_start_date = dt.datetime(2020, 1, 2)
rss_start_date = dt.datetime(2020, 7, 2)

# This is the time period where the entire parameter tuning ends
#final_end_date = dt.datetime(2020, 7, 2)
final_end_date = dt.datetime(2020, 10, 22)

# You can change this based on your case
# if there is no time blocks you want to define, then change_dates = [dt.datetime(start date of the simulation)]
#change_dates = [dt.datetime(2020, 1, 2)]

change_dates = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 7, 2)]
'''
'''
rss_start_date = dt.datetime(2020, 10, 21)

final_end_date = dt.datetime(2021, 2, 20)

change_dates = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 7, 2),
                dt.datetime(2020, 7, 9),
                dt.datetime(2020, 7, 23),
                dt.datetime(2020, 9, 17),
                dt.datetime(2020, 10, 1),
                dt.datetime(2020, 10, 8),
                dt.datetime(2020, 10, 15),
                dt.datetime(2020, 10, 22)]
'''
'''
rss_start_date = dt.datetime(2021, 2, 21)
final_end_date = dt.datetime(2021, 5, 20)

change_dates = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 7, 2),
                dt.datetime(2020, 7, 9),
                dt.datetime(2020, 7, 23),
                dt.datetime(2020, 9, 17),
                dt.datetime(2020, 10, 1),
                dt.datetime(2020, 10, 8),
                dt.datetime(2020, 10, 15),
                dt.datetime(2020, 10, 22),
                dt.datetime(2020, 10, 28),
                dt.datetime(2020, 11, 4),
                dt.datetime(2020, 11, 11),
                dt.datetime(2020, 11, 18),
                dt.datetime(2020, 12, 2),
                dt.datetime(2020, 12, 23),
                dt.datetime(2021, 1, 6),
                dt.datetime(2021, 2, 20)]
'''

'''
rss_start_date = dt.datetime(2021, 5, 21)
final_end_date = dt.datetime(2021, 8, 20)

change_dates = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 7, 2),
                dt.datetime(2020, 7, 9),
                dt.datetime(2020, 7, 23),
                dt.datetime(2020, 9, 17),
                dt.datetime(2020, 10, 1),
                dt.datetime(2020, 10, 8),
                dt.datetime(2020, 10, 15),
                dt.datetime(2020, 10, 22),
                dt.datetime(2020, 10, 28),
                dt.datetime(2020, 11, 4),
                dt.datetime(2020, 11, 11),
                dt.datetime(2020, 11, 18),
                dt.datetime(2020, 12, 2),
                dt.datetime(2020, 12, 23),
                dt.datetime(2021, 1, 6),
                dt.datetime(2021, 2, 20),
                dt.datetime(2021, 2, 28),
                dt.datetime(2021, 3, 7),
                dt.datetime(2021, 4, 4),
                dt.datetime(2021, 5, 20)]
'''

rss_start_date = dt.datetime(2021, 8, 21)
final_end_date = dt.datetime(2021, 11, 20)

change_dates = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 7, 2),
                dt.datetime(2020, 7, 9),
                dt.datetime(2020, 7, 23),
                dt.datetime(2020, 9, 17),
                dt.datetime(2020, 10, 1),
                dt.datetime(2020, 10, 8),
                dt.datetime(2020, 10, 15),
                dt.datetime(2020, 10, 22),
                dt.datetime(2020, 10, 28),
                dt.datetime(2020, 11, 4),
                dt.datetime(2020, 11, 11),
                dt.datetime(2020, 11, 18),
                dt.datetime(2020, 12, 2),
                dt.datetime(2020, 12, 23),
                dt.datetime(2021, 1, 6),
                dt.datetime(2021, 2, 20),
                dt.datetime(2021, 2, 28),
                dt.datetime(2021, 3, 7),
                dt.datetime(2021, 4, 4),
                dt.datetime(2021, 5, 20),
                dt.datetime(2021, 6, 25),
                dt.datetime(2021, 7, 2),
                dt.datetime(2021, 8, 6),
                dt.datetime(2021, 8, 20)]



time_frame = (calbrienstick.cal.calendar.index(rss_start_date), calbrienstick.cal.calendar.index(final_end_date))
param_fitting = ParameterFitting(calbrienstick,
                                 vaccines,
                                variables,
                                initial_guess = None,
                                bounds = None,
                                objective_weights = None,
                                time_frame = time_frame,
                                change_dates = change_dates,
                                transmission_reduction = [],
                                cocoon = [])


model_selection(param_fitting, objective, paramdir, output_path_rss)