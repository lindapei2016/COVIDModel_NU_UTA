from DataObjects import City, Vaccine
from ParamFittingTools_Smart import ParameterFitting
from utils_auto_param_tune import model_selection,model_plot
import os
# Import other Python packages
import datetime as dt

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_smart_v2"
#param_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt/st_4_2020-03-05_2020-07-02_transmission_lsq_estimated_data.csv"
#param_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt/st_11_2020-03-05_2020-10-22_transmission_lsq_estimated_data.csv"
#param_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt/st_19_2020-03-05_2021-02-20_transmission_lsq_estimated_data.csv"
#param_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt/st_23_2020-03-05_2021-05-20_transmission_lsq_estimated_data.csv"
#param_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt/st_27_2020-03-05_2021-08-20_transmission_lsq_estimated_data.csv"
param_path = "/Users/shuotaodiao/PycharmProjects/seir_wastewater_smart_v2/instances/calbrienstick/backward_pt/st_29_2020-03-05_2021-11-20_transmission_lsq_estimated_data.csv"
#objective = "ToIHT_history"
objective = "wastewater_viral_load"
variables = ["transmission_reduction", "viral_shedding_profile"]
#output_png = os.path.join(rootdir, "model_selection/ToIHT_2020_01_02_2021_02_20.png")
#output_png = os.path.join(rootdir, "model_selection/VL_2020_01_02_2021_02_20.png")
#output_png = os.path.join(rootdir, "model_selection/ToIHT_2020_01_02_2021_05_20.png")
#output_png = os.path.join(rootdir, "model_selection/VL_2020_01_02_2021_05_20.png")
#output_png = os.path.join(rootdir, "model_selection/ToIHT_2020_01_02_2021_08_20.png")
#output_png = os.path.join(rootdir, "model_selection/VL_2020_01_02_2021_08_20.png")
#output_png = os.path.join(rootdir, "model_selection/ToIHT_2020_01_02_2021_11_20.png")
output_png = os.path.join(rootdir, "model_selection/VL_2020_01_02_2021_11_20.png")
'''
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

viral_shedding_param = [(87.2317, 5)]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20)]
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

transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105, 0.7356415113451352, 0.6844052639693269, 0.5997681854886096, 0.6480350017827944]
cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105, 0.7356415113451352, 0.6844052639693269, 0.5997681854886096, 0.6480350017827944]

viral_shedding_param = [(87.2317, 5), (53.2020, 3), (56.5276, 3.1368)]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20), dt.datetime(2021, 5, 20), dt.datetime(2021, 8, 20)]

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


transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105, 0.7356415113451352, 0.6844052639693269, 0.5997681854886096, 0.6480350017827944, 0.6614577304024093, 0.5990408895744399, 0.539209170688709, 0.5877817745266677]
cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105, 0.7356415113451352, 0.6844052639693269, 0.5997681854886096, 0.6480350017827944, 0.6614577304024093, 0.5990408895744399, 0.539209170688709, 0.5877817745266677]
viral_shedding_param = [(87.2317, 5), (53.2020, 3), (56.5276, 3.1368), (55.7357, 3.1160)]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20), dt.datetime(2021, 5, 20), dt.datetime(2021, 8, 20), dt.datetime(2021, 11, 20)]


time_frame = (calbrienstick.cal.calendar.index(rss_start_date), calbrienstick.cal.calendar.index(final_end_date))
viral_start_index = calbrienstick.cal.calendar.index(dt.datetime(2020, 10, 21))
param_fitting = ParameterFitting(calbrienstick,
                                 vaccines,
                                variables,
                                initial_guess = None,
                                bounds = None,
                                objective_weights = None,
                                time_frame = time_frame,
                                change_dates = change_dates,
                                transmission_reduction = [],
                                cocoon = [],
                                viral_shedding_param=viral_shedding_param,
                                viral_shedding_profile_end_date=viral_shedding_profile_end_date)


model_plot(param_fitting, objective, param_path, "upper right", viral_start_index, output_png)