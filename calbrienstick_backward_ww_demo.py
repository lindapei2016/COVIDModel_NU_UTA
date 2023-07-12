from DataObjects import City, Vaccine
from utils_auto_param_tune import auto_pt_backward

# Import other Python packages
import datetime as dt
import os

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick"
part_num = 6
correct_method = "Kpower"
output_path_sol = os.path.join(rootdir, "sol_part{}_{}.txt".format(part_num, correct_method))
output_path_transmission = os.path.join(rootdir, "transmission_part{}_{}.csv".format(part_num, correct_method))
output_path_sim_hosp_admin = os.path.join(rootdir, "sim_hosp_admin_part{}_{}.csv".format(part_num, correct_method))
output_path_vsp = os.path.join(rootdir, "viral_shedding_profile_part{}_{}.csv".format(part_num, correct_method))
output_path_viral_load = os.path.join(rootdir, "viral_load_part{}_{}.csv".format(part_num, correct_method))

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
# If there is no fixed kappa, then this date would need to be the same as the start of the simulation
#rss_start_date = dt.datetime(2020, 7, 2)
rss_start_date = dt.datetime(2020, 10, 21)
# This is the time period where the entire parameter tuning ends
#final_end_date = dt.datetime(2020, 10, 22) # end date must be consistent with the jason file
final_end_date = dt.datetime(2021, 2, 20)

# You can change this based on your case
# if there is no time blocks you want to define, then change_dates = [dt.datetime(start date of the simulation)]


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

viral_shedding_param = [None]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20)]
transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075]
cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075]
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

#transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129]
#transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075]
transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105]
#cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129]
#cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075]
cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105]

viral_shedding_param = [(87.2317, 5), None]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20), dt.datetime(2021, 5, 20)]
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

viral_shedding_param = [(87.2317, 5), (53.2020, 3), None]
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
viral_shedding_param = [(87.2317, 5), (53.2020, 3), (56.5276, 3.1368), None]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20), dt.datetime(2021, 5, 20), dt.datetime(2021, 8, 20), dt.datetime(2021, 11, 20)]

# weights for the forward parameter tunes
# Since we are only fitting tramsission and cocoon, the only weights that matters is the hospital admission weight and kappa_weight
# kappa_weight is designed to minimize the change from kappa_{t-1} to kappa_{t}
'''
objective_weights = {"ToIHT_history": 1,
                     "kappa_weight": 7 * 50}
'''

objective_weights = {"ToIHT_history": 1,
                    "kappa_weight": 7 * 50,
                     "log_wastewater_viral_load": 8}

# This is the json solution file. It is the starting point of the backward parameter tuning.
#solution_file = "./instances/calbrienstick/forward_pt/st_7_2020-03-05_2020-10-22_lsq_estimated_data.json"
#solution_file = "./instances/calbrienstick/forward_pt/st_7_2020-03-05_2021-02-20_lsq_estimated_data.json"
#solution_file = "./instances/calbrienstick/forward_pt/st_7_2020-03-05_2021-05-20_lsq_estimated_data.json"
#solution_file = "./instances/calbrienstick/forward_pt/st_7_2020-03-05_2021-08-20_lsq_estimated_data.json"
solution_file = "./instances/calbrienstick/forward_pt/st_7_2020-03-05_2021-11-20_lsq_estimated_data.json"
auto_pt_backward(solution_file,
                     calbrienstick,
                     vaccines,
                     objective_weights,
                     transmission_reduction,
                     cocoon,
                     change_dates,
                     rss_start_date,
                     final_end_date,
                 viral_shedding_param=viral_shedding_param,
                 viral_shedding_profile_end_date=viral_shedding_profile_end_date
                 )
