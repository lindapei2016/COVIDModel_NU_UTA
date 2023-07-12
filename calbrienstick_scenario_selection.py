# =====================
# Peak during the time period from Mar. 2020 to May 2020
# =====================


from DataObjects import City, Vaccine
from ParamFittingTools_Smart import ParameterFitting
from utils_auto_param_tune import model_selection, model_plot
import os
# Import other Python packages
import datetime as dt

rootdir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_sample_paths_v2"
rootdir2 = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_sample_paths_analysis_v2"
rss_start_date = dt.datetime(2020, 1, 2)
final_end_date = dt.datetime(2021, 11, 20)
num_days_ahead = 21

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
                dt.datetime(2021, 8, 20),
                dt.datetime(2021, 10, 16),
                dt.datetime(2021, 11, 20) + dt.timedelta(num_days_ahead)]

transmission_reduction = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105, 0.7356415113451352, 0.6844052639693269, 0.5997681854886096, 0.6480350017827944, 0.6614577304024093, 0.5990408895744399, 0.539209170688709, 0.5877817745266677, 0.6795943787169452, 0.5778998501198671]
cocoon = [0.6323772214438947, 0.48268730976076496, 0.803901455727129, 0.7626716767054642, 0.7726894549926526, 0.7612420187265064, 0.7708473977072722, 0.7263190222814302, 0.6691537532977219, 0.6509354388399075, 0.7554377692684465, 0.6950819054624483, 0.7521599579416107, 0.8220591066121877, 0.7365477886009473, 0.7912432295517325, 0.7216806291061568, 0.7849013491945105, 0.7356415113451352, 0.6844052639693269, 0.5997681854886096, 0.6480350017827944, 0.6614577304024093, 0.5990408895744399, 0.539209170688709, 0.5877817745266677, 0.6795943787169452, 0.5778998501198671]

viral_shedding_param = [(87.2317, 5), (53.2020, 3), (56.5276, 3.1368), (55.7357, 3.1160)]
viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20), dt.datetime(2021, 5, 20), dt.datetime(2021, 8, 20), dt.datetime(2021, 11, 20) + dt.timedelta(num_days_ahead)]



calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json",
                     "variant.json",
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

variables = ["transmission_reduction", "viral_shedding_profile"]

time_frame = (calbrienstick.cal.calendar.index(rss_start_date), calbrienstick.cal.calendar.index(final_end_date))
viral_start_index = calbrienstick.cal.calendar.index(dt.datetime(2020, 10, 21))
param_fitting = ParameterFitting(calbrienstick,
                                 vaccines,
                                 variables,
                                 initial_guess=None,
                                 bounds=None,
                                 objective_weights=None,
                                 time_frame=time_frame,
                                 change_dates=change_dates,
                                 transmission_reduction=transmission_reduction,
                                 cocoon=cocoon,
                                 viral_shedding_param=viral_shedding_param,
                                 viral_shedding_profile_end_date=viral_shedding_profile_end_date)

count_sample_paths = 98
write_csv_file = open(os.path.join(rootdir2, "r_squared_summary.csv"), "a")
for random_seed in range(1664, 1800):
    r_sqaured, flag = param_fitting.simulate_rng(rootdir, random_seed=random_seed, key="ToIHT_history",
                                                 r_squared_threshold=0.65, num_days_ahead=num_days_ahead)
    print("Random Seed: {}, R Squared: {}".format(random_seed, r_sqaured))
    if flag is True:
        print("If accepted (Y/N)?: Y")
        count_sample_paths += 1
        write_csv_file.write("{}, {}\n".format(random_seed, r_sqaured))
    else:
        print("If accepted (Y/N)?: N")
    if count_sample_paths >= 100:
        break

write_csv_file.close()
print("Number of feasible sample paths: {}".format(count_sample_paths))

