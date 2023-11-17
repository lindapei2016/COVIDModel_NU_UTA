##
# This script illustrates how to run the deterministic simulation with wastewater policy
##

from Engine_SimObjects_Wastewater import MultiTierPolicy_Wastewater
from Engine_DataObjects_Wastewater import City, TierInfo, Vaccine
from Engine_SimModel_Wastewater import SimReplication
from Tools_InputOutput import export_rep_to_json
from Tools_Plot import plot_from_file_ww

import datetime as dt
import pandas as pd

# Import other Python packages
import numpy as np
import os

from pathlib import Path

base_path = Path(__file__).parent

# extra functions
def extend_transmission_reduction(change_dates, tr_reduc, cocoon_reduc):
    """
    Extend the transmission reduction into a dataframe with the corresponding dates.
    """
    date_list = []
    tr_reduc_extended, cocoon_reduc_extended = [], []

    for idx in range(len(change_dates[:-1])):
        tr_reduc_extended.extend([tr_reduc[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend(
            [
                str(change_dates[idx] + dt.timedelta(days=x))
                for x in range((change_dates[idx + 1] - change_dates[idx]).days)
            ]
        )
        cocoon_reduc_extended.extend(
            [cocoon_reduc[idx]] * (change_dates[idx + 1] - change_dates[idx]).days
        )

    d = {
        "date": pd.to_datetime(date_list),
        "transmission_reduction": tr_reduc_extended,
        "cocooning": cocoon_reduc_extended,
    }
    df_transmission = pd.DataFrame(data=d)
    return df_transmission


###############################################################################
# wastewater threshold
#wastewater_thresholds = (-1, -1.0, -1.0, 399999999.0, 999999999.0)
wastewater_thresholds = (-1, 79999999.0, 159999999.0, 319999999.0, 959999999.0)
# initialize the city object
calbrienstick = City("calbrienstick", "calbrienstick_test_IHT.json", "calendar.csv", "setup_data_Final_new.json",
                     "variant_post_pandemic_test1.json",
                    "transmission_null.csv",
                    "IH_null.csv",  # hospitalization file name
                    "icu_null.csv",
                    "calbrienstick_hosp_admin_est_Katelyn_after_202303.csv",  # ToIHT
                    "death_null.csv",
                    "home_death_null.csv",
                    "variant_prevalence_post_pandemic.csv",
                    "calbrienstick_viral_merge_dpcr_qcpr_v2_test2.csv",
                    "viral_shedding_profile_corr.json")

# Transmission Reduction, Cocooning
part_num = 13
final_end_date = dt.datetime(2023, 8, 16)

change_dates_pt = {}
transmission_reduction_pt = {}
cocoon_pt = {}
for i in range(1, part_num):
    change_dates_pt[i] = []
    transmission_reduction_pt[i] = []
    cocoon_pt[i] = []

# part 1 period: Jan. 2, 2020 - Oct. 21, 2020
change_dates_pt[1] = [dt.datetime(2020, 1, 2),
                dt.datetime(2020, 2, 27),
                dt.datetime(2020, 3, 5),
                dt.datetime(2020, 3, 19),
                dt.datetime(2020, 3, 26),
                dt.datetime(2020, 5, 21),
                dt.datetime(2020, 6, 11),
                dt.datetime(2020, 10, 8),
                dt.datetime(2020, 10, 21)]
# part 2 period: Oct. 21, 2020, Feb. 20, 2021
change_dates_pt[2] = [dt.datetime(2020, 11, 4), dt.datetime(2021, 1, 13), dt.datetime(2021, 2, 20)]
# part 3 period: Fec. 21, 2021, May. 20, 2021
change_dates_pt[3] = [dt.datetime(2021, 2, 27), dt.datetime(2021, 3, 6), dt.datetime(2021, 4, 3), dt.datetime(2021, 5, 20)]
# part 4 period: May. 21, 2021, Aug. 20, 2021
change_dates_pt[4] = [dt.datetime(2021, 7, 1), dt.datetime(2021, 8, 20)]
# part 5 period: Aug. 21, 2021, Nov. 16, 2021
change_dates_pt[5] = [dt.datetime(2021, 10, 15), dt.datetime(2021, 11, 16)]
# part 6 period: Nov. 16, 2021, Feb. 16, 2022
change_dates_pt[6] = [dt.datetime(2021, 12, 28), dt.datetime(2022, 2, 16)]
# part 7 period: Feb. 16, 2022, May. 16, 2022
change_dates_pt[7] = [dt.datetime(2022, 3, 9), dt.datetime(2022, 5, 16)]
# part 8 period: May. 16, 2022, Aug. 16, 2022
change_dates_pt[8] = [dt.datetime(2022, 6, 16), dt.datetime(2022, 7, 16), dt.datetime(2022, 8, 16)]
# part 9 period: Aug. 16, 2022, Nov. 16, 2022
change_dates_pt[9] = [dt.datetime(2022, 9, 16), dt.datetime(2022, 10, 16), dt.datetime(2022, 11, 16)]
# part 10 period: Nov. 16 2022, Feb. 16, 2023
change_dates_pt[10] = [dt.datetime(2022, 12, 16), dt.datetime(2023, 1, 16), dt.datetime(2023, 2, 16)]
# part 11 period: Feb. 16, 2023, May. 16, 2023
change_dates_pt[11] = [dt.datetime(2023, 3, 16), dt.datetime(2023, 4, 16), dt.datetime(2023, 5, 16)]
# part 12 period: May. 16, 2023, Aug. 16, 2023
change_dates_pt[12] = [dt.datetime(2023, 6, 16), dt.datetime(2023, 7, 16), dt.datetime(2023, 8, 16)]


transmission_reduction_pt[1] = [0.6390371782202954, 0.5516637746067924, 0.4593874352740992, 0.5431581035156047, 0.8010305164649246, 0.8650087984865349, 0.7704266683825446, 0.7022778460719028]
cocoon_pt[1] = [0.6390371782202954, 0.5516637746067924, 0.4593874352740992, 0.5431581035156047, 0.8010305164649246, 0.8650087984865349, 0.7704266683825446, 0.7022778460719028]

transmission_reduction_pt[2] = [0.7071045652992998, 0.7827570544989622, 0.7787963600659638]
cocoon_pt[2] = [0.7071045652992998, 0.7827570544989622, 0.7787963600659638]

transmission_reduction_pt[3] = [0.814847140552781, 0.7426532922510829, 0.641859302260311, 0.7027421062417172]
cocoon_pt[3] = [0.814847140552781, 0.7426532922510829, 0.641859302260311, 0.7027421062417172]

transmission_reduction_pt[4] = [0.7074411039930196, 0.5669710578471897]
cocoon_pt[4] = [0.7074411039930196, 0.5669710578471897]

transmission_reduction_pt[5] = [0.716166004155231, 0.6500987928460659]
cocoon_pt[5] = [0.716166004155231, 0.6500987928460659]

transmission_reduction_pt[6] = [0.5462696682946308, 0.7395507152540185]
cocoon_pt[6] = [0.5462696682946308, 0.7395507152540185]

transmission_reduction_pt[7] = [0.7817589839729123, 0.6953047385645258]
cocoon_pt[7] = [0.7817589839729123, 0.6953047385645258]

transmission_reduction_pt[8] = [0.7465481833959726, 0.7269208821585026, 0.7046844112369961]
cocoon_pt[8] = [0.7465481833959726, 0.7269208821585026, 0.7046844112369961]

transmission_reduction_pt[9] = [0.6894160473940694, 0.6505859845073149, 0.6117343380597784]
cocoon_pt[9] = [0.6894160473940694, 0.6505859845073149, 0.6117343380597784]

transmission_reduction_pt[10] = [0.46838615919842624, 0.26398925931825507, 0]
cocoon_pt[10] = [0.46838615919842624, 0.26398925931825507, 0]

transmission_reduction_pt[11] = [0, 0.06530968385930111, 0.2572336438257812]
cocoon_pt[11] = [0, 0.06530968385930111, 0.2572336438257812]

transmission_reduction_pt[12] = [0.4561448089044957, 0.5724309574843542, 0.5827562136593519]
cocoon_pt[12] = [0.4561448089044957, 0.5724309574843542, 0.5827562136593519]

change_dates = change_dates_pt[1]
tr_reduc = transmission_reduction_pt[1]
cocoon_reduc = cocoon_pt[1]
for i in range(2, part_num):
    change_dates = change_dates + change_dates_pt[i]
    tr_reduc = tr_reduc + transmission_reduction_pt[i]
    cocoon_reduc = cocoon_reduc + cocoon_pt[i]

viral_shedding_param = [(88.2202949284742, 5),
                        (53.29762348814473, 3.0291887544213423),
                        (57.01558877635972, 3.222699787919262),
                        (57.07113028608089, 3.2629926768862725),
                        (52.070491827093655, 3.0000000000000164),
                        (79.79330168005905, 4.999999999941512),
                        (77.00439043232691, 4.999999999999996),
                        (75.97691611154939, 4.999999999999524),
                        (70.99696336506713, 4.664767517262472),
                        (76.48024848874383, 4.99999999995958),
                        (77.9743482318601, 4.999999999999994)]

viral_shedding_profile_end_date = [dt.datetime(2021, 2, 20),
                                   dt.datetime(2021, 5, 20),
                                   dt.datetime(2021, 8, 20),
                                   dt.datetime(2021, 11, 16),
                                   dt.datetime(2022, 2, 16),
                                   dt.datetime(2022, 5, 16),
                                   dt.datetime(2022, 8, 16),
                                   dt.datetime(2022, 11, 16),
                                   dt.datetime(2023, 2, 16),
                                   dt.datetime(2023, 5, 16),
                                   final_end_date]

df_transmission = extend_transmission_reduction(change_dates, tr_reduc, cocoon_reduc)
transmission_reduction = [
            (d, tr)
            for (d, tr) in zip(
                df_transmission["date"], df_transmission["transmission_reduction"]
            )
        ]
calbrienstick.cal.load_fixed_transmission_reduction(transmission_reduction)

cocooning = [
            (d, c) for (d, c) in zip(df_transmission["date"], df_transmission["cocooning"])
        ]
calbrienstick.cal.load_fixed_cocooning(cocooning)

## Viral Shedding Profile
calbrienstick.viral_shedding_date_period_map(viral_shedding_profile_end_date)
calbrienstick.load_fixed_viral_shedding_param(viral_shedding_profile_end_date, viral_shedding_param)

wastewater_tiers = TierInfo("calbrienstick", "tiers_wastewater.json")


vaccines = Vaccine(
    calbrienstick,
    "calbrienstick",
    "vaccines.json",
    "calbrienstick_booster_allocation_Sonny_after_202305.csv",
    "calbrienstick_vaccine_allocation_Sonny_after_202305.csv",
)


# wastewater policy

wastetwater_policy =  MultiTierPolicy_Wastewater(calbrienstick,
                                                     wastewater_tiers,
                                                     wastewater_thresholds)

seed = -1
rep = SimReplication(calbrienstick, vaccines, wastetwater_policy, seed, flag_wastewater_policy=True)

history_end_time = dt.datetime(2021, 11, 1)  # use fixed transmission value until history en time.
simulation_end_time = dt.datetime(2022, 3, 1)

fixed_kappa_end_date = calbrienstick.cal.calendar.index(history_end_time)
rep.simulate_time_period(calbrienstick.cal.calendar.index(simulation_end_time),fixed_kappa_end_date=fixed_kappa_end_date)


rep.output_hospital_history_var("ToIHT", f"{base_path}/retrospective_analysis/seed{seed}/ToIHT{str(wastetwater_policy)}.csv")
rep.output_viral_load(f"{base_path}/retrospective_analysis/seed{seed}/viral_load{str(wastetwater_policy)}.csv")
rep.output_tier_history(f"{base_path}/retrospective_analysis/seed{seed}/tier_history{str(wastetwater_policy)}.csv")