###############################################################################
# Wastewater based policy
# This script contains beginning-to-end sample path generation
#   and evaluation of CDC policies. User can specify which CDC policies
#   they would like to evaluate.
# Can split up sample path generation and policy evaluation on
#   parallel processors using ''mpi4py.''
# The number of sample paths generated (and number of replications
#   that each policy is evaluated on) is
#       num_processors_evaluation x sample_paths_generated_per_processor
#           (num_processors_evaluation) is inferred from mpi call
#           (sample_paths_generated_per_processor is a variable that is
#       specified in the code)

###############################################################################

import copy
from Engine_SimObjects_Wastewater import MultiTierPolicy_Wastewater
from Engine_DataObjects_Wastewater import City, TierInfo, Vaccine
from Engine_SimModel_Wastewater import SimReplication
import Tools_InputOutput_Wastewater
import Tools_Optimization_Wastewater

import datetime as dt
import pandas as pd

# Import other Python packages
import numpy as np
import glob
import os

from mpi4py import MPI
from pathlib import Path

base_path = Path(__file__).parent

comm = MPI.COMM_WORLD
num_processors_evaluation = comm.Get_size()
rank = comm.Get_rank()

output_dir = "policy_evaluation"
##
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

num_stages = 5

###############################################################################

# OPTIONS
# Toggle True/False or specify values for customization

# Change to False if sample paths have already been generated
need_sample_paths = True

# Different than num_processors_evaluation because
#   num_processors_sample_path is used for naming/distinguishing
#   states .json files
num_processors_sample_paths = 1 #300
sample_paths_generated_per_processor = 2

# Change to False if evaluation is already done
need_evaluation = True

# If only interested in evaluating on subset of reps
num_reps_evaluated_per_policy = 2 #300 Sonny's notes:
# important, need to be consistent with sample_paths_generated_per_processor and num_processors_sample_paths

# Reps offset
# Rep number to start on
reps_offset = 0

# If True, only test 2 policies
using_test_set_only = False

# Change to True if also want to automatically parse files
need_parse = True

# Assume that the number of processors >= 4
# When True, for parsing, will use 4 processors and give
#   1 peak to each processor
split_peaks_amongst_processors = False


###############################################################################

# Step 1: generate sample paths
# For each parallel processor, obtain 1 sample path for
#   each of the 4 peaks
# First timepoint of 25 is just to speed up sample path generation
#   using timeblocks method
# Timepoints corresponding to 93, 276, 502, and 641 correspond to
#   start of 4 peaks
if need_sample_paths:
    Tools_Optimization_Wastewater.get_sample_paths(calbrienstick,
                                        vaccines,
                                        0.65,
                                        #-np.inf,
                                        sample_paths_generated_per_processor,
                                        #timepoints=(25, 93, 276, 502, 641),
                                        timepoints=(101, 326, 466, 598),
                                        processor_rank=rank,
                                        save_intermediate_states=True,
                                        storage_folder_name="states",
                                        fixed_kappa_end_date=763)

    # Force synchronization step so that all sample paths actually exist
    #   before evaluation begins
    # Otherwise, might have situation where one processor finishes
    #   sample paths earlier and begins evaluation, but other processors
    #   have not finished sample paths and thus their sample paths files do not
    #   yet exist, causing file read errors
    comm.Barrier()
    if rank == 0:
        print("Sample path generation completed.")


###############################################################################

# Step 2: create list of policy objects
# plot viral load per person vs hospital admissions per 100k to find feasible thresholds
wastewater_thresholds_array = Tools_Optimization_Wastewater.thresholds_generator((-1, 1e9, 8e7),
                                                                                 (-1, 1e9, 8e7),
                                                                                 (-1, 1e9, 8e7),
                                                                                 (-1, 1e9, 8e7),
                                                                                 flag_tie_allowed=False)

wastetwater_policies = []
for wastewater_thresholds in wastewater_thresholds_array:
    wastetwater_policy =  MultiTierPolicy_Wastewater(calbrienstick,
                                                     wastewater_tiers,
                                                     wastewater_thresholds)
    wastetwater_policies.append(wastetwater_policy)

# output policy
writePolicy = open('/Users/shuotaodiao/PycharmProjects/COVIDModel_Wastewater/grid_search_policies.csv', 'w')
writePolicy.write("policy_id,threshold1,threshold2,threshold3,threshold4,threshold5\n")
for policy_idx in range(len(wastetwater_policies)):
    writePolicy.write("{}".format(policy_idx))
    for threshold_idx in range(num_stages):
        writePolicy.write(",{}".format(wastetwater_policies[policy_idx].lockdown_thresholds[threshold_idx]))
    writePolicy.write("\n")
writePolicy.close()

###############################################################################

# Step 3: create dictionary, where each entry corresponds to a peak
#   and contains list of SimReplication objects with loaded sample paths
#   for that peak
reps_per_peak_dict = {}
#peaks_dates_strs = ["2020-05-31", "2020-11-30", "2021-07-14", "2021-11-30"]
peaks_dates_strs = ["2020-04-12", "2020-11-23", "2021-04-12", "2021-08-22"] # Calbrienstick
if need_evaluation:
    for peak in np.arange(4): # number of peaks
        reps = []
        for p in np.arange(num_processors_sample_paths):
            for sample_path_number in np.arange(sample_paths_generated_per_processor):
                prefix = str(p) + "_" + str(sample_path_number) + "_" + peaks_dates_strs[peak] + "_"

                # Create a rep with no policy attached
                # Will edit the random number generator later, so seed does not matter
                rep = SimReplication(calbrienstick, vaccines, None, 1000, flag_wastewater_policy=True)
                Tools_InputOutput_Wastewater.import_rep_from_json(rep,
                                                       base_path / "states" / (prefix + "sim.json"),
                                                       base_path / "states" / (prefix + "v0.json"),
                                                       base_path / "states" / (prefix + "v1.json"),
                                                       base_path / "states" / (prefix + "v2.json"),
                                                       base_path / "states" / (prefix + "v3.json"),
                                                       None,
                                                       base_path / "states" / (prefix + "epi_params.json"))
                reps.append(rep)
            #print("len(reps): {}".format(len(reps)))
            #print("num_reps_evaluated_per_policy + reps_offset: {}".format(num_reps_evaluated_per_policy + reps_offset))
            if len(reps) >= num_reps_evaluated_per_policy + reps_offset:
                break
        reps_per_peak_dict[peaks_dates_strs[peak]] = reps[reps_offset:]


###############################################################################

# Step 4: split policies amongst processors and create RNG for each processor
# Some processors have base_assignment
# Others have base_assignment + 1
num_policies = len(wastetwater_policies)
base_assignment = int(np.floor(num_policies / num_processors_evaluation))
leftover = num_policies % num_processors_evaluation

slicepoints = np.append([0],
                        np.cumsum(np.append(np.full(leftover, base_assignment + 1),
                                            np.full(num_processors_evaluation - leftover, base_assignment))))

# Assume that seeds 0 through num_processors_evaluation-1 inclusively
#   were used for sample path generation
# But for safety we start from seed 1000 to start sampling (rather than
#   start from num_processors_evaluation, because if sample path generation
#   is run using, say, 300 processors, but evaluation is run using, say,
#   100 processors, then there will be seed overlap
# Right now, use a different bit generator for every parallel processor

bit_generator = np.random.MT19937(1000 + rank)

###############################################################################

# Step 5: evaluate policies
# TODO need to set up the peak dates
#peaks_start_times = [93, 276, 502, 641]
peaks_start_times = [79, 274, 425, 547]
#peaks_end_times = [215, 397, 625, 762]
peaks_end_times = [140, 424, 546, 653]
policy_ids_to_evaluate = np.arange(slicepoints[rank], slicepoints[rank + 1])
#print(policy_ids_to_evaluate)
#peaks_total_hosp_beds = [3026, 3791, 3841, 3537] # TODO need to get peak total beds for Calbrienstick
peaks_total_hosp_beds = [7624, 7624, 7624, 7624]
if need_evaluation:
    for peak in np.arange(4):

        reps = reps_per_peak_dict[peaks_dates_strs[peak]]
        end_time = peaks_end_times[peak]

        for policy_id in policy_ids_to_evaluate:
            print("peak: {}, policy id/total: {}/{}".format(peak, policy_id, len(policy_ids_to_evaluate)))

            if peak == 0 or peak == 1:
                policy = wastetwater_policies[policy_id]
            else:
                policy = wastetwater_policies[policy_id]

            cost_per_rep = []
            feasibility_per_rep = []
            stage_days_per_rep = [[] for _ in range(num_stages)]
            ICU_violation_patient_days_per_rep = []
            surge_days_per_rep = []

            rep_counter = 0

            for rep in reps:

                rep_counter += 1

                new_rep = copy.deepcopy(rep)

                epi_rand = copy.deepcopy(rep.epi_rand)
                epi_rand.random_params_dict = rep.epi_rand.random_params_dict
                epi_rand.setup_base_params()

                new_rep.epi_rand = epi_rand

                new_rep.policy = policy
                new_rep.rng = np.random.Generator(bit_generator)

                cost, feasibility, stage_days, ICU_violation_patient_days, surge_days \
                    = Tools_Optimization_Wastewater.evaluate_one_policy_one_sample_path_wasetwater(policy, new_rep, end_time + 1)
                for i in range(num_stages):
                    stage_days_per_rep[i].append(stage_days[i])
                ICU_violation_patient_days_per_rep.append(ICU_violation_patient_days)
                cost_per_rep.append(cost)
                feasibility_per_rep.append(feasibility)

                policy.reset()
                #print("debug rep_counter: {}".format(rep_counter))
                #print(stage_days_per_rep)
                # Every 10 replications, save output Sonny's notes: every 100 reps?
                if rep_counter == 1 or rep_counter % 100 == 0 or rep_counter == num_reps_evaluated_per_policy:
                    for i in range(num_stages):
                        #pass
                        np.savetxt(os.path.join(output_dir, "peak" + str(peak) + "_policy" + str(policy_id) + "_stage{}_days.csv".format(i + 1)),
                               np.array(stage_days_per_rep[i]), delimiter=",")
                    #pass
                    np.savetxt(os.path.join(output_dir,"peak" + str(peak) + "_policy" + str(policy_id) + "_ICU_violation_patient_days.csv"),
                               np.array(ICU_violation_patient_days_per_rep), delimiter=",")
                    np.savetxt(os.path.join(output_dir, "peak" + str(peak) + "_policy" + str(
                        policy_id) + "_cost.csv"), np.array(cost_per_rep), delimiter=",")
                    np.savetxt(os.path.join(output_dir, "peak" + str(peak) + "_policy" + str(
                        policy_id) + "_feasibility.csv"), np.array(feasibility_per_rep), delimiter=",")

    comm.Barrier()
    if rank == 0:
        print("Evaluation completed.")



