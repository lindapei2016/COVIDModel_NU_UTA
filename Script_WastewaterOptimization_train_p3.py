import datetime as dt
import pandas as pd

# Import other Python packages
import numpy as np
import glob # Sonny's notes: packages for finding files recursively
import os

input_dir = "/Users/shuotaodiao/PycharmProjects/COVIDModel_Wastewater/policy_evaluation"
input_dir2 = "/Users/shuotaodiao/PycharmProjects/COVIDModel_Wastewater"
output_dir = "/Users/shuotaodiao/PycharmProjects/COVIDModel_Wastewater"
# compute the objective
num_peaks = 4
num_stages = 5
num_scenarios = 0
policy_dic = {}
# compute total objective
for cur_peak in range(num_peaks):
    cur_cost_pd = pd.read_csv(os.path.join(input_dir, "aggregated_peak{}_cost.csv".format(cur_peak)), index_col=0)
    if num_scenarios < 1:
        num_scenarios = len(cur_cost_pd.index)
    #print(cur_cost_pd.info())
    for col in cur_cost_pd.columns:
        if col in policy_dic:
            policy_dic[col]["cost"] += cur_cost_pd.loc[:, col].mean()
        else:
            policy_dic[col] = {"cost": cur_cost_pd.loc[:, col].mean(),
                               "num_feasible_cases": 0}

# evaluate ICU violation
for cur_peak in range(num_peaks):
    cur_feasibility_pd = pd.read_csv(os.path.join(input_dir, "aggregated_peak{}_feasibility.csv".format(cur_peak)), index_col=0)
    for col in cur_feasibility_pd.columns:
        policy_dic[col]["num_feasible_cases"] += cur_feasibility_pd.loc[:, col].sum()


min_obj = np.inf
min_obj_feasible = np.inf
policy_id = None
policy_id_feasible = None
num_feasible_polcies = 0
for key in policy_dic.keys():
    if policy_dic[key]["cost"] < min_obj:
        min_obj = policy_dic[key]["cost"]
        policy_id = key
    if policy_dic[key]["num_feasible_cases"] >= num_scenarios * num_peaks:
        num_feasible_polcies += 1
        if policy_dic[key]["cost"] < min_obj_feasible:
            min_obj_feasible = policy_dic[key]["cost"]
            policy_id_feasible = key

policies = pd.read_csv(os.path.join(input_dir2, "grid_search_policies.csv"))
print(policies.info())

with open(os.path.join(output_dir, "grid_search_summary.txt"), 'w') as writeFile:
    print("=================================================================================", file=writeFile)
    print("Total number of scenarios: {}".format(num_scenarios), file=writeFile)
    print("There are {} feasible polices out of {} polices".format(num_feasible_polcies, len(policy_dic)), file=writeFile)
    print("Minimum objective cost of all the feasible polices: {}".format(min_obj_feasible), file=writeFile)
    print("Policy ID with Minimum Objective Cost of all the feasible policies: {}".format(policy_id_feasible), file=writeFile)
    print(policies.loc[policies["policy_id"] == int(policy_id_feasible)], file=writeFile)
    print("=================================================================================", file=writeFile)
    print("If ICU violation is ignored.", file=writeFile)
    print("Minimum objective cost: {}".format(min_obj), file=writeFile)
    print("Policy ID with Minimum Objective Cost: {}".format(policy_id), file=writeFile)
    print(policies.loc[policies["policy_id"] == int(policy_id)], file=writeFile)
    print("=================================================================================", file=writeFile)



