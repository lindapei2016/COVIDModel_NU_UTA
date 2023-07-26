###############################################################################
#
# LP's WIP parsing script scratchpad
# Right now very messy and redundant -- but successfully parses
#   aggregated .csv files after running "parse" option in Script_CDCOptimization.py
# Assumes there is a folder called "singleindicator_aggregated" with the relevant aggregated
#   .csv files
#
###############################################################################

import pandas as pd
import numpy as np
import time
import glob
import itertools
import matplotlib.pyplot as plt

from pathlib import Path

base_path = Path(__file__).parent

###############################################################################

def thresholds_generator(stage2_info, stage3_info, stage4_info, stage5_info):
    """
    Creates a list of 5-tuples, where each 5-tuple has the form
        (-1, t2, t3, t4, t5) with 0 <= t2 <= t3 <= t4 <= t5 < inf.
    The possible values t2, t3, t4, and t5 can take come from
        the grid generated by stage2_info, stage3_info, stage4_info,
        and stage5_info respectively.
    Stage 1 threshold is always fixed to -1 (no social distancing).

    :param stage2_info: [3-tuple] with elements corresponding to
        start point, end point, and step size
        for candidate values for stage 2
    :param stage3_info: same as above but for stage 3
    :param stage4_info: same as above but for stage 4
    :param stage5_info: same as above but for stage 5
    :return: [array] of 5-tuples
    """

    # Create an array (grid) of potential thresholds for each stage
    stage2_options = np.arange(stage2_info[0], stage2_info[1], stage2_info[2])
    stage3_options = np.arange(stage3_info[0], stage3_info[1], stage3_info[2])
    stage4_options = np.arange(stage4_info[0], stage4_info[1], stage4_info[2])
    stage5_options = np.arange(stage5_info[0], stage5_info[1], stage5_info[2])

    # Using Cartesian products, create a list of 5-tuple combos
    stage_options = [stage2_options, stage3_options, stage4_options, stage5_options]
    candidate_feasible_combos = []
    for combo in itertools.product(*stage_options):
        candidate_feasible_combos.append((-1,) + combo)

    # Eliminate 5-tuples that do not satisfy monotonicity constraint
    # However, ties in thresholds are allowed
    feasible_combos = []
    for combo in candidate_feasible_combos:
        if np.all(np.diff(combo) >= 0):
            feasible_combos.append(combo)

    return feasible_combos


###############################################################################

# Better to replace this with a .csv file of policies or something

single_indicator_policies = True

policies = []

case_threshold = 200

non_surge_hosp_adm_thresholds_array = thresholds_generator((-1, 0, 1),
                                                           (-1, 0, 1),
                                                           (1, 51, 1),
                                                           (1, 51, 1))
non_surge_staffed_thresholds_array = thresholds_generator((-1, 0, 1),
                                                          (-1, 0, 1),
                                                          (0.01, 0.51, 0.01),
                                                          (0.01, 0.51, 0.01))

if single_indicator_policies:
    for non_surge_hosp_adm_thresholds in non_surge_hosp_adm_thresholds_array:

        hosp_adm_thresholds = {"non_surge": (non_surge_hosp_adm_thresholds[2],
                                             non_surge_hosp_adm_thresholds[3],
                                             non_surge_hosp_adm_thresholds[4]),
                               "surge": (-1,
                                         -1,
                                         non_surge_hosp_adm_thresholds[3])}
        staffed_thresholds = {"non_surge": (np.inf,
                                            np.inf,
                                            np.inf),
                              "surge": (-1,
                                        -1,
                                        np.inf)}
        policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

    for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:

        staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                            non_surge_staffed_thresholds[3],
                                            non_surge_staffed_thresholds[4]),
                              "surge": (-1,
                                        -1,
                                        non_surge_staffed_thresholds[3])}

        hosp_adm_thresholds = {"non_surge": (np.inf,
                                             np.inf,
                                             np.inf),
                               "surge": (-1,
                                         -1,
                                         np.inf)}
        policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))
else:
    for non_surge_hosp_adm_thresholds in non_surge_hosp_adm_thresholds_array:

        hosp_adm_thresholds = {"non_surge": (non_surge_hosp_adm_thresholds[2],
                                             non_surge_hosp_adm_thresholds[3],
                                             non_surge_hosp_adm_thresholds[4]),
                               "surge": (-1,
                                         -1,
                                         non_surge_hosp_adm_thresholds[3])}

        for non_surge_staffed_thresholds in non_surge_staffed_thresholds_array:

            staffed_thresholds = {"non_surge": (non_surge_staffed_thresholds[2],
                                                non_surge_staffed_thresholds[3],
                                                non_surge_staffed_thresholds[4]),
                                  "surge": (-1,
                                            -1,
                                            non_surge_staffed_thresholds[3])}
            policies.append((case_threshold, hosp_adm_thresholds, staffed_thresholds))

breakpoint()

###############################################################################


def make_weighted_sum(df1, df2, df3, df4, weight1, weight2, weight3, weight4):
    weighted_df1 = df1 * weight1
    weighted_df2 = df2 * weight2
    weighted_df3 = df3 * weight3
    weighted_df4 = df4 * weight4
    summed_df = weighted_df1.add(weighted_df2, axis=1).add(weighted_df3, axis=1).add(weighted_df4, axis=1)
    return summed_df


def find_optimal_weighted_sum(summed_df):
    min_val = summed_df.mean().iloc[summed_df.mean().argmin()]
    min_ix = int(summed_df.mean().index[summed_df.mean().argmin()])
    return min_val, min_ix, policies[min_ix]


###############################################################################

# Read data from all 4 peaks and store into dictionaries

stage1_days_dict = {}
stage2_days_dict = {}
stage3_days_dict = {}
ICU_violation_patient_days_dict = {}
feasibility_dict = {}

for peak in np.arange(4):
    stage1_days_df = pd.read_csv(base_path / "singleindicator_aggregated" / ("aggregated_peak" + str(peak) + "_stage1_days.csv"),
                                 index_col=0)
    stage2_days_df = pd.read_csv(base_path / "singleindicator_aggregated" / ("aggregated_peak" + str(peak) + "_stage2_days.csv"),
                                 index_col=0)
    stage3_days_df = pd.read_csv(base_path / "singleindicator_aggregated" / ("aggregated_peak" + str(peak) + "_stage3_days.csv"),
                                 index_col=0)
    ICU_violation_patient_days_df = pd.read_csv(base_path / "singleindicator_aggregated" /
                                                ("aggregated_peak" + str(peak) + "_ICU_violation_patient_days.csv"),
                                                index_col=0)
    feasibility_df = pd.read_csv(base_path / "singleindicator_aggregated" / ("aggregated_peak" + str(peak) + "_feasibility.csv"),
                                 index_col=0)
    stage1_days_dict[str(peak)] = stage1_days_df
    stage2_days_dict[str(peak)] = stage2_days_df
    stage3_days_dict[str(peak)] = stage3_days_df
    ICU_violation_patient_days_dict[str(peak)] = ICU_violation_patient_days_df
    feasibility_dict[str(peak)] = feasibility_df
    # breakpoint()

###############################################################################

# Find unconstrained optimization optimal across peaks

for w in [i for i in range(1,11)] + [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, np.inf]:

    cost_dfs_per_peak = []
    ICU_violation_patient_days_per_peak = []
    feasibility_dfs_per_peak = []

    for peak in np.arange(4):

        stage1_days_df = stage1_days_dict[str(peak)]
        stage2_days_df = stage2_days_dict[str(peak)]
        stage3_days_df = stage3_days_dict[str(peak)]
        ICU_violation_patient_days_df = ICU_violation_patient_days_dict[str(peak)]

        feasibility_df = feasibility_dict[str(peak)]

        cost_df = make_weighted_sum(stage1_days_df, stage2_days_df, stage3_days_df, ICU_violation_patient_days_df,
                                    1, 10, 100, w)
        cost_dfs_per_peak.append(cost_df)

        ICU_violation_patient_days_per_peak.append(ICU_violation_patient_days_df)

        feasibility_dfs_per_peak.append(feasibility_df)
        # print(cost_df.mean().mean())

    cost_df_across_peaks = pd.concat(cost_dfs_per_peak)
    ICU_violation_patient_days_across_peaks = pd.concat(ICU_violation_patient_days_per_peak)
    optimal_val, optimal_ix, optimal_policy = find_optimal_weighted_sum(cost_df_across_peaks)

    feasibility_dfs_across_peaks = pd.concat(feasibility_dfs_per_peak)

    # 1/0 depending on if feasible across all peaks
    feasibility_all_peaks = feasibility_dfs_per_peak[0] + feasibility_dfs_per_peak[1] + \
        feasibility_dfs_per_peak[2] + feasibility_dfs_per_peak[3]
    feasibility_all_peaks = feasibility_all_peaks[feasibility_all_peaks >= 4]

    print(w, optimal_policy[1]["non_surge"], optimal_policy[2]["non_surge"],
          ICU_violation_patient_days_across_peaks.mean()[str(optimal_ix)],
          feasibility_all_peaks.sum()[str(optimal_ix)]/400)

breakpoint()

###############################################################################

# Find constrained optimization optimal per-peak

# index of df.mean() is a str holding the integer policy_id
# Once we identify the argmin of the mean cost, need to convert to actual policy
#   based on thresholds generator

# Comment from Dave:
# Weights (10^0, 10^1, and 10^2).
# Then you make sure the solution stays the same when you change the weights to 10^0, 10^2, 10^4.

performance_measures_strs = ["cost", "feasibility", "icu_violation_patient_days",
                             "stage1_days", "stage2_days", "stage3_days"]

min_cost_policies_per_peak = []
min_cost_per_peak = []

for peak in np.arange(4):

    feasibility_df = feasibility_dict[str(peak)]

    feasible_sols = feasibility_df.mean()[feasibility_df.mean() > 0.95].index

    stage1_days_df = stage1_days_dict[str(peak)]
    stage2_days_df = stage2_days_dict[str(peak)]
    stage3_days_df = stage3_days_dict[str(peak)]
    icu_violation_patient_days_df = ICU_violation_patient_days_dict[str(peak)]

    cost_df = make_weighted_sum(stage1_days_df, stage2_days_df, stage3_days_df, ICU_violation_patient_days_df,
                                1, 10, 100, 0)
    cost_df = cost_df[feasible_sols]

    print(feasibility_df.mean().max())
    print(len(feasible_sols))

    if len(feasible_sols) > 0:

        min_cost = cost_df.mean().iloc[cost_df.mean().argmin()]
        min_cost_ix = cost_df.mean().index[cost_df.mean().argmin()]
        min_cost_policy = policies[int(min_cost_ix)]

        min_cost_policies_per_peak.append(min_cost_policy)
        min_cost_per_peak.append(min_cost)

        print(min_cost_policy)

    else:
        min_cost_policies_per_peak.append(None)
        min_cost_per_peak.append(np.inf)

###############################################################################

# Find constrained optimization optimal across peaks

feasible_sols_per_peak = []
cost_dfs_per_peak = []

for peak in np.arange(4):
    feasibility_df = feasibility_dict[str(peak)]

    feasible_sols = feasibility_df.mean()[feasibility_df.mean() > 0.95].index
    feasible_sols_per_peak.append(feasible_sols)

    stage1_days_df = stage1_days_dict[str(peak)]
    stage2_days_df = stage2_days_dict[str(peak)]
    stage3_days_df = stage3_days_dict[str(peak)]
    icu_violation_patient_days_df = ICU_violation_patient_days_dict[str(peak)]

    cost_df = make_weighted_sum(stage1_days_df, stage2_days_df, stage3_days_df, ICU_violation_patient_days_df,
                                1, 10, 100, 0)
    cost_dfs_per_peak.append(cost_df)

cost_df_across_peaks = pd.concat(cost_dfs_per_peak)

feasible_sols_across_peak = list(set.intersection(*[set(x) for x in feasible_sols_per_peak]))

average_cost_across_peaks = cost_df_across_peaks[feasible_sols_across_peak].mean()

min_cost = average_cost_across_peaks[average_cost_across_peaks.argmin()]
min_cost_ix = average_cost_across_peaks.index[average_cost_across_peaks.argmin()]
min_cost_policy = policies[int(min_cost_ix)]

print("ACROSS PEAKS MIN COST POLICY")
print(min_cost_policy)

for peak in np.arange(4):
    feasibility_df = feasibility_dict[str(peak)]
    print(feasibility_df[min_cost_ix].mean())

for peak in np.arange(4):
    stage2_days_df = stage2_days_dict[str(peak)]
    if peak == 0:
        stage2_days_df_across_peaks = stage2_days_df
    else:
        stage2_days_df_across_peaks = stage2_days_df_across_peaks.add(stage2_days_df)

    print(stage2_days_df[min_cost_ix].quantile(0.5))

for peak in np.arange(4):
    stage3_days_df = stage3_days_dict[str(peak)]
    if peak == 0:
        stage3_days_df_across_peaks = stage3_days_df
    else:
        stage3_days_df_across_peaks = stage3_days_df_across_peaks.add(stage3_days_df)

    print(stage3_days_df[min_cost_ix].quantile(0.5))

breakpoint()

###############################################################################

# Get info on feasible policies

feasible_sols_across_peak = np.asarray(feasible_sols_across_peak, dtype=int)
np.asarray(policies)[feasible_sols_across_peak]

feasible_non_surge_hosp_adm_thresholds = []
feasible_non_surge_staffed_thresholds = []

for policy in np.asarray(policies)[feasible_sols_across_peak]:
    if policy[1]["non_surge"][0] < np.inf:
        feasible_non_surge_hosp_adm_thresholds.append(policy[1]["non_surge"])
    elif policy[2]["non_surge"][0] < np.inf:
        feasible_non_surge_staffed_thresholds.append(policy[2]["non_surge"])

feasible_non_surge_hosp_adm_thresholds.sort()
feasible_non_surge_staffed_thresholds.sort()

ix_increasing_cost_sorted = np.array(average_cost_across_peaks.sort_values().index, dtype=int)
for ix in ix_increasing_cost_sorted:
    if policies[ix][2]["non_surge"][0] != np.inf:
        print(policies[ix])

# non surge hosp adm threshold #1 must be <= 10
# non surge staffed threshold #1 must be <= .12

# best hosp admits only policy is also constrained optimal policy with weights 1/10/100
# (200, {'non_surge': (-1, 1, 17), 'surge': (-1, -1, 1)}, {'non_surge': (inf, inf, inf), 'surge': (-1, -1, inf)})

# best staffed beds only policy
# (200, {'non_surge': (inf, inf, inf), 'surge': (-1, -1, inf)}, {'non_surge': (-1, 0.01, 0.25), 'surge': (-1, -1, 0.01)})

breakpoint()

###############################################################################

for peak in np.arange(4):
    print(np.average(stage3_days_dict[str(peak)].mean()[feasible_sols_across_peak]))

for peak in np.arange(4):
    print(np.average(stage2_days_dict[str(peak)].mean()[feasible_sols_across_peak]))

breakpoint()

# Get variances of number of days in yellow and red

for peak in np.arange(4):
    plt.clf()
    plt.hist(stage2_days_dict[str(peak)].var()[feasible_sols_across_peak], color="gold")
    plt.title("Sample variance of days in yellow for feasible policies for peak " + str(peak+1))
    plt.xlabel("Sample variance of days in yellow stage")
    plt.ylabel("Number of policies")
    plt.savefig("stage2_variance_peak" + str(peak+1) + "_hist.png", dpi=1200)

for peak in np.arange(4):
    plt.clf()
    plt.hist(stage3_days_dict[str(peak)].var()[feasible_sols_across_peak], color="red")
    plt.title("Sample variance of days in red for feasible policies for peak " + str(peak+1))
    plt.xlabel("Sample variance of days in red stage")
    plt.ylabel("Number of policies")
    plt.savefig("stage3_variance_peak" + str(peak+1) + "_hist.png", dpi=1200)

breakpoint()

for peak in np.arange(4):
    plt.clf()
    plt.hist(stage3_days_dict[str(peak)].mean()[feasible_sols_across_peak], color="red")
    plt.title("Average days in red for feasible policies for peak " + str(peak+1))
    plt.xlabel("Average days in red stage")
    plt.ylabel("Number of policies")
    plt.savefig("stage3_peak" + str(peak+1) + "_hist.png", dpi=1200)

for peak in np.arange(4):
    plt.clf()
    plt.hist(stage2_days_dict[str(peak)].mean()[feasible_sols_across_peak], color="gold")
    plt.title("Average days in yellow for feasible policies for peak " + str(peak+1))
    plt.xlabel("Average days in yellow stage")
    plt.ylabel("Number of policies")
    plt.savefig("stage2_peak" + str(peak+1) + "_hist.png", dpi=1200)

plt.clf()
plt.hist(stage3_days_df_across_peaks.mean()[feasible_sols_across_peak], color="red")
plt.title("Average across-peak days in red for feasible policies")
plt.xlabel("Average across-peak days in red stage")
plt.ylabel("Number of policies")
plt.savefig("stage3_across_peaks_hist.png", dpi=1200)

plt.clf()
plt.hist(stage2_days_df_across_peaks.mean()[feasible_sols_across_peak], color="gold")
plt.title("Average across-peak days in yellow for feasible policies")
plt.xlabel("Average across-peak days in yellow stage")
plt.ylabel("Number of policies")
plt.savefig("stage2_across_peaks_hist.png", dpi=1200)

plt.clf()
plt.hist(average_cost_across_peaks[feasible_sols_across_peak])
plt.title("Average monetary cost for feasible policies")
plt.xlabel("Average monetary cost")
plt.ylabel("Number of policies")
plt.savefig("cost_across_peaks_hist.png", dpi=1200)

breakpoint()
