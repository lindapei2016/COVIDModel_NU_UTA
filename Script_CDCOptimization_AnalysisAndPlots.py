#######################################
############## IMPORTS ################
#######################################

import pandas as pd
import numpy as np
import time
import glob
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import scipy.stats
import seaborn as sns

from prettytable import PrettyTable

from pathlib import Path

base_path = Path(__file__).parent

#######################################
############## OPTIONS ################
#######################################

# Note -- have dataframe of 12172 policies with 300 reps
#   and also have dataframe of ~200-300ish policies with 2000 reps
#   that survived after KN and feasibility (multiple rounds)
# Sometimes want to use the first dataframe! Especially to
#   look at sensitivity and whatnot
# Will use the latter dataframe for tables of the optimal
#   solutions

aggregated_files_folder_name = "Results_09292023_NoCaseThreshold_12172Policies_BetterSubset_4000Reps"

full_df_files_prefix = "4000reps_"
# full_df_files_prefix = "300reps_"

need_plot_correlation = True
correlation_files_folder_name = "CDC_tier_history"

need_table_best_policies = False
need_table_cross_validation_peaks = False

need_plot_changing_coordinate = False
need_plot_pareto = False
need_plot_heatmap = False

#############################################
############## READ-IN FILES ################
#############################################

full_df_dict = {}
for peak in np.arange(3):
    full_df_dict[str(peak)] = pd.read_csv(base_path / aggregated_files_folder_name /
                                          (full_df_files_prefix + "full_df_peak" + str(peak) + ".csv"), index_col=0)
full_df_across_peaks = pd.read_csv(
    base_path / aggregated_files_folder_name / (full_df_files_prefix + "full_df_across_peaks.csv"),
    index_col=0)


# Dataframes must have following columns
# hosp1
# hosp2
# beds1
# beds2
# cost
# cost_stderror
# stage2
# stage2_stderror
# stage3
# stage3_stderror
# violation
# violation_stderror
# feasibility
# feasibility_stderror

# cost, stage2, stage3, violation, feasibility are sample means
# for full_df_across_peaks -- these are sample means across peaks
# cost depends on weights for blue/yellow/red days and for ICU violation penalty

breakpoint()

#################################################
############## CORRELATION PLOTS ################
#################################################

peaks_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]

# manually summed from austin_setup.json file
population_total = 128527 + 9350 + 327148 + 37451 + 915894 + 156209 + 249273 + 108196 + 132505 + 103763

# Maps peaks to lists of lists
tier_history_dict = {}
ICU_history_dict = {}
hospital_admits_history_dict = {}

if need_plot_correlation:
    for peak in np.arange(3):

        tier_history_filenames = glob.glob("**/*rank0_peak" + str(peak) + "*CDC*tier_history*")
        tier_history_dict[peak] = []
        for filename in tier_history_filenames:
            tier_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None))[peaks_start_times[peak]:])

        ICU_history_filenames = glob.glob("**/*rank0_peak" + str(peak) + "*CDC*ICU_history*")
        ICU_history_dict[peak] = []
        for filename in ICU_history_filenames:
            ICU_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None).rolling(7).mean())[peaks_start_times[peak]:])

        hospital_admits_history_filenames = glob.glob("**/*rank0_peak" + str(peak) + "*CDC*hospital_admits*")
        hospital_admits_history_dict[peak] = []
        for filename in hospital_admits_history_filenames:
            hospital_admits_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None).rolling(7).sum())[peaks_start_times[peak]:]*1e5/population_total)

        t = np.arange(peaks_end_times[peak] - peaks_start_times[peak])

        fig, ax1 = plt.subplots(layout="constrained")
        color = 'tab:cyan'
        ax1.set_xlabel('Days since start of peak')
        ax1.set_ylabel('Percent staffed beds occupied (7-day avg)')
        for rep in np.arange(100):
            if rep == 0:
                ax1.plot(t, ICU_history_dict[peak][rep] / 1100, color=color, alpha=0.2, label="Staffed beds")
            else:
                ax1.plot(t, ICU_history_dict[peak][rep]/1100, color=color, alpha=0.2)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.legend(loc=2)

        color = 'tab:purple'
        ax2.set_ylabel('Hospital admits (7-day avg per 100k)')  # we already handled the x-label with ax1
        for rep in np.arange(100):
            if rep == 0:
                ax2.plot(t, hospital_admits_history_dict[peak][rep], color=color, linestyle="--", alpha=0.3, label="Hospital admits")
            else:
                ax2.plot(t, hospital_admits_history_dict[peak][rep], color=color, linestyle="--", alpha=0.3)
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax1.set_title("Indicators under CDC policy for peak " + str(peak + 1))

        ax1.margins(x=0)
        ax2.margins(x=0)
        ax1.margins(y=0)
        ax2.margins(y=0)

        ax2.legend(loc = 1)

        plt.subplots_adjust(top=0.88)
        plt.savefig("indicators_correlation_peak" + str(peak + 1) + ".png", dpi=1200)
        plt.show()
        breakpoint()

#########################################
############## FUNCTIONS ################
#########################################

def lookup_policy(hosp1, hosp2, beds1, beds2, df):
    '''
    Looks up policy in df based on threshold values for two indicators
    Returns row of df where hosp1, hosp2, beds1, beds2 values
        match corresponding columns "hosp1", "hosp2", "beds1", "beds2"
    Compares absolute value of difference to epsilon because sometimes
        a literal match does not work due to formatting issues
    If policy is not in df, returns an empty dataframe

    Subroutine for get_df_of_perturbations()

    :param hosp1 [int] 1st threshold value for hospital admits
    :param hosp2 [int] 2nd threshold value for hospital admits
    :param beds1 [float in [0,1]] 1st threshold value for percent staffed beds
    :param beds2 [float in [0,1]] 2nd threshold value for percent staffed beds
    :param df -- dataframe with structure specified at beginning of document
        (with columns "hosp1", "hosp2", "beds1", "beds2", "cost", "cost_stderror",
        "stage2", "stage2_stderror", "stage3", "stage3_stderror", "violation",
        "violation_stderror", "feasibility", "feasibility_stderror")
    '''
    eps = 1e-6

    hosp1_condition = (np.abs(df["hosp1"] - hosp1) < eps) | (df["hosp1"] == hosp1)
    hosp2_condition = (np.abs(df["hosp2"] - hosp2) < eps) | (df["hosp2"] == hosp2)
    beds1_condition = (np.abs(df["beds1"] - beds1) < eps) | (df["beds1"] == beds1)
    beds2_condition = (np.abs(df["beds2"] - beds2) < eps) | (df["beds2"] == beds2)

    return df[(hosp1_condition) & (hosp2_condition) & (beds1_condition) & (beds2_condition)]


def get_df_of_perturbations(base_policy, threshold_to_perturb, threshold_values, df):
    '''
    Returns subset of df with threshold values fixed at base_policy except for
        threshold_to_perturb, which takes values in threshold_values
    Used to make figures for sensitivity analysis

    :base_policy [4-tuple] -- (hosp1, hosp2, beds1, beds2) corresponding to
        1st & 2nd thresholds for hospital admits and 1st & 2nd thresholds for
        percent staffed beds
    :threshold_to_perturb [str] -- must be "hosp1", "hosp2", "beds1", or "beds2"
        corresponding to column name in df corresponding to thresholds
    :threshold_values [list-like] -- values of threshold_to_perturb
    :param df [DataFrame] -- see parameter description in lookup_policy function
    '''

    hosp1 = base_policy[0]
    hosp2 = base_policy[1]
    beds1 = base_policy[2]
    beds2 = base_policy[3]

    list_of_dfs = []

    for val in threshold_values:
        if threshold_to_perturb == "hosp1":
            hosp1 = val
        elif threshold_to_perturb == "hosp2":
            hosp2 = val
        elif threshold_to_perturb == "beds1":
            beds1 = val
        elif threshold_to_perturb == "beds2":
            beds2 = val

        list_of_dfs.append(lookup_policy(hosp1, hosp2, beds1, beds2, df))

    df_of_perturbations = pd.concat(list_of_dfs)

    return df_of_perturbations


def make_plot_changing_coordinate(threshold_to_perturb,
                                  columns_to_plot,
                                  column_labels,
                                  colors_to_plot,
                                  df,
                                  fig_filename,
                                  xlabel,
                                  ylabel,
                                  title):
    '''
    Creates lineplot of multiple performance measures, as specified by columns_to_plot
    x-axis is value in threshold_to_perturb column
    y-axis is value of performance measure in columns_to_plot
    error bars are created using plus/minus 1 standard error

    To be used as intended, this function should act on output of get_df_of_perturbations(),
        a subset of the original full dataframe with all policies' data in which
        every value in threshold_to_perturb column is unique

    :param threshold_to_perturb [str] -- must be "hosp1", "hosp2", "beds1", or "beds2"
        corresponding to column name in df corresponding to thresholds
    :param columns_to_plot [list-like of strings] -- list-like structure with
        names of columns in df
    :param column_labels [list-like of strings] -- list-like structure with
        strings -- the ith string in column_labels will be the label of the ith
        column in columns_to_plot
    :param colors_to_plot [list-like of strings] -- must be same length as
        colors_to_plot -- the ith column in columns_to_plot is assigned the ith
        color in colors_to_plot -- format should be valid color name recognized
        by matplotlib or an RGB color code starting with #
    :param df [DataFrame] -- see parameter description in lookup_policy function --
        must have those same columns
    :param fig_filename [str] -- filename of figure, including filetype
    :param xlabel [str] -- label for x-axis
    :param ylabel [str] -- label for y-axis
    :param title [str] -- title for graph
    '''

    plt.clf()
    num_columns = len(columns_to_plot)
    for i in range(num_columns):
        sns.scatterplot(x=df[threshold_to_perturb],
                        y=df[columns_to_plot[i]],
                        label=column_labels[i],
                        color=colors_to_plot[i])
        plt.fill_between(df[threshold_to_perturb],
                         df[columns_to_plot[i]] - df[columns_to_plot[i] + "_stderror"],
                         df[columns_to_plot[i]] + df[columns_to_plot[i] + "_stderror"],
                         alpha=0.25, color=colors_to_plot[i])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fig_filename, dpi=1200)


######################################
############## TABLES ################
######################################

def add_policy_pretty_table(policy,
                            table,
                            first_field_value,
                            second_field_value):
    '''
    Note that rounding is hardcoded
    See :param table -- fills in rows -- column data corresponds to
        columns in order that is specified below

    :param policy [row of full_df DataFrame] -- where full_df DataFrame is of the type of
        pandas DataFrame loaded above -- see section on "READ-IN FILES" for
        column name specifications
    :param table [PrettyTable object] -- assuming field_names
        "", "", "Cost", "Yellow Days", "Red Days", "Violation", "Feasibility"
        (first two field names can vary, and their entries are specified below)
    :param first_field_value [str] -- entry for first_field_value (corresponding to first "" column)
    :param second_field_value [str] -- entry for second_field_value (corresponding to second "" column)
    '''

    table_contents = [first_field_value, second_field_value,
                      str(np.round(policy["cost"], -1)) + " (" + str(
                          np.round(policy["cost_stderror"], 0)) + ")",
                      str(np.round(policy["stage2"], 1)) + " (" + str(
                          np.round(policy["stage2_stderror"], 1)) + ")",
                      str(np.round(policy["stage3"], 1)) + " (" + str(
                          np.round(policy["stage3_stderror"], 1)) + ")",
                      str(np.round(policy["violation"], 2)) + " (" + str(
                          np.round(policy["violation_stderror"], 2)) + ")",
                      str(np.round(policy["feasibility"], 3)) + " (" + str(
                          np.round(policy["feasibility_stderror"], 3)) + ")"]
    table.add_row(table_contents)


if need_table_best_policies:
    table = PrettyTable()
    table.field_names = ["", "Thresholds", "Cost", "Yellow Days", "Red Days", "Violation", "Feasibility"]
    table.title = "Lowest-cost policies"
    for peak in np.arange(3):
        best_policy = full_df_dict[str(peak)].sort_values("cost").iloc[0]
        thresholds = "(" + str(np.round(best_policy["hosp1"], 0)) + \
                     ", " + str(np.round(best_policy["hosp2"], 0)) + \
                     ")  & (" + str(np.round(best_policy["beds1"], 1)) + \
                     ", " + str(np.round(best_policy["beds2"], 1)) + ")"
        add_policy_pretty_table(best_policy, table, "Peak " + str(peak + 1), thresholds)
    best_policy = full_df_across_peaks.sort_values("cost").iloc[0]
    thresholds = "(" + str(np.round(best_policy["hosp1"], 0)) + \
                 ", " + str(np.round(best_policy["hosp2"], 0)) + \
                 ")  & (" + str(np.round(best_policy["beds1"], 1)) + \
                 ", " + str(np.round(best_policy["beds2"], 1)) + ")"
    add_policy_pretty_table(best_policy, table, "Across-peaks", thresholds)
    print(table)

if need_table_cross_validation_peaks:
    table = PrettyTable()
    table.field_names = ["Thresholds", "Peaks", "Cost", "Yellow Days", "Red Days", "Violation", "Feasibility"]
    table.title = "Lowest-cost policy per-peak performance on other peaks"
    for peak in np.arange(3):
        best_policy = full_df_dict[str(peak)].sort_values("cost").iloc[0]
        thresholds = "(" + str(np.round(best_policy["hosp1"], 0)) + \
                     ", " + str(np.round(best_policy["hosp2"], 0)) + \
                     ")  & (" + str(np.round(best_policy["beds1"], 1)) + \
                     ", " + str(np.round(best_policy["beds2"], 1)) + ")"
        for peak_cv in np.arange(3):
            policy = lookup_policy(best_policy["hosp1"],
                                   best_policy["hosp2"],
                                   best_policy["beds1"],
                                   best_policy["beds2"],
                                   full_df_dict[str(peak_cv)])
            # Remove extraneous formatting that clogs the table
            policy = policy.iloc[0]
            print(policy)
            if peak_cv == 0:
                add_policy_pretty_table(policy, table, thresholds, "Peak " + str(peak_cv + 1))
            else:
                add_policy_pretty_table(policy, table, "", "Peak " + str(peak_cv + 1))
    print(table)

################################################
############## COORDINATE PLOTS ################
################################################

if need_plot_changing_coordinate:
    vary_threshold1_df = get_df_of_perturbations((3, 14, np.inf, np.inf), "hosp1", np.arange(13), full_df_across_peaks)

    title = "Hospital-admits-only policies with optimal 2nd threshold (at 14)"
    xlabel = "1st threshold"
    ylabel = "Number of days (average)"
    make_plot_changing_coordinate("hosp1",
                                  ("stage2", "stage3", "violation"),
                                  ("Stage 2 Days", "Stage 3 Days", "ICU Violation"),
                                  ("gold", "red", "purple"),
                                  vary_threshold1_df,
                                  "vary_threshold1_days.png",
                                  xlabel,
                                  ylabel,
                                  title)

    # Adding economic and health costs and corresponding standard errors to vary_threshold1_df
    vary_threshold1_df["econ_cost"] = 1 * vary_threshold1_df["stage1"] + 10 * vary_threshold1_df["stage2"] + 100 * \
                                      vary_threshold1_df["stage3"]
    vary_threshold1_df["econ_cost_stderror"] = 10 * vary_threshold1_df["stage2_stderror"] + 100 * vary_threshold1_df[
        "stage3_stderror"]
    vary_threshold1_df["health_cost"] = 330 * vary_threshold1_df["violation"]
    vary_threshold1_df["health_cost_stderror"] = 330 * vary_threshold1_df["violation_stderror"]

    title = "Hospital-admits-only policies with optimal 2nd threshold (at 14)"
    xlabel = "1st threshold"
    ylabel = "Cost (average)"
    make_plot_changing_coordinate("hosp1",
                                  ("econ_cost", "health_cost", "cost"),
                                  ("Economic cost", "Health cost", "Total cost"),
                                  ("#43C59E", "#48BEFF", "#14453D"),
                                  vary_threshold1_df,
                                  "vary_threshold1_costs.png",
                                  xlabel,
                                  ylabel,
                                  title)

    title = "Hospital-admits-only policies with optimal 1st threshold (at 3)"
    xlabel = "2nd threshold"
    ylabel = "Number of days (average)"

    vary_threshold2_df = get_df_of_perturbations((3, 20, np.inf, np.inf), "hosp2", np.arange(20), full_df_across_peaks)

    make_plot_changing_coordinate("hosp2",
                                  ("stage2", "stage3", "violation"),
                                  ("Stage 2 Days", "Stage 3 Days", "ICU Violation"),
                                  ("gold", "red", "purple"),
                                  vary_threshold2_df,
                                  "vary_threshold2_days.png",
                                  xlabel,
                                  ylabel,
                                  title)

    vary_threshold2_df["econ_cost"] = 1 * vary_threshold2_df["stage1"] + 10 * vary_threshold2_df["stage2"] + 100 * \
                                      vary_threshold2_df["stage3"]
    vary_threshold2_df["econ_cost_stderror"] = 10 * vary_threshold2_df["stage2_stderror"] + 100 * vary_threshold2_df[
        "stage3_stderror"]
    vary_threshold2_df["health_cost"] = 330 * vary_threshold2_df["violation"]
    vary_threshold2_df["health_cost_stderror"] = 330 * vary_threshold2_df["violation_stderror"]

    title = "Hospital-admits-only policies with optimal 1st threshold (at 3)"
    xlabel = "2nd threshold"
    ylabel = "Number of days (average)"
    make_plot_changing_coordinate("hosp2",
                                  ("econ_cost", "health_cost", "cost"),
                                  ("Economic cost", "Health cost", "Total cost"),
                                  ("#43C59E", "#48BEFF", "#14453D"),
                                  vary_threshold2_df,
                                  "vary_threshold2_costs.png",
                                  xlabel,
                                  ylabel,
                                  title)

############################################
############## PARETO PLOTS ################
############################################

if need_plot_pareto:
    # Note the clunky if/elif is hardcoded for the case of overlapping points
    #   based on 1000 reps -- but when not in that case, does not affect the code
    # TODO: convert this into a function...

    full_df_across_peaks_copy = full_df_across_peaks.copy(deep=True)
    full_df_across_peaks_copy.drop(columns="cost", inplace=True)
    # full_df_feasible_across_peaks = full_df_across_peaks_copy[full_df_across_peaks_copy["feasibility"] > 0.95]

    econ_costs = []
    health_costs = []
    feasibilities = []
    yellow_days = []
    red_days = []

    actual_cost_argmin = full_df_across_peaks.sort_values("cost").index[0]
    cost_argmins = []

    for w in np.concatenate((np.arange(1, 1001), np.arange(2, 11) * 1000)):
        full_df_across_peaks_copy["cost"] = 10 * full_df_across_peaks_copy["stage2"] + \
                                            100 * full_df_across_peaks_copy["stage3"] + \
                                            w * full_df_across_peaks_copy["violation"]
        cost_argmin = full_df_across_peaks_copy.sort_values("cost").index[0]
        econ_cost = 1 * full_df_across_peaks_copy["stage1"].iloc[cost_argmin] + \
                    10 * full_df_across_peaks_copy["stage2"].iloc[cost_argmin] + \
                    100 * full_df_across_peaks_copy["stage3"].iloc[cost_argmin]
        health_cost = full_df_across_peaks_copy["violation"].iloc[cost_argmin]

        econ_costs.append(econ_cost)
        health_costs.append(health_cost)
        feasibilities.append(full_df_across_peaks_copy["feasibility"].iloc[cost_argmin])
        yellow_days.append(full_df_across_peaks_copy["stage2"].iloc[cost_argmin])
        red_days.append(full_df_across_peaks_copy["stage3"].iloc[cost_argmin])

        cost_argmins.append(cost_argmin)

    # FYI based on analyzing cost_argmins -- at ICU penalty 78 onwards,
    #   optimal policy coincides with our across-peak optimal policy under
    #   ICU penalty w = 330 (which was chosen based on 300 replications
    #   and 12172 policies, including many infeasible policies)
    # See note in slidedeck that if we were to compute optimal ICU penalty
    #   based on smaller set of policies (selected based on low cost and
    #   feasibility), the penalty would be smaller -- and that intuitively makes sense

    # values of ICU penalty such that optimal solution changes
    #   and econ_costs changes
    changepoints_weights = []
    changepoints_econ_costs = []
    changepoints_health_costs = []
    changepoints_feasibilities = []
    changepoints_yellow_days = []
    changepoints_red_days = []
    for unique_econ_cost in set(econ_costs):
        changepoints_weights.append(int(np.concatenate((np.arange(1, 1001), np.arange(2, 11) * 1000))[np.argwhere(econ_costs == unique_econ_cost)[0]]))
        changepoints_econ_costs.append(econ_costs[int(np.argwhere(econ_costs == unique_econ_cost)[0])])
        changepoints_health_costs.append(health_costs[int(np.argwhere(econ_costs == unique_econ_cost)[0])])
        changepoints_feasibilities.append(feasibilities[int(np.argwhere(econ_costs == unique_econ_cost)[0])])
        changepoints_yellow_days.append(yellow_days[int(np.argwhere(econ_costs == unique_econ_cost)[0])])
        changepoints_red_days.append(red_days[int(np.argwhere(econ_costs == unique_econ_cost)[0])])

    plt.clf()
    plt.scatter(np.array(econ_costs), np.array(health_costs))
    for i in range(len(changepoints_weights)):
        if changepoints_weights[i] == 11:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_health_costs[i]),
                         xycoords='data',
                         xytext=(-12, 2), textcoords='offset points')
        elif changepoints_weights[i] == 17:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_health_costs[i]),
                         xycoords='data',
                         xytext=(3, 3), textcoords='offset points')
        else:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_health_costs[i]),
                         xycoords='data',
                         xytext=(3, 3), textcoords='offset points')
    plt.title("Optimal economic costs and ICU violations under various ICU penalties")
    plt.xlabel("Economic cost")
    sns.despine()
    plt.ylabel("ICU violation: total patient days above ICU capacity")
    plt.savefig("paretoplot.png", dpi=1200)
    plt.show()

    plt.clf()
    plt.scatter(np.array(econ_costs), np.array(feasibilities))
    for i in range(len(changepoints_weights)):
        if changepoints_weights[i] == 11:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_feasibilities[i]),
                         xycoords='data',
                         xytext=(-12, 2), textcoords='offset points')
        elif changepoints_weights[i] == 17:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_feasibilities[i]),
                         xycoords='data',
                         xytext=(3, 3), textcoords='offset points')
        else:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_feasibilities[i]),
                         xycoords='data',
                         xytext=(3, 3), textcoords='offset points')
    plt.title("Optimal economic costs and feasibilities under various ICU penalties")
    plt.xlabel("Economic cost")
    plt.ylabel("Feasibility on all 3 peaks")
    sns.despine()
    plt.savefig("paretoplot_feasibilities.png", dpi=1200)
    plt.show()

    plt.clf()
    plt.scatter(np.array(econ_costs), np.array(yellow_days), color="gold", label="Yellow Stage Days")
    plt.scatter(np.array(econ_costs), np.array(red_days), color="red", label="Red Stage Days")
    for i in range(len(changepoints_weights)):
        if changepoints_weights[i] == 11:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_red_days[i]),
                         xycoords='data',
                         xytext=(-12, 2), textcoords='offset points')
        elif changepoints_weights[i] == 17:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_red_days[i]),
                         xycoords='data',
                         xytext=(3, 3), textcoords='offset points')
        else:
            plt.annotate(str(changepoints_weights[i]),
                         xy=(changepoints_econ_costs[i], changepoints_red_days[i]),
                         xycoords='data',
                         xytext=(3, 3), textcoords='offset points')
    plt.title("Optimal economic costs and days per stage under various ICU penalties")
    plt.xlabel("Economic cost")
    plt.ylabel("Average days across peaks")
    plt.legend()
    sns.despine()
    plt.savefig("paretoplot_days.png", dpi=1200)
    plt.show()

############################################
############## HEATMAP PLOTS ###############
############################################

if need_plot_heatmap:

    breakpoint()

    # Heat map for 2652 single-indicator policies
    # First 1326 policies are hosp adms only
    # Second 1326 policies are staffed beds only
    # First threshold: 51 options, second threshold: 26 options

    # Note: showing heatmap for two-indicator policies is not super interpretable
    # Note: showing full heatmap for whole range of single-indicator policies
    #   is not high resolution enough to distinguish policies!
    # --> Therefore we show partial heatmap with smaller subset of single-indicator
    #   policies

    # To hack this for-loop, I made peak 3 the across-peaks result (we have no peak 3 anymore)
    for peak in np.arange(4):
        # for peak in [3]:

        if peak < 3:
            base_df = full_df_dict[str(peak)]
            title_suffix = "peak " + str(peak + 1)
        else:
            base_df = full_df_across_peaks
            title_suffix = "across peaks"

        hosp_df = base_df[:1326]

        # Need to reshape into upper triangular array!
        base_array = np.full((51, 51), 0)
        upper_triangular_indices = np.triu_indices(51, m=51)

        hosp_df.sort_values(by=["hosp1", "hosp2"], inplace=True)
        cost = hosp_df["cost"]

        for i in np.arange(1326):
            base_array[upper_triangular_indices[0][i]][upper_triangular_indices[1][i]] = cost.loc[i]

        smaller_array = base_array[:20, :20]
        mask_array = np.full((20, 20), False)
        mask_array[np.tril_indices(20, m=20)] = True

        # Note: we really do need to filter out bad policies
        #   with cost greater than 12000 (corresponding to
        #   all 120 days in red, ignoring ICU penalty) --
        #   otherwise the resolution / color gradient is very
        #   hard to distinguish
        for i in range(20):
            for j in range(20):
                if smaller_array[i][j] > 12000:
                    mask_array[i][j] = True

        labels = np.full((20, 20), "", dtype="str")
        labels[smaller_array == np.min(smaller_array[smaller_array > 0])] = "*"

        plt.clf()
        s = sns.heatmap(smaller_array,
                        xticklabels=np.arange(20),
                        yticklabels=np.arange(20),
                        cmap="magma_r",
                        mask=mask_array,
                        annot=labels,
                        fmt="").set(title="Cost of hospital-admits-only policies, " + title_suffix,
                                    xlabel="Threshold 2 (yellow-red)",
                                    ylabel="Threshold 1 (blue-yellow)")
        plt.savefig("hosp_only_heatmap_" + str(peak) + ".png", dpi=1200)
        plt.show()

        # Staffed beds only single-indicator policies

        beds_df = base_df[1326:2652]

        # Need to reshape into upper triangular array!
        base_array = np.full((51, 51), 0)
        upper_triangular_indices = np.triu_indices(51, m=51)

        beds_df.sort_values(by=["beds1", "beds2"], inplace=True)
        cost = beds_df["cost"]

        for i in np.arange(1326):
            base_array[upper_triangular_indices[0][i]][upper_triangular_indices[1][i]] = cost.loc[i + 1326]

        matplotlib.cm.get_cmap("magma_r").set_bad("white")

        smaller_array = base_array[:20, :50]
        mask_array = np.full((20, 50), False)
        mask_array[np.tril_indices(20, m=50)] = True

        for i in range(20):
            for j in range(50):
                if smaller_array[i][j] > 12000:
                    mask_array[i][j] = True

        labels = np.full((20, 50), "", dtype="str")
        labels[smaller_array == np.min(smaller_array[smaller_array > 0])] = "*"

        smaller_array = pd.DataFrame(smaller_array)
        smaller_array.index = np.arange(20)
        smaller_array.columns = np.arange(50)
        for col in smaller_array.columns:
            if col > 35:
                smaller_array.drop(columns=col, inplace=True)

        mask_array = mask_array[:20, :36]
        labels = labels[:20, :36]

        plt.clf()
        ax = sns.heatmap(smaller_array,
                         # xticklabels=0.5,
                         # yticklabels=np.arange(20) / 100,
                         cmap="magma_r",
                         mask=mask_array,
                         annot=labels,
                         fmt="").set(title="Cost of staffed-beds-only policies, " + title_suffix,
                                     xlabel="Threshold 2 (yellow-red) -- percentages",
                                     ylabel="Threshold 1 (blue-yellow) -- percentages")
        plt.savefig("beds_only_heatmap_" + str(peak) + ".png", dpi=1200)
        plt.show()

        # breakpoint()

        # Can add vmin and vmax to heatmap to specify min/max values
