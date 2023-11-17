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

# aggregated_files_folder_name = "Results_09292023_NoCaseThreshold_12172Policies_BetterSubset_4000Reps"
aggregated_files_folder_name = "3000I"
# aggregated_files_folder_name = "coordinate"
# aggregated_files_folder_name = "heatmaps"

aggregated_files_prefix = "8000_aggregated_"

full_df_files_prefix = "8000reps_"
# full_df_files_prefix = "coordinate_"

optimal_policy_files_folder_name = "optimal"
need_table_optimal_policy = False
need_stage3_days_plot_optimal_policy = False
need_tiers_plot_optimal_policy = False

correlation_files_folder_name = "CDC"
need_plot_correlation = False

need_table_best_policies = True
need_table_cross_validation_peaks = True
need_plot_regret = False

need_plot_changing_coordinate = False
need_plot_pareto = False
need_plot_heatmap = False

num_reps = 8000

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

#######################################################
############## OPTIMAL POLICY ANALYSIS ################
#######################################################

# Also want Dali plot?

peaks_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]

peaks_total_hosp_beds = [3026, 3791, 3841, 3537]

# manually summed from austin_setup.json file
population_total = 128527 + 9350 + 327148 + 37451 + 915894 + 156209 + 249273 + 108196 + 132505 + 103763

# Maps peaks to lists of lists
tier_history_dict = {}
ICU_history_dict = {}
IH_history_dict = {}
ToIHT_history_dict = {}

first_stage3_day_dict = {}
last_stage3_day_dict = {}

if need_table_optimal_policy:
    for peak in np.arange(3):
        first_stage3_day_list = []
        last_stage3_day_list = []
        tier_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*optimalpolicyacrosspeaks*tier_history*")
        tier_history_dict[peak] = []
        for filename in tier_history_filenames:
            tier_history = np.asarray(pd.read_csv(filename, header=None))[peaks_start_times[peak]:]
            tier_history_dict[peak].append(tier_history)
            first_stage3_day_list.append(np.argwhere(tier_history == 2)[0][0])
            last_stage3_day_list.append(np.argwhere(tier_history == 2)[-1][0])
        first_stage3_day_dict[peak] = first_stage3_day_list
        last_stage3_day_dict[peak] = last_stage3_day_list
        print(np.average(np.array(first_stage3_day_list)))
        print(np.average(np.array(last_stage3_day_list)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Get a table on...
    # Average start time of lockdown
    # Average end time of lockdown
    # Average number of lockdowns
    # Average length of longest lockdowns

    num_changes_dict = {}
    longest_lockdown_dict = {}

    for peak in np.arange(3):
        num_changes_list = []
        longest_lockdown_list = []
        for tier_history in tier_history_dict[peak]:
            num_changes_counter = 0
            longest_lockdown_length = 0
            current_lockdown_length = 0
            for i in range(len(tier_history)):
                if tier_history[i] == 2:
                    current_lockdown_length += 1
                else:
                    if current_lockdown_length > longest_lockdown_length:
                        longest_lockdown_length = current_lockdown_length
                    current_lockdown_length = 0
                if i == 0:
                    continue
                else:
                    if tier_history[i] != tier_history[i - 1]:
                        num_changes_counter += 1
            num_changes_list.append(num_changes_counter)
            longest_lockdown_list.append(longest_lockdown_length)
        num_changes_dict[peak] = np.array(num_changes_list)
        print(np.average(np.array(num_changes_list)))
        longest_lockdown_dict[peak] = np.array(longest_lockdown_list)
        print(np.average(np.array(longest_lockdown_list)))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    num_reps = len(num_changes_list)

    table = PrettyTable()
    table.field_names = ["", "First Red Day", "Last Red Day", "Number of Stage Changes", "Days of Longest Lockdown"]
    table.title = "Profile statistics of across-peak optimal policy"
    for peak in np.arange(3):
        first_stage3_day_list = np.array(first_stage3_day_dict[peak])
        last_stage3_day_list = np.array(last_stage3_day_dict[peak])
        num_changes_list = np.array(num_changes_dict[peak])
        longest_lockdown_list = np.array(longest_lockdown_dict[peak])
        table.add_row(("Peak " + str(peak + 1),
                       str(np.round(np.average(first_stage3_day_list), 1)) + " (" + str(
                           np.round(np.sqrt(np.var(first_stage3_day_list) / num_reps), 1)) + ")",
                       str(np.round(np.average(last_stage3_day_list), 1)) + " (" + str(
                           np.round(np.sqrt(np.var(last_stage3_day_list) / num_reps), 1)) + ")",
                       str(np.round(np.average(num_changes_list), 1)) + " (" + str(
                           np.round(np.sqrt(np.var(num_changes_list) / num_reps), 1)) + ")",
                       str(np.round(np.average(longest_lockdown_list), 1)) + " (" + str(
                           np.round(np.sqrt(np.var(longest_lockdown_list) / num_reps), 1)) + ")"))
    print(table)

# breakpoint()

if need_stage3_days_plot_optimal_policy:
    for peak in np.arange(3):

        ICU_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*optimalpolicyacrosspeaks*ICU_history*")
        ICU_history_dict[peak] = []
        for filename in ICU_history_filenames:
            ICU_history_dict[peak].append(
                np.asarray(pd.read_csv(filename, header=None).rolling(7).mean())[peaks_start_times[peak]:])

        IH_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*optimalpolicyacrosspeaks*IH_history*")
        IH_history_dict[peak] = []
        for filename in IH_history_filenames:
            IH_history_dict[peak].append(
                np.asarray(pd.read_csv(filename, header=None).rolling(7).mean())[peaks_start_times[peak]:])

        ToIHT_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*optimalpolicyacrosspeaks*ToIHT_history*")
        ToIHT_history_dict[peak] = []
        for filename in ToIHT_history_filenames:
            ToIHT_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None).rolling(7).sum())[
                                            peaks_start_times[peak]:] * 1e5 / population_total)

        t = np.arange(peaks_end_times[peak] - peaks_start_times[peak])

        fig, ax1 = plt.subplots(layout="constrained")
        color = 'tab:cyan'
        ax1.set_xlabel('Days since start of peak')
        ax1.set_ylabel('Percent staffed beds occupied (7-day avg)')
        for rep in np.arange(50):
            if rep == 0:
                ax1.plot(t, (ICU_history_dict[peak][rep] + IH_history_dict[peak][rep]) / peaks_total_hosp_beds[peak],
                         color=color, alpha=0.2, label="Staffed beds")
            else:
                ax1.plot(t, (ICU_history_dict[peak][rep] + IH_history_dict[peak][rep]) / peaks_total_hosp_beds[peak],
                         color=color, alpha=0.2)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.legend(loc=2)

        color = 'tab:purple'
        ax2.set_ylabel('Hospital admits (7-day sum per 100k)')  # we already handled the x-label with ax1
        for rep in np.arange(50):
            if rep == 0:
                ax2.plot(t, ToIHT_history_dict[peak][rep], color=color, linestyle="--", alpha=0.3,
                         label="Hospital admits")
            else:
                ax2.plot(t, ToIHT_history_dict[peak][rep], color=color, linestyle="--", alpha=0.3)
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax1.set_title("Indicators under across-peak optimal policy for peak " + str(peak + 1))

        ax1.margins(x=0)
        ax2.margins(x=0)
        ax1.margins(y=0)
        ax2.margins(y=0)

        ax2.legend(loc=1)

        plt.subplots_adjust(top=0.88)
        plt.savefig("indicators_optimal_correlation_peak" + str(peak + 1) + ".png", dpi=1200)
        plt.show()
        # breakpoint()

if need_tiers_plot_optimal_policy:
    for peak in np.arange(3):

        tier_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*optimalpolicyacrosspeaks*tier_history*")
        tier_history_dict[peak] = []
        for filename in tier_history_filenames[:50]:
            tier_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None))[peaks_start_times[peak]+1:])

        t = np.arange(peaks_end_times[peak] - peaks_start_times[peak])

        tier_history_df = pd.DataFrame(np.squeeze(np.asarray(tier_history_dict[peak])))

        tier_history_blue_days = (tier_history_df == 0).sum()
        tier_history_yellow_days = (tier_history_df == 1).sum()
        tier_history_red_days = (tier_history_df == 2).sum()

        blue_days_percent = np.array(tier_history_blue_days)/50
        yellow_days_percent = np.array(tier_history_yellow_days) / 50
        red_days_percent = np.array(tier_history_red_days) / 50

        # There were no days in blue
        plt.stackplot(np.arange(len(blue_days_percent)),
                      red_days_percent,
                      yellow_days_percent,
                      colors=["red", "gold"],
                      labels=["Days in Red Stage", "Days in Yellow Stage"])
        plt.ylabel("Proportion of sample paths")
        plt.xlabel("Days since start of peak")
        plt.title("Across-Peak Optimal Days in Yellow and Red Stages, Peak " + str(peak + 1) + "")
        plt.legend()
        plt.margins(0, 0)
        plt.savefig("dali_plot_across_peak_optimal_peak" + str(peak+1) + ".png", dpi=1200)
        plt.clf()

        # Plot the distribution of clarity ratings, conditional on carat
        # sns.displot(data=tier_history_df, x="day", hue="tier", kind="kde", height=6, multiple="fill", clip=(0, None), palette="ch:rot=-.25,hue=1,light=.75")

#################################################
############## CORRELATION PLOTS ################
#################################################

# Use IH + ICU for numerator of staffed beds (7 day avg)
# Use ToIHT for hospital admissions! (7 day total)

# For some reason I only have 50 reps, not sure why...
#   but plots with 50 reps are fine too

peaks_start_times = [93, 276, 502, 641]
peaks_end_times = [215, 397, 625, 762]

peaks_total_hosp_beds = [3026, 3791, 3841, 3537]

# manually summed from austin_setup.json file
population_total = 128527 + 9350 + 327148 + 37451 + 915894 + 156209 + 249273 + 108196 + 132505 + 103763

# Maps peaks to lists of lists
tier_history_dict = {}
ICU_history_dict = {}
IH_history_dict = {}
ToIHT_history_dict = {}

if need_plot_correlation:
    for peak in np.arange(3):

        # breakpoint()

        tier_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*CDCimplementation*tier_history*")
        tier_history_dict[peak] = []
        for filename in tier_history_filenames:
            tier_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None))[peaks_start_times[peak]:])

        ICU_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*CDCimplementation*ICU_history*")
        ICU_history_dict[peak] = []
        for filename in ICU_history_filenames:
            ICU_history_dict[peak].append(
                np.asarray(pd.read_csv(filename, header=None).rolling(7).mean())[peaks_start_times[peak]:])

        IH_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*CDCimplementation*IH_history*")
        IH_history_dict[peak] = []
        for filename in IH_history_filenames:
            IH_history_dict[peak].append(
                np.asarray(pd.read_csv(filename, header=None).rolling(7).mean())[peaks_start_times[peak]:])

        ToIHT_history_filenames = glob.glob("**/*rank*_peak" + str(peak) + "*CDCimplementation*ToIHT_history*")
        ToIHT_history_dict[peak] = []
        for filename in ToIHT_history_filenames:
            ToIHT_history_dict[peak].append(np.asarray(pd.read_csv(filename, header=None).rolling(7).sum())[
                                            peaks_start_times[peak]:] * 1e5 / population_total)

        t = np.arange(peaks_end_times[peak] - peaks_start_times[peak])

        fig, ax1 = plt.subplots(layout="constrained")
        color = 'tab:cyan'
        ax1.set_xlabel('Days since start of peak')
        ax1.set_ylabel('Percent staffed beds occupied (7-day avg)')
        for rep in np.arange(50):
            if rep == 0:
                ax1.plot(t, (ICU_history_dict[peak][rep] + IH_history_dict[peak][rep]) / peaks_total_hosp_beds[peak],
                         color=color, alpha=0.2, label="Staffed beds")
            else:
                ax1.plot(t, (ICU_history_dict[peak][rep] + IH_history_dict[peak][rep]) / peaks_total_hosp_beds[peak],
                         color=color, alpha=0.2)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.legend(loc=2)

        color = 'tab:purple'
        ax2.set_ylabel('Hospital admits (7-day sum per 100k)')  # we already handled the x-label with ax1
        for rep in np.arange(50):
            if rep == 0:
                ax2.plot(t, ToIHT_history_dict[peak][rep], color=color, linestyle="--", alpha=0.3,
                         label="Hospital admits")
            else:
                ax2.plot(t, ToIHT_history_dict[peak][rep], color=color, linestyle="--", alpha=0.3)
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax1.set_title("Indicators under CDC policy for peak " + str(peak + 1))

        ax1.margins(x=0)
        ax2.margins(x=0)
        ax1.margins(y=0)
        ax2.margins(y=0)

        ax2.legend(loc=1)

        plt.subplots_adjust(top=0.88)
        plt.savefig("indicators_correlation_peak" + str(peak + 1) + ".png", dpi=1200)
        plt.show()
        # breakpoint()


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
                                  xticks,
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
    :param xticks [list] -- list for x-axis ticks
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
    plt.xticks(xticks)
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
            # print(policy)
            if peak_cv == 0:
                add_policy_pretty_table(policy, table, thresholds, "Peak " + str(peak_cv + 1))
            else:
                add_policy_pretty_table(policy, table, "", "Peak " + str(peak_cv + 1))
    # And for across-peak optimal
    # This is redundant and could also be considered peak 3 and combined with
    #   loop above, but I'm just doing it this way...
    best_policy = full_df_across_peaks.sort_values("cost").iloc[0]
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
        policy = policy.iloc[0]
        if peak_cv == 0:
            add_policy_pretty_table(policy, table, thresholds, "Peak " + str(peak_cv + 1))
        else:
            add_policy_pretty_table(policy, table, "", "Peak " + str(peak_cv + 1))
    print(table)

# Note I'm hardcoding this -- across-peak best policy has ID 25 (in original policies generated)
if need_plot_regret:

    colors = ["palevioletred", "darkviolet", "cornflowerblue"]

    for peak in np.arange(3):
        stage2_days = pd.read_csv(base_path /
                                  aggregated_files_folder_name /
                                  (aggregated_files_prefix + "aggregated_peak" + str(peak) + "_stage2_days.csv"),
                                  index_col=0)["25"]
        stage3_days = pd.read_csv(base_path /
                                  aggregated_files_folder_name /
                                  (aggregated_files_prefix + "aggregated_peak" + str(peak) + "_stage3_days.csv"),
                                  index_col=0)["25"]
        ICU_violation_patient_days = pd.read_csv(base_path /
                                                 aggregated_files_folder_name /
                                                 (aggregated_files_prefix + "aggregated_peak" + str(
                                                     peak) + "_ICU_violation_patient_days.csv"),
                                                 index_col=0)["25"]

        # sns.displot(stage2_days, kind="kde", bw_adjust=0.5, color="gold", label="Yellow Stage Days")
        # sns.displot(stage3_days, kind="kde", bw_adjust=0.5)
        # density_plot.axes[0][0].axvline(stage2_days.mean(), c="k", ls="--")

        fig = sns.kdeplot(stage3_days, fill=True, alpha=0.2, color=colors[peak], label="Peak " + str(peak+1), bw_adjust=0.6)

    plt.legend()
    plt.xlabel("Days in Red Stage")
    plt.title("Distribution of Days in Red Stage of Across-Peak Optimal Policy")
    plt.savefig("density_across_peak_optimal.png", dpi=1200)
    plt.clf()

################################################
############## COORDINATE PLOTS ################
################################################

# We accidentally simulated (1, 14, np.inf, np.inf) twice in the way we generated policies, oops
# One of them seemed to get bad random variates and was pretty noisy and had a slightly higher cost
#   (although distribution of days in each stage was basically the same)
# Dropped the higher cost one

if need_plot_changing_coordinate:
    full_df_across_peaks.drop(index=1, inplace=True)

    vary_threshold1_df = get_df_of_perturbations((1, 14, np.inf, np.inf), "hosp1", np.arange(1, 15),
                                                 full_df_across_peaks)

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
                                  np.arange(1, 15),
                                  title)

    # breakpoint()

    # Adding economic and health costs and corresponding standard errors to vary_threshold1_df
    vary_threshold1_df["econ_cost"] = 1 * vary_threshold1_df["stage1"] + 10 * vary_threshold1_df["stage2"] + 100 * \
                                      vary_threshold1_df["stage3"]
    vary_threshold1_df["econ_cost_stderror"] = 10 * vary_threshold1_df["stage2_stderror"] + 100 * vary_threshold1_df[
        "stage3_stderror"]
    vary_threshold1_df["health_cost"] = 503 * vary_threshold1_df["violation"]
    vary_threshold1_df["health_cost_stderror"] = 503 * vary_threshold1_df["violation_stderror"]

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
                                  np.arange(1, 15),
                                  title)

    title = "Hospital-admits-only policies with optimal 1st threshold (at 1)"
    xlabel = "2nd threshold"
    ylabel = "Number of days (average)"

    vary_threshold2_df = get_df_of_perturbations((1, 14, np.inf, np.inf), "hosp2", np.arange(1, 21),
                                                 full_df_across_peaks)

    make_plot_changing_coordinate("hosp2",
                                  ("stage2", "stage3", "violation"),
                                  ("Stage 2 Days", "Stage 3 Days", "ICU Violation"),
                                  ("gold", "red", "purple"),
                                  vary_threshold2_df,
                                  "vary_threshold2_days.png",
                                  xlabel,
                                  ylabel,
                                  np.arange(1, 21),
                                  title)

    vary_threshold2_df["econ_cost"] = 1 * vary_threshold2_df["stage1"] + 10 * vary_threshold2_df["stage2"] + 100 * \
                                      vary_threshold2_df["stage3"]
    vary_threshold2_df["econ_cost_stderror"] = 10 * vary_threshold2_df["stage2_stderror"] + 100 * vary_threshold2_df[
        "stage3_stderror"]
    vary_threshold2_df["health_cost"] = 503 * vary_threshold2_df["violation"]
    vary_threshold2_df["health_cost_stderror"] = 503 * vary_threshold2_df["violation_stderror"]

    title = "Hospital-admits-only policies with optimal 1st threshold (at 1)"
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
                                  np.arange(1, 21),
                                  title)

    # breakpoint()

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
        full_df_across_peaks_copy["cost"] = 1 * full_df_across_peaks_copy["stage1"] + \
                                            10 * full_df_across_peaks_copy["stage2"] + \
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
        changepoints_weights.append(int(np.concatenate((np.arange(1, 1001), np.arange(2, 11) * 1000))[
                                            np.argwhere(econ_costs == unique_econ_cost)[0]]))
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

# Maybe add "X" to infeasible policies?
#

if need_plot_heatmap:

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
        # for peak in [4]:
        # for peak in [3]:

        if peak < 3:
            base_df = full_df_dict[str(peak)]
            title_suffix = "peak " + str(peak + 1)
        else:
            base_df = full_df_across_peaks
            title_suffix = "across peaks"

        hosp_df = base_df[:161]

        base_array = np.full((21, 21), 0)
        mask_array = np.full((21, 21), True)
        cost = hosp_df["cost"]

        labels = np.full((21, 21), "", dtype="str")
        lowest_cost_policy_df = hosp_df.sort_values("cost").iloc[0]
        labels[int(lowest_cost_policy_df["hosp1"])][int(lowest_cost_policy_df["hosp2"])] = "*"

        for i in np.arange(161):
            base_array[int(hosp_df.iloc[i]["hosp1"])][int(hosp_df.iloc[i]["hosp2"])] = cost.iloc[i]
            if cost.iloc[i] < 30000:
                mask_array[int(hosp_df.iloc[i]["hosp1"])][int(hosp_df.iloc[i]["hosp2"])] = False
            if hosp_df["feasibility"].iloc[i] < 0.95:
                labels[int(hosp_df.iloc[i]["hosp1"])][int(hosp_df.iloc[i]["hosp2"])] = "x"

        plt.clf()
        s = sns.heatmap(base_array,
                        xticklabels=np.arange(21),
                        yticklabels=np.arange(21),
                        cmap="magma_r",
                        mask=mask_array,
                        annot=labels,
                        annot_kws={'fontsize': 12, 'color': 'k', 'alpha': 1,
                                   'rotation': 'vertical', 'verticalalignment': 'center'},
                        fmt="").set(title="Cost of hospital-admits-only policies, " + title_suffix,
                                    xlabel="Threshold 2 (yellow-red)",
                                    ylabel="Threshold 1 (blue-yellow)")
        plt.savefig("hosp_only_heatmap_" + str(peak) + ".png", dpi=1200)
        plt.show()

        # Staffed beds only single-indicator policies

        beds_df = base_df[161:int(161 + 45)]

        base_array = np.full((11, 11), 0)
        mask_array = np.full((11, 11), True)
        cost = beds_df["cost"]

        labels = np.full((11, 11), "", dtype="str")
        lowest_cost_policy_df = beds_df.sort_values("cost").iloc[0]
        labels[int(lowest_cost_policy_df["beds1"] * 100)][int(lowest_cost_policy_df["beds2"] * 100)] = "*"

        for i in np.arange(45):
            base_array[int(beds_df.iloc[i]["beds1"] * 100)][int(beds_df.iloc[i]["beds2"] * 100)] = cost.iloc[i]
            if cost.iloc[i] < 30000:
                mask_array[int(beds_df.iloc[i]["beds1"] * 100)][int(beds_df.iloc[i]["beds2"] * 100)] = False
            if beds_df["feasibility"].iloc[i] < 0.95:
                labels[int(beds_df.iloc[i]["beds1"] * 100)][int(beds_df.iloc[i]["beds2"] * 100)] = "x"

        plt.clf()
        ax = sns.heatmap(base_array,
                         cmap="magma_r",
                         mask=mask_array,
                         annot=labels,
                         annot_kws={'fontsize': 12, 'color': 'k', 'alpha': 1,
                                    'rotation': 'vertical', 'verticalalignment': 'center'},
                         fmt="").set(title="Cost of staffed-beds-only policies, " + title_suffix,
                                     xlabel="Threshold 2 (yellow-red) -- percentages",
                                     ylabel="Threshold 1 (blue-yellow) -- percentages")
        plt.savefig("beds_only_heatmap_" + str(peak) + ".png", dpi=1200)
        plt.show()

        # breakpoint()

        # Can add vmin and vmax to heatmap to specify min/max values
