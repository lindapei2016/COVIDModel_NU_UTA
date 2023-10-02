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

from pathlib import Path

base_path = Path(__file__).parent

#############################################
############## READ-IN FILES ################
#############################################

aggregated_files_folder_name = "Results_09292023_NoCaseThreshold_12172Policies_BetterSubset_1000Reps"

full_df_dict = {}
for peak in np.arange(3):
    full_df_dict[str(peak)] = pd.read_csv(base_path / aggregated_files_folder_name /
                                          ("full_df_peak" + str(peak) + ".csv"), index_col=0)
full_df_across_peaks = pd.read_csv(base_path / aggregated_files_folder_name / "full_df_across_peaks.csv",
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

##########################################
############## SCRATCHPAD ################
##########################################

breakpoint()

need_plot_changing_coordinate = False

if need_plot_changing_coordinate:
    vary_threshold1_df = get_df_of_perturbations((3,14,np.inf,np.inf), "hosp1", np.arange(13), full_df_across_peaks)

    title = "Hospital-admits-only policies with optimal 2nd threshold (at 14)"
    xlabel = "1st threshold"
    ylabel = "Number of days (average)"
    make_plot_changing_coordinate("hosp1",
                                  ("stage2", "stage3", "violation"),
                                  ("Stage 2 Days", "Stage 3 Days", "ICU Over Capacity Patient Days"),
                                  ("gold", "red", "purple"),
                                  vary_threshold1_df,
                                  "vary_threshold1_days.png",
                                  xlabel,
                                  ylabel,
                                  title)

    # Adding economic and health costs and corresponding standard errors to vary_threshold1_df
    vary_threshold1_df["econ_cost"] = 10 * vary_threshold1_df["stage2"] + 100 * vary_threshold1_df["stage3"]
    vary_threshold1_df["econ_cost_stderror"] = 10 * vary_threshold1_df["stage2_stderror"] + 100 * vary_threshold1_df["stage3_stderror"]
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

    vary_threshold2_df = get_df_of_perturbations((3,20,np.inf,np.inf), "hosp2", np.arange(20), full_df_across_peaks)

    make_plot_changing_coordinate("hosp2",
                                  ("stage2", "stage3", "violation"),
                                  ("Stage 2 Days", "Stage 3 Days", "ICU Over Capacity Patient Days"),
                                  ("gold", "red", "purple"),
                                  vary_threshold2_df,
                                  "vary_threshold2_days.png",
                                  xlabel,
                                  ylabel,
                                  title)

    vary_threshold2_df["econ_cost"] = 10 * vary_threshold2_df["stage2"] + 100 * vary_threshold2_df["stage3"]
    vary_threshold2_df["econ_cost_stderror"] = 10 * vary_threshold2_df["stage2_stderror"] + 100 * vary_threshold2_df["stage3_stderror"]
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

full_df_across_peaks_copy = full_df_across_peaks.copy(deep=True)
full_df_across_peaks_copy.drop(columns="cost", inplace=True)
# full_df_feasible_across_peaks = full_df_across_peaks_copy[full_df_across_peaks_copy["feasibility"] > 0.95]

econ_costs = []
health_costs = []

for w in np.arange(401):
    full_df_across_peaks_copy["cost"] = 10 * full_df_across_peaks_copy["stage2"] + \
                                        100 * full_df_across_peaks_copy["stage3"] + \
                                        w * full_df_across_peaks_copy["violation"]
    cost_argmin = full_df_across_peaks_copy.sort_values("cost").index[0]
    econ_cost = 10 * full_df_across_peaks_copy["stage2"].iloc[cost_argmin] + \
                100 * full_df_across_peaks_copy["stage3"].iloc[cost_argmin]
    health_cost = full_df_across_peaks_copy["violation"].iloc[cost_argmin]

    econ_costs.append(econ_cost)
    health_costs.append(health_cost)

plt.clf()
plt.scatter(np.array(econ_costs), np.array(health_costs))
plt.show()

breakpoint()