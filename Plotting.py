###############################################################################

# Plotting.py
# This script has an exemplary plotting function. Plot_Manager.py has the Plot class that has different
# plotting styles etc.
# plot_from_file() create objects of the Plot class to generate couple different plots for retrospective analysis
# Feel free to copy the below function to a new script and modify for your own use.

# Nazlican Arslan 2022

###############################################################################

import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.colors as pltcolors

from Plot_Manager import Plot, find_central_path, BarPlot
from Report_Manager import Report
from InputOutputTools import import_stoch_reps_for_reporting
from Report_Manager import sim_history_key_stats

base_path = Path(__file__).parent
path_to_plot = base_path / "plots"
real_data_file_names = {}

surge_colors = ['moccasin', 'pink']


def plot_from_file(seeds, num_reps, instance, real_history_end_date, equivalent_thresholds, policy_name, tier_colors,
                   storage_folder_name):

    # Read the simulation outputs:
    sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, real_history_end_date, instance,
                                                                  policy_name, storage_folder_name)

    # Choose the central path among 300 sample paths:
    central_path_id = find_central_path(sim_outputs["ICU_history"],
                                        sim_outputs["IH_history"],
                                        instance.real_IH_history,
                                        instance.cal.calendar.index(real_history_end_date))

    for key, val in sim_outputs.items():
        # Create plots for each output:
        print(key)
        if hasattr(instance, f"real_{key}"):
            real_data = getattr(instance, f"real_{key}")
        else:
            real_data = None

        if key == "ICU_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
            plot.dali_plot(policy_outputs["tier_history"], tier_colors, instance.icu)

        elif key == "ToIHT_history":
            if "surge_history" in policy_outputs.keys():
                plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id,
                            color=('k', 'silver'))
                plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                              ["non_surge", "surge"],
                                              equivalent_thresholds,
                                              tier_colors)

                plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_sum", policy_name,
                            central_path_id,
                            color=('k', 'silver'))
                plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                              ["non_surge", "surge"],
                                              policy_outputs["hosp_adm_thresholds"][0],
                                              tier_colors)

            if "active_indicator_history" in policy_outputs.keys():
                # The background colors in the plot show the proportion of sample  paths where
                # a certain  CDC  indicator is active, e.g., prescribe a higher stage-alert level.
                # Turquoise color indicates hospital admission and percent of staffed bed indicators
                # prescribe the same alert level.
                plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_sum", policy_name, central_path_id,
                            color=('k', 'silver'))
                plot.dali_plot(policy_outputs["active_indicator_history"],
                               ["cornflowerblue", "navy", "turquoise"])

                active_ind_given_red = []
                for i, hist in enumerate(policy_outputs["active_indicator_history"]):
                    active_ind_given_red.append(
                        list(np.array(hist)[np.where(np.array(policy_outputs["tier_history"][i]) == 1)]))
                # Report the percent active CDC indicator
                print(f"Active indicator percentages: {sim_history_key_stats(active_ind_given_red, 3)}")

            if "CDC" not in policy_outputs["policy_type"][0]:
                plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id,
                            color=('k', 'silver'))
                plot.horizontal_plot(policy_outputs["lockdown_thresholds"][0], tier_colors)

                plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_sum", policy_name, central_path_id,
                            color=('k', 'silver'))
                plot.horizontal_plot(equivalent_thresholds, tier_colors)

        elif key == "IH_history":
            # ToDo: Fix IH_history:
            val_updated = [[ai + bi for (ai, bi) in zip(v, sim_outputs["ICU_history"][i])]
                           for i, v in enumerate(val)]

            if "surge_history" in policy_outputs.keys():
                plot = Plot(instance, real_history_end_date, real_data, val_updated, f"{key}_average", policy_name,
                            central_path_id, color=('k', 'silver'))
                plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                              ["non_surge", "surge"],
                                              policy_outputs["staffed_bed_thresholds"][0],
                                              tier_colors)

            plot = Plot(instance, real_history_end_date, real_data, val, f"{key}", policy_name, central_path_id,
                        color=('k', 'silver'))

            plot.vertical_plot(policy_outputs["tier_history"], tier_colors, instance.hosp_beds)
        elif key == "ToIY_history" and "surge_history" in policy_outputs.keys():
            # ToDo: Fix the data reading part here:
            filename = 'austin_real_case.csv'
            real_data = pd.read_csv(
                str(instance.path_to_data / filename),
                parse_dates=["date"],
                date_parser=pd.to_datetime,
            )["admits"]
            plot = Plot(instance, real_history_end_date, real_data, val, "ToIY_history_sum", policy_name,
                        central_path_id)
            plot.vertical_plot(policy_outputs["surge_history"], surge_colors, policy_outputs["case_threshold"])
            plot.dali_plot(policy_outputs["surge_history"], surge_colors, policy_outputs["case_threshold"])

        # The remaining plots are useful during parameter fitting:
        # elif key == "D_history":
        #     real_data = np.cumsum(
        #         np.array([ai + bi for (ai, bi) in zip(instance.real_ToIYD_history, instance.real_ToICUD_history)]))
        #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        # elif key == "ToIYD_history":
        #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        # elif key == "ToICUD_history":
        #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        # elif key == "S_history":
        #     real_data = None
        #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name,
        #                 central_path_id)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)


def report_from_file(seeds, num_reps, instance, history_end_date, stats_end_date, policy_name, tier_colors,
                     report_template, storage_folder_name):
    sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, history_end_date, instance,
                                                                  policy_name, storage_folder_name)
    report = Report(instance, sim_outputs, policy_outputs, history_end_date, stats_end_date, tier_colors,
                    report_template)
    report_data = report.build_report()


def bar_plot_from_file(seeds,
                       num_reps,
                       instance,
                       history_end_date_list,
                       stats_end_date_list,
                       labels_dict,
                       tier_colors_dict1,
                       tier_colors_dict2,
                       report_template_dict,
                       storage_folder_name):
    plot_data_dict_ax1 = {}
    plot_data_dict_ax2 = {}
    for t, sim_end_time in enumerate(stats_end_date_list):
        for pname, rt in report_template_dict.items():
            sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, history_end_date_list[t],
                                                                          instance,
                                                                          pname, storage_folder_name)
            report = Report(instance,
                            sim_outputs,
                            policy_outputs,
                            history_end_date_list[t],
                            sim_end_time,
                            tier_colors_dict1[pname],
                            rt)

            report_data = report.build_report()
            plot_data_dict_ax1[f"{pname}_{t}"] = sim_history_key_stats(policy_outputs["tier_history"],
                                                                       len(tier_colors_dict1[pname]))
            plot_data_dict_ax2[f"{pname}_{t}"] = report_data['MEAN-ICU-PEAK']

    plot_info_filename = "bar_plot_info.json"
    stats_plot = BarPlot(instance,
                         plot_info_filename,
                         labels_dict,
                         plot_data_dict_ax1,
                         tier_colors_dict1,
                         plot_data_dict_ax2,
                         tier_colors_dict2)
    stats_plot.plot()

# def active_bar_plot_from_file(seeds,
#                               num_reps,
#                               instance,
#                               history_end_date_list,
#                               stats_end_date_list,
#                               labels_dict,
#                               tier_colors_dict1,
#                               storage_folder_name):
#     plot_data_dict_ax1 = {}
#     for t, sim_end_time in enumerate(stats_end_date_list):
#         for pname, rt in labels_dict.items():
#             sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, history_end_date_list[t],
#                                                                           instance,
#                                                                           pname, storage_folder_name)
#
#             plot_data_dict_ax1[f"{pname}_{t}"] = sim_history_key_stats(policy_outputs["active_indicator_history"],
#                                                                        3)
#
#     plot_info_filename = "bar_plot_active_indicator_info.json"
#     stats_plot = BarPlot(instance, plot_info_filename, labels_dict)
#     stats_plot.plot(plot_data_dict_ax1, tier_colors_dict1)
