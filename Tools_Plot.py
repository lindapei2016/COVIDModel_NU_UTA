# Plots will be saved in folder called "plots" in current working directory
# If such a folder does not exist, the code will automaticallyc reate one

from matplotlib import pyplot as plt, colors
from pathlib import Path
import numpy as np
from datetime import datetime as dt
import calendar as py_cal
import json
import pandas as pd
import os

from Tools_InputOutput import import_stoch_reps_for_reporting
from Tools_Report import sim_history_key_stats, Report

base_path = Path(__file__).parent
path_to_plot = base_path / "plots"
real_data_file_names = {}

directory_exists = os.path.exists(path_to_plot)
if not directory_exists:
    os.makedirs(path_to_plot)

surge_colors = ('moccasin', 'pink')

plt.rcParams["font.size"] = "18"


######################################################################################

# Plotting Module

def find_central_path(sim_data_ICU, sim_data_IH, real_data, T_real: int):
    # Trim real data to T_real time steps
    real_data = real_data[: T_real]

    # Compute simulated data
    num_rep = len(sim_data_ICU)
    sim_data = []
    for s in range(num_rep):
        sim_ICU = np.sum(sim_data_ICU[s], axis=(1, 2))[:T_real]
        sim_IH = np.sum(sim_data_IH[s], axis=(1, 2))[:T_real]
        sim_total = sim_ICU + sim_IH
        sim_data.append(sim_total)

    # Compute R-squared for each simulation
    rsq = []
    for sim in sim_data:
        numerator = np.sum((sim - real_data) ** 2)
        denominator = np.sum((real_data - np.mean(real_data)) ** 2)
        rsq.append(1 - numerator / denominator)

    # Find index of simulation with highest R-squared
    central_path_id = np.argmax(rsq)
    return central_path_id


def moving_avg(data, n_day, percent=1):
    """
    Take the n-day moving average of data.
    (Add percentage for percent of staffed inpatient beds).
    :return:
    """

    return [data[max(0, i - n_day): i].mean() / percent if i - n_day > 0 else 0 for i in range(len(data))]


def moving_sum(data, n_day, total_population):
    """
    Take the n-day moving sum per 100k of the simulation data.
    :return:
    """
    return [data[max(0, i - n_day): i].sum() * 100000 / total_population if i - n_day > 0 else 0 for i in
            range(len(data))]


######################################################################################


class Plot:
    """
    This class is for plotting historical simulation results/projections etc. for a single simulation run.
    Plot a list of sample paths in the same figure with different plot backgrounds.
    """

    def __init__(self,
                 instance: object,
                 real_history_end_date,
                 real_data: list,
                 sim_data: list,
                 var: str,
                 policy_name,
                 central_path=0,
                 color=('teal', 'paleturquoise'),
                 text_size=28):

        self.instance = instance
        self.real_history_end_date = real_history_end_date
        # TODO: fix this part later:
        if sim_data is not None:
            if var == "ToIY_history_sum":
                self.sim_data = [np.sum(s, axis=(1, 2)) * 0.6 for s in sim_data]
            else:
                self.sim_data = [np.sum(s, axis=(1, 2)) for s in sim_data]
        else:
            self.sim_data = None

        self.var = var
        self.policy_name = policy_name
        self.central_path = central_path
        self.T = len(np.sum(sim_data[0], axis=(1, 2)))
        self.T_real = (real_history_end_date - instance.simulation_start_date).days
        self.real_data = np.array(real_data[0:self.T_real]) if real_data is not None else None
        self.text_size = text_size

        with open(str(base_path / "instances" / f"{instance.city}" / "plot_info.json"), "r") as input_file:
            data = json.load(input_file)
            self.y_lim = data["y_lim"]
            self.compartment_names = data["compartment_names"]

        self.path_to_plot = base_path / "plots"

        self.fig, (self.ax1, self.year_ax, self.actions_ax) = plt.subplots(3, 1, figsize=(17, 9),
                                                                           gridspec_kw={'height_ratios': [10, 0.7, 1.1]}
                                                                           )

        plt.subplots_adjust(hspace=0.15)
        self.policy_ax = self.ax1.twinx()
        self.base_plot(color)

    def base_plot(self, color):
        """
        The base plotting function sets the common plot design for different type of plots.
        :return:
        """
        if self.var == 'IH_history_average':
            percent = self.instance.total_hosp_beds
        else:
            percent = 1
        if 'Seven-day Average' in self.compartment_names[self.var]:
            n_day = self.instance.moving_avg_len
            # Compute moving averages for simulation data
            self.sim_data = np.apply_along_axis(
                lambda s: moving_avg(s, n_day, percent),
                axis=1,
                arr=self.sim_data
            )
            if self.real_data is not None:
                self.real_data = moving_avg(self.real_data.copy(), n_day, percent)

        elif 'Seven-day Sum' in self.compartment_names[self.var]:
            n_day = self.instance.moving_avg_len
            N = np.sum(self.instance.N, axis=(0, 1))
            self.sim_data = np.apply_along_axis(
                lambda s: moving_sum(s, n_day, N),
                axis=1,
                arr=self.sim_data
            )

            if self.real_data is not None:
                self.real_data = moving_sum(self.real_data.copy(), n_day, N)

        if self.real_data is not None:
            real_h_plot = self.ax1.scatter(range(self.T_real), self.real_data[0:self.T_real], color='maroon',
                                           zorder=100, s=15)

        self.ax1.plot(range(self.T), np.transpose(np.array(self.sim_data)), color[1])
        self.ax1.plot(range(self.T), self.sim_data[self.central_path], color[0])

        # plot a vertical line to separate history from projections:
        self.ax1.vlines(self.T_real, 0, self.y_lim[self.var], colors='k', linewidth=3)
        # self.ax1.vlines(self.instance.cal.calendar.index(dt(2021, 11, 30)), 0, self.y_lim[self.var], colors='k',
        #                 linewidth=3)

        # self.actions_ax.vlines(self.T_real, 0, self.y_lim[self.var], colors='k',
        #                        linewidth=3)
        # Plot styling:
        # Axis limits:
        self.ax1.set_ylim(0, self.y_lim[self.var])
        self.year_ax.set_ylim(0, 1)
        self.policy_ax.set_ylim(0, self.y_lim[self.var])
        self.ax1.set_ylabel(self.compartment_names[self.var])
        self.actions_ax.set_xlim(0, self.T)
        self.year_ax.set_xlim(0, self.T)
        self.set_x_axis()
        # Order of layers
        self.ax1.set_zorder(self.policy_ax.get_zorder() + 10)  # put ax in front of policy_ax
        self.ax1.patch.set_visible(False)  # hide the 'canvas'
        # Plot margins
        self.actions_ax.margins(0)
        self.year_ax.margins(0)
        self.ax1.margins(0)
        self.policy_ax.margins(0)

        # Axis ticks.
        if "Percent" in self.compartment_names[self.var]:
            self.ax1.yaxis.set_ticks(np.arange(0, self.y_lim[self.var] + 0.01, 0.2))
            self.ax1.yaxis.set_ticklabels(
                [f' {np.round(t * 100)}%' for t in np.arange(0, self.y_lim[self.var] + 0.01, 0.2)],
                rotation=0,
                fontsize=22)

    def set_x_axis(self):
        """
        Set the months and years on the x-axis of the plot.
        """
        # Axis ticks: write the name of the month on the x-axis:
        axis = self.ax1  # self.actions_ax
        axis.xaxis.set_ticks(
            [t for t, d in enumerate(self.instance.cal.calendar) if (d.day == 1 and d.month % 2 == 1)])
        axis.xaxis.set_ticklabels(
            [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(self.instance.cal.calendar) if
             (d.day == 1 and d.month % 2 == 1)],
            rotation=0,
            fontsize=22)

        for tick in axis.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
        axis.tick_params(axis='y', labelsize=self.text_size, length=5, width=2)
        axis.tick_params(axis='x', length=5, width=2)

        # Clean up the action_ax and policy_ax to write the years there:
        self.actions_ax.spines['top'].set_visible(False)
        self.actions_ax.spines['bottom'].set_visible(False)
        self.actions_ax.spines['left'].set_visible(False)
        self.actions_ax.spines['right'].set_visible(False)

        self.year_ax.spines['top'].set_visible(False)
        self.year_ax.spines['bottom'].set_visible(False)
        self.year_ax.spines['left'].set_visible(False)
        self.year_ax.spines['right'].set_visible(False)

        self.policy_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False,
            labelright=False)  # labels along the bottom edge are off

        self.actions_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the top edge are off
            labelleft=False,
            labelbottom=False,
            labelright=False
        )  # labels along the bottom edge are off

        self.year_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the top edge are off
            labelleft=False,
            labelbottom=False,
            labelright=False
        )

        year_list = {2020, 2021, 2022}  # fix this part later.
        year_ticks = {}
        for year in year_list:
            t1 = self.instance.cal.calendar.index(dt(year, 6, 15))
            t2 = self.instance.cal.calendar.index(dt(year, 2, 1)) if dt(year, 2,
                                                                        1) in self.instance.cal.calendar else self.T + 1
            if t1 <= self.T:
                year_ticks[year] = t1
            elif t1 > self.T >= t2:
                year_ticks[year] = t2
        # write down the year on the plot axis:
        for year in year_ticks:
            self.year_ax.annotate(year,
                                  xy=(year_ticks[year], 0.0),
                                  xycoords='data',
                                  color='k',
                                  annotation_clip=True,
                                  fontsize=self.text_size - 2)

    def horizontal_plot(self, thresholds, tier_colors):
        """
        Plot the policy thresholds horizontally with corresponding policy colors.
        This plotting is  used with the indicator we keep track.
        For instance plot 7-day avg. hospital admission with the hospital admission thresholds in the background.

        Color the plot only for the part with projections where the transmission reduction is not fixed.
        """
        for id_tr, tr in enumerate(thresholds):
            u_color = tier_colors[id_tr]
            u_alpha = 0.6
            u_lb = tr
            u_ub = thresholds[id_tr + 1] if id_tr + 1 < len(thresholds) else self.y_lim[self.var]
            if u_lb >= -1 and u_ub >= 0:
                self.policy_ax.fill_between(range(self.T_real, self.T + 1),
                                            u_lb,
                                            u_ub,
                                            color=u_color,
                                            alpha=u_alpha,
                                            linewidth=0.0,
                                            step='pre')

        self.save_plot("horizontal")

    def changing_horizontal_plot(self, surge_history, surge_states, thresholds, tier_colors: dict):
        """
        Plot the policy thresholds horizontally with corresponding policy colors.
        This plotting is used when plotting the CDC staged-alert system. The thresholds change over time
        according to the case count.
        Color the plot only for the part with projections where the transmission reduction is not fixed.
        """
        for u, state in enumerate(surge_states):
            fill = [True if s == u else False for s in surge_history[self.central_path][self.T_real:self.T]]
            fill = [True if fill[i] or fill[i - 1] else False for i in range(len(fill))]
            for id_tr, tr in enumerate(thresholds[state]):
                u_color = tier_colors[id_tr]
                u_alpha = 0.6
                u_lb = tr
                u_ub = thresholds[state][id_tr + 1] if id_tr + 1 < len(thresholds[state]) else self.y_lim[self.var]
                if u_lb >= -1 and u_ub >= 0:
                    self.policy_ax.fill_between(range(self.T_real, self.T),
                                                u_lb,
                                                u_ub,
                                                color=u_color,
                                                alpha=u_alpha,
                                                linewidth=0.0,
                                                where=fill,
                                                step='pre')

        self.save_plot("changing_horizontal")

    def save_plot(self, plot_type):
        """ Save the plot in a png format to /plots directory. """
        plt.savefig(
            self.path_to_plot / f"{self.real_history_end_date.date()}_{self.policy_name}_{self.instance.total_hosp_beds}_{self.var}_{plot_type}.png",
            bbox_inches='tight'
        )

    def vertical_plot(self, tier_history, tier_colors: dict, cap_limit=0):
        """
        Plot the historical policy vertically with corresponding policy colors.
        The historical policy can correspond to the five tiers or to the surge tiers in the CDC system.

        Color the plot only for the part with projections where the transmission reduction is not fixed.

        Parameters
        ----------
        tier_history: historical alert-stages determined by the policy.
        tier_colors: colors corresponding to each staged-alert level, e.g. blue, yellow etc.
        cap_limit: capacity of a hospital recourse. e.g. ICU or general ward capacity.

        Returns None
        """
        policy_start_date = tier_history[0].count(None)
        for u in range(len(tier_colors)):
            u_color = tier_colors[u]
            u_alpha = 0.6
            fill = np.array(tier_history[self.central_path][policy_start_date:self.T]) == u
            fill = [True if fill[i] or fill[i - 1] else False for i in range(len(fill))]
            self.policy_ax.fill_between(range(policy_start_date, self.T),
                                        0,
                                        self.y_lim[self.var],
                                        where=fill,
                                        color=u_color,
                                        alpha=u_alpha,
                                        linewidth=0)

            # Plot a horizontal black line to indicate the resource capacity:
            self.ax1.hlines(cap_limit, 0, self.T, colors='k', linewidth=3)
        self.save_plot("vertical")

    def dali_plot(self,
                  tier_history,
                  tier_colors: dict,
                  cap_limit=0,
                  legend=None,
                  tier_history2=None,
                  tier_colors2=None):
        """
        Plot the tier history colors. Different sample paths may be in different stages during the same time period.
        color the background to tier color according the percent of paths in that particular tier during a particular
        time. (e.g. if 40% of paths are in blue for a certain day then 40% of the background will be blue for that day.)

        Color the plot only for the part with projections where the transmission reduction is not fixed.

        Parameters
        ----------
        tier_history: historical alert-stages determined by the policy.
        tier_colors: colors corresponding to each staged-alert level, e.g. blue, yellow etc.
        cap_limit: capacity of a hospital recourse. e.g. ICU or general ward capacity.
        legend: list of legend labels for the background colors.
        Returns None
        """
        bottom_tier = 0
        policy_start_date = tier_history.count(None)
        for u in range(len(tier_colors)):
            color_fill = (sum(np.array(t[policy_start_date:self.T]) == u for t in tier_history)
                          / (len(tier_history) - policy_start_date)) * self.y_lim[self.var]
            self.policy_ax.bar(range(policy_start_date, self.T),
                               color_fill,
                               color=tier_colors[u],
                               bottom=bottom_tier,
                               width=1,
                               alpha=0.6,
                               linewidth=0,
                               label=legend[u] if legend is not None else None)
            bottom_tier += np.array(color_fill)

        if tier_history2 is not None:
            bottom_tier2 = 0
            for u in range(len(tier_colors2)):
                color_fill2 = (sum(np.array(t[policy_start_date:self.T]) == u for t in tier_history2)
                               / (len(tier_history2) - policy_start_date)) * self.y_lim[self.var]

                self.actions_ax.bar(range(policy_start_date, self.T),
                                    color_fill2,
                                    color=tier_colors2[u],
                                    bottom=bottom_tier2,
                                    width=1,
                                    alpha=0.6,
                                    linewidth=0)
                bottom_tier2 += np.array(color_fill2)

        # Plot a horizontal black line to indicate the resource capacity:
        self.ax1.hlines(cap_limit, 0, self.T, colors='k', linewidth=3)

        # Plot legends if the labels are provided:
        if legend is not None:
            lines, labels = self.policy_ax.get_legend_handles_labels()
            label_line_dict = {label: line for label, line in zip(labels, lines)}

            unique_labels = list(label_line_dict.keys())
            unique_lines = list(label_line_dict.values())

            unique_labels = [unique_labels[1], unique_labels[2], unique_labels[-1], unique_labels[-2]]
            unique_lines = [unique_lines[1], unique_lines[2], unique_lines[-1], unique_lines[-2]]

            self.policy_ax.legend(unique_lines, unique_labels, loc="upper left", fontsize="15")
        self.save_plot("dali")


class BarPlot:
    """
    This class plots histogram of days spent in each stage to visually compare different policies.
    """

    def __init__(self, instance,
                 plot_info_filename: str,
                 label_dict: dict,
                 data_ax1,
                 tier_colors1: dict,
                 data_ax2=None,
                 tier_colors2=None):

        self.instance = instance
        self.label_dict = label_dict
        self.data_ax1 = data_ax1
        self.data_ax2 = data_ax2
        self.tier_colors1 = tier_colors1
        self.tier_colors2 = tier_colors2
        self.path_to_plot = base_path / "plots"
        self.fig, (self.ax, self.timeline_ax) = plt.subplots(2, 1,
                                                             figsize=(17, 9),
                                                             gridspec_kw={'height_ratios': [10, 1.1]})

        with open(str(base_path / "instances" / f"{instance.city}" / plot_info_filename), "r") as input_file:
            self.plot_info = json.load(input_file)
        if data_ax2 is not None:
            self.ax2 = self.ax.twinx()  # instantiate a second axes that shares the same x-axis for peak ICU demand.

        self.xticks_location = []

    def plot(self):
        self.ax.set_ylabel(self.plot_info["ylabel_ax1"])
        self.ax.set_ylim(self.plot_info["ylim_ax1"][0], self.plot_info["ylim_ax1"][1])

        if self.data_ax2 is not None:
            self.ax2.set_ylabel(self.plot_info["ylabel_ax2"])
            self.ax2.set_ylim(self.plot_info["ylim_ax2"][0], self.plot_info["ylim_ax2"][1])

        self.plot_tier_bar()
        self.create_x_ticks()
        self.create_legend()
        self.save_to_file()

    def plot_tier_bar(self):
        """
        Plots histograms of average proportion of days each tier was active.
        :return: the plot filename.
        """
        count = 0
        ite = 0
        color_labels_ax1 = self.plot_info["color_labels_ax1"]
        color_labels_ax2 = self.plot_info["color_labels_ax2"]
        for pname, rdata in self.data_ax1.items():
            val = [i + count for i in range(len(self.tier_colors1[pname[:-2]]))]
            self.xticks_location += [(len(val)) / 2 + count]
            col = list(self.tier_colors1[pname[:-2]].values())
            labels_ax1 = [color_labels_ax1[i] for i in self.tier_colors1[pname[:-2]].values()]
            self.ax.bar(val, rdata, color=col, label=labels_ax1)

            if self.data_ax2 is not None:
                labels_ax2 = [color_labels_ax2[i] for i in self.tier_colors2[pname[:-2]].values()]
                self.ax2.bar(len(val) + count, self.data_ax2[pname], color="gray", label=labels_ax2)

            count += len(val) + 2 + ite % 2
            ite += 1

    def create_x_ticks(self):
        # Clean up the timeline_ax to write the years or peak names there:
        self.timeline_ax.spines['top'].set_visible(False)
        self.timeline_ax.spines['bottom'].set_visible(False)
        self.timeline_ax.spines['left'].set_visible(False)
        self.timeline_ax.spines['right'].set_visible(False)
        self.ax.patch.set_visible(False)  # hide the 'canvas'
        self.timeline_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

        year_ticks = self.plot_info["year_ticks"]
        self.ax.xaxis.set_ticks(self.xticks_location)
        self.ax.xaxis.set_ticklabels([pname for pname in self.label_dict.values()] * len(year_ticks),
                                     rotation=0, fontsize="12")
        self.timeline_ax.xaxis.set_ticks(self.xticks_location)
        self.timeline_ax.xaxis.set_ticklabels([pname for pname in self.label_dict.values()] * len(year_ticks),
                                              rotation=0,
                                              fontsize="12")

        start, end = min(self.xticks_location), max(self.xticks_location)
        ticks = np.arange(start + 2, end + 2, (end - start) / len(year_ticks))
        for i, key in enumerate(year_ticks):
            self.timeline_ax.annotate(key,
                                      xy=(ticks[i], 0),
                                      xycoords='data',
                                      color='k',
                                      annotation_clip=True,
                                      fontsize=20)

        self.ax.tick_params(axis='y', labelsize=28, length=5, width=2)
        self.ax.tick_params(axis='x', length=5, width=2)

        self.ax.set_zorder(self.timeline_ax.get_zorder() + 10)  # put ax in front of policy_ax

    def create_legend(self):
        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()

        labels = labels1 + labels2
        lines = lines1 + lines2

        label_line_dict = {label: line for label, line in zip(labels, lines)}

        # Sort the label_line_dict based on the stages:
        color_map = self.plot_info["color_map"]
        sorted_label_line_dict = {k: v for k, v in sorted(label_line_dict.items(), key=lambda x: color_map[x[0]])}

        unique_labels = list(sorted_label_line_dict.keys())
        unique_lines = list(sorted_label_line_dict.values())

        self.ax2.legend(unique_lines, unique_labels, loc="upper right", fontsize="15")

    def save_to_file(self):
        plot_filename = self.path_to_plot / f"{self.instance.city}_{self.plot_info['plot_name']}_bar_plot.png"
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(plot_filename)
        return plot_filename


def plot_from_file(seeds, num_reps, instance, real_history_end_date, equivalent_thresholds, policy_name, tier_colors,
                   storage_folder_name):
    '''
    plot_from_file() create objects of the Plot class to generate couple different plots for retrospective analysis
    '''
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
                indicator_hist = []
                for i, ind_hist in enumerate(policy_outputs["active_indicator_history"]):
                    temp_hist = [h if policy_outputs["surge_history"][i][j] == 0 or h is None else h + 3
                                 for j, h in enumerate(ind_hist)]
                    indicator_hist.append(temp_hist)

                plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_sum", policy_name, central_path_id,
                            color=('k', 'silver'))
                # plot.dali_plot(indicator_hist,
                #                ["blue", "navy", "turquoise", "yellow", "skyblue", "purple"])
                legend = ["hosp. adm (case count low)",
                          "staffed bed (case count low)",
                          "both (case count low)",
                          "hosp. adm (case count high)",
                          "staffed bed (case count high)",
                          "both (case count high)"]

                plot.dali_plot(indicator_hist,
                               ["tab:red", "tab:blue", "tab:green", "tab:pink", "tab:orange", "tab:purple"],
                               0,
                               legend,
                               [policy_outputs["tier_history"][central_path_id]],
                               tier_colors)

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

            plot.vertical_plot(policy_outputs["tier_history"], tier_colors, instance.total_hosp_beds)
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

#         # The remaining plots are useful during parameter fitting:
#         # elif key == "D_history":
#         #     real_data = np.cumsum(
#         #         np.array([ai + bi for (ai, bi) in zip(instance.real_ToIYD_history, instance.real_ToICUD_history)]))
#         #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
#         #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
#         # elif key == "ToIYD_history":
#         #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
#         #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
#         # elif key == "ToICUD_history":
#         #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name, central_path_id)
#         #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
#         # elif key == "S_history":
#         #     real_data = None
#         #     plot = Plot(instance, real_history_end_date, real_data, val, key, policy_name,
#         #                 central_path_id)
#         #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)


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