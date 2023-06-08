###############################################################################
# main_det_alert_systems.py
# This script contains examples of how to run plot various plots.

# Nazlican Arslan 2023
###############################################################################

from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from InputOutputTools import export_rep_to_json
from Plotting import plot_from_file
from InputOutputTools import import_stoch_reps_for_reporting
from Plot_Manager import Plot, find_central_path, BarPlot

# Import other Python packages
import datetime as dt
from pathlib import Path
import os

# Let's first run a deterministic sample path with Austin's staged-alert system.
# The beginning is similar to main_det_alert_systems.py. You can skip to the plotting part.

base_path = Path(__file__).parent
###############################################################################
# Create a city object:
austin = City(
    "austin",
    "calendar.csv",
    "austin_setup.json",
    "variant.json",
    "transmission.csv",
    "austin_hospital_home_timeseries.csv",
    "variant_prevalence.csv"
)

vaccines = Vaccine(
    austin,
    "austin",
    "vaccines.json",
    "booster_allocation_fixed.csv",
    "vaccine_allocation_fixed.csv",
)
tiers_austin = TierInfo("austin", "tiers5_opt_Final.json")
thresholds_austin = (-1, 15, 25, 50)  # Austin's system has one indicator.
mtp = MultiTierPolicy(austin, tiers_austin, thresholds_austin, "green")

history_end_time = dt.datetime(2020, 5, 30)  # use fixed transmission value until history en time.
simulation_end_time = dt.datetime(2020, 10, 1)

# Define the deterministic simulation with CDC system, you can define for Austin system with mtp object:
seed = -1
rep = SimReplication(austin, vaccines, mtp, seed)
rep.fixed_kappa_end_date = austin.cal.calendar.index(history_end_time)
rep.simulate_time_period(austin.cal.calendar.index(simulation_end_time))
temp_path = f"{base_path}/input_output_folder/austin"
is_exist = os.path.exists(temp_path)
if not is_exist:
    # Create a new directory because it does not exist
    os.makedirs(temp_path)
    print("The /input_output_folder directory is created!")

# but you can define your own convention and define the name of the output files in a different way:
base_filename = f"{base_path}/input_output_folder/austin/{seed}_1_{history_end_time.date()}_{str(mtp)}"
export_rep_to_json(
    rep,
    f"{base_filename}_sim_updated.json",
    f"{base_filename}_v0.json",
    f"{base_filename}_v1.json",
    f"{base_filename}_v2.json",
    f"{base_filename}_v3.json",
    f"{base_filename}_policy.json"
)

###############################################################################
# Now let's plot what we have simulated. Normally I use the Plotting.ph as a helper function but let's skip that
# for a simple example.
# First I read the json files that I recorded at the end of simulation.
# Read the simulation outputs:

# This is a helper function I used to plot a bunch of simulation outcomes. The import_stoch_reps_reporting
# collect multiple sample path outputs and combine them in a list. So this is again a helper function.
sim_outputs, policy_outputs = import_stoch_reps_for_reporting([seed], 1, history_end_time, austin,
                                                              str(mtp), "input_output_folder/austin")

# Now I have my simulation outputs I can start plotting.
# Here this example is only for the deterministic path. So there is only one sample path.
# But you can plot more than one. In the stochastic plots we highlight one representative sample path that has the
# highest rsq value. You can find the central or representative path with the following function:

# Choose the central path among 300 sample paths (you don't need this here really but just to illustrate):
central_path_id = find_central_path(sim_outputs["ICU_history"],
                                    sim_outputs["IH_history"],
                                    austin.real_IH_history,
                                    austin.cal.calendar.index(history_end_time))

# The plotting file is very flexible you can plot any compartment you keep track of.
# And you can plot in different formats. Let's plot hospitalization data in different versions:
ToIHT_hist = sim_outputs["ToIHT_history"]  # grab the hospitalization data from the dictionary.
real_ToIHT_hist = austin.real_ToIHT_history  # let's also grab the real historical data.

# Let's define a Plot object:
plot = Plot(austin, history_end_time, real_ToIHT_hist, ToIHT_hist, "ToIHT_history", str(mtp), central_path_id,
            color=('k', 'silver'))

# Let's first plot the thresholds with horizontal lines in the background:
tier_colors_ctp = {0: "blue", 1: "yellow", 2: "orange", 3: "red"}  # define the plot colors you would like for each tier.

# I use horizontal_plot to plot horizontal ones:
plot.horizontal_plot(policy_outputs["lockdown_thresholds"][0], tier_colors_ctp)

# Now let's plot ICU with Dali plots:
ICU_hist = sim_outputs["ICU_history"]
real_ICU_hist = austin.real_ICU_history
plot_icu_dali = Plot(austin, history_end_time, real_ICU_hist, ICU_hist, "ICU_history", str(mtp), central_path_id)
plot_icu_dali.dali_plot(policy_outputs["tier_history"], tier_colors_ctp, austin.icu)

# You can instead plot ICU with vertical background:
plot_icu_vertical = Plot(austin, history_end_time, real_ICU_hist, ICU_hist, "ICU_history", str(mtp), central_path_id)
plot_icu_vertical.vertical_plot(policy_outputs["tier_history"], tier_colors_ctp, austin.icu)

# There are even more plotting options in the Plot class.

