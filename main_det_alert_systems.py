###############################################################################
# main_det_alert_systems.py
# This script contains examples of how to run the deterministic simulation
# for retrospective staged-alert system evaluation.

# Nazlican Arslan 2023
###############################################################################

from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from InputOutputTools import export_rep_to_json
from Plotting import plot_from_file

# Import other Python packages
import datetime as dt
from pathlib import Path
import os

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
###############################################################################
# Define the staged-alert policy to run the simulation with.
# This script has Austin's and the CDC's staged-alert systems.

# tiers file contains the staged-alert levels with the corresponding transmission reductions:
# CDC system has three levels and Austin system has four levels.
tiers_CDC = TierInfo("austin", "tiers_CDC.json")
tiers_austin = TierInfo("austin", "tiers5_opt_Final.json")

# Define the threshold values for each indicator:
thresholds_austin = (-1, 0, 15, 25, 50)  # Austin's system has one indicator.
mtp = MultiTierPolicy(austin, tiers_austin, thresholds_austin, "green")

# the CDC's system has three indicators:
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, 10, 20), "surge": (-1, -1, 10)}
staffed_thresholds = {"non_surge": (-1, 0.1, 0.15), "surge": (-1, -1, 0.1)}

# The CDC's system uses case count as an indicator.
# We estimated real case count data as 40% of ToIY for Austin.
percentage_cases = 0.4
ctp = CDCTierPolicy(austin, tiers_CDC, case_threshold, hosp_adm_thresholds, staffed_thresholds, percentage_cases)

###############################################################################
# Define the time horizon you want to run the policy on:
# This example run the system for the first peak:
history_end_time = dt.datetime(2020, 5, 30)  # use fixed transmission value until history en time.
simulation_end_time = dt.datetime(2020, 10, 1)

# Define the deterministic simulation with CDC system, you can define for Austin system with mtp object:
seed = -1
rep = SimReplication(austin, vaccines, ctp, seed)
rep.simulate_time_period(austin.cal.calendar.index(simulation_end_time), austin.cal.calendar.index(history_end_time))

# Save the simulation output to files. (I have my own directory /input_output_folder/austin for storing the files)
# Check if the directory exists:
temp_path = f"{base_path}/input_output_folder/austin"
# Check whether the specified path exists or not
is_exist = os.path.exists(temp_path)
if not is_exist:
    # Create a new directory because it does not exist
    os.makedirs(temp_path)
    print("The /input_output_folder directory is created!")

# but you can define your own convention and define the name of the output files in a different way:
base_filename = f"{base_path}/input_output_folder/austin/{seed}_1_{history_end_time.date()}_{str(ctp)}"
export_rep_to_json(
    rep,
    f"{base_filename}_sim_updated.json",
    f"{base_filename}_v0.json",
    f"{base_filename}_v1.json",
    f"{base_filename}_v2.json",
    f"{base_filename}_v3.json",
    f"{base_filename}_policy.json"
)

# Read the outputs and plot the results:
tier_colors_ctp = {0: "blue", 1: "gold", 2: "red"}  # define the plot colors you would like for each tier.

# ToIHT indicators of Austin and CDC system are different. Austin has 7-day average of hosp adm. CDC has 7-day total
# per 100k. I used this equivalent_thresholds to convert one set of threshold to the equivalent format.
equivalent_thresholds_cdc = {"non_surge": (-1, 28.57, 57.14), "surge": (-1, -1, 28.57)}
equivalent_thresholds_austin = [-1, 4.8, 8.1, 16.1]

# This is a helper function I used to plot a bunch of simulation outcomes:
plot_from_file([seed],
               1,  # we have one sample path.
               austin,
               history_end_time,
               equivalent_thresholds_cdc,
               str(ctp),
               tier_colors_ctp,
               "input_output_folder/austin")
# You can check the plots created in the /plots directory.
