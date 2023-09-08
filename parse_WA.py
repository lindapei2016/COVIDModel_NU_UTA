###############################################################################
# Script_Washington.py
# This script parses the output of Script_Washington to generate

# Karan Agrawal 2023
###############################################################################
from pathlib import Path
import numpy as np
import glob
import pandas as pd
import csv

base_path = Path(__file__).parent
file_path = base_path / "output"

# Parsing Code
for peak in np.arange(4):

    cost_dict = []
    feasibility_dict = []
    ICU_violation_patient_days_dict = []
    stage1_days_dict = []
    stage2_days_dict = []
    stage3_days_dict = []
    ICU_peak_dict = []
    IHT_peak_dict = []
    triggerCase_dict = []
    triggerHosp_dict = []
    triggerIcu_dict = []
    triggerCaseAndHosp_dict = []

    performance_measures_dicts = [cost_dict, feasibility_dict, ICU_violation_patient_days_dict,
                                  stage1_days_dict, stage2_days_dict, stage3_days_dict, ICU_peak_dict, IHT_peak_dict,
                                  triggerCase_dict, triggerHosp_dict, triggerIcu_dict, triggerCaseAndHosp_dict]

    cost_filenames = glob.glob("output/peak" + str(peak) + "*cost.csv")
    feasibility_filenames = glob.glob("output/peak" + str(peak) + "*feasibility.csv")
    ICU_violation_patient_days_filenames = glob.glob("output/peak" + str(peak) + "*ICU_violation_patient_days.csv")
    stage1_days_filenames = glob.glob("output/peak" + str(peak) + "*stage1_days.csv")
    stage2_days_filenames = glob.glob("output/peak" + str(peak) + "*stage2_days.csv")
    stage3_days_filenames = glob.glob("output/peak" + str(peak) + "*stage3_days.csv")
    ICU_peak_filenames = glob.glob("output/peak" + str(peak) + "*ICU_peak.csv")
    IHT_peak_filenames = glob.glob("output/peak" + str(peak) + "*IHT_peak.csv")
    triggerCase_filenames = glob.glob("output/peak" + str(peak) + "*triggerCase.csv")
    triggerHosp_filenames = glob.glob("output/peak" + str(peak) + "*triggerHosp.csv")
    triggerIcu_filenames = glob.glob("output/peak" + str(peak) + "*triggerIcu.csv")
    triggerCaseAndHosp_filenames = glob.glob("output/peak" + str(peak) + "*triggerCaseAndHosp.csv")

    num_performance_measures = len(performance_measures_dicts)

    performance_measures_filenames = [cost_filenames, feasibility_filenames, ICU_violation_patient_days_filenames,
                                      stage1_days_filenames, stage2_days_filenames, stage3_days_filenames,
                                      ICU_peak_filenames, IHT_peak_filenames, triggerCase_filenames,
                                      triggerHosp_filenames, triggerIcu_filenames, triggerCaseAndHosp_filenames]

    performance_measures_strs = ["cost", "feasibility", "icu_violation_patient_days",
                                 "stage1_days", "stage2_days", "stage3_days", "ICU_peaks", "IHT_peaks",
                                 "triggerCase_days", "triggerHosp_days", "triggerIcu_days", "triggerCaseAndHosp_days"]

    for performance_measures_id in range(num_performance_measures):
        values = []
        for filename in performance_measures_filenames[performance_measures_id]:
            df = pd.read_csv(filename, header=None)
            values = df[0].astype(int).values
            performance_measures_dicts[performance_measures_id].append(values)

    for performance_measures_id in range(num_performance_measures):
        df = np.concatenate(performance_measures_dicts[performance_measures_id])
        np.savetxt("TEST_aggregated_peak" + str(peak) + "_" + str(performance_measures_strs[performance_measures_id]) \
                   + ".csv", df, delimiter=",", fmt="%d")

        # find summary statistics and write to file
        mean_value = np.mean(df)
        median_value = np.median(df)
        percentiles = np.percentile(df, [5, 95])
        lower_bound = percentiles[0]
        upper_bound = percentiles[1]
        percentile_95 = np.percentile(df, 95)

        # Create a list of (name, value) pairs for each statistic
        statistics = [
            ("Mean", mean_value),
            ("Median", median_value),
            ("90% Prediction Interval Lower Bound", lower_bound),
            ("90% Prediction Interval Upper Bound", upper_bound),
            ("95th Percentile", percentile_95)
        ]

        # Specify the file path where you want to save the CSV file
        file_path = str(peak) + "_" + str(performance_measures_strs[performance_measures_id]) + "summary_statistics.csv"

        # Write the statistics to a CSV file
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(statistics)
