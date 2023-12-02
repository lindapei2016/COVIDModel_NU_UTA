###############################################################################

# Misc parsing stuff :O

# Misc routines run on local computer to combine/reorganize files
#   created from different experiments

###############################################################################

import copy
import pandas as pd
import Tools_Optimization_Utilities

from pathlib import Path

# Import other Python packages
import numpy as np
import glob

base_path = Path(__file__).parent

###############################################################################

import matplotlib.pyplot as plt

aggregated_files_folder_name = "3000I"
aggregated_files_prefix = "8000reps_aggregated_peak"

breakpoint()

for peak in np.arange(3):
    for suffix in ["_stage2_days.csv", "_stage3_days.csv", "_ICU_violation_patient_days.csv"]:
        df = pd.read_csv(base_path / aggregated_files_folder_name /
                         (aggregated_files_prefix + str(peak) + suffix), index_col=0)
        plt.clf()
        plt.hist(df["25"])
        plt.show()
        breakpoint()

breakpoint()