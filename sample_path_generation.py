from mpi4py import MPI

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from InputOutputTools import import_rep_from_json, export_rep_to_json
from OptTools import get_sample_paths, thresholds_generator, \
    evaluate_policies_on_sample_paths, evaluate_single_policy_on_sample_path

import numpy as np
import datetime as dt
import copy
import itertools
import json

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_total_workers = size - 1
master_rank = size - 1

################################################################################

austin = City("austin",
              "calendar.csv",
              "austin_setup.json",
              "variant.json",
              "transmission.csv",
              "austin_hospital_home_timeseries.csv",
              "variant_prevalence.csv")

tiers = TierInfo("austin", "tiers4.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

get_sample_paths(austin,
                 vaccines,
                 0.75,
                 50,
                 timepoints=(10, 20, 30, 40, 50, 93),
                 processor_rank=rank,
                 save_intermediate_states=False,
                 storage_folder_name="states",
                 fixed_kappa_end_date=763)