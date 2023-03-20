###############################################################################
# This module contains unit tests on functions used in retrospective analysis.
# ToDO: Add unittest to compare the outputs.
# Nazlican Arslan 2023
###############################################################################

from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from OptTools import evaluate_single_policy_on_sample_path, get_sample_paths
from InputOutputTools import export_rep_to_json
from Plotting import plot_from_file

# Import other Python packages
import datetime as dt
import unittest
from pathlib import Path


class TestFixedCDCPolicy(unittest.TestCase):
    """
    This class tests the pipeline for running the CDC retrospective analysis on four major peaks in Austin.
    """
    def setUp(self):
        """
        Set up the instance for the particular experiment.
        Returns None
        -------

        """
        self.city = City(
            "austin",
            "austin_test_IHT.json",
            "calendar.csv",
            "setup_data_Final.json",
            "variant.json",
            "transmission.csv",
            "austin_real_hosp_updated.csv",
            "austin_real_icu_updated.csv",
            "austin_hosp_ad_updated.csv",
            "austin_real_death_from_hosp_updated.csv",
            "austin_real_death_from_home.csv",
            "variant_prevalence.csv"
        )
        self.tiers = TierInfo("austin", "tiers_CDC.json")
        self.vaccines = Vaccine(
            self.city,
            "austin",
            "vaccines.json",
            "booster_allocation_fixed.csv",
            "vaccine_allocation_fixed.csv",
        )
        case_threshold = 200
        hosp_adm_thresholds = {"non_surge": (-1, 10, 20), "surge": (-1, -1, 10)}
        staffed_thresholds = {"non_surge": (-1, 0.1, 0.15), "surge": (-1, -1, 0.1)}
        percentage_cases = 0.4
        self.policy = CDCTierPolicy(self.city, self.tiers, case_threshold, hosp_adm_thresholds, staffed_thresholds,
                                    percentage_cases)

        self.policy_name = f"CDC_{case_threshold}"

    def test_sample_path_files(self):
        """
        - Test if files are created successfully for each sample path with each time brackets with get_sample_paths()
        functions.
        - Test if files are created successfully after policy evaluation for each time brackets with
        evaluate_single_policy_on_sample_path().
        The files will be generated within the /tests directory. The function will give an assertion error
        if the files are not created correctly.

        Returns None
        -------

        """
        time_points = [dt.datetime(2020, 5, 30),
                       dt.datetime(2020, 11, 30),
                       dt.datetime(2021, 7, 14),
                       dt.datetime(2021, 11, 30),
                       dt.datetime(2022, 3, 30)
                       ]
        time_points = [self.city.cal.calendar.index(date) for date in time_points]
        seeds = [1, 2]  # seeds for path generations
        new_seeds = [3, 4]  # seeds for policy evaluations.

        # Generate new sample paths for each peak:
        for seed in seeds:
            get_sample_paths(self.city,
                             self.vaccines,
                             0.75,
                             2,
                             seed,
                             "",
                             True,
                             time_points[-1],
                             time_points)

        # Check if the files are successfully created:
        for seed in seeds:
            for rep in range(2):
                for time in time_points:
                    filename = f"{seed}_{rep + 1}_{self.city.cal.calendar[time].date()}_sim.json"
                    print(filename)
                    with self.subTest(filename=filename):
                        self.assertTrue(Path(filename).is_file())

        # Evaluate the CDC policy on the new sample paths for each peak:
        simulation_end_points = [dt.datetime(2020, 9, 30),
                                 dt.datetime(2021, 3, 31),
                                 dt.datetime(2021, 11, 14),
                                 dt.datetime(2022, 3, 31)
                                 ]
        simulation_end_points = [self.city.cal.calendar.index(date) for date in simulation_end_points]
        for t in range(4):
            for i in range(2):
                for rep in range(2):
                    evaluate_single_policy_on_sample_path(self.city,
                                                          self.vaccines,
                                                          self.policy,
                                                          simulation_end_points[t],
                                                          time_points[t],
                                                          new_seeds[i],
                                                          2,
                                                          f"{seeds[i]}_",
                                                          "tests",
                                                          "tests")
        # Check if the new files are successfully created:
        for seed in seeds:
            for rep in range(2):
                for time in time_points[:-1]:
                    filename = f"{seed}_{rep + 1}_{self.city.cal.calendar[time].date()}_{self.policy_name}_sim_updated.json"
                    print(filename)
                    with self.subTest(filename=filename):
                        self.assertTrue(Path(filename).is_file())

                    filename = f"{seed}_{rep + 1}_{self.city.cal.calendar[time].date()}_{self.policy_name}_policy.json"
                    with self.subTest(filename=filename):
                        self.assertTrue(Path(filename).is_file())


if __name__ == '__main__':
    unittest.main()
