###############################################################################

import numpy as np
import pandas as pd

# Work-in-progress

# KN Procedure
# See "A Fully Sequential Procedure for Indifference-Zone
#   Selection in Simulation" by Kim & Nelson (2001)

# Currently for CDC optimization, doing KN per-peak
#   and also across-peak -- take UNION of surviving policies
#   on each of the peaks and across-peak for further simulation
#   -- this also allows useful analysis such as seeing how
#   the best policy in peak i performs on peak j

# Example dataset to initialize KN is dataset of costs
#   -- see Script_CDCOptimization_FileParsing.py to see how
#   such a cost dataframe is generated and what its structure
#   is like

# Since we are interested in feasible policies (with low
#   ICU violation probability), can reduce the number of
#   policies evaluated by KN by including in dataset only policies
#   that are estimated to be feasible (for that specific peak
#   or for across-peak average).

# IMPORTANT NOTE: num_original_policies should be the same
#   (for a given peak or across-peak average)
#   even if KN is called again on a smaller dataset / subset,
#   unless previous replications are thrown out. This note
#   is important to preserve the statistical guarantee.

# TODO: something confusing is happening with the dataframe
#   column names -- full_df uses integers for indexing and
#   cost_dfs_per_peak has columns that are strings -- figure this out
#   and make more consistent

# KN statistical guarantees

# Probability of correct selection (PCS)
# ^ theoretically proven
# Under the assumption that the true best is iz_param
#   away from the second best, KN returns the true best with
#   probability 1 - alpha.

# Probability of good selection (PGS)
# ^ empirically seems to be the case
# KN returns either the true best or a policy within
#   iz_param of the true best, with probability 1 - alpha.

class KN():

    '''
    Fully sequential ranking and selection procedure to obtain
        the system (policy) with MINIMUM performance.

    Uses pairwise comparisons to eliminate inferior policies.

    Assumes independent and normally distributed output to get
        theoretical statistical guarantee, but empirically is
        decently robust to nonnormality. Can also use batch means
        to get approximately normally distributed output.
    '''

    def __init__(self,
                 dataset,
                 iz_param,
                 num_original_policies,
                 num_reference_policies=100,
                 alpha=0.05):

        '''
        :param dataset [DataFrame] -- columns are [str] corresponding to
            policy IDs, rows are scalar values containing output
            (nth row corresponds to nth i.i.d. replication)
            -- values in dataset are used to conduct optimization
        :param iz_param [positive scalar] -- differences less than
            iz_param are considered practically insignificant.
        :param num_original_policies [integer] -- total number of policies
        :param num_reference_policies [integer] -- number of policies
            used to compare against every other policy. Normally this is
            equal to k, so that there are O(k^2) comparisons and every
            policy is compared to every other policy. This is prohibitively
            slow in a serial setting, so to speed up comparisons we can
            conduct a subset of comparisons, by comparing policies to
            the top (lowest sample means) num_reference_policies
        :param alpha [float] -- value in (0,1-1/num_original_policies) --
            error rate. Probability of correct selection is 1-alpha.
        '''

        self.dataset = dataset
        self.iz_param = iz_param
        self.num_original_policies = num_original_policies
        self.num_reference_policies = min(num_reference_policies, len(dataset.columns))
        self.alpha = alpha

        # Recommended value of constant
        self.c = 1

        # Initial sample size (number of replications for sample variance estimation)
        self.n0 = 100

        # Compute parameters for pairwise
        self.eta = (1 / 2) * (((2 * self.alpha) /
                               (self.num_original_policies - 1)) ** (-2 / (self.n0 - 1)) - 1)
        self.hsquared = 2 * self.c * self.eta * (self.n0 - 1)

        # Initialize surviving policies and eliminated policies
        self.surviving_policies = self.dataset.columns
        self.eliminated_policies = []

        # Create dictionary that holds all the pairwise variances
        self.var_of_diff_dict = {}
        self.compute_all_var_of_diff()

    def compute_all_var_of_diff(self):

        var_of_diff_dict = {}

        # Save instance attributes as local variables to speed up access
        dataset = self.dataset
        dataset_surviving_policies_mean = dataset[self.surviving_policies].mean()
        n0 = self.n0

        for i in range(self.num_reference_policies):

            reference = dataset_surviving_policies_mean.sort_values().index[i]

            reference_output = dataset[reference][:n0]
            reference_mean = reference_output.mean()

            for policy_id in self.surviving_policies:

                var_of_diff_dict[(policy_id, reference)] = np.sum((dataset[policy_id][:n0] - reference_output - (dataset[policy_id][:n0].mean() - reference_mean)) ** 2) / (n0 - 1)

        self.var_of_diff_dict = var_of_diff_dict

    def pairwise_comparisons(self, rep_number):

        '''
        rep_number [int] -- replication number at which to do pairwise comparisons
            -- must be larger than n0
        '''

        # Save instance attributes as local variables to speed up access
        dataset = self.dataset
        dataset_surviving_policies_mean = dataset[self.surviving_policies][:rep_number].mean()
        iz_param = self.iz_param
        n0 = self.n0
        c = self.c
        hsquared = self.hsquared
        var_of_diff_dict = self.var_of_diff_dict

        new_eliminated_policies = []

        for i in range(self.num_reference_policies):

            reference = dataset_surviving_policies_mean.sort_values().index[i]

            # Use only the replications up to rep_number
            reference_output = dataset[reference][:rep_number]
            reference_mean = reference_output.mean()

            for policy_id in self.surviving_policies:

                    if policy_id == reference:
                        continue
                    else:
                        wiggle_room = max(0, (iz_param / (2 * c * rep_number)) *
                                          (hsquared * var_of_diff_dict[(policy_id, reference)] / (iz_param ** 2) - rep_number))

                        if dataset_surviving_policies_mean[policy_id] > reference_mean + wiggle_room:
                            new_eliminated_policies.append(policy_id)

        self.eliminated_policies = set(self.eliminated_policies).union(set(new_eliminated_policies))

        self.surviving_policies = set(self.surviving_policies).difference(self.eliminated_policies)

# Old
# num_feasible_policies_100_reps = [7037, 7399, 6104, 6002]

#         # The way I set this up is that we simulate the same set of policies per peak, so
#         #   we are just grabbing the columns from the 0th (1st) peak
#         non_eliminated_policies = set(stage3_days_dict[str(0)].columns.astype(int)).difference(
#             set(eliminated_policies_ix))
#         feasible_policies = set(feasible_policies_df_current_peak.index)
#         non_eliminated_feasible_policies = list(non_eliminated_policies.intersection(feasible_policies))
#
#         np.savetxt("non_eliminated_feasible_policies_peak" + str(peak) + ".csv",
#                    np.array(non_eliminated_feasible_policies).astype("int"))
#
# breakpoint()

###############################################################################

# This is stuff for Rinott -- I will clean this up later -- LP

# reps_needed = []
#
# # peak 3 is across-peak!
# # for peak in np.arange(4):
# for peak in np.arange(4):
#
#     # Peak-specific k, eta, hsquared
#     k = num_feasible_policies_100_reps[peak]
#
#     Z = np.random.normal(size=(int(1e5), k - 1))
#     Y = np.random.chisquare(df=(n0 - 1), size=(int(1e5), k - 1))
#     C = np.random.chisquare(df=(n0 - 1), size=int(1e5))
#     C = np.reshape(C, (len(C), 1))
#     Cmat = np.repeat(C, k - 1, axis=1)
#     denom = np.sqrt((n0 - 1) * (1 / Y + 1 / Cmat))
#     H = np.sort(np.max(Z * denom, axis=1))
#     rinott_constant = np.quantile(H, 1 - 0.05 / 2.0)
#
#     subset_policies_ix = pd.read_csv("w503_5000reps_non_eliminated_feasible_policies_peak" + str(peak) + ".csv",
#                                      header=None)
#     subset_policies_ix = np.array(subset_policies_ix, dtype="int")
#
#     for ix in subset_policies_ix:
#         # breakpoint()
#         if peak <= 2:
#             var = np.sum((cost_dfs_per_peak[peak][ix][:n0] - np.average(cost_dfs_per_peak[peak][ix][:n0])) ** 2) / (
#                     n0 - 1)
#         elif peak == 3:
#             current_ix_costs = (cost_dfs_per_peak[0][ix][:n0] +
#                                 cost_dfs_per_peak[1][ix][:n0] +
#                                 cost_dfs_per_peak[2][ix][:n0]) / 3
#             var = np.sum((current_ix_costs - np.average(current_ix_costs[:n0])) ** 2) / (n0 - 1)
#
#         if peak == 3:
#             reps_needed.append(rinott_constant ** 2 * var / iz_param ** 2)
#
# breakpoint()

