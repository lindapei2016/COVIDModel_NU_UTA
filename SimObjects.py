###############################################################################

# SimObjects.py

###############################################################################

import numpy as np

datetime_formater = "%Y-%m-%d %H:%M:%S"


###############################################################################
# Common simple functions:


def find_tier(thresholds, stat):
    """
    Calculate the new tier according to the tier statistics.
    :param thresholds: the tier thresholds.
    :param stat: the critical statistics that would determine the next tier.
    :return: the new tier.
    """
    counter = 0
    lb_threshold = 0
    for lt in thresholds:
        if stat >= lt:
            lb_threshold = counter
            counter += 1
            if counter == len(thresholds):
                break

    return lb_threshold

def find_tier_WA_case(thresholds_case, stat_case):
    """
    Calculate the new tier according to the tier statistics.
    :param thresholds: the tier thresholds.
    :param stat: the critical statistics that would determine the next tier.
    :return: the new tier.
    """

    if stat_case > thresholds_case[2]:
        return 2
    elif stat_case < thresholds_case[1]:
        return 0
    else:
        return 1

def find_tier_WA_hosp(thresholds_hosp, stat_hosp):
    """
    Calculate the new tier according to the tier statistics.
    :param thresholds: the tier thresholds.
    :param stat: the critical statistics that would determine the next tier.
    :return: the new tier.
    """

    if stat_hosp > thresholds_hosp[2]:
        return 2
    elif stat_hosp < thresholds_hosp[1]:
        return 0
    else:
        return 1

def find_tier_WA_icu(thresh_ICU, stat_ICU):
    """
    Calculate the new tier according to the tier statistics.
    :param thresholds: the tier thresholds.
    :param stat: the critical statistics that would determine the next tier.
    :return: the new tier.
    """

    if stat_ICU > thresh_ICU[1]:
        return 2
    else:
        return 0

def find_tier_WA(tier_case, tier_hosp, tier_icu):
    """
    Calculate the new tier according to the tier statistics.
    :param thresholds: the tier thresholds.
    :param stat: the critical statistics that would determine the next tier.
    :return: the new tier.
    """

    if tier_icu == 2:
        return 2
    elif tier_case == 2 and tier_hosp == 2:
        return 2
    elif tier_case == 1 or tier_hosp == 1:
        return 0
    else:
        return 1

def find_ind_WA(tier_case, tier_hosp, tier_icu):
    """
        Calculate the active indicator based on tier statistics.
        :param thresholds: the tier thresholds.
        :param stat: the critical statistics that would determine the next tier.
        :return: the active indicator (1 = cases, 2 = hosp, 3 = icu, 4 = case and hosp).
    """
    if tier_icu == 2:
        return 3
    elif tier_case > tier_hosp:
        return 1
    elif tier_case < tier_hosp:
        return 2
    else:
        return 4




###############################################################################
# Modules:

class CDCTierPolicy:
    """
    CDC's community levels.
    CDC system includes three tiers. Green and orange stages are deprecated but maintained
    for code consistency with our system.
    CDC system includes three indicators;
        1. Case counts (new COVID-19 Cases Per 100,000 people in the past 7 days.),
        2. Hospital admissions (new COVID-19 admissions per 100,000 population (7-day total)),
        3. Percent hospital beds (percent of staffed inpatient beds occupied by COVID-19 patients (7-day average)).

        Depending on the case counts thresholds, the hospital admissions and percent hospital beds thresholds
    changes. I think of this as follows, when there is a surge of cases the other two thresholds are stricter
    but when there is no surge of cases the other two thresholds are more relax.
        The history for the case counts is written as self.surge_history to indicate which set of thresholds are active
    for hospital admissions and percent hospital beds.
        The new tier will be stricter of what hospital admission and percent hospital beds thresholds are indicating.
    """

    def __init__(self, instance,
                 tiers,
                 case_threshold,
                 hosp_adm_thresholds,
                 staffed_bed_thresholds,
                 percentage_cases=0.4):
        """
        :param instance:
        :param tiers: (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    }
        :param case_threshold: (Surge threshold).
        :param hosp_adm_thresholds: (dict of dict) thresholds
                   { non_surge : thresholds level when case counts is below the case threshold
                    surge : thresholds level when case counts is above the case threshold
                   }
        :param staffed_bed_thresholds: (dict of dict) similar entries as the hosp_adm_thresholds.
        :param percentage_cases: the CDC system uses total case counts as an indicators. However, we don't have a direct
        interpretation of case counts in the model. We estimate the real total case count as some percentage of people
        entering symptomatic compartment (ToIY). We use percentage_case to adjust ToIY.
        """
        self._instance = instance
        self.tiers = tiers.tier
        self.case_threshold = case_threshold
        self.hosp_adm_thresholds = hosp_adm_thresholds
        self.staffed_bed_thresholds = staffed_bed_thresholds
        self.percentage_cases = percentage_cases
        self.tier_history = None
        self.surge_history = None
        self.active_indicator_history = []

    def reset(self):
        self.tier_history = None
        self.surge_history = None
        self.active_indicator_history = []

    def __repr__(self):
        return f"CDC_{self.case_threshold}_{self.hosp_adm_thresholds['non_surge'][0]}_{self.staffed_bed_thresholds['non_surge'][0]}_{self.percentage_cases}"

    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]
            self.surge_history = [None for i in range(t)]
            self.active_indicator_history = [None for i in range(t)]
        if len(self.tier_history) > t:
            return

        ToIHT = np.array(ToIHT)
        IH = np.array(IH)
        ToIY = np.array(ToIY)
        ICU = np.array(ICU)

        # Compute daily admissions moving sum
        moving_avg_start = np.maximum(0, t - self._instance.moving_avg_len)
        hos_adm_total = ToIHT.sum((1, 2))
        hosp_adm_sum = 100000 * hos_adm_total[moving_avg_start:].sum() / N.sum((0, 1))

        # Compute 7-day total new cases:
        N = self._instance.N
        ToIY_total = ToIY.sum((1, 2))
        ToIY_total = ToIY_total[moving_avg_start:].sum() * 100000 / np.sum(N, axis=(0, 1))

        # Compute 7-day average percent of COVID beds:
        IH_total = IH.sum((1, 2)) + ICU.sum((1, 2))
        IH_avg = IH_total[moving_avg_start:].mean() / self._instance.hosp_beds

        current_tier = self.tier_history[t - 1]

        # Decide on the active hospital admission and staffed bed thresholds depending on the estimated
        # case count level:
        if ToIY_total * self.percentage_cases < self.case_threshold:
            hosp_adm_thresholds = self.hosp_adm_thresholds["non_surge"]
            staffed_bed_thresholds = self.staffed_bed_thresholds["non_surge"]
            surge_state = 0
        else:
            hosp_adm_thresholds = self.hosp_adm_thresholds["surge"]
            staffed_bed_thresholds = self.staffed_bed_thresholds["surge"]
            surge_state = 1

        # find hosp admission new tier:
        hosp_adm_tier = find_tier(hosp_adm_thresholds, hosp_adm_sum)

        # find staffed bed new tier:
        staffed_bed_tier = find_tier(staffed_bed_thresholds, IH_avg)

        # choose the stricter tier among tiers the two indicators suggesting:
        new_tier = max(hosp_adm_tier, staffed_bed_tier)
        # keep track of the active indicator for indicator statistics:
        if hosp_adm_tier > staffed_bed_tier:
            active_indicator = 0
        elif hosp_adm_tier < staffed_bed_tier:
            active_indicator = 1
        else:
            active_indicator = 2

        if current_tier != new_tier:  # bump to the next tier
            t_end = t + self.tiers[new_tier]["min_enforcing_time"]
        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]
        self.surge_history += [surge_state for i in range(t_end - t)]
        self.active_indicator_history += [active_indicator for i in range(t_end - t)]


class MultiTierPolicy:
    """
    A multi-tier policy allows for multiple tiers of lock-downs.
    Attrs:
        tiers (list of dict): a list of the tiers characterized by a dictionary
            with the following entries:
                {
                    "transmission_reduction": float [0,1)
                    "cocooning": float [0,1)
                    "school_closure": int {0,1}
                }

        lockdown_thresholds (list of list): a list with the thresholds for every
            tier. The list must have n-1 elements if there are n tiers. Each threshold
            is a list of values for evert time step of simulation.
        community_transmission: (deprecated) CDC's old community transmission threshold for staging.
                                Not in use anymore.
    """

    def __init__(self, instance, tiers, lockdown_thresholds, community_transmission):
        self._instance = instance
        self.tiers = tiers.tier

        self.community_transmission = community_transmission
        self.lockdown_thresholds = lockdown_thresholds
        self.tier_history = None

    def reset(self):
        self.tier_history = None

    def __repr__(self):
        return str(self.lockdown_thresholds)

    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        """
        Function that makes an instance of a policy a callable.

        """
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]

        if len(self.tier_history) > t:
            return

        ToIHT = np.array(ToIHT)
        ToIY = np.array(ToIY)

        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - self._instance.moving_avg_len)

        if len(ToIHT) > 0:
            criStat_total = ToIHT.sum((1, 2))
            criStat_avg = criStat_total[moving_avg_start:].mean()
        else:
            criStat_avg = 0

        # Compute new cases per 100k:
        if len(ToIY) > 0:
            ToIY_avg = (
                    ToIY.sum((1, 2))[moving_avg_start:].sum()
                    * 100000
                    / np.sum(N, axis=(0, 1))
            )
        else:
            ToIY_avg = 0
        # find new tier
        new_tier = find_tier(self.lockdown_thresholds, criStat_avg)

        # Check if community_transmission rate is included:
        if self.community_transmission == "blue":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else:
                        new_tier = 2
            elif new_tier == 1:
                if ToIY_avg > 10:
                    new_tier = 2
        elif self.community_transmission == "green":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else:
                        new_tier = 2

        if len(self.tier_history) > 0:
            current_tier = self.tier_history[t - 1]
        else:
            current_tier = new_tier

        if current_tier != new_tier:  # bump to the next tier
            t_end = t + self.tiers[new_tier]["min_enforcing_time"]
        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]


class MultiTierPolicyWA:
    """
    A multi-tier policy allows for multiple tiers of lock-downs.
    Attrs:
        tiers (list of dict): a list of the tiers characterized by a dictionary
            with the following entries:
                {
                    "transmission_reduction": float [0,1)
                    "cocooning": float [0,1)
                    "school_closure": int {0,1}
                }

        lockdown_thresholds_case (list of list): a list with the thresholds for every
            tier (case counts). The list must have n-1 elements if there are n tiers. Each threshold
            is a list of values for evert time step of simulation.
        lockdown_thresholds_hosp (list of list): a list with the thresholds for every
            tier (hospitalizations). The list must have n-1 elements if there are n tiers. Each threshold
            is a list of values for evert time step of simulation.
        community_transmission: (deprecated) CDC's old community transmission threshold for staging.
                                Not in use anymore.
    """

    def __init__(self, instance, tiers, lockdown_thresholds_case, lockdown_thresholds_hosp, lockdown_thresholds_ICU):
        self._instance = instance
        self.tiers = tiers.tier
        self.lockdown_thresholds_case = lockdown_thresholds_case
        self.lockdown_thresholds_hosp = lockdown_thresholds_hosp
        self.lockdown_thresholds_ICU = lockdown_thresholds_ICU
        self.tier_history = None
        self.active_indicator_history = []

    def reset(self):
        self.tier_history = None
        self.active_indicator_history = []

    def __repr__(self):
        return str(self.lockdown_thresholds_case)

    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        """
        Function that makes an instance of a policy a callable.

        """
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]
            self.active_indicator_history = [None for i in range(t)]

        if len(self.tier_history) > t:
            return

        ToIY = np.array(ToIY)
        ToIHT = np.array(ToIHT)
        ICU = np.array(ICU)

        # Set moving average ranges
        moving_avg_start_hosp = np.maximum(0, t - 7)
        moving_avg_start_case = np.maximum(0, t - 14)

        # find hospitalizations 7 day avg per 100k
        if len(ToIHT) > 0:
            criStatIHT_total = ToIHT.sum((1, 2))
            criStatIHT_avg = criStatIHT_total[moving_avg_start_hosp:].sum()
            ToIHT_avg = 100000 * criStatIHT_avg / N.sum((0, 1))
        else:
            criStatIHT_avg = 0
            ToIHT_avg = 100000 * criStatIHT_avg / N.sum((0, 1))

        # Compute 14 day case avg per 100k:
        if len(ToIY) > 0:
            ToIY_avg = (
                    ToIY.sum((1, 2))[moving_avg_start_case:].sum()
                    * 100000
                    / np.sum(N, axis=(0, 1))
            )
        else:
            ToIY_avg = 0

        # Compute % of ICU capacity reached:
        ICU_amt = ICU.sum((1, 2))[t:].sum() / self._instance.icu

        # find new tier
        new_tier_case = find_tier_WA_case(self.lockdown_thresholds_case, ToIY_avg)
        new_tier_hosp = find_tier_WA_hosp(self.lockdown_thresholds_hosp, ToIHT_avg)
        new_tier_icu = find_tier_WA_icu(self.lockdown_thresholds_ICU, ICU_amt)
        new_tier = find_tier_WA(new_tier_case, new_tier_hosp, new_tier_icu)

        active_indicator = find_ind_WA(new_tier_case, new_tier_hosp, new_tier_icu)

        if len(self.tier_history) > 0:
            current_tier = self.tier_history[t - 1]
        else:
            current_tier = new_tier

        if current_tier != new_tier:  # bump to the next tier
            t_end = t + self.tiers[new_tier]["min_enforcing_time"]
        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]
        self.active_indicator_history += [active_indicator for i in range(t_end - t)]


class VaccineGroup:
    def __init__(
            self,
            v_name,
            v_beta_reduct,
            v_tau_reduct,
            v_pi_reduct,
            N, I0, A, L, step_size):

        """
        Define each vaccine status as a group. Define each set of compartments for vaccine group.
        """

        self.v_beta_reduct = v_beta_reduct
        self.v_tau_reduct = v_tau_reduct
        self.v_pi_reduct = v_pi_reduct
        self.v_name = v_name

        if self.v_name == "unvax":
            self.v_in = ()
            self.v_out = ("v_first",)

        elif self.v_name == "first_dose":
            self.v_in = ("v_first",)
            self.v_out = ("v_second",)

        elif self.v_name == "second_dose":
            self.v_in = ("v_second", "v_booster")
            self.v_out = ()

        else:
            self.v_in = ()
            self.v_out = ("v_booster",)

        self.N = N
        self.I0 = I0

        self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")
        self.tracking_vars = (
            "IYIH",
            "IYICU",
            "IHICU",
            "ToICU",
            "ToIHT",
            "ToICUD",
            "ToIYD",
            "ToIA",
            "ToIY",
            "ToRS",
            "ToSS",
            "ToR"
        )

        for attribute in self.state_vars:
            setattr(self, attribute, np.zeros((A, L)))
            setattr(self, "_" + attribute, np.zeros((step_size + 1, A, L)))

        for attribute in self.tracking_vars:
            setattr(self, attribute, np.zeros((A, L)))
            setattr(self, "_" + attribute, np.zeros((step_size, A, L)))

        if self.v_name == "unvax":
            # Initial Conditions (assumed)
            self.PY = self.I0
            self.R = 0
            self.S = self.N - self.PY - self.IY

        for attribute in self.state_vars:
            vars(self)["_" + attribute][0] = getattr(self, attribute)

    def variant_update(self, params, prev):
        """
        Update efficacy according to variant of concern efficacy
        """
        self.v_beta_reduct = self.v_beta_reduct * (1 - prev) + params[
            ('v_beta_reduct', self.v_name)]  # efficacy against infection.
        self.v_tau_reduct = self.v_tau_reduct * (1 - prev) + params[
            ('v_tau_reduct', self.v_name)]  # efficacy against symptomatic infection.

    def get_total_population(self, total_risk_groups):
        """
        :param total_risk_groups: total number of compartments for age-risk groups.
        :return: the total population in a certain vaccine group (S+E+IY+PY+..).
        """
        N = 0
        for attribute in self.state_vars:
            N += getattr(self, attribute)
        N = N.reshape((total_risk_groups, 1))
        return N
