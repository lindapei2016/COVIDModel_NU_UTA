###############################################################################

# SimModel.py
# This module contains the SimReplication class. Each instance holds
#   a City instance, an EpiSetup instance, a Vaccine instance,
#   VaccineGroup instance(s), and optionally a MultiTierPolicy instance.

###############################################################################

import numpy as np
from SimObjects import VaccineGroup
import copy
import datetime as dt

datetime_formater = "%Y-%m-%d %H:%M:%S"
import time


###############################################################################

def discrete_approx(rate, timestep):
    return 1 - np.exp(-rate / timestep)


class SimReplication:
    def __init__(self, instance, vaccine, policy, rng_seed):
        """
        :param instance: [obj] instance of City class
        :param vaccine: [obj] instance of Vaccine class
        :param policy: [obj] instance of MultiTierPolicy
            class, or [None]
        :param rng_seed: [int] or [None] either a
            non-negative integer, -1, or None
        """

        # Save arguments as attributes
        self.instance = instance
        self.vaccine = vaccine
        self.policy = policy
        self.rng_seed = rng_seed

        self.step_size = self.instance.config["step_size"]
        self.t_historical_data_end = len(self.instance.real_IH_history)

        # A is the number of age groups
        # L is the number of risk groups
        # Many data arrays in the simulation have dimension A x L
        A = self.instance.A
        L = self.instance.L

        # Important steps critical to initializing a replication
        # Initialize random number generator
        # Sample random parameters
        # Create new VaccineGroup instances
        self.init_rng()
        self.init_epi()
        self.init_vaccine_groups()

        # Initialize data structures to track ICU, IH, ToIHT, ToIY
        # These statistics or data we look at changes a lot over time
        # better to keep them in a list to modify.
        self.history_vars = ("ICU",
                             "IH",
                             "D",
                             "R",
                             "ToIHT",
                             "ToIY",
                             "ToICUD",
                             "ToIYD",
                             "ToRS",
                             "ToSS",
                             "S",
                             "SE" # Apr. 19, 2023, Sonny, track flow from S to E
                             )

        # Keep track of the total number of immune evasion:
        self.ToRS_immune = []  # waned natural immunity
        self.ToSS_immune = []  # waned vaccine induced immunity

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        # The next t that is simulated (automatically gets updated after simulation)
        # This instance has simulated up to but not including time next_t
        self.next_t = 0

        # Tuples of variable names for organization purposes
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
            "SE" # Apr. 19, 2023, Sonny, add tracking variable to track number of people exposed
        ) # note that both VaccineGroup and SimReplication define tracking_vars and state_vars,
        # these may cause conflict if they are inconsistent in two objects
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apr. 19, 2023, Sonny, tracking wastewater signal
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.wastewater_viral_load = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def init_rng(self):
        """
        Assigns self.rng to a newly created random number generator
            initialized with seed self.rng_seed.
        If self.rng_seed is None (not specified) or -1, then self.rng
            is set to None, so no random number generator is created
            and the simulation will run deterministically.

        :return: [None]
        """

        if self.rng_seed is not None:
            if self.rng_seed >= 0:
                self.rng = np.random.default_rng(self.rng_seed)
            else:
                self.rng = None
        else:
            self.rng = None

    def init_epi(self):
        """
        Assigns self.epi_rand to an instance of EpiSetup that
            inherits some attribute values (primitives) from
            the "base" object self.instance.base_epi and
            also generates new values for other attributes.
        These new values come from randomly sampling
            parameters using the random number generator
            self.rng.
        If no random number generator is given, these
            randomly sampled parameters are set to the
            expected value from their distributions.
        After random sampling, some basic parameters
            are updated.

        :return: [None]
        """

        # Create a deep copy of the "base" EpiSetup instance
        #   to inherit some attribute values (primitives)
        epi_rand = copy.deepcopy(self.instance.base_epi)

        # On this copy, sample random parameters and
        #   do some basic updating based on the results
        #   of this sampling
        epi_rand.sample_random_params(self.rng)
        epi_rand.setup_base_params()

        # Assign self.epi_rand to this copy
        self.epi_rand = epi_rand

    def init_vaccine_groups(self):
        """
        Creates 4 vaccine groups:
            group 0 / "unvax": unvaccinated
            group 1 / "first_dose": partially vaccinated
            group 2 / "second_dose": fully vaccinated
            group 3 / "waned": waning efficacy

        We assume there is one type of vaccine with 2 doses.
        After 1 dose, individuals move from group 0 to 1.
        After 2 doses, individuals move from group 1 to group 2.
        After efficacy wanes, individuals move from group 2 to group 3.
        After booster shot, individuals move from group 3 to group 2.
                 - one type of vaccine with two-doses

        :return: [None]
        """

        N = self.instance.N
        I0 = self.instance.I0
        A = self.instance.A
        L = self.instance.L
        step_size = self.instance.config["step_size"]

        self.vaccine_groups = []

        self.vaccine_groups.append(VaccineGroup("unvax", 0, 0, 0, N, I0, A, L, step_size))
        for key in self.vaccine.beta_reduct:
            self.vaccine_groups.append(
                VaccineGroup(
                    key,
                    self.vaccine.beta_reduct[key],
                    self.vaccine.tau_reduct[key],
                    self.vaccine.pi_reduct[key],
                    N, I0, A, L, step_size
                )
            )
        self.vaccine_groups = tuple(self.vaccine_groups)

    def compute_cost(self):
        """
        If a policy is attached to this replication, return the
            cumulative cost of its enforced tiers (from
            the end of the historical data time period to the
            current time of the simulation).
        If no policy is attached to this replication, return
            None.

        :return: [float] or [None] cumulative cost of the
            attached policy's enforced tiers (returns None
            if there is no attached policy)
        """

        if self.policy:
            return sum(
                self.policy.tiers[i]["daily_cost"]
                for i in self.policy.tier_history
                if i is not None
            )
        else:
            return None

    def compute_feasibility(self):
        """
        If a policy is attached to this replication, return
            True/False if the policy is estimated to be
            feasible (from the end of the historical data time period
            to the current time of the simulation).
        If no policy is attached to this replication or the
            current time of the simulation is still within
            the historical data time period, return None.

        :return: [Boolean] or [None] corresponding to whether
            or not the policy is estimated to be feasible
        """

        if self.policy is None:
            return None
        elif self.next_t < self.t_historical_data_end:
            return None

        # Check whether ICU capacity has been violated
        if np.any(
                np.array(self.ICU_history).sum(axis=(1, 2))[self.t_historical_data_end:]
                > self.instance.icu
        ):
            return False
        else:
            return True

    def compute_rsq(self):
        """
        Return R-squared type statistic based on historical hospital
            data (see pg. 10 in Yang et al. 2021), comparing
            thus-far-simulated hospital numbers (starting from t = 0
            up to the current time of the simulation) to the
            historical data hospital numbers (over this same time
            interval).

        Note that this statistic is not exactly R-squared --
            and as a result it takes values outside of [-1, 1].

        :return: [float] current R-squared value
        """

        f_benchmark = self.instance.real_IH_history

        IH_sim = np.array(self.ICU_history) + np.array(self.IH_history)
        IH_sim = IH_sim.sum(axis=(2, 1))
        IH_sim = IH_sim[: self.t_historical_data_end]

        if self.next_t < self.t_historical_data_end:
            IH_sim = IH_sim[: self.next_t]
            f_benchmark = f_benchmark[: self.next_t]

        rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark)) ** 2)) / sum(
            (np.array(f_benchmark) - np.mean(np.array(f_benchmark))) ** 2
        )

        return rsq

    def simulate_time_period(self, time_end, fixed_kappa_end_date=0):

        """
        Advance the simulation model from time_start up to
            but not including time_end.

        Calls simulate_t as a subroutine for each t between
            time_start and self.next_t, the last point at which it
            left off.

        Note that if a simulation replication is being run at timepoints
        t > fixed_kappa_end_date, there must be a MultiTierPolicy attached.

        :param time_end: [int] nonnegative integer -- time t (number of days)
            to simulate up to.
        :param fixed_kappa_end_date: the end date for fixed transmission
            reduction (kappa). The staged-alert policy will be called after
            this date. This is generally the end date of historical data, too.
            Make sure the transmission.csv file has reduction values until and
            including fixed_kappa_end_date. If fixed_kappa_end_date is 0,
            the staged-alert policy will be called from the start of the
            simulation date. If fixed_kappa_end_date is 0,
            then staged-alert policy dictates fixed transmission reduction
            (kappa), even if historical transmission reduction data
            is available.
        :return: [None]
        """

        # Begin where the simulation last left off
        time_start = self.next_t

        # Call simulate_t as subroutine from time_start to time_end
        for t in range(time_start, time_end):

            self.next_t += 1

            self.simulate_t(t, fixed_kappa_end_date)
            # print(t)

            A = self.instance.A
            L = self.instance.L

            # Clear attributes in self.state_vars + self.tracking_vars
            #   since these only hold the most recent values
            for attribute in self.state_vars + self.tracking_vars:
                setattr(self, attribute, np.zeros((A, L)))

            # Update attributes in self.state_vars + self.tracking_vars --
            #   their values are the sums of the same attributes
            #   across all vaccine groups

            for attribute in self.state_vars + self.tracking_vars:
                sum_across_vaccine_groups = 0
                for v_group in self.vaccine_groups:
                    assert (getattr(v_group, attribute) > -1e-2).all(), \
                        f"fPop negative value of {getattr(v_group, attribute)} " \
                        f"on compartment {v_group.v_name}.{attribute} at time {self.instance.cal.calendar[t]}, {t}"

                    sum_across_vaccine_groups += getattr(v_group, attribute)

                    # LP note -- commented this assertion because we
                    #   already have total_imbalance assertion later --
                    #   might want to uncomment or consider making this a toggle
                    #   for debugging purposes
                    # assert (getattr(v_group, attribute) > -1e-2).all(), \
                    #     f"fPop negative value of {getattr(v_group, attribute)} " \
                    #     f"on compartment {v_group.v_name}.{attribute} at time " \
                    #     f"{self.instance.cal.calendar[t]}, {t}"

                setattr(self, attribute, sum_across_vaccine_groups)

            # We are interested in the history, not just current values, of
            #   certain variables -- save these current values
            for attribute in self.history_vars:
                getattr(self, f"{attribute}_history").append(getattr(self, attribute))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Apr. 19, 2023, Sonny, simulating wastewater signal
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.instance.flag_wastewater_sim is True:
                _w = 0
                for i in range(self.instance.viral_shedding_profile['num_days']):
                    if t - i >= time_start:
                        _w += np.sum(self.instance.viral_shedding_profile["shedding_function_val"][i]
                                     * self.SE_history[t - i])
                self.wastewater_viral_load.append(_w)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # LP note -- might want to comment if not debugging and
            #   want speedier simulation
            total_imbalance = np.sum(
                self.S
                + self.E
                + self.IA
                + self.IY
                + self.R
                + self.D
                + self.PA
                + self.PY
                + self.IH
                + self.ICU
            ) - np.sum(self.instance.N)

            assert (
                    np.abs(total_imbalance) < 1e-2
            ), f"fPop unbalanced {total_imbalance} at time {self.instance.cal.calendar[t]}, {t}"

    def simulate_t(self, t_date, fixed_kappa_end_date):

        """
        Advance the simulation 1 timepoint (day).

        Subroutine called in simulate_time_period

        :param t_date: [int] nonnegative integer corresponding to
            current timepoint to simulate.
        :param fixed_kappa_end_date: see simulate_time_period
        :return: [None]
        """

        # Get dimensions (number of age groups,
        #   number of risk groups,
        A = self.instance.A
        L = self.instance.L
        N = self.instance.N

        calendar = self.instance.cal.calendar

        t = t_date

        epi = copy.deepcopy(self.epi_rand)

        if t <= fixed_kappa_end_date:
            # If the transmission reduction is fixed don't call the policy object.
            phi_t = epi.effective_phi(
                self.instance.cal.schools_closed[t],
                self.instance.cal.fixed_cocooning[t],
                self.instance.cal.fixed_transmission_reduction[t],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )
        else:
            self.policy(
                t,
                self.ToIHT_history,
                self.IH_history,
                self.ToIY_history,
                self.ICU_history,
            )
            current_tier = self.policy.tier_history[t]
            phi_t = epi.effective_phi(
                self.policy.tiers[current_tier]["school_closure"],
                self.policy.tiers[current_tier]["cocooning"],
                self.policy.tiers[current_tier]["transmission_reduction"],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )

        if calendar[t] >= self.instance.variant_start:
            days_since_variant_start = (calendar[t] - self.instance.variant_start).days
            new_epi_params_coef, new_vax_params, var_prev = self.instance.variant_pool.update_params_coef(
                days_since_variant_start, epi.sigma_E)
            # Assume immune evasion starts with the variants.
            immune_evasion = self.instance.variant_pool.immune_evasion(epi.immune_evasion, calendar[t])
            for v_groups in self.vaccine_groups:
                if v_groups.v_name != 'unvax':
                    v_groups.variant_update(new_vax_params, var_prev)
            epi.variant_update_param(new_epi_params_coef)
        else:
            immune_evasion = 0

        if self.instance.otherInfo == {}:
            rd_start = dt.datetime.strptime(
                self.instance.config["rd_start"], datetime_formater
            )
            rd_end = dt.datetime.strptime(
                self.instance.config["rd_end"], datetime_formater
            )
            if self.instance.cal.calendar.index(
                    rd_start
            ) < t <= self.instance.cal.calendar.index(rd_end):
                epi.update_icu_params(self.instance.config["rd_rate"])
        else:
            epi.update_icu_all(t, self.instance.otherInfo)

        step_size = self.step_size
        get_binomial_transition_quantity = self.get_binomial_transition_quantity

        rate_E = discrete_approx(epi.sigma_E, step_size)

        rate_IAR = discrete_approx(np.full((A, L), epi.gamma_IA), step_size)
        rate_PAIA = discrete_approx(np.full((A, L), epi.rho_A), step_size)
        rate_PYIY = discrete_approx(np.full((A, L), epi.rho_Y), step_size)
        rate_IHICU = discrete_approx(epi.nu * epi.etaICU, step_size)
        rate_IHR = discrete_approx((1 - epi.nu) * epi.gamma_IH, step_size)
        rate_ICUD = discrete_approx(epi.nu_ICU * epi.mu_ICU, step_size)
        rate_ICUR = discrete_approx((1 - epi.nu_ICU) * epi.gamma_ICU, step_size)
        rate_immune = discrete_approx(immune_evasion, step_size)

        for _t in range(step_size):
            # Dynamics for dS

            for v_groups in self.vaccine_groups:
                dSprob_sum = np.zeros((5, 2))

                for v_groups_temp in self.vaccine_groups:
                    # Vectorized version for efficiency. For-loop version commented below
                    temp1 = (
                            np.matmul(np.diag(epi.omega_PY), v_groups_temp._PY[_t, :, :])
                            + np.matmul(np.diag(epi.omega_PA), v_groups_temp._PA[_t, :, :])
                            + epi.omega_IA * v_groups_temp._IA[_t, :, :]
                            + epi.omega_IY * v_groups_temp._IY[_t, :, :]
                    )

                    temp2 = np.sum(N, axis=1)[np.newaxis].T
                    temp3 = np.divide(
                        np.multiply(epi.beta * phi_t / step_size, temp1), temp2
                    )

                    dSprob = np.sum(temp3, axis=(2, 3))
                    dSprob_sum = dSprob_sum + dSprob

                if v_groups.v_name in {"second_dose"}:
                    # If there is immune evasion, there will be two outgoing arc from S_vax. Infected people will
                    # move to E compartment. People with waned immunity will go the S_waned compartment.
                    # _dS: total rate for leaving S compartment.
                    # _dSE: adjusted rate for entering E compartment. Sonny note: "used for wastewater"
                    # _dSR: adjusted rate for entering S_waned (self.vaccine_groups[3]._S) compartment.
                    _dSE = get_binomial_transition_quantity(
                        v_groups._S[_t],
                        (1 - v_groups.v_beta_reduct) * dSprob_sum,
                    )
                    _dSR = get_binomial_transition_quantity(v_groups._S[_t], rate_immune)
                    _dS = _dSR + _dSE

                    E_out = get_binomial_transition_quantity(v_groups._E[_t], rate_E)
                    v_groups._E[_t + 1] = v_groups._E[_t] + _dSE - E_out

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Apr. 19, 2023, Sonny, track SE
                    v_groups._SE[_t] = _dSE
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    self.vaccine_groups[3]._S[_t + 1] = (
                            self.vaccine_groups[3]._S[_t + 1] + _dSR
                    )
                    v_groups._ToSS[_t] = _dSR
                else:
                    _dS = get_binomial_transition_quantity(
                        v_groups._S[_t], (1 - v_groups.v_beta_reduct) * dSprob_sum
                    )
                    # Dynamics for E
                    E_out = get_binomial_transition_quantity(v_groups._E[_t], rate_E)
                    v_groups._E[_t + 1] = v_groups._E[_t] + _dS - E_out

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Apr. 19, 2023, Sonny, track SE
                    v_groups._SE[_t] = _dS
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    v_groups._ToSS[_t] = 0

                immune_escape_R = get_binomial_transition_quantity(v_groups._R[_t], rate_immune)
                self.vaccine_groups[3]._S[_t + 1] = self.vaccine_groups[3]._S[_t + 1] + immune_escape_R
                v_groups._ToRS[_t] = immune_escape_R
                v_groups._S[_t + 1] = v_groups._S[_t + 1] + v_groups._S[_t] - _dS

                # Dynamics for PY
                EPY = get_binomial_transition_quantity(
                    E_out, epi.tau * (1 - v_groups.v_tau_reduct)
                )
                PYIY = get_binomial_transition_quantity(v_groups._PY[_t], rate_PYIY)
                v_groups._PY[_t + 1] = v_groups._PY[_t] + EPY - PYIY

                # Dynamics for PA
                EPA = E_out - EPY
                PAIA = get_binomial_transition_quantity(v_groups._PA[_t], rate_PAIA)
                v_groups._PA[_t + 1] = v_groups._PA[_t] + EPA - PAIA

                # Dynamics for IA
                IAR = get_binomial_transition_quantity(v_groups._IA[_t], rate_IAR)
                v_groups._IA[_t + 1] = v_groups._IA[_t] + PAIA - IAR

                # Dynamics for IY
                rate_IYR = discrete_approx(
                    np.array(
                        [
                            [
                                (1 - epi.pi[a, l] * (1 - v_groups.v_pi_reduct)) * epi.gamma_IY * (1 - epi.alpha_IYD)
                                for l in range(L)
                            ]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )
                rate_IYD = discrete_approx(
                    np.array(
                        [
                            [(1 - epi.pi[a, l] * (1 - v_groups.v_pi_reduct)) * epi.gamma_IY * epi.alpha_IYD for l in
                             range(L)]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )
                IYR = get_binomial_transition_quantity(v_groups._IY[_t], rate_IYR)
                IYD = get_binomial_transition_quantity(v_groups._IY[_t] - IYR, rate_IYD)

                rate_IYH = discrete_approx(
                    np.array(
                        [
                            [(epi.pi[a, l]) * (1 - v_groups.v_pi_reduct) * epi.Eta[a] * epi.pIH for l in range(L)]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )
                rate_IYICU = discrete_approx(
                    np.array(
                        [
                            [(epi.pi[a, l]) * (1 - v_groups.v_pi_reduct) * epi.Eta[a] * (1 - epi.pIH) for l in range(L)]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )

                v_groups._IYIH[_t] = get_binomial_transition_quantity(
                    v_groups._IY[_t] - IYR - IYD, rate_IYH
                )
                v_groups._IYICU[_t] = get_binomial_transition_quantity(
                    v_groups._IY[_t] - IYR - IYD - v_groups._IYIH[_t], rate_IYICU
                )
                v_groups._IY[_t + 1] = (
                        v_groups._IY[_t]
                        + PYIY
                        - IYR
                        - IYD
                        - v_groups._IYIH[_t]
                        - v_groups._IYICU[_t]
                )

                # Dynamics for IH
                IHR = get_binomial_transition_quantity(v_groups._IH[_t], rate_IHR)
                v_groups._IHICU[_t] = get_binomial_transition_quantity(
                    v_groups._IH[_t] - IHR, rate_IHICU
                )
                v_groups._IH[_t + 1] = (
                        v_groups._IH[_t] + v_groups._IYIH[_t] - IHR - v_groups._IHICU[_t]
                )

                # Dynamics for ICU
                ICUR = get_binomial_transition_quantity(v_groups._ICU[_t], rate_ICUR)
                ICUD = get_binomial_transition_quantity(
                    v_groups._ICU[_t] - ICUR, rate_ICUD
                )
                v_groups._ICU[_t + 1] = (
                        v_groups._ICU[_t]
                        + v_groups._IHICU[_t]
                        - ICUD
                        - ICUR
                        + v_groups._IYICU[_t]
                )
                v_groups._ToICU[_t] = v_groups._IYICU[_t] + v_groups._IHICU[_t]
                v_groups._ToIHT[_t] = v_groups._IYICU[_t] + v_groups._IYIH[_t]

                # Dynamics for R
                v_groups._R[_t + 1] = (v_groups._R[_t] + IHR + IYR + IAR + ICUR - immune_escape_R)

                # Dynamics for D
                v_groups._D[_t + 1] = v_groups._D[_t] + ICUD + IYD
                v_groups._ToICUD[_t] = ICUD
                v_groups._ToIYD[_t] = IYD
                v_groups._ToIA[_t] = PAIA
                v_groups._ToIY[_t] = PYIY

        for v_groups in self.vaccine_groups:
            # End of the daily discretization
            for attribute in self.state_vars:
                setattr(
                    v_groups,
                    attribute,
                    getattr(v_groups, "_" + attribute)[step_size].copy(), # Sonny's note, deep copy value at last
                ) # time step

            for attribute in self.tracking_vars:
                setattr(
                    v_groups, attribute, getattr(v_groups, "_" + attribute).sum(axis=0) # Sonny's note, deep copy
                ) # tracking variables, since tracking variables track the flow among compartments, so we take
                    # the sum
        if t >= self.vaccine.vaccine_start_time:
            self.vaccine_schedule(t, rate_immune)

        for v_groups in self.vaccine_groups: # Sonny's note, refresh and update v_group

            for attribute in self.state_vars:
                setattr(v_groups, "_" + attribute, np.zeros((step_size + 1, A, L)))
                vars(v_groups)["_" + attribute][0] = vars(v_groups)[attribute]

            for attribute in self.tracking_vars:
                setattr(v_groups, "_" + attribute, np.zeros((step_size, A, L)))

    def vaccine_schedule(self, t_date, rate_immune):
        """
        Mechanically move people between compartments for daily vaccination at the end of a day. We only move people
        between the susceptible compartments but assume that people may receive vaccine while they are already infected
        of recovered (in that case we assume that the vaccine is wasted). For vaccine amount of X, we adjust
        it by X * (S_v/N_v) where S_v is the total population in susceptible for vaccine status v and N_v is
        the total eligible population (see Vaccine.get_num_eligible for more detail) for vaccination in
        vaccine status v.
            The vaccination assumption is slightly different for booster dose. People don't mechanically move to the
        waned compartment, we draw binomial samples with a certain rate, but they get booster mechanically. That's why
        the definition of total eligible population is different. We adjust the vaccine amount X as
        X * (S_waned/(N_waned + N_fully_vax)) where N_waned and N_fully_wax is the total population in waned and
        fully vax compartments (see VaccineGroup.get_total_population). We estimate the probability of being waned and
        susceptible with S_waned/(N_waned + N_fully_vax).
        :param t_date: the current day simulated.
        :param rate_immune: immune evasion rate. Adjust the vaccine amount if there is immune evasion.
        """
        A = self.instance.A
        L = self.instance.L
        N = self.instance.N

        calendar = self.instance.cal.calendar
        t = t_date
        S_before = np.zeros((5, 2))
        S_temp = {}
        N_temp = {}

        for v_groups in self.vaccine_groups:
            S_before += v_groups.S
            S_temp[v_groups.v_name] = v_groups.S
            N_temp[v_groups.v_name] = v_groups.get_total_population(A * L)

        for v_groups in self.vaccine_groups:
            out_sum = np.zeros((A, L))
            S_out = np.zeros((A * L, 1))
            N_out = np.zeros((A * L, 1))

            for vaccine_type in v_groups.v_out:
                event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                if event is not None:
                    S_out = np.reshape(
                        self.vaccine.vaccine_allocation[vaccine_type][event][
                            "assignment"
                        ],
                        (A * L, 1),
                    )

                    if v_groups.v_name == "waned":
                        N_out = N_temp["waned"] + N_temp["second_dose"]
                    else:
                        N_out = self.vaccine.get_num_eligible(
                            N,
                            A * L,
                            v_groups.v_name,
                            v_groups.v_in,
                            v_groups.v_out,
                            calendar[t],
                        )
                    ratio_S_N = np.array(
                        [
                            0 if N_out[i] == 0 else float(S_out[i] / N_out[i])
                            for i in range(len(N_out))
                        ]
                    ).reshape((A, L))

                    out_sum += (ratio_S_N * S_temp[v_groups.v_name]).astype(int)

            in_sum = np.zeros((A, L))
            S_in = np.zeros((A * L, 1))
            N_in = np.zeros((A * L, 1))
            for vaccine_type in v_groups.v_in:
                for v_g in self.vaccine_groups:
                    if (
                            v_g.v_name
                            == self.vaccine.vaccine_allocation[vaccine_type][0]["from"]
                    ):
                        v_temp = v_g

                event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                if event is not None:
                    S_in = np.reshape(
                        self.vaccine.vaccine_allocation[vaccine_type][event][
                            "assignment"
                        ],
                        (A * L, 1),
                    )

                    if v_temp.v_name == "waned":
                        N_in = N_temp["waned"] + N_temp["second_dose"]
                    else:
                        N_in = self.vaccine.get_num_eligible(
                            N,
                            A * L,
                            v_temp.v_name,
                            v_temp.v_in,
                            v_temp.v_out,
                            calendar[t],
                        )

                    ratio_S_N = np.array(
                        [
                            0 if N_in[i] == 0 else float(S_in[i] / N_in[i])
                            for i in range(len(N_in))
                        ]
                    ).reshape((A, L))

                    in_sum += (ratio_S_N * S_temp[v_temp.v_name]).astype(int)

            v_groups.S = v_groups.S + (np.array(in_sum - out_sum))
            S_after = np.zeros((5, 2))

        for v_groups in self.vaccine_groups:
            S_after += v_groups.S

        imbalance = np.abs(np.sum(S_before - S_after, axis=(0, 1)))

        assert (imbalance < 1e-2).all(), (
            f"fPop inbalance in vaccine flow in between compartment S "
            f"{imbalance} at time {calendar[t]}, {t}"
        )

    def reset(self):

        '''
        In-place "resets" the simulation by clearing simulation data
            and reverting the time simulated to 0.
        Does not reset the random number generator, so
            simulating a replication that has been reset
            pulls new random numbers from where the random
            number generator last left off.
        Does not reset or resample the previously randomly sampled
            epidemiological parameters either.

        :return: [None]
        '''

        self.init_vaccine_groups()

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apr. 19, 2023, Sonny, reset viral load
        self.wastewater_viral_load = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.next_t = 0

    def get_binomial_transition_quantity(self, n, p):

        '''
        Either returns mean value of binomial distribution
            with parameters n, p or samples from that
            distribution, depending on whether self.rng is
            specified (depending on if simulation is run
            deterministically or not).

        :param n: [float] nonnegative value -- can be non-integer
            if running simulation deterministically, but must
            be integer to run simulation stochastically since
            it is parameter of binomial distribution
        :param p: [float] value in [0,1] corresponding to
            probability parameter in binomial distribution
        :return: [int] nonnegative integer that is a realization
            of a binomial random variable
        '''

        if self.rng is None:
            return n * p
        else:
            return self.rng.binomial(np.round(n).astype(int), p)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apr. 19, 2023, Sonny, Output Viral Load
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def output_viral_load(self, path):
        writeFile = open(path, 'w')
        writeFile.write(",date,viral_load\n")
        for idx, viral_load in enumerate(self.wastewater_viral_load):
            writeFile.write("{},".format(idx + 1))
            writeFile.write("{},".format((self.instance.start_date + dt.timedelta(days=idx)).strftime("%m/%d/%Y")))
            writeFile.write("{}\n".format(viral_load))
        writeFile.close()


    def output_IH_history(self, num_days, path):
        writeFile = open(path, 'w')
        writeFile.write(",date,hospitalized\n")
        for idx in range(num_days):
            writeFile.write("{},".format(idx + 1))
            writeFile.write("{},".format((self.instance.start_date + dt.timedelta(days=idx)).strftime("%m/%d/%Y")))
            writeFile.write("{}\n".format(np.sum(self.IH_history[idx])))
        writeFile.close()


    def output_hospital_history_var(self, history_var, path):
        writeFile = open(path, 'w')
        writeFile.write(",date,hospitalized\n")
        sample_path = np.array(getattr(self, f"{history_var}_history"))
        sample_path = np.sum(sample_path, axis=(1,2))
        for idx in range(len(sample_path)):
            writeFile.write("{},".format(idx + 1))
            writeFile.write("{},".format((self.instance.start_date + dt.timedelta(days=idx)).strftime("%m/%d/%Y")))
            writeFile.write("{}\n".format(sample_path[idx]))
        writeFile.close()


    def output_SE_hisory(self, path):
        writeFile = open(path, 'w')
        writeFile.write(",date,exposed\n")
        sample_path = np.array(self.SE_history)
        sample_path = np.sum(sample_path, axis=(1,2))
        for idx in range(len(sample_path)):
            writeFile.write("{},".format(idx + 1))
            writeFile.write("{},".format((self.instance.start_date + dt.timedelta(days=idx)).strftime("%m/%d/%Y")))
            writeFile.write("{}\n".format(sample_path[idx]))
        writeFile.close()


    def output_IHICU_history(self, path):
        writeFile = open(path, 'w')
        writeFile.write(",date,hospitalized\n")
        IH = np.array(self.IH_history)
        ICU = np.array(self.ICU_history)
        IH = np.sum(IH, axis=(1,2))
        ICU = np.sum(ICU, axis=(1,2))
        IHICU = IH + ICU
        for idx in range(len(IHICU)):
            writeFile.write("{},".format(idx + 1))
            writeFile.write("{},".format((self.instance.start_date + dt.timedelta(days=idx)).strftime("%m/%d/%Y")))
            writeFile.write("{}\n".format(IHICU[idx]))
        writeFile.close()