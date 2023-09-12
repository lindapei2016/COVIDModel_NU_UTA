###############################################################################

# ParamFittingTools.py

###############################################################################

from Engine_SimModel_Wastewater import SimReplication
import numpy as np
from scipy.optimize import least_squares
import datetime as dt
import pandas as pd
import json
import time

# extra packages
import matplotlib.pyplot as plt
from pathlib import Path

variant_list = ["delta", "omicron"]


class ParameterFitting:
    def __init__(self, city: object,
                 vaccines: object,
                 variables: list,
                 initial_guess: list,
                 bounds: tuple,
                 objective_weights: dict,
                 time_frame: tuple,
                 change_dates=None,
                 transmission_reduction=None,
                 cocoon=None,
                 alpha_change_dates=None, # backward and forward by Guyi, parameter tuning for alpha
                 alpha_IHs=None,
                 alpha_gamma_ICUs=None,
                 alpha_IYDs=None,
                 alpha_mu_ICUs=None,
                 is_auto=False,
                 is_forward=False,
                 step_size=0,
                 viral_shedding_param=None, # Jun. 9, 2023, Sonny, lits of tuples
                 viral_shedding_profile_end_date=None): # Jun. 10, 2023, Sonny, list of end dates in datetime format
        """
        ToDo: This version assume transmission reduction and cocooning is same in the recent fit, change it later.

        :param city:
        :param vaccines:
        :param variables: the name of variables that will be fitted to the data. If you would like to fit
        transmission reduction please input "transmission_reduction" as the last element of the dictionary.
        :param initial_guess: the list initial guesses for the variables. It is very crucial to input good initial
        values for the least square fit to work.
        :param bounds: tuple of list of bounds for the variables. (list_of_lb, list_of_ub)
        :param objective_weights: dictionary of data included in the objective and their respective obj weights.
        :param time_frame: tuple of start and end date for the period we would like to fit the data.
        :param change_dates: if we are fitting transmission reduction we need to input the dates where the
        social distancing behaviour changes.
        :param transmission_reduction: the list of transmission reductions. We don't fit the transmission reduction
        from scratch. The list contains the fixed values that are already fitted.
        If the input is None, then fit the transmission reduction.
        :param cocoon: similar to transmission_reduction
        """
        self.result = None
        self.city = city
        self.vaccines = vaccines
        self.variables = variables
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.objective_weights = objective_weights
        self.change_dates = change_dates
        self.time_frame = time_frame
        self.transmission_reduction = transmission_reduction
        self.cocoon = cocoon
        # Backward and Forward by Guyi
        self.alpha_IHs = alpha_IHs
        self.alpha_gamma_ICUs = alpha_gamma_ICUs
        self.alpha_IYDs = alpha_IYDs
        self.alpha_mu_ICUs = alpha_mu_ICUs

        self.is_auto = is_auto
        self.is_forward = is_forward
        self.step_size = step_size
        self.alpha_change_dates = alpha_change_dates
        # Jun. 9, 2023, Sonny time sensitive viral shedding profile
        self.viral_shedding_param = None
        self.viral_shedding_profile_end_date = None
        self.vsp_start_period = None
        self.vsp_num_fit_period = None
        self.vsp_num_fit_param = 0
        if viral_shedding_param is not None:
            self.viral_shedding_param = viral_shedding_param # tuple [(param1, param2,...), (param1, param2,...),...] Jun. 9, 2023 Sonny
            self.viral_shedding_profile_end_date = viral_shedding_profile_end_date
            # create shedding date to period map
            self.city.viral_shedding_date_period_map(viral_shedding_profile_end_date)
            # find the start period of fitting viral shedding profile
            self.vsp_start_period = 0
            for i in range(len(viral_shedding_param)):
                if viral_shedding_param[i] is None:
                    self.vsp_start_period = i
                    break
            self.vsp_num_fit_period = len(viral_shedding_param) - self.vsp_start_period

            self.vsp_num_fit_param = self.vsp_num_fit_period * self.city.viral_shedding_profile["param_dim"]
            # load fixed vsp parameters
            self.city.load_fixed_viral_shedding_param(viral_shedding_profile_end_date, viral_shedding_param)
            # debug
            print("Debug")
            print("VSP Parameter Dimension: {}".format(self.city.viral_shedding_profile["param_dim"]))
            print("Number of VSP Parameters for fitting: {}".format(self.vsp_num_fit_param))
            print("Start period for fitting: {}".format(self.vsp_start_period))

        if transmission_reduction is not None:
            assert ("transmission_reduction" in transmission_reduction
                    and transmission_reduction.keys().index("transmission_reduction") == len(transmission_reduction),
                    "transmission_reduction must be the last element of the variable list!")

        self.rep = SimReplication(city, vaccines, None, -1)

    def run_fit(self, path=None, csv_viral_shedding_path=None, csv_transmission_reduction_path=None,
                csv_hosp_admin=None, csv_viral_load=None): # add path option for outputting the solution
        """
        Function that runs the parameter fitting.
        """
        # add time stamps for computing time elapsed
        start = time.time()
        res = self.least_squares_fit()
        print(res.x)
        x_variables = res.x
        print("SSE:", res.cost)
        solution = {}
        solution_tr = {}
        solution_alpha = {}
        rel_idx = 0  # Apr. 19, 2023, Sonny, for tracking the index of x variables
        writeSol = None
        write_viral_shedding = None
        write_transmission_reduction = None
        if path is not None:
            writeSol = open(path, 'w')
        if csv_viral_shedding_path is not None:
            write_viral_shedding = open(csv_viral_shedding_path, 'w')
            write_viral_shedding.write("end_date")
            for i in range(self.city.viral_shedding_profile["param_dim"]):
                write_viral_shedding.write(",param{}".format(i + 1))
            write_viral_shedding.write("\n")
        if csv_transmission_reduction_path is not None:
            write_transmission_reduction = open(csv_transmission_reduction_path, 'w')
        for idx, var in enumerate(self.variables):
            if var == "transmission_reduction":
                tr_reduc, cocoon_reduc = self.create_transmission_reduction(x_variables[rel_idx:])
                df_transmission = self.extend_transmission_reduction(tr_reduc, cocoon_reduc)
                # save transmission reduction into a csv file and a jason file by Guyi
                # save them into a csv file:
                if self.is_auto:
                    if self.is_forward:
                        dir = "forward_pt/"
                    else:
                        dir = "backward_pt/"
                    file_name = dir + "st_{}_{}_{}_transmission_lsq_estimated_data.csv".format(self.step_size,
                                                                                           self.change_dates[1].date(),
                                                                                          self.change_dates[-1].date())
                else:
                    file_name = "{}_{}_transmission_lsq_estimated_data.csv".format(self.change_dates[1].date(),
                                                                                          self.change_dates[-1].date())
                file_path = self.city.path_to_data / file_name
                df_transmission.to_csv(file_path, index=False)
                end_date = []
                for date in self.change_dates[1:]:
                    end_date.append(str(date - dt.timedelta(days=1)))
                table = pd.DataFrame(
                    {
                        "start_date": self.change_dates[:-1],
                        "end_date": end_date,
                        "contact_reduction": tr_reduc,
                        "cocoon": cocoon_reduc,
                    }
                )
                solution[var] = tr_reduc
                solution_tr[var] = tr_reduc # Sonny
                print(table)


                if self.is_auto:
                    file_name = dir + "st_{}_{}_{}_lsq_transmission_data.json".format(self.step_size,
                                                                                      self.change_dates[1].date(),
                                                                                      self.change_dates[-1].date())
                else:
                    file_name = "{}_{}_lsq_transmission_data.json".format(self.change_dates[1].date(),
                                                                                      self.change_dates[-1].date())
                with open(self.city.path_to_data / file_name, 'w') as f:
                    json.dump(solution_tr, f)
                # end save transmission reduction into a csv file and a jason file by Guyi
                # extra solution outputs
                if path is not None:
                    for index, date in enumerate(self.change_dates[:-1]):
                        writeSol.write("start_date: {}, ".format(date.strftime("%m/%d/%Y")))
                        writeSol.write("end_date: {}, ".format(end_date[index]))
                        writeSol.write("transmission_reduction: {}, ".format(tr_reduc[index]))
                        writeSol.write("cocoon: {}\n".format(cocoon_reduc[index]))
                if csv_transmission_reduction_path is not None:
                    write_transmission_reduction.write("start_date,end_date,transmission_reduction,cocoon\n")
                    for index, date in enumerate(self.change_dates[:-1]):
                        write_transmission_reduction.write("{},".format(date.strftime("%m/%d/%Y")))
                        write_transmission_reduction.write("{},".format(end_date[index]))
                        write_transmission_reduction.write("{},".format(tr_reduc[index]))
                        write_transmission_reduction.write("{}\n".format(cocoon_reduc[index]))
            elif var == "alphas": # Alpha fitting by Guyi
                alpha_IHs, alpha_gamma_ICUs, alpha_IYDs, alpha_mu_ICUs = self.create_alphas(x_variables[idx:idx+ len(self.alpha_IHs) * 4])
                df_alphas = self.extend_alphas(alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs)
                # save them into a csv file:
                if self.is_auto:
                    if self.is_forward:
                        dir = "forward_pt/"
                    else:
                        dir = "backward_pt/"
                    file_name = dir + "st_{}_{}_{}_alpha_estimated_data.csv".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
                else:
                    file_name = "{}_{}_alpha_estimated_data.csv".format(self.change_dates[1].date(),
                                                                                    self.change_dates[-1].date())
                file_path = self.city.path_to_data / file_name
                df_alphas.to_csv(file_path, index=False)
                end_date = []
                for date in self.change_dates[1:]:
                    end_date.append(str(date - dt.timedelta(days=1)))
                table = pd.DataFrame(
                    {
                        "start_date": self.change_dates[:-1],
                        "end_date": end_date,
                        "alpha_gamma_ICU": alpha_gamma_ICUs,
                        "alpha_IH": alpha_IHs,
                        "alpha_IYD": alpha_IYDs,
                        "alpha_mu_ICU": alpha_mu_ICUs,
                    }
                )
                print(table)
                solution["alpha_gamma_ICU"] = alpha_gamma_ICUs
                solution["alpha_IH"] = alpha_IHs
                solution["alpha_IYD"] = alpha_IYDs
                solution["alpha_mu_ICU"] = alpha_mu_ICUs
                solution_alpha["alpha_gamma_ICU"] = alpha_gamma_ICUs # Sonny
                solution_alpha["alpha_IH"] = alpha_IHs # Sonny
                solution_alpha["alpha_IYD"] = alpha_IYDs # Sonny
                solution_alpha["alpha_mu_ICU"] = alpha_mu_ICUs # Sonny
                file_name = dir + "st_{}_{}_{}_lsq_alpha_data.json".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
                with open(self.city.path_to_data / file_name, 'w') as f:
                    json.dump(solution_alpha, f)
                # End Alpha fitting by Guyi
            elif var == "viral_shedding_profile":
                num_shedding_days = self.city.viral_shedding_profile["num_days"]
                vsp_param = x_variables[idx:(idx + self.vsp_num_fit_param)]
                rel_idx += self.vsp_num_fit_param
                if path is not None:
                    writeSol.write(
                        "Viral Shedding Function: {}\n".format(self.city.viral_shedding_profile["shedding_function"]))
                for p in range(len(self.city.viral_shedding_profile["param"])):
                    if path is not None:
                        writeSol.write("End Date: {}, Viral Shedding Profile:".format(self.city.viral_shedding_profile["param"][p]["end_date"].strftime("%m/%d/%Y")))
                        if csv_viral_shedding_path is not None:
                            write_viral_shedding.write("{},".format(self.city.viral_shedding_profile["param"][p]["end_date"].strftime("%m/%d/%Y")))
                        for i in range(self.city.viral_shedding_profile["param_dim"] - 1):
                            writeSol.write(" {},".format(self.city.viral_shedding_profile["param"][p]["param"][i]))
                            if csv_viral_shedding_path is not None:
                                write_viral_shedding.write("{},".format(self.city.viral_shedding_profile["param"][p]["param"][i]))
                        writeSol.write(" {}\n".format(self.city.viral_shedding_profile["param"][p]["param"][self.city.viral_shedding_profile["param_dim"] - 1]))
                        if csv_viral_shedding_path is not None:
                            write_viral_shedding.write(
                                "{}\n".format(self.city.viral_shedding_profile["param"][p]["param"][self.city.viral_shedding_profile["param_dim"] - 1]))
                solution[var] = vsp_param.tolist()
            else:
                print(f"{var} = {x_variables[idx]}")
                solution[var] = x_variables[idx]
                if path is not None:
                    writeSol.write(f"{var} = {x_variables[idx]}\n")

        end = time.time()
        # May 22, 2023 Sonny
        if path is not None:
            if "log_wastewater_viral_load" in self.objective_weights.keys():
                writeSol.write("Note: Log viral load is used for computing least squares.\n")
            writeSol.write("Time elapsed(s): {}\n".format(end - start))
            writeSol.close()
        if csv_viral_shedding_path is not None:
            write_viral_shedding.close()
        if csv_transmission_reduction_path is not None:
            write_transmission_reduction.close()

        # May 29, 2023 Sonny output simulated hospital admissions
        if csv_hosp_admin is not None or csv_viral_load is not None:
            self.rep.simulate_time_period(self.time_frame[1], self.time_frame[1])
            if csv_hosp_admin is not None:
                self.rep.output_hospital_history_var("ToIHT", csv_hosp_admin)
            if csv_viral_load is not None:
                self.rep.output_viral_load(csv_viral_load)

        solution["change_dates"] = [str(x.date()) for x in self.change_dates]
        if self.is_auto:
            file_name = dir + "st_{}_{}_{}_lsq_estimated_data.json".format(self.step_size, self.change_dates[1].date(),
                                                                       self.change_dates[-1].date())
        else:
            file_name = "{}_{}_lsq_estimated_data.json".format(self.change_dates[1].date(),
                                                                       self.change_dates[-1].date())
        with open(self.city.path_to_data / file_name, 'w') as f:
            json.dump(solution, f)
        return solution


    def least_squares_fit(self):
        """
        Function that runs the least squares fit
        """
        result = least_squares(
            self.residual_error,
            self.initial_guess,
            bounds=self.bounds,
            method="trf",
            verbose=2,
        )
        return result

    # Sonny's notes: the data stream looks like: |other variables |alpha |viral shedding profile |transmission reduction |
    def residual_error(self, x_variables): # Sonny's notes: explain how to group variables?
        print("new value: ", x_variables)
        rel_idx = 0 # Apr. 19, 2023, Sonny, for tracking the index of x variables
        for idx, var in enumerate(self.variables):
            if var == "alphas": # Guyi
                alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs = self.create_alphas(
                    x_variables[rel_idx:rel_idx + len(self.alpha_IHs) * 4])
                df_alphas = self.extend_alphas(alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs)
                # print(df_alphas["alpha_IH"])
                self.city.otherInfo["alpha_gamma_ICU"] = df_alphas["alpha_gamma_ICU"]
                self.city.otherInfo["alpha_IH"] = df_alphas["alpha_IH"]
                self.city.otherInfo["alpha_IYD"] = df_alphas["alpha_IYD"]
                self.city.otherInfo["alpha_mu_ICU"] = df_alphas["alpha_mu_ICU"]
                # update the rel_idx
                rel_idx += len(self.alpha_IHs) * 4
            elif hasattr(self.city.base_epi, var):
                setattr(self.city.base_epi, var, x_variables[idx])
                rel_idx += 1 # Apr. 19, 2023, Sonny, for tracking the index of x variables
            elif var == "transmission_reduction": # July, may be way to optimize the fitting spead, there is reconstruction
                # of transmission reduction rate
                # Apr. 19, 2023, Sonny, rel_idx is for tracking the index of x variables
                tr_reduc, cocoon_reduc = self.create_transmission_reduction(x_variables[rel_idx:])
                df_transmission = self.extend_transmission_reduction(tr_reduc, cocoon_reduc)
                transmission_reduction = [
                    (d, tr)
                    for (d, tr) in zip(
                        df_transmission["date"], df_transmission["transmission_reduction"]
                    )
                ]
                #print("transmission reduction")
                #print(transmission_reduction)
                self.city.cal.load_fixed_transmission_reduction(transmission_reduction)
                cocooning = [
                    (d, c) for (d, c) in zip(df_transmission["date"], df_transmission["cocooning"])
                ]
                self.city.cal.load_fixed_cocooning(cocooning)
            elif var.split()[0] in variant_list:
                if var.split()[1] in {"start_date", "end_date", "peak_date"}:
                    self.city.variant_pool.variants_data['epi_params']["immune_evasion"][var.split()[0]][
                        var.split()[1]] = self.city.variant_start + dt.timedelta(days=int(x_variables[idx]))
                else:
                    self.city.variant_pool.variants_data['epi_params'][var.split()[1]][var.split()[0]] = x_variables[idx]
                rel_idx += 1
            elif var == "viral_shedding_profile": # Apr. 19, 2023, Sonny, include viral shedding profile variable
                viral_shedding_param = x_variables[rel_idx: (rel_idx + self.vsp_num_fit_param)]
                #print("debug viral shedding param")
                #print(viral_shedding_param)
                self.city.load_fitting_viral_shedding_param(viral_shedding_param, self.vsp_start_period, self.viral_shedding_profile_end_date)
                rel_idx += self.vsp_num_fit_param
        # Simulate the system with the new variables:
        self.rep.simulate_time_period(self.time_frame[1], self.time_frame[1])

        residual_error = []
        # Calculate the residual error:
        for key, var in self.objective_weights.items():
            if key == "kappa_weight":
                #residual_error.extend([self.objective_weights["kappa_weight"] * (x_variables[i] - x_variables[i+1]) for i in range(len(x_variables)-1)])
                # need to adjust for viral shedding profile
                residual_error.extend([self.objective_weights["kappa_weight"] * (x_variables[i] - x_variables[i+1]) for i in range(self.vsp_num_fit_param, len(x_variables)-1)])
            else:
                real_data = getattr(self.city, f"real_{key}")[self.time_frame[0]: self.time_frame[1]]
                if key == "wastewater_viral_load":
                    sim_data = getattr(self.rep, key)[self.time_frame[0]: self.time_frame[1]]
                elif key == "log_wastewater_viral_load":
                    sim_data = getattr(self.rep, key)[self.time_frame[0]: self.time_frame[1]]
                else:
                    sim_data = np.sum(np.array(getattr(self.rep, key)), axis=(1, 2))[self.time_frame[0]: self.time_frame[1]]

                error = [var * (a_i - b_i) for a_i, b_i in zip(real_data, sim_data)]
                #print("debug")
                #print(key)
                #print(real_data)
                #print(sim_data)
                # compute sum of error squared
                error_np = np.array(error)
                error_np = np.dot(error_np,error_np)
                print("{} residual sum of squares: {}".format(key, error_np))
                #print(sim_data)
                residual_error.extend(error)

        self.rep.reset()
        return residual_error

    def create_transmission_reduction(self, x_variables):
        """
        If we are optimizing the transmission reduction values convert the x_variables
        into transmission reduction list.
        :return:
        """
        i = 0
        tr_reduc = []
        for tr in self.transmission_reduction:
            if tr is None:
                tr_reduc.append(x_variables[i])
                i += 1
            else:
                tr_reduc.append(tr)

        cocoon_reduc = []
        i = 0
        for tr in self.cocoon:
            if tr is None:
                cocoon_reduc.append(x_variables[i])
                i += 1
            else:
                cocoon_reduc.append(tr)
        return tr_reduc, cocoon_reduc

    def extend_transmission_reduction(self,  tr_reduc, cocoon_reduc):
        """
        Extend the transmission reduction into a dataframe with the corresponding dates.
        """
        change_dates = self.change_dates
        date_list = []
        tr_reduc_extended, cocoon_reduc_extended = [], []
        for idx in range(len(change_dates[:-1])):
            tr_reduc_extended.extend([tr_reduc[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
            date_list.extend(
                [
                    str(change_dates[idx] + dt.timedelta(days=x))
                    for x in range((change_dates[idx + 1] - change_dates[idx]).days)
                ]
            )
            cocoon_reduc_extended.extend(
                [cocoon_reduc[idx]] * (change_dates[idx + 1] - change_dates[idx]).days
            )

        d = {
            "date": pd.to_datetime(date_list),
            "transmission_reduction": tr_reduc_extended,
            "cocooning": cocoon_reduc_extended,
        }
        df_transmission = pd.DataFrame(data=d)
        return df_transmission

    # load transmission reduction from csv file, Sonny
    def load_transmission_reduction(self, path):
        df_transmission_cocoon = pd.read_csv(path)
        df_transmission_cocoon["date"] = pd.to_datetime(df_transmission_cocoon["date"], format="mixed")
        transmission_reduction = [
            (d, tr)
            for (d, tr) in zip(
                df_transmission_cocoon["date"], df_transmission_cocoon["transmission_reduction"]
            )
        ]
        self.city.cal.load_fixed_transmission_reduction(transmission_reduction)
        cocooning = [
            (d, c) for (d, c) in zip(df_transmission_cocoon["date"], df_transmission_cocoon["cocooning"])
        ]
        self.city.cal.load_fixed_cocooning(cocooning)
        return df_transmission_cocoon["date"]

    # compute residual sum of squares, Sonny
    def compute_rss(self, key):
        self.rep.simulate_time_period(self.time_frame[1], self.time_frame[1])
        real_data = getattr(self.city, f"real_{key}")[self.time_frame[0]: self.time_frame[1]]
        sim_data = np.sum(np.array(getattr(self.rep, key)), axis=(1, 2))[self.time_frame[0]: self.time_frame[1]]
        self.rep.reset() # reset the sim model
        error = [(a_i - b_i) for a_i, b_i in zip(real_data, sim_data)]
        # compute sum of error squared
        error_np = np.array(error)
        rss = np.dot(error_np, error_np)
        print("{} residual sum of squares: {}".format(key, rss))
        return rss


    # plot residual sum of squares, Sonny
    def plot_rss(self, key, dates, location="upper left", viral_start_index = 0, output_path = None):
        key_label_map = {"ToIHT_history": "Hospital Admissions", "wastewater_viral_load": "Viral Load (N1 GC)"}
        self.rep.simulate_time_period(self.time_frame[1], self.time_frame[1])
        real_data = getattr(self.city, f"real_{key}")[:self.time_frame[1]]
        if key == "wastewater_viral_load" or key == "log_wastewater_viral_load":
            sim_data = getattr(self.rep, key)
            plt.scatter(dates[viral_start_index: self.time_frame[1]], real_data[viral_start_index: self.time_frame[1]], c="g", label="real_{}".format(key))
            plt.plot(dates, sim_data, color="k", label="simulated_{}".format(key))
            plt.ylim([0, 6e15]) # move it to the argument later
        else:
            sim_data = np.sum(np.array(getattr(self.rep, key)), axis=(1, 2))
            plt.scatter(dates, real_data, c="b", label="Real {}".format(key_label_map[key]))
            plt.plot(dates, sim_data, color = "r", label="Simulated {}".format(key_label_map[key]))
            plt.ylim([0, 500])  # move it to the argument later
        plt.legend(loc=location)
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.ylabel(key_label_map[key])
        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


    # simulate
    def simulate_rng(self, root_dir, random_seed = -1, key = "ToIHT_history", r_squared_threshold = -1, num_days_ahead = 14):
        df_transmission = self.extend_transmission_reduction(self.transmission_reduction, self.cocoon)
        transmission_reduction = [
            (d, tr)
            for (d, tr) in zip(
                df_transmission["date"], df_transmission["transmission_reduction"]
            )
        ]
        self.city.cal.load_fixed_transmission_reduction(transmission_reduction)
        cocooning = [
            (d, c) for (d, c) in zip(df_transmission["date"], df_transmission["cocooning"])
        ]
        self.city.cal.load_fixed_cocooning(cocooning)

        # reset rep
        self.rep = SimReplication(self.city, self.vaccines, None, random_seed)

        # simulate
        end_date_index = self.city.cal.calendar.index(self.viral_shedding_profile_end_date[-1])
        self.rep.simulate_time_period(end_date_index, end_date_index)
        #self.rep.simulate_time_period(self.time_frame[1] + num_days_ahead, self.time_frame[1] + num_days_ahead)

        real_data = getattr(self.city, f"real_{key}")[:self.time_frame[1]]
        sim_data = np.sum(np.array(getattr(self.rep, key)), axis=(1, 2))[:self.time_frame[1]]

        error = [(a_i - b_i) for a_i, b_i in zip(real_data, sim_data)]
        # compute sum of error squared
        error_np = np.array(error)
        rss = np.dot(error_np, error_np)
        real_data = np.array(real_data)
        mean_real_data = np.mean(real_data)
        tmp = real_data - mean_real_data # total sum of squares
        tss = np.dot(tmp, tmp)
        r_squared = 1 - rss / tss
        flag = False
        if r_squared_threshold < 0:
            cur_folder_path = root_dir + "/seed_{}".format(random_seed)
            Path(cur_folder_path).mkdir(parents=True, exist_ok=True)
            self.rep.output_hospital_history_var("ToIHT", cur_folder_path + "/ToIHT.csv")
            self.rep.output_viral_load(cur_folder_path + "/viral_load.csv")
            flag = True
        elif r_squared >= r_squared_threshold:
            cur_folder_path = root_dir + "/seed_{}".format(random_seed)
            Path(cur_folder_path).mkdir(parents=True, exist_ok=True)
            self.rep.output_hospital_history_var("ToIHT", cur_folder_path + "/ToIHT.csv")
            self.rep.output_viral_load(cur_folder_path + "/viral_load.csv")
            flag = True

        return r_squared, flag


    # Guyi
    def create_alphas(self, x_variables):
        """
        If we are optimizing the alpha values convert the x_variables
        into alpha list.
        :return:
        """
        i = 0
        alpha_IHs = []
        alpha_gamma_ICUs = []
        alpha_IYDs = []
        alpha_mu_ICUs = []
        for alpha_IH, alpha_gamma_ICU, alpha_IYD, alpha_mu_ICU in zip(self.alpha_IHs, self.alpha_gamma_ICUs,
                                                                      self.alpha_IYDs, self.alpha_mu_ICUs):
            if alpha_gamma_ICU is None:
                alpha_gamma_ICUs.append(x_variables[i])
                i += 1
            else:
                alpha_gamma_ICUs.append(alpha_gamma_ICU)

            if alpha_IH is None:
                alpha_IHs.append(x_variables[i])
                i += 1
            else:
                alpha_IHs.append(alpha_IH)

            if alpha_IYD is None:
                alpha_IYDs.append(x_variables[i])
                i += 1
            else:
                alpha_IYDs.append(alpha_IYD)

            if alpha_mu_ICU is None:
                alpha_mu_ICUs.append(x_variables[i])
                i += 1
            else:
                alpha_mu_ICUs.append(alpha_mu_ICU)

        return alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs

    #Guyi
    def extend_alphas(self, alpha_gamma_ICU, alpha_IH, alpha_IYD, alpha_mu_ICU):
        change_dates = self.alpha_change_dates
        date_list = []
        alpha_ICU_extended, alpha_IH_extended, alpha_IYD_extended, alpha_mu_ICU_extended = [], [], [], []
        for idx in range(len(change_dates[:-1])):
            date_list.extend(
                [
                    str(change_dates[idx] + dt.timedelta(days=x))
                    for x in range((change_dates[idx + 1] - change_dates[idx]).days)
                ]
            )
            alpha_ICU_extended.extend([alpha_gamma_ICU[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
            alpha_IYD_extended.extend(
                [alpha_IYD[idx]] * (change_dates[idx + 1] - change_dates[idx]).days
            )
            alpha_mu_ICU_extended.extend(
                [alpha_mu_ICU[idx]] * (change_dates[idx + 1] - change_dates[idx]).days
            )
            alpha_IH_extended.extend(
                [alpha_IH[idx]] * (change_dates[idx + 1] - change_dates[idx]).days
            )

        d = {
            "date": pd.to_datetime(date_list),
            "alpha_gamma_ICU": alpha_ICU_extended,
            "alpha_IH": alpha_IH_extended,
            "alpha_IYD": alpha_IYD_extended,
            "alpha_mu_ICU": alpha_mu_ICU_extended,

        }
        df_alphas = pd.DataFrame(data=d)
        return df_alphas
