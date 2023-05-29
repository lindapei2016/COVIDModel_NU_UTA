###############################################################################

# ParamFittingTools.py

###############################################################################

from SimModel import SimReplication
import numpy as np
from scipy.optimize import least_squares
import datetime as dt
import pandas as pd
import json

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
                 alpha_change_dates=None,
                 alpha_IHs=None,
                 alpha_gamma_ICUs=None,
                 alpha_IYDs=None,
                 alpha_mu_ICUs=None,
                 is_auto = False,
                 is_forward=False,
                 step_size=0):
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
        self.alpha_IHs = alpha_IHs
        self.alpha_gamma_ICUs = alpha_gamma_ICUs
        self.alpha_IYDs = alpha_IYDs
        self.alpha_mu_ICUs = alpha_mu_ICUs

        self.is_auto = is_auto
        self.is_forward = is_forward
        self.step_size = step_size
        self.alpha_change_dates = alpha_change_dates

        if transmission_reduction is not None:
            assert ("transmission_reduction" in transmission_reduction
                    and transmission_reduction.keys().index("transmission_reduction") == len(transmission_reduction),
                    "transmission_reduction must be the last element of the variable list!")

        self.rep = SimReplication(city, vaccines, None, -1)

    def run_fit(self):
        """
        Function that runs the parameter fitting.
        """
        res = self.least_squares_fit()
        x_variables = res.x
        print("SSE:", res.cost)
        solution = {}
        dir = ""
        for idx, var in enumerate(self.variables):
            if var == "transmission_reduction":
                tr_reduc, cocoon_reduc = self.create_transmission_reduction(x_variables[idx:])
                df_transmission = self.extend_transmission_reduction(tr_reduc, cocoon_reduc)
                # save them into a csv file:
                if self.is_auto:
                    if self.is_forward:
                        dir = "forward_pt/"
                    else:
                        dir = "backward_pt/"
                file_name = dir + "st_{}_{}_{}_transmission_lsq_estimated_data.csv".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
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
                print(table)
                file_name = dir + "st_{}_{}_{}_lsq_transmission_data.json".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
                with open(self.city.path_to_data / file_name, 'w') as f:
                    json.dump(solution, f)
            elif var == "alphas":
                alpha_IHs, alpha_gamma_ICUs, alpha_IYDs, alpha_mu_ICUs = self.create_alphas(x_variables[idx:idx+ len(self.alpha_IHs) * 4])
                df_alphas = self.extend_alphas(alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs)
                # save them into a csv file:
                if self.is_auto:
                    if self.is_forward:
                        dir = "forward_pt/"
                    else:
                        dir = "backward_pt/"
                file_name = dir + "st_{}_{}_{}_alpha_estimated_data.csv".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
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
                file_name = dir + "st_{}_{}_{}_lsq_alpha_data.json".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
                with open(self.city.path_to_data / file_name, 'w') as f:
                    json.dump(solution, f)

            else:
                print(f"{var} = {x_variables[idx]}")
                solution[var] = x_variables[idx]
        solution["change_dates"] = [str(x.date()) for x in self.change_dates]

        file_name = dir + "st_{}_{}_{}_lsq_estimated_data.json".format(self.step_size, self.change_dates[1].date(), self.change_dates[-1].date())
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

    def residual_error(self, x_variables):
        print("new value: ", x_variables)
        for idx, var in enumerate(self.variables):
            if var == "alphas":
                alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs = self.create_alphas(x_variables[idx:idx+ len(self.alpha_IHs) * 4])
                df_alphas = self.extend_alphas(alpha_gamma_ICUs, alpha_IHs, alpha_IYDs, alpha_mu_ICUs)
                # print(df_alphas["alpha_IH"])
                self.city.otherInfo["alpha_gamma_ICU"] = df_alphas["alpha_gamma_ICU"]
                self.city.otherInfo["alpha_IH"] = df_alphas["alpha_IH"]
                self.city.otherInfo["alpha_IYD"] = df_alphas["alpha_IYD"]
                self.city.otherInfo["alpha_mu_ICU"] = df_alphas["alpha_mu_ICU"]

            elif hasattr(self.city.base_epi, var):
                setattr(self.city.base_epi, var, x_variables[idx])
            elif var == "transmission_reduction":
                tr_reduc, cocoon_reduc = self.create_transmission_reduction(x_variables[idx:])
                df_transmission = self.extend_transmission_reduction(tr_reduc, cocoon_reduc)
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
            elif var.split()[0] in variant_list:
                if var.split()[1] == "start_date":
                    self.city.variant_pool.variants_data['epi_params']["immune_evasion"][var.split()[0]][
                        "start_date"] = self.city.variant_start + dt.timedelta(days=int(x_variables[idx]))
                elif var.split()[1] == "days":
                    self.city.variant_pool.variants_data['epi_params']["immune_evasion"][var.split()[0]]["days"] = int(
                        x_variables[idx])
                    self.city.variant_pool.variants_data['epi_params']["immune_evasion"][var.split()[0]][
                        "peak_date"] = self.city.variant_pool.variants_data['epi_params']["immune_evasion"][var.split()[0]][
                                           "start_date"] + dt.timedelta(days=int(x_variables[idx]))
                else:
                    self.city.variant_pool.variants_data['epi_params'][var.split()[1]][var.split()[0]] = x_variables[idx]

        # Simulate the system with the new variables:
        self.rep.simulate_time_period(self.time_frame[1], self.time_frame[1])

        residual_error = []
        # Calculate the residual error:
        for key, var in self.objective_weights.items():
            if key == "kappa_weight":
                residual_error.extend([self.objective_weights["kappa_weight"] * (x_variables[i] - x_variables[i+1]) for i in range(len(x_variables)-1)])
            else: 
                real_data = getattr(self.city, f"real_{key}")[self.time_frame[0]: self.time_frame[1] + 1]
                # For the Cook county case there is an arc from IH to D
                if self.city == "cook" and key == "ToICUD_history":
                    sim_data_ICUD = np.sum(np.array(getattr(self.rep, key)), axis=(1, 2))[self.time_frame[0]: self.time_frame[1] + 1]
                    sim_data_IHD = np.sum(np.array(getattr(self.rep, "ToIHD_history")), axis=(1, 2))[self.time_frame[0]: self.time_frame[1] + 1]
                    sim_data = sim_data_ICUD + sim_data_IHD
                else:
                    sim_data = np.sum(np.array(getattr(self.rep, key)), axis=(1, 2))[self.time_frame[0]: self.time_frame[1] + 1]
                error = [var * (a_i - b_i) for a_i, b_i in zip(real_data, sim_data)]
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
        for alpha_IH, alpha_gamma_ICU, alpha_IYD, alpha_mu_ICU in zip(self.alpha_IHs, self.alpha_gamma_ICUs, self.alpha_IYDs, self.alpha_mu_ICUs):
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