# Input files Overview
Below contains descriptions for input files used in the current Austin Model under Austin directory. 
To model a new city, create a new directory under the name of the new city and update the files for your city model 
in the new directory.

## setup_data_Final.json
This file has information about population size, epidemiological parameters etc. for a city instance. 
- "start_date" and "end_date": start and end date of the simulation
- "hosp_beds": hospital bed capacity.
- "icu": ICU bed capacity.
- "epi_params":To learn more about epi_params please check Haoxiang et al.
- "school_closure": school closure calendar of Austin. The model distinguishes if the schools are open or closed. 
Don't forget to add the school closure schedule if you run the simulation for later dates.


## transmission.csv

Fixed transmission reduction and cocooning values. The file has additional information about changing hospital dynamics.
Care of COVID-19 patient improved over time. Refer to Haoxiang et al. for more information about parameters.

## tiers5_opt_Final.json or tiers_CDC.json
These files contain information about staged-alert system tiers. tiers5_opt_Final.json contain the tier information 
about the Austin's staged-alert system. tiers_CDC.json contains information about the CDC community thresholds. 
The keywords can be different depending on the staged-alert system that has been used.

The file intended to be used dynamically. It is possible to change number of stages or input values.

## variant.json

This file includes the information about changes in model parameters when a new variant emerges.
Each variant can have different epidemiological dynamics. Therefore, it can effect different model parameters.
It is possible to input different parameters to this file to model different variants. 

Below is a list of parameters that are currently updated for the Austin model under Delta and Omicron variants. These 
are not static keywords and can be modified according to the need of a modeler. We made these updates according to 
literature and our own parameter fitting.

Example entry: Let's say we want to update "param_name" such that the new value if 50% of the original value 
under variant "variant_name". Then enter the following:

"param_name": {"variant_name": 0.5}

Important Note: "immune_evasion", "sigma_E" and "vax_params" are exceptions to the format above!! 
See explanations below.

Keywords:
- "immune_evasion": In the current Austin model we assume there is immune evasion only when a new variant emerges. 
This key word includes all the information related to the immune evasion due to emergence of new variant.
  
  1. "half-life": we assume immunity wane at an exponential rate with a certain half-life. 
  2. "start_date": we use a triangular shape to adjust immune evasion (as the new variant start circulating the immune 
  evasion will increase. After a while as more people got infected with the new variant the immune evasion rate will 
  decrease.)
  3. "peak_date": the date where immune evasion peak.
  4. "end_date": the date immune evasion stops.

- "alpha_gamma_ICU", "alpha_IH", "alpha_mu_ICU", "alpha_IYD": These parameters are fitted for each variant
to adjust the hospital dynamics. See Haoxiang et al. for definitions.
- "beta": (transmission rate) the variants have higher transmission rate than the wild-type.
- "sigma_E": (exposed rate = (1 / incubation_period)) shorter incubation period with variants.
We use triangle distribution to sample incubation_period. With Delta and Omicron the mean 
incubation period is 1.5 days shorter. The input in the file contains that information.
- "gamma_ICU0": (recovery rate from ICU) additional changes to the hospital dynamic.
- "YHR": (percent of symptomatic infectious that go to the hospital) Hospitalization rate increased with Delta variant.
-  "YHR_overall": similar to.
- "vax_params": Vaccine efficacy decreased with Delta and Omicron variants. 
  1. "v_beta_reduct": (reduction in transmission)
  2. "v_tau_reduct": (reduction in symptomatic infection)
  3. "v_pi_reduct": (reduction in hospitalisation)
  
  For example, input entry of "first_dose": { "delta": 0.95, "omicron": 0.85} means vaccine efficacy after the first 
dose of vaccine was 95% and 85% for delta and omicron, respectively.

## variant_prevalence.csv
 
We assume that the prevalence of a new variant follows an S-shaped logistic curve. This file contains the points from 
that logistic curve for different variants.

## vaccines.json

This files contain information about the vaccine efficacy etc.
- "beta_reduct": vaccine efficacy against infection under different vaccine levels.
- "tau_reduct": vaccine efficacy against symptomatic infection under different vaccine levels.
- "pi_reduct": vaccine efficacy against hospitalization under different vaccine levels.
- "second_dose_time": number of days between first and second dose.
- "effect_time": vaccine start protecting after effect_time amount of days.

## vaccine_allocation_fixed.csv
Fixed vaccine allocation percentages for each age risk group for first dose vaccination. 
(The data is from Texas DSHS.)

## booster_allocation_fixed.csv

Fixed booster allocation data, we assume the booster schedule is similar to first dose schedule but with uptake rate. 

## calendar.csv
