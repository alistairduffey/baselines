Final, production version of code for Duffey et al., 2024, investigating impact of transience during baseline period for assessment of SAI. 

Files:


01_preprocess.py
Inputs: 
* .nc files of 1pctCO2 and abrupt-2xCO2 runs for tas
* list of 9 models with abrupt-2xCO2 runs, and their respective file paths on JASMIN
Outputs (saved in /int_outs/):
* dataframe with time and associated CO2 concentration at time of crossing for the trasient scenario, for each model and ensemble member.
* dataframe with pr at time of crossing for the two scenarios, for each model and ensemble member.
* dataframe with land and sea temp. anomalies at time of crossing for the two scenario, for each model and ensemble member.


