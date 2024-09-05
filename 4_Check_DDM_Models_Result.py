#!/usr/bin/env python3
#
#This script processes hierarchical drift-diffusion models (HDDM) by:
#Loading Model Chains: Loads and concatenates model chains (e.g., m1, m2, am, pam, slp).
#Model Evaluation: Computes Deviance Information Criterion (DIC) and saves posterior traces.
#Model Summaries: Generates detailed summaries with convergence diagnostics (r_hat), saved as CSV files.
#LOO Cross-Validation: Performs Leave-One-Out (LOO) cross-validation and removes outliers based on Pareto k diagnostics.
#Model Comparison: Compares models using LOO and deviance, saving results and comparison plots.
#The script ensures robust model evaluation and outputs results for further analysis.
# -----------------------------------------------------
# Author: Yikang Liu
# -----------------------------------------------------
# example: python 1_extract_data.py
# -----------------------------------------------------
# This analysis is conducted in dockerddm. For more details, refer to:
# Pan, W., Geng, H., Zhang, L., Fengler, A., Frank, M., Zhang, R.-Y., & Chuan-Peng, H. (2022, November 1). A Hitchhiker’s Guide to Bayesian Hierarchical Drift-Diffusion Modeling with dockerHDDM. PsyArXiv. https://doi.org/10.31234/osf.io/6uzga


import kabuki
import sys
import os
import hddm
import matplotlib.pyplot as plt
import pickle
from glob import glob
from kabuki.analyze import gelman_rubin
import arviz as az
import numpy as np
import pandas as pd


scripts_dir = '/home/jovyan/scripts'
sys.path.append(scripts_dir)


from HDDMarviz import HDDMarviz
from InferenceDataFromHDDM import InferenceDataFromHDDM
from plot_ppc_by_cond import plot_ppc_by_cond

# Import necessary libraries and modules
import os
from glob import glob
import hddm
import kabuki
import pandas as pd
import numpy as np
import arviz as az
from InferenceDataFromHDDM import InferenceDataFromHDDM

# Load the behavioral data
dfs = pd.read_csv('dfs2ddm.csv')

# Model comparison by DIC
# Load model chains for models 1 through 5, concatenate them, and calculate the DIC (Deviance Information Criterion)
models_1 = []
for model_path in glob(os.path.join(os.getcwd(),'temp/m1_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_1.append(modelx)
m1 = kabuki.utils.concat_models(models_1)
print(f"DIC is : {m1.dic}")

models_2 = []
for model_path in glob(os.path.join(os.getcwd(),'temp/m2_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_2.append(modelx)
m2 = kabuki.utils.concat_models(models_2)
print(f"DIC is : {m2.dic}")

models_3 = []
for model_path in glob(os.path.join(os.getcwd(),'temp/m3_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_3.append(modelx)
m3 = kabuki.utils.concat_models(models_3)
print(f"DIC is : {m3.dic}")

models_4 = []
for model_path in glob(os.path.join(os.getcwd(),'temp/m4_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_4.append(modelx)
m4 = kabuki.utils.concat_models(models_4)
print(f"DIC is : {m4.dic}")

models_5 = []
for model_path in glob(os.path.join(os.getcwd(),'temp/m5_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_5.append(modelx)
m5 = kabuki.utils.concat_models(models_5)
print(f"DIC is : {m5.dic}")

# Save the traces from model 5
m5.get_traces().to_csv('m5_traces.csv', index=False)

# Load and process trials data for different models (slps, ams, pams)
models_slps = []
for model_path in glob(os.path.join(os.getcwd(),'temp/slps_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_slps.append(modelx)
m_slps = kabuki.utils.concat_models(models_slps)
print(f"DIC is : {m_slps.dic}")
m_slps.get_traces().to_csv('m_slps_traces.csv', index=False)

# Similarly for ams and pams
models_am = []
for model_path in glob(os.path.join(os.getcwd(),'temp/ams_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_am.append(modelx)
m_ams = kabuki.utils.concat_models(models_am)
print(f"DIC is : {m_ams.dic}")
m_ams.get_traces().to_csv('m_ams_traces.csv', index=False)

models_pam = []
for model_path in glob(os.path.join(os.getcwd(),'temp/pams_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_pam.append(modelx)
m_pams = kabuki.utils.concat_models(models_pam)
print(f"DIC is : {m_pams.dic}")
m_pams.get_traces().to_csv('m_pams_traces.csv', index=False)

# Process binning models (slp_bin, am_bin, pam_bin) similarly
models_slp_bin = []
for model_path in glob(os.path.join(os.getcwd(),'temp/slp_bin_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_slp_bin.append(modelx)
m_slp_bin = kabuki.utils.concat_models(models_slp_bin)
print(f"DIC is : {m_slp_bin.dic}")
m_slp_bin.get_traces().to_csv('m_slp_bin_traces.csv', index=False)

models_am_bin = []
for model_path in glob(os.path.join(os.getcwd(),'temp/am_bin_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_am_bin.append(modelx)
m_am_bin = kabuki.utils.concat_models(models_am_bin)
print(f"DIC is : {m_am_bin.dic}")
m_am_bin.get_traces().to_csv('m_am_bin_traces.csv', index=False)

# Process condition models (slp_cond, am_cond, pam_cond) similarly
models_slp_cond = []
for model_path in glob(os.path.join(os.getcwd(),'temp/slp_cond_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_slp_cond.append(modelx)
m_slp_cond = kabuki.utils.concat_models(models_slp_cond)
print(f"DIC is : {m_slp_cond.dic}")
m_slp_cond.get_traces().to_csv('m_slp_cond_traces.csv', index=False)

# Similarly for am_cond and pam_cond
models_am_cond = []
for model_path in glob(os.path.join(os.getcwd(),'temp/am_cond_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_am_cond.append(modelx)
m_am_cond = kabuki.utils.concat_models(models_am_cond)
print(f"DIC is : {m_am_cond.dic}")
m_am_cond.get_traces().to_csv('m_am_cond_traces.csv', index=False)

models_pam_cond = []
for model_path in glob(os.path.join(os.getcwd(),'temp/pam_cond_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_pam_cond.append(modelx)
m_pam_cond = kabuki.utils.concat_models(models_pam_cond)
print(f"DIC is : {m_pam_cond.dic}")
m_pam_cond.get_traces().to_csv('m_pam_cond_traces.csv', index=False)

# Infer HDDM object to arviz object
# Compute InferenceData objects for all models (m1 to m5, and joint models)
infdata_m1 = InferenceDataFromHDDM(models_1, nppc=300, save_name="infdata_m1")
infdata_m2 = InferenceDataFromHDDM(models_2, nppc=300, save_name="infdata_m2")
infdata_m3 = InferenceDataFromHDDM(models_3, nppc=300, save_name="infdata_m3")
infdata_m4 = InferenceDataFromHDDM(models_4, nppc=300, save_name="infdata_m4")
infdata_m5 = InferenceDataFromHDDM(models_5, nppc=300, save_name="infdata_m5")
infdata_slps = InferenceDataFromHDDM(models_slps, nppc=300, save_name="infdata_slps")
infdata_ams = InferenceDataFromHDDM(models_am, nppc=300, save_name="infdata_ams")
infdata_pams = InferenceDataFromHDDM(models_pam, nppc=300, save_name="infdata_pams")
infdata_slp_bin = InferenceDataFromHDDM(models_slp_bin, nppc=300, save_name="infdata_slp_bin")
infdata_am_bin = InferenceDataFromHDDM(models_am_bin, nppc=300, save_name="infdata_am_bin")
infdata_pam_bin = InferenceDataFromHDDM(models_pam_bin, nppc=300, save_name="infdata_pam_bin")
infdata_slp_cond = InferenceDataFromHDDM(models_slp_cond, nppc=300, save_name="infdata_slp_cond")
infdata_am_cond = InferenceDataFromHDDM(models_am_cond, nppc=300, save_name="infdata_am_cond")
infdata_pinfdata_pam_cond = InferenceDataFromHDDM(models_pam_cond, nppc=300, save_name="infdata_pam_cond")

# Load the inference data from netCDF files for further analysis and comparisons
infdata_m1 = az.from_netcdf('infdata_m1_netcdf')
infdata_m2 = az.from_netcdf('infdata_m2_netcdf')
infdata_m3 = az.from_netcdf('infdata_m3_netcdf')
infdata_m4 = az.from_netcdf('infdata_m4_netcdf')
infdata_m5 = az.from_netcdf('infdata_m5_netcdf')
infdata_am_bin = az.from_netcdf('infdata_am_bin_netcdf')
infdata_am_cond = az.from_netcdf('infdata_am_cond_netcdf')
infdata_ams = az.from_netcdf('infdata_ams_netcdf')
infdata_pam_bin = az.from_netcdf('infdata_pam_bin_netcdf')
infdata_pam_cond = az.from_netcdf('infdata_pam_cond_netcdf')
infdata_pams = az.from_netcdf('infdata_pams_netcdf')
infdata_slp_bin = az.from_netcdf('infdata_slp_bin_netcdf')
infdata_slp_cond = az.from_netcdf('infdata_slp_cond_netcdf')
infdata_slps = az.from_netcdf('infdata_slps_netcdf')

# Generate summaries for each inference data object and sort them by the 'r_hat' statistic (used to assess convergence)
m1_summary = az.summary(infdata_m1, round_to=4)
m1_summary.sort_values('r_hat')
m2_summary = az.summary(infdata_m2, round_to=4)
m2_summary.sort_values('r_hat')
m3_summary = az.summary(infdata_m3, round_to=4)
m3_summary.sort_values('r_hat')
m4_summary = az.summary(infdata_m4, round_to=4)
m4_summary.sort_values('r_hat')
m5_summary = az.summary(infdata_m5, round_to=4)
m5_summary.sort_values('r_hat')
am_bin_summary = az.summary(infdata_am_bin, round_to=4)
am_bin_summary.sort_values('r_hat')
am_cond_summary = az.summary(infdata_am_cond, round_to=4)
am_cond_summary.sort_values('r_hat')
ams_summary = az.summary(infdata_ams, round_to=4)
ams_summary.sort_values('r_hat')
pam_bin_summary = az.summary(infdata_pam_bin, round_to=4)
pam_bin_summary.sort_values('r_hat')
pam_cond_summary = az.summary(infdata_pam_cond, round_to=4)
pam_cond_summary.sort_values('r_hat')
pams_summary = az.summary(infdata_pams, round_to=4)
pams_summary.sort_values('r_hat')
slp_bin_summary = az.summary(infdata_slp_bin, round_to=4)
slp_bin_summary.sort_values('r_hat')
slp_cond_summary = az.summary(infdata_slp_cond, round_to=4)
slp_cond_summary.sort_values('r_hat')
slps_summary = az.summary(infdata_slps, round_to=4)
slps_summary.sort_values('r_hat')

# Save all the summaries to CSV files
m1_summary.to_csv('m1_summary.csv')
m2_summary.to_csv('m2_summary.csv')
m3_summary.to_csv('m3_summary.csv')
m4_summary.to_csv('m4_summary.csv')
m5_summary.to_csv('m5_summary.csv')
am_bin_summary.to_csv('am_bin_summary.csv')
am_cond_summary.to_csv('am_cond_summary.csv')
ams_summary.to_csv('ams_summary.csv')
pam_bin_summary.to_csv('pam_bin_summary.csv')
pam_cond_summary.to_csv('pam_cond_summary.csv')
pams_summary.to_csv('pams_summary.csv')
slp_bin_summary.to_csv('slp_bin_summary.csv')
slp_cond_summary.to_csv('slp_cond_summary.csv')
slps_summary.to_csv('slps_summary.csv')



# Model comparison by LOO-CV
# Calculate Leave-One-Out (LOO) cross-validation for model comparison
loo_m1 = az.loo(infdata_m1, pointwise=True)
loo_m2 = az.loo(infdata_m2, pointwise=True)
loo_m3 = az.loo(infdata_m3, pointwise=True)
loo_m4 = az.loo(infdata_m4, pointwise=True)
loo_m5 = az.loo(infdata_m5, pointwise=True)
loo_am_bin = az.loo(infdata_am_bin, pointwise=True)
loo_am_cond = az.loo(infdata_am_cond, pointwise=True)
loo_ams = az.loo(infdata_ams, pointwise=True)
loo_pam_bin = az.loo(infdata_pam_bin, pointwise=True)
loo_pam_cond = az.loo(infdata_pam_cond, pointwise=True)
loo_pams = az.loo(infdata_pams, pointwise=True)
loo_slp_bin = az.loo(infdata_slp_bin, pointwise=True)
loo_slp_cond = az.loo(infdata_slp_cond, pointwise=True)
loo_slps = az.loo(infdata_slps, pointwise=True)

# Identify outliers based on Pareto k diagnostics (k > 0.7 indicates problematic points)
outliers1 = loo_m1.pareto_k.where(loo_m1.pareto_k >= 0.7, drop=True).trial_idx.values
outliers2 = loo_m2.pareto_k.where(loo_m2.pareto_k >= 0.7, drop=True).trial_idx.values
outliers3 = loo_m3.pareto_k.where(loo_m3.pareto_k >= 0.7, drop=True).trial_idx.values
outliers4 = loo_m4.pareto_k.where(loo_m4.pareto_k >= 0.7, drop=True).trial_idx.values
outliers5 = loo_m5.pareto_k.where(loo_m5.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_am_bin = loo_am_bin.pareto_k.where(loo_am_bin.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_am_cond = loo_am_cond.pareto_k.where(loo_am_cond.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_ams = loo_ams.pareto_k.where(loo_ams.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_pam_bin = loo_pam_bin.pareto_k.where(loo_pam_bin.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_pam_cond = loo_pam_cond.pareto_k.where(loo_pam_cond.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_pams = loo_pams.pareto_k.where(loo_pams.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_slp_bin = loo_slp_bin.pareto_k.where(loo_slp_bin.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_slp_cond = loo_slp_cond.pareto_k.where(loo_slp_cond.pareto_k >= 0.7, drop=True).trial_idx.values
outliers_slps = loo_slps.pareto_k.where(loo_slps.pareto_k >= 0.7, drop=True).trial_idx.values

# Combine all the identified outliers into a single list
outliers = np.unique(np.concatenate((
    outliers1, outliers2, outliers3, outliers4, outliers5, outliers_am_bin,
    outliers_am_cond, outliers_ams, outliers_pam_bin, outliers_pam_cond,
    outliers_pams, outliers_slp_bin, outliers_slp_cond, outliers_slps), axis=0))

# Remove the outliers from the dataset for reanalysis
new_indx = dfs.index.values[~np.isin(dfs.index.values, outliers)]
infdata_m1 = infdata_m1.isel(trial_idx=new_indx)
infdata_m2 = infdata_m2.isel(trial_idx=new_indx)
infdata_m3 = infdata_m3.isel(trial_idx=new_indx)
infdata_m4 = infdata_m4.isel(trial_idx=new_indx)
infdata_m5 = infdata_m5.isel(trial_idx=new_indx)
infdata_am_bin = infdata_am_bin.isel(trial_idx=new_indx)
infdata_am_cond = infdata_am_cond.isel(trial_idx=new_indx)
infdata_ams = infdata_ams.isel(trial_idx=new_indx)
infdata_pam_bin = infdata_pam_bin.isel(trial_idx=new_indx)
infdata_pam_cond = infdata_pam_cond.isel(trial_idx=new_indx)
infdata_pams = infdata_pams.isel(trial_idx=new_indx)
infdata_slp_bin = infdata_slp_bin.isel(trial_idx=new_indx)
infdata_slp_cond = infdata_slp_cond.isel(trial_idx=new_indx)
infdata_slps = infdata_slps.isel(infdata_slps.isel(trial_idx=new_indx))

# Compare all models using LOO (Leave-One-Out) cross-validation and deviance as the scale
model_comparison = az.compare({
    'm1': infdata_m1,
    'm2': infdata_m2,
    'm3': infdata_m3,
    'm4': infdata_m4,
    'm5': infdata_m5,
    'am_bin': infdata_am_bin,
    'am_cond': infdata_am_cond,
    'ams': infdata_ams,
    'pam_bin': infdata_pam_bin,
    'pam_cond': infdata_pam_cond,
    'pams': infdata_pams,
    'slp_bin': infdata_slp_bin,
    'slp_cond': infdata_slp_cond,
    'slps': infdata_slps
}, ic='loo', scale='deviance')

# Save the model comparison results to a CSV file
model_comparison.to_csv('model_comparison_results.csv')




