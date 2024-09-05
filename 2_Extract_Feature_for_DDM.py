#!/usr/bin/env python3
#
# This script is to extract CPP features and prepare data for DDM modeling.
# -----------------------------------------------------

import os
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import numpy as np
import pandas as pd
import pickle
from utils import *

# Check if the preprocessed data already exists (CSV files)
# If these files exist, load them; otherwise, proceed with data preprocessing.
if os.path.exists('../Data/results/dfs_resp.csv') and os.path.exists('../Data/results/dfs_stim.csv') and os.path.exists('../Data/results/dfs_resp_sm.csv') and os.path.exists('../Data/results/dfs_resp_fc.csv'):
    # Load preprocessed response and stimulus data from CSV files
    dfs_resp = pd.read_csv('../Data/results/dfs_resp.csv')
    dfs_stim = pd.read_csv('../Data/results/dfs_stim.csv')
    dfs_resp_sm = pd.read_csv('../Data/results/dfs_resp_sm.csv')
    dfs_resp_fc = pd.read_csv('../Data/results/dfs_resp_fc.csv')
    
    # Rename columns of the loaded DataFrame (convert string column names to integers)
    dfs_resp.rename(columns={col: int(col) for col in dfs_resp.columns if col.isdigit()}, inplace=True)
    dfs_stim.rename(columns={col: int(col) for col in dfs_stim.columns if col.isdigit()}, inplace=True)
    dfs_resp_sm.rename(columns={col: int(col) for col in dfs_resp_sm.columns if col.isdigit()}, inplace=True)

else:
    # If preprocessed data doesn't exist, process the raw EEG data
    
    # Load EEG data from .fif files in the preprocessed_data folder
    folder_path = '../Data/preprocessed_data'
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.fif')]
    raw_data_list = []
    
    # Read each .fif file and append the raw data to raw_data_list
    for file_path in file_list:
        raw_data = mne.io.read_raw_fif(file_path)
        raw_data_list.append(raw_data)  # raw_data_list is a list of raw EEG data
    
    # Load behavioral data from a CSV file (behavioral data is in tab-separated format)
    rawdata = pd.read_csv('../Data/results/beh_preprocessed.csv', sep='\t')
    
    # Convert raw EEG data into epochs for response-locked and stimulus-locked events
    dfs_resp = epoch2df(raw_data_list, rawdata=rawdata, tmin=-3, tmax=1, baseline=(None, 0), locked='resp')  # Response-locked
    dfs_stim = epoch2df(raw_data_list, rawdata=rawdata, tmin=-1.5, tmax=1.5, baseline=(-1.5, -1), locked='stim')  # Stimulus-locked
    
    # Apply smoothing to the response-locked data
    dfs_resp_sm = smooth(dfs_resp)
    
    # Add features to the response-locked data (including the smoothed data)
    dfs_resp_fc = add_feature(dfs_resp, dfs_resp_sm)
    
    # Drop columns representing raw time points (0 to 2049) to simplify the data
    dfs_resp_fc.drop(columns=np.arange(0, 2049), inplace=True)
    
    # Save the processed DataFrames to CSV files for future use
    dfs_resp.to_csv('../Data/results/dfs_resp.csv', index=False)
    dfs_stim.to_csv('../Data/results/dfs_stim.csv', index=False)
    dfs_resp_sm.to_csv('../Data/results/dfs_resp_sm.csv', index=False)
    dfs_resp_fc.to_csv('../Data/results/dfs_resp_fc.csv', index=False)

# Load the processed data from the CSV file
dfs_resp_fc = pd.read_csv('../Data/results/dfs_resp_fc.csv')

# Copy the 'correct' column to the 'response' column
dfs_resp_fc['response'] = dfs_resp_fc['correct']

# Keep the 'rt' column unchanged (this line has no actual effect and can be removed unless specific processing is needed)
dfs_resp_fc['rt'] = dfs_resp_fc['rt']

# Select specific features for further processing and create a new DataFrame
dfs2ddm = dfs_resp_fc.loc[:, ['subj_idx', 'coherence', 'slp', 'am', 'pkl', 'rt', 'slp_quantile', 'am_quantile', 'prioritized', 'response', 'pam', 'pam_quantile']]

# Normalize the features by subject and coherence (subtract the mean and divide by the standard deviation within each group)
dfs2ddm['ams'] = dfs2ddm.groupby(['subj_idx', 'coherence']).am.apply(lambda x: (x - x.mean()) / x.std())
dfs2ddm['slps'] = dfs2ddm.groupby(['subj_idx', 'coherence']).slp.apply(lambda x: (x - x.mean()) / x.std())
dfs2ddm['pams'] = dfs2ddm.groupby(['subj_idx', 'coherence']).pam.apply(lambda x: (x - x.mean()) / x.std())

# Bin the features based on quantiles and calculate the mean for each bin
# Bin the 'am' feature by 'am_quantile' and calculate the mean for each bin
mean_am = dfs2ddm.groupby(['subj_idx', 'coherence', 'am_quantile']).ams.mean().reset_index()
mean_am.rename(columns={'ams': 'am_bin'}, inplace=True)  # Rename the binned column
dfs2ddm = pd.merge(dfs2ddm, mean_am, on=['subj_idx', 'coherence', 'am_quantile'], how='left')

# Bin the 'slp' feature by 'slp_quantile' and calculate the mean for each bin
mean_slp = dfs2ddm.groupby(['subj_idx', 'coherence', 'slp_quantile']).slps.mean().reset_index()
mean_slp.rename(columns={'slps': 'slp_bin'}, inplace=True)
dfs2ddm = pd.merge(dfs2ddm, mean_slp, on=['subj_idx', 'coherence', 'slp_quantile'], how='left')

# Bin the 'pam' feature by 'pam_quantile' and calculate the mean for each bin
mean_pam = dfs2ddm.groupby(['subj_idx', 'coherence', 'pam_quantile']).pams.mean().reset_index()
mean_pam.rename(columns={'pams': 'pam_bin'}, inplace=True)
dfs2ddm = pd.merge(dfs2ddm, mean_pam, on=['subj_idx', 'coherence', 'pam_quantile'], how='left')

# Group the features by condition and calculate the mean for each condition
# Group the 'am' feature by 'prioritized' condition and calculate the mean
mean_am = dfs2ddm.groupby(['subj_idx', 'coherence', 'prioritized']).ams.mean().reset_index()
mean_am.rename(columns={'ams': 'am_cond'}, inplace=True)
dfs2ddm = pd.merge(dfs2ddm, mean_am, on=['subj_idx', 'coherence', 'prioritized'], how='left')

# Group the 'slp' feature by 'prioritized' condition and calculate the mean
mean_slp = dfs2ddm.groupby(['subj_idx', 'coherence', 'prioritized']).slps.mean().reset_index()
mean_slp.rename(columns={'slps': 'slp_cond'}, inplace=True)
dfs2ddm = pd.merge(dfs2ddm, mean_slp, on=['subj_idx', 'coherence', 'prioritized'], how='left')

# Group the 'pam' feature by 'prioritized' condition and calculate the mean
mean_pam = dfs2ddm.groupby(['subj_idx', 'coherence', 'prioritized']).pams.mean().reset_index()
mean_pam.rename(columns={'pams': 'pam_cond'}, inplace=True)
dfs2ddm = pd.merge(dfs2ddm, mean_pam, on=['subj_idx', 'coherence', 'prioritized'], how='left')

# Save the final processed data to a CSV file
dfs2ddm.to_csv('../Data/results/dfs2ddm.csv', index=False)
