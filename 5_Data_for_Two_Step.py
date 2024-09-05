import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import numpy as np
import pandas as pd

#This script prepares the necessary data for evaluating the relationship between drift rate and other behavioral features across different coherence conditions.


# Load preprocessed behavioral data and model trace data from CSV files
dfs2ddm = pd.read_csv('../Data/results/dfs2ddm.csv')
m4trc = pd.read_csv('../Data/model_trace/m4_traces.csv')

# Extract the first and second columns corresponding to the 'v' values from the model trace (excluding 'sub' and 'std')
vint = m4trc.filter(regex='^v(?!.*(sub|std)).*$').iloc[:, 0]  # The intercept for 'v'
vslp = m4trc.filter(regex='^v(?!.*(sub|std)).*$').iloc[:, 1]  # The slope for 'v'

# Calculate the mean 'v' for the low coherence condition across subjects (for the intercept part)
vlow_sub = m4trc.filter(regex='^v_In.*sub.*$').mean(axis=0).to_numpy()

# Calculate the mean 'v' for the high coherence condition by adding the intercept and condition-related value
vhigh_sub = m4trc.filter(regex='^v_In.*sub.*$').mean(axis=0).to_numpy() + m4trc.filter(regex='^v_C.*sub.*$').mean(axis=0).to_numpy()

# Group the behavioral data by subject and coherence, and calculate the mean for each feature
# The features include slope (slp), peak amplitude (pam), amplitude (am), and reaction time (rt)
slp_v = dfs2ddm.groupby(['subj_idx', 'coherence'])[['slp', 'pam', 'am', 'rt']].mean().reset_index()

# Assign the computed 'v' values for the low coherence condition to the corresponding rows
slp_v.loc[(slp_v['coherence'] == 'low'), 'v'] = vlow_sub

# Assign the computed 'v' values for the high coherence condition to the corresponding rows
slp_v.loc[(slp_v['coherence'] == 'high'), 'v'] = vhigh_sub

# Save the combined feature and 'v' data to a new CSV file for further analysis
slp_v.to_csv('../Data/results/slp_v_m4.csv', index=False)
