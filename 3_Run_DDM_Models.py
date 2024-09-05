#!/usr/bin/env python3
#
# This script is designed to run multiple models in parallel for behavioral and joint model analysis.
# The models are built and defined in an external script called 'model.py'.
# -----------------------------------------------------
# Author: Yikang Liu
# -----------------------------------------------------
# This analysis is conducted in dockerddm. For more details, refer to:
# Pan, W., Geng, H., Zhang, L., Fengler, A., Frank, M., Zhang, R.-Y., & Chuan-Peng, H. (2022, November 1). A Hitchhiker’s Guide to Bayesian Hierarchical Drift-Diffusion Modeling with dockerHDDM. PsyArXiv. https://doi.org/10.31234/osf.io/6uzga








# Import necessary libraries for modeling, parallel processing, and data handling
import kabuki
import sys
import os
import hddm
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
import arviz as az
import numpy as np
import pandas as pd
from model import *  # Import custom model definitions from 'model.py'

# Set up the path to custom scripts directory
scripts_dir = '/home/jovyan/scripts'
sys.path.append(scripts_dir)

# Import additional utilities for handling data and plotting
from HDDMarviz import HDDMarviz
from InferenceDataFromHDDM import InferenceDataFromHDDM
from plot_ppc_by_cond import plot_ppc_by_cond

# Load the preprocessed data from a CSV file
df = hddm.load_csv('./dfs2ddm.csv', sep=',')

# Run behavioral models (m1 to m5) using parallel processing
# Each model is run with 4000 samples, a burn-in period of 2000, and thinning set to 1
m1 = Parallel(n_jobs=4)(delayed(run_m1)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/m1') for i in range(4))
m2 = Parallel(n_jobs=4)(delayed(run_m2)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/m2') for i in range(4))
m3 = Parallel(n_jobs=4)(delayed(run_m3)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/m3') for i in range(4))
m4 = Parallel(n_jobs=4)(delayed(run_m4)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/m4') for i in range(4))
m5 = Parallel(n_jobs=4)(delayed(run_m5)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/m5') for i in range(4))

# Run joint models across 9 different pipelines using parallel processing
# Each model is again run with 4000 samples, a burn-in of 2000, and thinning set to 1
slps = Parallel(n_jobs=4)(delayed(slps)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/slps') for i in range(4))
ams = Parallel(n_jobs=4)(delayed(ams)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/ams') for i in range(4))
pams = Parallel(n_jobs=4)(delayed(pams)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/pams') for i in range(4))
slp_bin = Parallel(n_jobs=4)(delayed(slp_bin)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/slp_bin') for i in range(4))
am_bin = Parallel(n_jobs=4)(delayed(am_bin)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/am_bin') for i in range(4))
pam_bin = Parallel(n_jobs=4)(delayed(pam_bin)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/pam_bin') for i in range(4))
slp_cond = Parallel(n_jobs=4)(delayed(slp_cond)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/slp_cond') for i in range(4))
am_cond = Parallel(n_jobs=4)(delayed(am_cond)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/am_cond') for i in range(4))
pam_cond = Parallel(n_jobs=4)(delayed(pam_cond)(id=i, df=df, samples=4000, burn=2000, thin=1, save_name='temp/pam_cond') for i in range(4))
