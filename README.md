CPP Multiverse Project - Stage 1 Registered Report

The stage 1 pre-registered report can be found in OSF：
https://osf.io/9ygx6/

We re-analyzed four publicly available datasets and used joint modeling techniques integrating the Drift Diffusion Model (DDM) with CPP data, investigating whether CPP serves as a robust ERP marker of evidence accumulation across various perceptual decision-making tasks.

The repository contains：
- 1_Preprocess_Data.py: Raw EEG/behavioral data preprocessing
- 2_Extract_Feature_for_DDM.py: CPP Feature extraction for DDM fitting
- 3_Run_DDM_Models.py: run DDM
- model.py: model specification
- 4_Check_DDM_Models_Result.py: Model diagnostics
- 5_Data_for_Two_Step.py: Prepares data for two-step analysis	
- 5_Two_Step.Rmd: Implements the two-stage correlation approach
- 6_Figure4a&s1.Rmd: Generates main result figures	
- 7_Sensitivity_analysis_run_model.ipynb: Tests model robustness for subjects 
- 8_Sensitivity_analysis_load_model.ipynb: Compares sensitivity results
Multiverse_Model_Data: Temp results of running DDM of multiverse CPP
Sentivity_Model_Data: Temp results of running DDM of sensitivity analysis

utils.py: 




Please cite the Registered Report when referencing this work. Citation format will be updated upon publication.
