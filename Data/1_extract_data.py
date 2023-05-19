#!/usr/bin/env python3
#
# This script is a batch script for extract the data from BIDS_data folder.
# -----------------------------------------------------
# Author: Yikang Liu
# Date: 2023-03-06
# -----------------------------------------------------
# example: python 1_extract_data.py
# -----------------------------------------------------


# load packages
import sys
import os
import re
import numpy as np
import pandas as pd

import mne

def main():
    # BIDS data extraction
    # ------------------------------------------------------
    # data location specification
    ds_root             = 'BIDS_data' # data target root directory
    tgt_dir             = '.' # group analysis results directory
    beh_results_dir     = os.path.join(tgt_dir, 'results')
    out_dir             = 'sourcedata-eeg_outside-MRT\\beh'
    in_dir              = 'sourcedata-eeg_inside-MRT\\beh'

    #  create results directory if it does not exist
    if not os.path.exists(beh_results_dir):
        os.makedirs(beh_results_dir)
    else:
        pass

    # subject directories
    sub_dir = ['sub-001', 
                    'sub-003', 
                    'sub-004', 
                    'sub-005', 
                    'sub-006', 
                    'sub-007', 
                    'sub-008', 
                    'sub-009', 
                    'sub-010', 
                    'sub-011', 
                    'sub-012',
                    'sub-013', 
                    'sub-014', 
                    'sub-015', 
                    'sub-016',
                    'sub-017']

    # BIDS format file name part labels
    BIDS_fn_label = []
    BIDS_fn_label.append('_task-pdm')      # BIDS file name task label. format: [_task-<task_label>]
    BIDS_fn_label.append('_acq-')          # BIDS file name acquisition label. format: [_acq-<label>]
    BIDS_fn_label.append('_run-0')         # BIDS file name run index. format: [_run-<index>]
    BIDS_fn_label.append('_beh')           # BIDS file name modality suffix. format: [_eeg\meg\bold]


    # Concatenate data from all runs
    dfs=pd.DataFrame()
    for i in sub_dir:
        for j in [1,2]: # run index
            df = pd.read_csv(os.path.join(ds_root, i, out_dir,i+BIDS_fn_label[0]+BIDS_fn_label[1]+'outsideMRT'+BIDS_fn_label[2]+str(j)+BIDS_fn_label[3]+'.tsv'),
                        sep='\t')
            df['subj_idx']=i
            df['run']=j
            dfs = pd.concat([dfs, df])
    dfs.to_csv(os.path.join(beh_results_dir, 'beh.csv'), sep='\t', index=False)

    # preprocess data
    df_raw = pd.read_csv( 'results/beh.csv',sep='\t')
    df_tgt = pd.DataFrame()
    df_tgt['condition']=df_raw['condition']
    df_tgt['coherence']=df_raw['condition'].map({1:'high',2:'high',3:'low',4:'low'})
    df_tgt['prioritized']=df_raw['condition'].map({1:'yes',2:'no',3:'yes',4:'no'})
    df_tgt['subj_idx']=df_raw['subj_idx'] 
    df_tgt['rt']=df_raw['response_time'] 
    df_tgt['key_press']=df_raw['key_press'].map({1:'car',2:'face'})
    df_tgt['correct']=df_raw['response_corr']
    df_tgt['run']=df_raw['run']
    car_images=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
    face_images=[19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]
    df_tgt['stim']=df_raw['image_index'].isin(car_images).astype(int).map({1:'car', 0:'face'})
    df_tgt.to_csv(os.path.join(beh_results_dir, 'beh_preprocessed.csv'), sep='\t', index=False)


    # EEG data extraction
    # ------------------------------------------------------
    # data preprocessing
    # data location specification
    ds_root             = 'BIDS_data' # data target root directory
    tgt_dir             = '.' 
    eeg_results_dir     = os.path.join(tgt_dir, 'preprocessed_data')
    out_dir             = 'sourcedata-eeg_outside-MRT\\eeg'
    in_dir              = 'sourcedata-eeg_inside-MRT\\eeg'

    #  create results directory if it does not exist
    if not os.path.exists(beh_results_dir):
        os.makedirs(beh_results_dir)
    else:
        pass

    # subject directories
    sub_dir = ['sub-001', 
                    'sub-003', 
                    'sub-004', 
                    'sub-005', 
                    'sub-006', 
                    'sub-007', 
                    'sub-008', 
                    'sub-009', 
                    'sub-010', 
                    'sub-011', 
                    'sub-012',
                    'sub-013', 
                    'sub-014', 
                    'sub-015', 
                    'sub-016',
                    'sub-017']

    # BIDS format file name part labels
    BIDS_fn_label = []
    BIDS_fn_label.append('_task-pdm')      # BIDS file name task label. format: [_task-<task_label>]
    BIDS_fn_label.append('_acq-')          # BIDS file name acquisition label. format: [_acq-<label>]
    BIDS_fn_label.append('_eeg')           # BIDS file name modality suffix. format: [_eeg\meg\bold]

    for i in sub_dir:
        # read raw data
        raw = mne.io.read_raw_brainvision(os.path.join(ds_root, i, out_dir,i+BIDS_fn_label[0]+BIDS_fn_label[1]+'outsideMRT'+BIDS_fn_label[2]+'.vhdr'))
        # set channel type
        raw.set_channel_types({'EOG':'eog'})   
        raw.set_channel_types({'ECG':'ecg'})   
        # resample
        raw.resample(512, npad="auto")    
        # filter
        raw.filter(1, 30, fir_design='firwin', picks=['eeg'])  
        # re-reference
        raw.set_eeg_reference('average')    
        # ica remove artifact
        ica = mne.preprocessing.ICA(n_components=50, random_state=97)
        ica.fit(raw) 
        ica.exclude = []                                   
        eog_indices, eog_scores = ica.find_bads_eog(raw)                                                         
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='ctps')               
        ica.exclude = eog_indices + ecg_indices 
        ica.apply(raw) 
        # save data
        raw.save(os.path.join(beh_results_dir, i+'_pre.fif'), overwrite=True)



if __name__ == '__main__':
    main()























