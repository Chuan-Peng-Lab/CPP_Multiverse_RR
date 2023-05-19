#!/usr/bin/env python3
#
# This script is a batch script for extract the data from BIDS_data folder.
# -----------------------------------------------------
# Author: Yikang Liu
# Date: 2023-03-06
# -----------------------------------------------------
# example: python 2_EEG.py --subj 1
# -----------------------------------------------------


import os
import mne
import pandas as pd
import numpy as np


def extract_feature(subj):

    subj_str = f'sub-{subj:03d}'
    raw = mne.io.read_raw_fif(os.path.join('preprocessed_data/'+subj_str+'_pre.fif'), preload=True)




    stimulus_dict = {
    'Stimulus/hc/p/left': 4,
    'Stimulus/hc/p/right': 5,
    'Stimulus/hc/np/left': 6,
    'Stimulus/hc/np/right': 7,
    'Stimulus/lc/p/left': 8,
    'Stimulus/lc/p/right': 9,
    'Stimulus/lc/np/left': 10,
    'Stimulus/lc/np/right': 11,
    }

    response_dict = {
    'Response/car': 2,
    'Response/face': 3,
    }

    stim_dict = {
        'Stimulus/car': 16,
        'Stimulus/face': 17,
    }

    data = pd.read_csv('results/beh_preprocessed.csv', sep='\t')
    data = data[data['subj_idx']==subj_str]



    # N170
    events_from_annot, event_dict = mne.events_from_annotations(raw, event_id={
        'Stimulus/S 10': 4,
        'Stimulus/S 11': 5,
        'Stimulus/S 20': 6,
        'Stimulus/S 21': 7,
        'Stimulus/S 30': 8,
        'Stimulus/S 31': 9,
        'Stimulus/S 40': 10,
        'Stimulus/S 41': 11,})

    if len(events_from_annot) != len(data):
        new_events = [1 if x == 4 or x == 5 else 2 if x == 6 or x == 7 else 3 if x == 8 or x == 9 else 4 for x in events_from_annot[:,2].tolist()]
        boo = np.zeros(len(data))

        
        diff_len = len(data) - len(new_events)
        new_events += [4] * diff_len
        
        boo = np.array(data['condition'] == new_events, dtype=np.int32)

        loc = []
        for i in range(len(boo)):
            boo = np.array(data['condition'] == new_events, dtype=np.int32)
            if not boo[i]:
                new_events[i+1:] = new_events[i:-1]
                loc.append(i)
                new_events[i] = data.iloc[i]['condition']
                boo[i] = 1 if data.iloc[i]['condition'] == new_events[i] else 0
        data.reset_index(drop=True, inplace=True)
        data_tmp = data.iloc[~data.index.isin(loc)]
    else:
        data_tmp = data

    for trial_num in range(data_tmp.shape[0]):
        if data_tmp.iloc[trial_num]['stim'] == 'car':
            events_from_annot[trial_num][2] = 16
        else:
            events_from_annot[trial_num][2] = 17

    tmin = -1
    tmax = 2
    baseline = -0.1
    epoch = mne.Epochs(raw, 
                    events_from_annot, 
                    event_id=stim_dict , 
                    tmin=tmin, 
                    tmax=tmax,
                    baseline = (baseline,0), 
                    preload=True, 
                    picks=['eeg'])


    data_tmp_face = data_tmp[data_tmp['stim']=='face']
    epoch_face = epoch['Stimulus/face']

    chs=['PO7', 'P7', 'P8','PO8']
    n170_time_window = [0.13, 0.17]
    n170_integ_window = [int(n170_time_window[0]*epoch.info['sfreq']), int(n170_time_window[1]*epoch.info['sfreq'])]
    epoch_face_slice = epoch_face.pick_channels(chs).crop(tmin=n170_time_window[0], tmax=n170_time_window[1])
    n170_amplitudes_range = np.mean(epoch_face_slice.get_data(),axis=1)

    # N170 amplitude
    n170_amplitudes = np.mean(n170_amplitudes_range,axis=1)
    data_tmp.loc[data_tmp['stim']=='face', 'n170_amplitude'] = n170_amplitudes
    data_tmp_n170_amplitudes = data_tmp.loc[data_tmp['stim']=='face', 'n170_amplitude']
    data_tmp['n170_amplitude'] = data_tmp_n170_amplitudes
    data['n170_amplitude'] = data_tmp_n170_amplitudes


    # N170 slope
    n170_slopes = np.apply_along_axis(lambda x: np.polyfit(range(len(x)), x, 1)[0], axis=1, arr=n170_amplitudes_range)
    data_tmp.loc[data_tmp['stim']=='face', 'n170_slopes'] = n170_slopes
    data_tmp_n170_slopes = data_tmp.loc[data_tmp['stim']=='face', 'n170_slopes']
    data_tmp['n170_slopes'] = data_tmp_n170_slopes
    data['n170_slopes'] = data_tmp_n170_slopes

    # CPP baseline
    tmin = -1
    tmax = 2
    baseline = -0.5
    epoch = mne.Epochs(raw, 
                    events_from_annot, 
                    event_id=stim_dict , 
                    tmin=tmin, 
                    tmax=tmax,
                    baseline = (baseline,0), 
                    preload=True, 
                    picks=['eeg'])
    chs=['CPz','CP1','CP2']
    baseline_time_window = [-0.5, 0]
    epoch_slice = epoch.pick_channels(chs).crop(tmin=baseline_time_window[0], tmax=baseline_time_window[1])
    baseline_range = np.mean(np.mean(epoch_slice.get_data(),axis=1),axis=1)
    data_tmp['baseline'] = baseline_range
    data['baseline'] = data_tmp['baseline']

    # cpp 
    events_from_annot, event_dict = mne.events_from_annotations(raw, event_id={'Stimulus/S  5': 2,
    'Stimulus/S  6': 3,})




    tmin = -1
    tmax = 2
    epoch = mne.Epochs(raw, 
                    events_from_annot, 
                    event_id=response_dict , 
                    tmin=tmin, 
                    tmax=tmax,
                    baseline = None,
                    preload=True, 
                    picks=['eeg'])


    chs=['CPz','CP1','CP2']
    cpp_time_window = [-0.18, -0.08]
    epoch_slice = epoch.pick_channels(chs).crop(tmin=cpp_time_window[0], tmax=cpp_time_window[1])
    cpp_amplitudes_range = np.mean(epoch_slice.get_data(),axis=1)

    if len(data)!= len(events_from_annot):
        new_events = events_from_annot[:,2].tolist()
        data.loc[data['key_press']=='car','key_press'] = 2
        data.loc[data['key_press']=='face','key_press'] = 3

        boo = np.zeros(len(data))

        # 找出长度不同的部分，并使用默认值 4 来填充 new_events 中的空缺
        diff_len = len(data) - len(new_events)
        new_events += [0] * diff_len

        # 根据条件判断条件是否满足，并将结果存储在 boo 中
        boo = np.array(data['key_press'] == new_events, dtype=np.int32)

        loc = []
        # 如果条件不满足，将 new_events 中的值顺延，使其与 data['condition'] 匹配
        for i in range(len(boo)):
            boo = np.array(data['key_press'] == new_events, dtype=np.int32)
            if not boo[i]:
                new_events[i+1:] = new_events[i:-1]
                loc.append(i)
                new_events[i] = data.iloc[i]['key_press']
                boo[i] = 1 if data.iloc[i]['key_press'] == new_events[i] else 0
        data.reset_index(drop=True, inplace=True)
        data_tmp = data.iloc[~data.index.isin(loc)]
        data.loc[data['key_press']==2,'key_press'] = 'car'
        data.loc[data['key_press']==3,'key_press'] = 'face' 
    else:
        pass   

    cpp_amplitudes = np.mean(cpp_amplitudes_range,axis=1)
    data_tmp['cpp_amplitude'] = cpp_amplitudes
    data['cpp_amplitude'] = data_tmp['cpp_amplitude']
    data['cpp_amplitude'] = data['cpp_amplitude'] - data['baseline']

    # cpp slope
    cpp_slopes = np.apply_along_axis(lambda x: np.polyfit(range(len(x)), x, 1)[0], axis=1, arr=cpp_amplitudes_range)
    data_tmp['cpp_slopes'] = cpp_slopes
    data['cpp_slopes'] = data_tmp['cpp_slopes']

    # cpp variance
    cpp_var = np.var(cpp_amplitudes_range,axis=1)
    data_tmp['cpp_var'] = cpp_var
    data['cpp_var'] = data_tmp['cpp_var']

    # cpp peak
    cpp_time_window = [-0.2, 0.1]
    epoch_slice = epoch.pick_channels(chs).crop(tmin=cpp_time_window[0], tmax=cpp_time_window[1])
    cpp_amplitudes_range = np.mean(epoch_slice.get_data(),axis=1)
    cpp_peak = np.max(cpp_amplitudes_range,axis=1) 
    data_tmp['cpp_peak'] = cpp_peak
    data['cpp_peak'] = data_tmp['cpp_peak']
    data['cpp_peak'] = data['cpp_peak'] - data['baseline']
    # cpp peak latency
    cpp_peak_latency = np.argmax(cpp_amplitudes_range,axis=1)/epoch.info['sfreq']+cpp_time_window[0] +data_tmp.rt
    data_tmp['cpp_peak_latency'] = cpp_peak_latency
    data['cpp_peak_latency'] = data_tmp['cpp_peak_latency']

    return data


if __name__ == '__main__':
    dfs=pd.DataFrame()
    for i in range(17):
        if i+1 != 2:
            df = extract_feature(subj=i+1)
            dfs = pd.concat([dfs,df]).reset_index(drop=True)
            dfs.to_csv('results/data_beh_eeg.csv', sep='\t', index=False)
        else:
            pass
