import os
import mne
import pandas as pd
import numpy as np




def permutate_var(subj, ntimes):
    
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

    data = pd.read_csv('data_beh_eeg.csv', sep='\t')
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
    tmax = 2.5
    baseline = -0.1
    epoch = mne.Epochs(raw, 
                    events_from_annot, 
                    event_id=stim_dict , 
                    tmin=tmin, 
                    tmax=tmax,
                    baseline = (baseline,0), 
                    preload=True, 
                    picks=['eeg'])
    # permutation
    data_n170_var = np.zeros(ntimes)
    data_cpp_var = np.zeros(ntimes)
    for i in range(ntimes):
        index_perm = np.random.permutation(data_tmp.index)
        data_perm = data_tmp.loc[index_perm].reset_index(drop=True)
        rt = data_perm['rt'].values

        chs=['PO7', 'P7', 'P8','PO8']
        time_window = np.array([[-0.18, -0.08]])+ rt.reshape(-1, 1)+ 0.2
        n_samples = int((0.1)*raw.info['sfreq'])
        epoch_slice = np.zeros((len(epoch), len(chs), n_samples))
        for j in range(len(epoch)):
            if pd.isna(data_perm['rt'][j]):
                epoch_slice[j] = epoch[j].pick_channels(chs).crop(tmin=time_window[j][0], tmax=time_window[j][1]).get_data()[:, :, :n_samples]*0
            else:
                epoch_slice[j] = epoch[j].pick_channels(chs).crop(tmin=time_window[j][0], tmax=time_window[j][1]).get_data()[:, :, :n_samples]

        n170_var_range = np.mean(np.mean(epoch_slice,axis=1),axis=1)
        n170_var_range = n170_var_range[np.where(n170_var_range!=0)]
        n170_var = np.var(n170_var_range)
        data_n170_var[i] = n170_var


        chs=['CPz', 'CP1', 'CP2']
        time_window = np.array([[-0.18, -0.08]])+ rt.reshape(-1, 1)+ 0.2
        epoch_slice = np.zeros((len(epoch), len(chs), n_samples))
        for j in range(len(epoch)):
            if pd.isna(data_perm['rt'][j]):
                epoch_slice[j] = epoch[j].pick_channels(chs).crop(tmin=time_window[j][0], tmax=time_window[j][1]).get_data()[:, :, :n_samples]*0
            else:
                epoch_slice[j] = epoch[j].pick_channels(chs).crop(tmin=time_window[j][0], tmax=time_window[j][1]).get_data()[:, :, :n_samples]

        cpp_var_range = np.mean(np.mean(epoch_slice,axis=1),axis=1)
        cpp_var_range = cpp_var_range[np.where(cpp_var_range!=0)]
        cpp_var = np.var(cpp_var_range)
        data_cpp_var[i] = cpp_var

    
    return  pd.DataFrame({str(subj):data_n170_var}), pd.DataFrame({str(subj):data_cpp_var})

        