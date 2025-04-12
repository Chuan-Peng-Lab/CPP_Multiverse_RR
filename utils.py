import os
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import numpy as np
import pandas as pd


def raw2epoch(raw, locked, tmin, tmax, baseline, picks = ['CPz','CP1','CP2']):
    '''
    the function to transform raw eeg data to epoch data
    ---------------------------------------------
    params:
        raw: raw data of one subject
        locked: the time point to lock the epoch, 'stim', 'cue', 'resp'
        picks: the channels to pick, default is ['CPz','CP1','CP2']
        tmin: the start time of epoch
        tmax: the end time of epoch
        baseline: the baseline of epoch
    ---------------------------------------------
    return: 
        (evoked1, evoked2): tuple of numpy array
            evoked1: evoked data of condition 1
            evoked2: evoked data of condition 2
    '''
    # Create events
    event, event_dict = mne.events_from_annotations(raw,event_id={
            'Stimulus/S  5': 2,
            'Stimulus/S  6': 3,
            'Stimulus/S 10': 4,
            'Stimulus/S 11': 5,
            'Stimulus/S 20': 6,
            'Stimulus/S 21': 7,
            'Stimulus/S 30': 8,
            'Stimulus/S 31': 9,
            'Stimulus/S 40': 10,
            'Stimulus/S 41': 11,
            'Stimulus/S 74': 13,
            'Stimulus/S 75': 14,
            'Stimulus/S 76': 15,})
    # Create epochs
    # locked to stimulus
    if locked == 'stim':
        cond1 = ['Stimulus/S 10', 'Stimulus/S 11', 'Stimulus/S 20', 'Stimulus/S 21']
        cond2 = ['Stimulus/S 30', 'Stimulus/S 31', 'Stimulus/S 40', 'Stimulus/S 41']
        epoch = mne.Epochs(raw, 
                    event, 
                    event_id=event_dict , 
                    tmin=tmin,
                    tmax=tmax,
                    baseline = baseline, 
                    preload=True, 
                    picks=picks)
        # transform to evoked data(numpy)
        evoked1 = epoch[cond1].get_data()
        evoked2 = epoch[cond2].get_data()
        evoked1 = np.mean(evoked1,axis=1)
        evoked2 = np.mean(evoked2,axis=1)
    # locked to cue
    elif locked == 'cue':
        cond1 = ['Stimulus/S 74', 'Stimulus/S 75']
        cond2 = ['Stimulus/S 76']
        epoch = mne.Epochs(raw, 
                    event, 
                    event_id=event_dict , 
                    tmin= tmin,
                    tmax= tmax,
                    baseline = baseline,
                    preload=True, 
                    picks=picks)
        # transform to evoked data(numpy)
        evoked1 = epoch[cond1].get_data()
        evoked2 = epoch[cond2].get_data()
        evoked1 = np.mean(evoked1,axis=1)
        evoked2 = np.mean(evoked2,axis=1)
    # locked to response
    elif locked == 'resp':    
        cond1 = ['Stimulus/S  5']
        cond2 = ['Stimulus/S  6']
        epoch = mne.Epochs(raw, 
                    event, 
                    event_id=event_dict , 
                    tmin=tmin,
                    tmax=tmax,
                    baseline = baseline, 
                    preload=True, 
                    picks=picks)
        # transform to evoked data(numpy)
        evoked1 = epoch[cond1].get_data()
        evoked2 = epoch[cond2].get_data()
        # average across channels(Cp1, Cp2, Cpz)
        evoked1 = np.mean(evoked1,axis=1)
        evoked2 = np.mean(evoked2,axis=1)
    return (evoked1, evoked2) # tuple of numpy array




def ERP_pop(raw_data_list, locked, **kwargs):
    '''
    the function to calculate ERP of all subjects
    ---------------------------------------------
    params:
        raw_data_list: list of raw data of all subjects
        locked: the time point to lock the epoch, 'stim', 'cue', 'resp'
    ---------------------------------------------
    return:
        (mean1, std1, mean2, std2): tuple of numpy array
            mean1: mean of condition 1
            std1: std of condition 1
            mean2: mean of condition 2
            std2: std of condition 2
    '''
    # transform raw data to epoch data
    result_list = [raw2epoch(raw, locked, **kwargs) for raw in raw_data_list]
    # get evoked data of condition 1 and condition 2
    evokeds1 = [result_list[0] for result_list in result_list]
    evokeds2 = [result_list[1] for result_list in result_list]
    # average across tirals(number of epochs)
    evokeds1 = [np.mean(evoked1,axis=0) for evoked1 in evokeds1]
    evokeds2 = [np.mean(evoked2,axis=0) for evoked2 in evokeds2]
    # transform to numpy array
    evoked1 = [np.array(evokeds1) for evokeds1 in evokeds1]
    evoked2 = [np.array(evokeds2) for evokeds2 in evokeds2]
    # average across subjects
    mean1 = np.mean(evoked1, axis=0)
    std1 = np.std(evoked1, axis=0)
    mean2 = np.mean(evoked2, axis=0)
    std2 = np.std(evoked2, axis=0)

    return (mean1, std1, mean2, std2) # tuple of numpy array



def epoch2df(raw_data_list,rawdata,tmin,tmax,baseline,locked='resp',picks=['CPz','CP1','CP2']):
    '''
    transform epoch data to dataframe and merge them into behavioral dataframe
    ----------
    params:
        raw_data_list: list of raw data
        rawdata: behavioral dataframe
        tmin: start time of epoch
        tmax: end time of epoch
        baseline: baseline of epoch
        locked: 'resp' or 'stim'
        picks: channels to be extracted

    ----------
    return:
        dfs: dataframe of merging epoch and behavioral data
    '''
    # initialize dataframe
    dfs = pd.DataFrame()
    # loop through each subject
    for i in range(len(raw_data_list)):
        # get raw eeg data 
        raw = raw_data_list[i]
        # get behavioral data of each subject
        data = rawdata.loc[rawdata['subj_idx']==np.unique(rawdata['subj_idx'])[i],:]
        
        if locked == 'resp':
            # get epoch data
            # get event
            event, event_dict = mne.events_from_annotations(raw,event_id={
                        'Stimulus/S  5': 2,
                        'Stimulus/S  6': 3})
            # get epoch data
            epoch = mne.Epochs(raw, 
                    event, 
                    event_id=event_dict , 
                    tmin=tmin,
                    tmax=tmax,
                    baseline = baseline,
                    preload=True, 
                    picks=picks)
            # get epoch data and average across channels
            epoch_df = pd.DataFrame(epoch.get_data().mean(axis=1))
            
            # process mismatched epoch and behavioral data
            #
            if len(data)!= len(event):
                # get event id
                new_events = event[:,2].tolist()
                # transform string to int for comparison
                data.loc[data['key_press']=='car','key_press'] = 2
                data.loc[data['key_press']=='face','key_press'] = 3
                # get the index of mismatched event
                boo = np.zeros(len(data)) # 1 for matched, 0 for mismatched
                diff_len = len(data) - len(new_events)
                new_events += [0] * diff_len
                # get the index of mismatched event to store in loc
                loc = [] 
                # loop through each event to check if it is matched
                for i in range(len(boo)):
                    boo = np.array(data['key_press'] == new_events, dtype=np.int32)
                    # add new event to new_events to match the length
                    if not boo[i]:
                        # shift the events after mismatched event to the next trial
                        new_events[i+1:] = new_events[i:-1]
                        # store the index of added event in loc
                        loc.append(i)
                        # add new event to the mismatched event
                        new_events[i] = data.iloc[i]['key_press']
                        # check if the new event is matched
                        boo[i] = 1 if data.iloc[i]['key_press'] == new_events[i] else 0
                # drop the nan response
                data.reset_index(drop=True, inplace=True)
                # drop the mismatched event
                data = data.iloc[~data.index.isin(loc)]
                # transform int back to string
                data.loc[data['key_press']==2,'key_press'] = 'car'
                data.loc[data['key_press']==3,'key_press'] = 'face' 
            else:
                # drop the nan response
                data.reset_index(drop=True, inplace=True)   
        else:
            # get epoch data
            # get event
            event, event_dict = mne.events_from_annotations(raw,event_id={
                'Stimulus/S 10': 4,
                'Stimulus/S 11': 5,
                'Stimulus/S 20': 6,
                'Stimulus/S 21': 7,
                'Stimulus/S 30': 8,
                'Stimulus/S 31': 9,
                'Stimulus/S 40': 10,
                'Stimulus/S 41': 11,})
            # get epoch data
            epoch = mne.Epochs(raw, 
                    event, 
                    event_id=event_dict , 
                    tmin=tmin, 
                    tmax=tmax,
                    baseline = baseline,
                    preload=True, 
                    picks=picks)
            # get epoch data and average across channels
            epoch_df = pd.DataFrame(epoch.get_data().mean(axis=1))
            
            # process mismatched epoch and behavioral data
            #
            if len(data)!= len(event):
                new_events = [1 if x == 4 or x == 5 else 2 if x == 6 or x == 7 else 3 if x == 8 or x == 9 else 4 for x in event[:,2].tolist()]
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
                data = data.iloc[~data.index.isin(loc)]
            else:
                data = data.reset_index(drop=True)

                
        # merge epoch and behavioral data along the column
        df = pd.concat([data,epoch_df],axis=1)
        # merge each subject's data
        dfs = pd.concat([dfs,df]).reset_index(drop=True)
        # drop nan
        dfs.dropna(axis=0,how='any',inplace=True)
        # drop outlier
        dfs = dfs.groupby(['subj_idx']).apply(lambda x: x[(x['rt'] > x['rt'].mean() - x['rt'].std()*3) & (x['rt'] < x['rt'].mean() + x['rt'].std()*3)]).reset_index(drop=True)
    return dfs

def sigtest(cond1,cond2,feature):
    '''
    This function is used to test the significance of the difference between two conditions
    the time window is 0.1s and the slide window is 0.01s
    ----------
    params:
        cond1: dataframe of condition 1
        cond2: dataframe of condition 2
        feature: feature to be tested
    ----------
    return:
        ts: time points of significant difference
    '''
    import scipy.stats as stats
    # initialize dataframe
    ts = []
    # transform time window to time points
    #tw = 0.1
    #tp = round((0.1*512)/2)
    # transform slide window to time points
    #sw = 0.01
    #sp = round(0.01*512)
    sp = 1
    # loop through each slide window
    for i in range(round(2049/sp)):
        #if i*sp+tp > 2048:
        #    break
        # get the feature of each slide window
        if feature == 'amplitude':
            #cond1_win = cond1.iloc[:,(i*sp-tp):(i*sp+tp)].mean(axis=1)
            #cond2_win = cond2.iloc[:,(i*sp-tp):(i*sp+tp)].mean(axis=1)
            cond1_win = cond1.iloc[:,i].mean(axis=1)
            cond2_win = cond2.iloc[:,i].mean(axis=1)
        # t-test
        _,p = stats.ttest_rel(cond1_win,cond2_win)
        # store the time points of significant difference
        if p < 0.05:
            ts.append(i*sp)
    return ts


def smooth(x):
    x2 = x.copy()
    # smooth the data
    for i in range(2049):
        if (round(i-0.1*512) < 0):
            x2.loc[:,i] = x.loc[:,0:round(i+0.1*512)].mean(axis=1)
        elif (round(i+0.1*512) > 2048):
            x2.loc[:,i] = x.loc[:,round(i-0.1*512):2048].mean(axis=1)
        
        else:
            x2.loc[:,i] = x.loc[:,round(i-0.1*512):round(i+0.1*512)].mean(axis=1)
    return x2

def add_feature(x,x2):
    x2['pam'] = x.loc[:,round((-0.05+3)*512):round((0.05+3)*512)].mean(axis=1)
    x2['slp'] = x2.loc[:, round((-0.18+3)*512):round((-0.08+3)*512)].apply(lambda x: np.polyfit(range(np.shape(x)[0]), x, 1)[0],axis=1)
    x2['am'] = x.loc[:,round((-0.18+3)*512):round((-0.08+3)*512)].mean(axis=1)
    x2['pkl'] = x2.rt + ((x2.loc[:,round((-0.1+3)*512):round((0.1+3)*512)].apply(lambda x: np.argmax(x),axis=1)+round((-0.1+3)*512))/512-3)
    # calculate the quantile of each subject
    x2.insert(loc=2, column='slp_quantile', value=x2.groupby(['subj_idx','coherence']).slp.apply(lambda x: pd.qcut(x, q=4, labels=['1st','2nd','3rd','4th'])))
    x2.insert(loc=2, column='am_quantile', value=x2.groupby(['subj_idx','coherence']).am.apply(lambda x: pd.qcut(x, q=4, labels=['1st','2nd','3rd','4th'])))
    x2.insert(loc=2, column='pam_quantile', value=x2.groupby(['subj_idx','coherence']).pam.apply(lambda x: pd.qcut(x, q=4, labels=['1st','2nd','3rd','4th'])))
    return x2

def sigtest(cond1,cond2,feature):
    '''
    This function is used to test the significance of the difference between two conditions
    the time window is 0.1s and the slide window is 0.01s
    ----------
    params:
        cond1: dataframe of condition 1
        cond2: dataframe of condition 2
        feature: feature to be tested
    ----------
    return:
        ts: time points of significant difference
    '''
    import scipy.stats as stats
    # initialize dataframe
    ts = []
    # transform time window to time points
    tw = 0.1
    tp = round((0.1*512)/2)
    # transform slide window to time points
    sw = 0.01
    sp = round(0.01*512)

    # loop through each slide window
    for i in range(round(2049/sp)):
        if i*sp+tp > 2048:
            break
        # get the feature of each slide window
        if feature == 'amplitude':
            cond1_win = cond1.iloc[:,(i*sp-tp):(i*sp+tp)].mean(axis=1)
            cond2_win = cond2.iloc[:,(i*sp-tp):(i*sp+tp)].mean(axis=1)
        elif feature == 'slope':
            cond1_win = cond1.iloc[:,(i*sp-tp):(i*sp+tp)].apply(lambda x: np.polyfit(range(len(x)),x,1)[0],axis=1) # np.ployfit(x,y,deg)[0], First order linear item
            cond2_win = cond2.iloc[:,(i*sp-tp):(i*sp+tp)].apply(lambda x: np.polyfit(range(len(x)),x,1)[0],axis=1)    
        elif feature == 'peak':
            cond1_win = cond1.iloc[:,(i*sp-tp):(i*sp+tp)].apply(lambda x: np.max(x),axis=1)
            cond2_win = cond2.iloc[:,(i*sp-tp)(i*sp+tp)].apply(lambda x: np.max(x),axis=1)
        # t-test
        _,p = stats.ttest_rel(cond1_win,cond2_win)
        # store the time points of significant difference
        if p < 0.05:
            ts.append(i*sp)
    return ts


def permutation(arr1, arr2, iter_num):
    """
    This function is to calculate the permutation test for two groups of data
        arr1: the first group of data
        arr2: the second group of data
        iter_num: the number of iteration
    """
    def rand_num(arr1, arr2, iter_num):

        target = arr1.mean() - arr2.mean()
        merged_arr = np.concatenate((arr1, arr2))

        np.random.shuffle(merged_arr)

        group1 = merged_arr[:len(merged_arr)//2]
        group2 = merged_arr[len(merged_arr)//2:]
        diff = group1.mean() - group2.mean()
        return target,diff

    def percentile(arr, target):
        arr = np.concatenate((arr,[target]))
        arr_sorted = sorted(arr)
        rank = arr_sorted.index(target) + 1
        pct = (rank / len(arr_sorted)) * 100
        return pct

    dist = pd.DataFrame({'value':np.zeros(iter_num)}).apply(lambda x: rand_num(arr1, arr2, iter_num)[1], axis=1)
    target = rand_num(arr1, arr2, iter_num)[0]
    return percentile(dist, target)


def bootstrap(arr1, arr2, n_bootstrap):
    '''
    This function is to calculate the bootstrap test for two groups of data
    '''
    data = np.concatenate((arr1, arr2))
    target = arr1.mean() - arr2.mean()
    n_bootstrap = 100000

    bootstrap = []
    for i in range(n_bootstrap):
        ar1 = np.random.choice(data, size=len(data)//2, replace=True)
        ar2 = np.random.choice(data, size=len(data)//2, replace=True)
        stat = ar1.mean()-ar2.mean()
        bootstrap.append(stat)
    arr = np.concatenate((bootstrap,[target]))
    arr_sorted = sorted(arr)
    rank = arr_sorted.index(target) + 1
    pct = (rank / len(arr_sorted)) * 100
    return pct