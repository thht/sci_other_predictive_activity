import numpy as np
import pandas as pd
import seaborn as sns
import mne

canon_subj_order =  ['19830114RFTM', '19921111BRHC', '19930630MNSU', '19930118IMSH',
       '19861231ANSR', '19851130EIFI', '19930524GNTA', '19871229CRBC',
       '19940403HIEC', '19900606KTAD', '19880328AGSG', '19950326IIQI',
       '19750430PNRK', '19950212BTKC', '19930423EEHB', '19960418GBSH',
       '11920812IONP', '19950604LLZM', '19800616MRGU', '19950905MCSH',
       '19891222GBHL', '19940930SZDT', '19960304SBPE', '19821223KRHR',
       '19920804CRLE', '19810726GDZN', '19960708HLHY', '19810918SLBR',
       '19940601IGSH', '19961118BRSH', '19901026KRKE', '19930621ATLI',
       '19910823SSLD']
corresp = dict( zip(canon_subj_order, np.arange(33) ) )
events_omission = [10,20,30,40]
events_sound = [ 1,2,3,4]

# trans mat ordered
#M = np.zeros((4,4))
#M += np.diag(4*[0.25])
#i = 0
#for c in range(4):
#    M[i,(i+1) % 4] = 0.75; i+= 1
# M[to,from]
colors_ordered = dict(zip(np.arange(4),  ['blue','cyan','yellow','red'] ) )
M = np.array([[0.25, 0.75, 0.  , 0.  ],
                [0.  , 0.25, 0.75, 0.  ],
                [0.  , 0.  , 0.25, 0.75],
                [0.75, 0.  , 0.  , 0.25]])

def events_simple_pred(events):
    assert events.ndim == 2
    r = np.zeros_like(events)
    assert np.min(events[:,2]) == 1
    assert np.max(events[:,2]) == 4

    r[:,:2] = events[:,:2]
    r[0,2] = events[0,2]
    for i in range(1, events.shape[0]):
        prev_stim = events[i-1,2]
        pred_stim = M[:,prev_stim-1].argmax() 
        r[i,2] = pred_stim + 1
    return r


def reorder(random_events, events, raw_rd, del_processed = True,
        cut_fl = 0):
    events = list(events)

    events_reord = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_reord = list()  # the reordered X (based on yor), first contains data extracted from raws
    new_sample = 0  # keep track of the current sample to create the reordered events
    # DQ: why 200?
    raw_reord.append(raw_Xrd[:, :dur])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    #random_events_processed = []
    orig_nums = list()
    new_sample+=dur

    # Romain's version
    if del_processed:
        random_events_numbers = np.arange(len(random_events)) # indices of random events
        for event in tqdm(events):
            # event[2] is the event code
            # note that random_events changes on every iteration potentially
            # random events is actually _yet unprocessed_ random events
            if event[2] in random_events[:, 2]:
                # take the index of the ordered event as it is present in random events (need to delete it later)
                # index of first not processed with save code
                index = random_events[:, 2].tolist().index(event[2])

                orig_nums.append(random_events_numbers[index])
                samp = random_events[index, 0] - first_samp
                # DQ: why 33?
                raw_reord.append(raw_Xrd[:, samp:samp+nsamples])
                random_events = np.delete(random_events, index, axis=0)
                random_events_numbers = np.delete(random_events_numbers, index, axis=0)

                # simple artificial sample indices
                events_reord.append([new_sample, 0, event[2]])
                new_sample+=nsamples
            else:
                pass
    else:
        random_events_aug = np.concatenate([  random_events, 
            np.arange(len(random_events))[:,None]], axis=1 )
        was_processed = np.zeros(len(random_events), dtype=bool )
        for event in tqdm(events):
            random_events_aug_sub = random_events_aug[~was_processed]
            inds = np.where( random_events_aug_sub[:,2] == event[2])[0]
            #inds2 = np.where(~was_processed[inds] )[0]
            if len(inds) == 0:
                continue
            else:
                evt = random_events_aug_sub[inds[0]]
                index = evt[3]  # index of random event in orig array

                orig_nums.append(index)
                samp = evt[0] - first_samp
                raw_reord.append(raw_Xrd[:, samp:samp+nsamples])

                was_processed[index] = True

                # simple artificial sample indices
                events_reord.append([new_sample, 0, event[2]])
                new_sample+=nsamples

    raw_reord.append(raw_Xrd[:, -dur:])  # end the reorderd random with the 2 last seconds of the random raw
    # will be used to define epochs_rd
    orig_nums_reord = np.array(orig_nums)  
    events_reord = np.array(events_reord)  
    # removing the first and last trials
    if cut_fl:
        orig_nums_reord = orig_nums_reord[1:-1]
        events_reord = events_reord[1:-1]
    raw_reord = np.concatenate(raw_reord, axis=1)
    raw_reord = mne.io.RawArray(raw_reord, raw_rd.info)

    epochs_reord = mne.Epochs(raw_reord, events_reord,
         event_id=events_sound,
         tmin=tmin, tmax=tmax, baseline=None, preload=True)

    return epochs_reord, orig_nums_reord


def create_seq(n_trials, structure):
    """
    This function create a sequence with n_trials and a specific structure from random to structure (see Demarchi et al. 2019)
    :param n_trials: number of trials
    :param structure: integer (from 0 to 3, 0 being random)
    :return: dataFrame
    """
    sounds = [0, 1, 2, 3]
    trials = np.arange(n_trials)
    if structure == 0:
        data = pd.DataFrame({'trials': trials,
                             'sound': np.random.randint(4, size=n_trials)})
    elif structure == 3:
        data = pd.DataFrame({'trials': trials,
                             'sound': np.zeros(len(trials), dtype=int)})
        for trial in trials:
            # Initialize the first key of each block (random)
            if trial == 0:
                data.sound.at[trial] = np.random.choice(sounds)
            else:
                rnd = np.random.random()
                if data.sound[trial-1] == sounds[0]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[1]
                    else:
                        data.sound.at[trial] = sounds[0]
                elif data.sound[trial-1] == sounds[1]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[2]
                    else:
                        data.sound.at[trial] = sounds[1]
                elif data.sound[trial-1] == sounds[2]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[3]
                    else:
                        data.sound.at[trial] = sounds[2]
                elif data.sound[trial-1] == sounds[3]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[0]
                    else:
                        data.sound.at[trial] = sounds[3]
    return data


def decod_stats(X):
    from mne.stats import permutation_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**12, n_jobs=6,
        verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)


def gat_stats(X):
    from mne.stats import spatio_temporal_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask',
        n_permutations=2**12, n_jobs=-1, verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_).T
