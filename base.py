import numpy as np
import pandas as pd
import seaborn as sns

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
