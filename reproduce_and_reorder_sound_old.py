from base import gat_stats
import os.path as op
import os, sys
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from tqdm import tqdm

import os.path as op
import mne
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import GeneralizingEstimator

from base import corresp

path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results0'
# list all data files for each condition
# DWARNING: this assumes same ordering of files (participants)
#MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
#MEG_mms = sorted([f for f in os.listdir(path_data) if 'midminus' in f])
#MEG_mps = sorted([f for f in os.listdir(path_data) if 'midplus' in f])
#MEG_ors = sorted([f for f in os.listdir(path_data) if 'ordered' in f])
## Start the main loop analysis - Loop on participant
#for meg_rd, meg_mm, meg_mp, meg_or in zip(MEG_rds, MEG_mms, MEG_mps, MEG_ors):
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_cv_rd_to_or_scores = list()
all_cv_rd_to_orrd_scores = list()
all_cv_rd_to_mm_scores = list()
all_cv_rd_to_mmrd_scores = list()
all_cv_rd_to_mp_scores = list()
all_cv_rd_to_mprd_scores = list()
# define tmin and tmax
tmin, tmax = -0.7, 0.7
del_processed = 1

print('sys argv = ',sys.argv)
if len(sys.argv) < 2:
    print('Print supply subject ID number from [0-32]')
    sys.exit(1)
#subjs_to_use = ['19750430PNRK']
sids_to_use = [int(sys.argv[1])]
#sids_to_use = [12]

import pandas as pd
rows = [ dict(zip(['subj','block','cond','path'], v[:-15].split('_') + [v])  ) for v in os.listdir(path_data)]
df = pd.DataFrame(rows)
df['sid'] = df['subj'].apply(lambda x: corresp[x])

df = df.query('sid.isin(@sids_to_use)')
#TODO: run with arg of bad subject

grp = df.groupby(['subj'])
assert grp.size().min() == grp.size().max()
assert grp.size().min() == 4

for g,inds in grp.groups.items():
    subdf = df.loc[inds]

    subdf= subdf.set_index('cond')
    subdf = subdf.drop(columns=['subj','block','sid'])

    meg_rd = subdf.to_dict('index')['random']['path']
    meg_mm = subdf.to_dict('index')['midminus']['path']
    meg_mp = subdf.to_dict('index')['midplus']['path']
    meg_or = subdf.to_dict('index')['ordered']['path']
    print(meg_rd, meg_mm, meg_mp, meg_or)

    # results folder where to save the scores for one participant
    ps = [p[:12] for p in [meg_rd, meg_mm, meg_mp, meg_or] ]
    assert len(set(ps) ) == 1

    participant = meg_or[:12]
    print('------------------------------------------------------')
    print('---------------------- Starting participant', participant)
    print('------------------------------------------------------')
    results_folder = op.join(path_results, participant, 'reorder_random')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    p0 = op.join( os.path.expandvars('$SCRATCH/memerr/demarchi') , meg_rd[:-15] )
    if op.exists(op.join(p0, 'flt_rd-raw.fif')):
        print('!!!!!   Loading precomputed filtered raws from ',p0)
        raw_rd = mne.io.read_raw_fif(op.join(p0, 'flt_rd-raw.fif'))
        raw_mm = mne.io.read_raw_fif(op.join(p0, 'flt_mm-raw.fif'))
        raw_mp = mne.io.read_raw_fif(op.join(p0, 'flt_mp-raw.fif'))
        raw_or = mne.io.read_raw_fif(op.join(p0, 'flt_or-raw.fif'))    
    else:
        # Add the path to each data file
        meg_rd1 = op.join(path_data, meg_rd)
        meg_mm1 = op.join(path_data, meg_mm)
        meg_mp1 = op.join(path_data, meg_mp)
        meg_or1 = op.join(path_data, meg_or)
        # Read raw files
        raw_rd = mne.io.read_raw_fif(meg_rd1, preload=True)
        raw_mm = mne.io.read_raw_fif(meg_mm1, preload=True)
        raw_mp = mne.io.read_raw_fif(meg_mp1, preload=True)
        raw_or = mne.io.read_raw_fif(meg_or1, preload=True)
        # Filter the raw
        print('Filtering ')
        raw_rd.filter(0.1, 30, n_jobs=-1)
        raw_mm.filter(0.1, 30, n_jobs=-1)
        raw_mp.filter(0.1, 30, n_jobs=-1)
        raw_or.filter(0.1, 30, n_jobs=-1)

        if not op.exists(p0):
            os.makedirs(p0)
        raw_rd.save( op.join(p0, 'flt_rd-raw.fif'), overwrite = True )
        raw_mm.save( op.join(p0, 'flt_mm-raw.fif'), overwrite = True )
        raw_mp.save( op.join(p0, 'flt_mp-raw.fif'), overwrite = True )
        raw_or.save( op.join(p0, 'flt_or-raw.fif'), overwrite = True )

    # Get events
    events_rd = mne.find_events(raw_rd, shortest_event=1)
    events_mm = mne.find_events(raw_mm, shortest_event=1)
    events_mp = mne.find_events(raw_mp, shortest_event=1)
    events_or = mne.find_events(raw_or, shortest_event=1)

    # Create epochs
    raw_rd.pick_types(meg=True, eog=False, ecg=False,
                  ias=False, stim=False, syst=False)
    raw_mm.pick_types(meg=True, eog=False, ecg=False,
                  ias=False, stim=False, syst=False)
    raw_mp.pick_types(meg=True, eog=False, ecg=False,
                  ias=False, stim=False, syst=False)
    raw_or.pick_types(meg=True, eog=False, ecg=False,
                  ias=False, stim=False, syst=False)

    epochs_rd = mne.Epochs(raw_rd, events_rd,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_mm = mne.Epochs(raw_mm, events_mm,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_mp = mne.Epochs(raw_mp, events_mp,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_or = mne.Epochs(raw_or, events_or,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # remove omission and following trials in random trials
    # get indices of epochs having special even types
    om_rd = np.where(np.isin(epochs_rd.events, [10, 20, 30, 40]))[0]
    # take next indices after them and sort indices
    om_fo_rd = np.sort(np.concatenate([om_rd, om_rd+1]))
    # if the last one is not an index, remov it
    if om_fo_rd[-1] == len(epochs_rd.events):
        om_fo_rd = np.delete(om_fo_rd, -1)
    # remove these indiece
    epochs_rd.drop(om_fo_rd)
    # Save an epochs_rd to start from at each iteration on the decoding the reorders
    epochs_rd_init = epochs_rd.copy()
    # remove omission and following trials in midminus trials
    om_mm = np.where(np.isin(epochs_mm.events, [10, 20, 30, 40]))[0]
    om_fo_mm = np.sort(np.concatenate([om_mm, om_mm+1]))
    if om_fo_mm[-1] == len(epochs_mm.events):
        om_fo_mm = np.delete(om_fo_mm, -1)
    epochs_mm.drop(om_fo_mm)
    # remove omission and following trials in midplus trials
    om_mp = np.where(np.isin(epochs_mp.events, [10, 20, 30, 40]))[0]
    om_fo_mp = np.sort(np.concatenate([om_mp, om_mp+1]))
    if om_fo_mp[-1] == len(epochs_mp.events):
        om_fo_mp = np.delete(om_fo_mp, -1)
    epochs_mp.drop(om_fo_mp)
    # remove omission and following trials in ordered trials
    om_or = np.where(np.isin(epochs_or.events, [10, 20, 30, 40]))[0]
    om_fo_or = np.sort(np.concatenate([om_or, om_or+1]))
    if om_fo_or[-1] == len(epochs_or.events):
        om_fo_or = np.delete(om_fo_or, -1)
    epochs_or.drop(om_fo_or)

    ################################################################
    # reorder random raw as ordered
    ################################################################

    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    random_events_aug = np.concatenate([  random_events, 
        np.arange(len(random_events))[:,None]], axis=1 )

    ordered_events = list(epochs_or.events)
    events_orrd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xorrd = list()  # the reordered X (based on yor), first contains data extracted from raws
    new_sample = 0  # keep track of the current sample to create the reordered events
    # DQ: why 200?
    raw_Xorrd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    #random_events_processed = []
    orig_nums = list()
    new_sample+=200
    if del_processed:
        random_events_numbers = np.arange(len(random_events)) # indices of random events
        for event in tqdm(ordered_events):
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
                raw_Xorrd.append(raw_Xrd[:, samp:samp+33])
                random_events = np.delete(random_events, index, axis=0)
                random_events_numbers = np.delete(random_events_numbers, index, axis=0)

                # simple artificial sample indices
                events_orrd.append([new_sample, 0, event[2]])
                new_sample+=33
            else:
                pass
    else:
        was_processed = np.zeros(len(random_events), dtype=bool )
        for event in tqdm(ordered_events):
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
                raw_Xorrd.append(raw_Xrd[:, samp:samp+33])

                was_processed[index] = True

                # simple artificial sample indices
                events_orrd.append([new_sample, 0, event[2]])
                new_sample+=33

    raw_Xorrd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    # will be used to define epochs_rd
    orig_nums_orrd = np.array(orig_nums)[1:-1]  # removing the first and last trials
    events_orrd = np.array(events_orrd)[1:-1]  # removing the first and last trials
    raw_Xorrd = np.concatenate(raw_Xorrd, axis=1)
    raw_orrd = mne.io.RawArray(raw_Xorrd, raw_rd.info)

    assert len(set(orig_nums_orrd)) > 4
    ################################################################
    # reorder random raw as midplus
    ################################################################

    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    midplus_events = list(epochs_mp.events)
    events_mprd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xmprd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xmprd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    new_sample+=200
    for event in tqdm(midplus_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_Xmprd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_mprd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    raw_Xmprd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    orig_nums_mprd = np.array(orig_nums)[1:-1]  # removing the first and last trials
    events_mprd = np.array(events_mprd)[1:-1]  # removing the first and last trials
    raw_Xmprd = np.concatenate(raw_Xmprd, axis=1)
    raw_mprd = mne.io.RawArray(raw_Xmprd, raw_rd.info)

    ################################################################
    # reorder random raw as midminus
    ################################################################

    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    midminus_events = list(epochs_mm.events)
    events_mmrd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xmmrd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xmmrd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    new_sample+=200
    for event in tqdm(midminus_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_Xmmrd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_mmrd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    raw_Xmmrd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    orig_nums_mmrd = np.array(orig_nums)[1:-1]  # removing the first and last trials
    events_mmrd = np.array(events_mmrd)[1:-1]  # removing the first and last trials
    raw_Xmmrd = np.concatenate(raw_Xmmrd, axis=1)
    raw_mmrd = mne.io.RawArray(raw_Xmmrd, raw_rd.info)

    ###################################################################
    ########################     CV
    ###################################################################
    print("------------   Starting CV")

    print("##############  Starting ordered")
    # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
    # create an epoch from the reordered raw random
    # DQ: why these event ids?
    epochs_orrd = mne.Epochs(raw_orrd, events_orrd,
                             event_id=[1, 2, 3, 4],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_orrd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_orrd)]
    # keep only the same number of trials in ordered
    epochs_or = epochs_or[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xor = epochs_or.get_data()
    yor = epochs_or.events[:, 2] # events orderd
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xorrd = epochs_orrd.get_data()
    yorrd = epochs_orrd.events[:, 2]
    # Initialize classifier
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_or_scores = list()
    cv_rd_to_orrd_scores = list()
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_rd_score = clf.score(Xrd[test_rd], yrd[test_rd])
        cv_rd_to_or_score = clf.score(Xor[test_rd], yor[test_rd])
        test_orrd = np.isin(orig_nums_orrd, test_rd)  # why sum(test_orrd) != len(test_rd)
        cv_rd_to_orrd_score = clf.score(Xorrd[test_orrd], yorrd[test_orrd])
        cv_rd_to_rd_scores.append(cv_rd_to_rd_score)
        cv_rd_to_or_scores.append(cv_rd_to_or_score)
        cv_rd_to_orrd_scores.append(cv_rd_to_orrd_score)
    cv_rd_to_rd_scores = np.array(cv_rd_to_rd_scores)
    cv_rd_to_or_scores = np.array(cv_rd_to_or_scores)
    cv_rd_to_orrd_scores = np.array(cv_rd_to_orrd_scores)
    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_rd_scores.npy'), cv_rd_to_rd_scores)
    np.save(op.join(results_folder, 'cv_rd_to_or_scores.npy'), cv_rd_to_or_scores)
    np.save(op.join(results_folder, 'cv_rd_to_orrd_scores.npy'), cv_rd_to_orrd_scores)
    # append to keep the results in the python session
    all_cv_rd_to_rd_scores.append(cv_rd_to_rd_scores.mean(0))
    all_cv_rd_to_or_scores.append(cv_rd_to_or_scores.mean(0))
    all_cv_rd_to_orrd_scores.append(cv_rd_to_orrd_scores.mean(0))

    ###################################################################
    ########################     CV
    ###################################################################
    print("#################  Starting midminus")

    # Run cross validation for the midminus (and reorder-midminus)
    # create an epoch from the reordered raw random
    epochs_mmrd = mne.Epochs(raw_mmrd, events_mmrd,
                             event_id=[1, 2, 3, 4],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_mmrd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_mmrd)]
    # keep only the same number of trials in midminus
    epochs_mm = epochs_mm[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xmm = epochs_mm.get_data()
    ymm = epochs_mm.events[:, 2]
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xmmrd = epochs_mmrd.get_data()
    ymmrd = epochs_mmrd.events[:, 2]
    # Initialize classifier
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train and test with cross-validation
    #cv_rd_to_rd_scores = list()
    cv_rd_to_mm_scores = list()
    cv_rd_to_mmrd_scores = list()
    # DTODO: check numbers of each class before stratifying
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_mm_score = clf.score(Xmm[test_rd], ymm[test_rd])
        test_mmrd = np.isin(orig_nums_mmrd, test_rd)  # why sum(test_mmrd) != len(test_rd)
        cv_rd_to_mmrd_score = clf.score(Xmmrd[test_mmrd], ymmrd[test_mmrd])
        cv_rd_to_mm_scores.append(cv_rd_to_mm_score)
        cv_rd_to_mmrd_scores.append(cv_rd_to_mmrd_score)
    cv_rd_to_mm_scores = np.array(cv_rd_to_mm_scores)
    cv_rd_to_mmrd_scores = np.array(cv_rd_to_mmrd_scores)
    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_mm_scores.npy'), cv_rd_to_mm_scores)
    np.save(op.join(results_folder, 'cv_rd_to_mmrd_scores.npy'), cv_rd_to_mmrd_scores)
    # append to keep the results in the python session
    all_cv_rd_to_mm_scores.append(cv_rd_to_mm_scores.mean(0))
    all_cv_rd_to_mmrd_scores.append(cv_rd_to_mmrd_scores.mean(0))

    ################################################################
    # Run cross validation for the midplus (and reorder-midplus)
    ################################################################
    print("################### Starting midplus")

    # create an epoch from the reordered raw random
    epochs_mprd = mne.Epochs(raw_mprd, events_mprd,
                             event_id=[1, 2, 3, 4],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_mprd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_mprd)]
    # keep only the same number of trials in midplus
    epochs_mp = epochs_mp[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xmp = epochs_mp.get_data()
    ymp = epochs_mp.events[:, 2]
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xmprd = epochs_mprd.get_data()
    ymprd = epochs_mprd.events[:, 2]
    # Initialize classifier
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_mp_scores = list()
    cv_rd_to_mprd_scores = list()
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_mp_score = clf.score(Xmp[test_rd], ymp[test_rd])
        test_mprd = np.isin(orig_nums_mprd, test_rd)  # why sum(test_mprd) != len(test_rd)
        cv_rd_to_mprd_score = clf.score(Xmprd[test_mprd], ymprd[test_mprd])
        cv_rd_to_mp_scores.append(cv_rd_to_mp_score)
        cv_rd_to_mprd_scores.append(cv_rd_to_mprd_score)
    cv_rd_to_mp_scores = np.array(cv_rd_to_mp_scores)
    cv_rd_to_mprd_scores = np.array(cv_rd_to_mprd_scores)
    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_mp_scores.npy'), cv_rd_to_mp_scores)
    np.save(op.join(results_folder, 'cv_rd_to_mprd_scores.npy'), cv_rd_to_mprd_scores)
    # append to keep the results in the python session
    all_cv_rd_to_mp_scores.append(cv_rd_to_mp_scores.mean(0))
    all_cv_rd_to_mprd_scores.append(cv_rd_to_mprd_scores.mean(0))

    del raw_rd, raw_mm, raw_mp, raw_or
    import gc; gc.collect()

## create arrays with cross-validated scores
# all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
# all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)
# all_cv_rd_to_orrd_scores = np.array(all_cv_rd_to_orrd_scores)
# all_cv_rd_to_mp_scores = np.array(all_cv_rd_to_mp_scores)
# all_cv_rd_to_mprd_scores = np.array(all_cv_rd_to_mprd_scores)
# all_cv_rd_to_mm_scores = np.array(all_cv_rd_to_mm_scores)
# all_cv_rd_to_mmrd_scores = np.array(all_cv_rd_to_mmrd_scores)
