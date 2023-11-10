from base import gat_stats
import os.path as op
import os
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from tqdm import tqdm

path_data = '/Users/romainquentin/Desktop/data/MEG_demarchi/MEG'
path_results = '/Users/romainquentin/Desktop/data/MEG_demarchi/results'
# list all data files for each condition
MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
MEG_mms = sorted([f for f in os.listdir(path_data) if 'midminus' in f])
MEG_mps = sorted([f for f in os.listdir(path_data) if 'midplus' in f])
MEG_ors = sorted([f for f in os.listdir(path_data) if 'ordered' in f])
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
# Start the main loop analysis - Loop on participant
for meg_rd, meg_mm, meg_mp, meg_or in zip(MEG_rds, MEG_mms, MEG_mps, MEG_ors):
    # results folder where to save the scores for one participant
    participant = meg_or[:12]
    results_folder = op.join(path_results, participant, 'reorder_random_omission')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Add the path to each data file
    meg_rd = op.join(path_data, meg_rd)
    meg_mm = op.join(path_data, meg_mm)
    meg_mp = op.join(path_data, meg_mp)
    meg_or = op.join(path_data, meg_or)
    # Read raw files
    raw_rd = mne.io.read_raw_fif(meg_rd, preload=True)
    raw_mm = mne.io.read_raw_fif(meg_mm, preload=True)
    raw_mp = mne.io.read_raw_fif(meg_rd, preload=True)
    raw_or = mne.io.read_raw_fif(meg_or, preload=True)
    # Get events 
    events_rd = mne.find_events(raw_rd, shortest_event=1)
    events_mm = mne.find_events(raw_mm, shortest_event=1)
    events_mp = mne.find_events(raw_mp, shortest_event=1)
    events_or = mne.find_events(raw_or, shortest_event=1)
    # Filter the raw
    raw_rd.filter(0.1, 30)
    raw_mm.filter(0.1, 30)
    raw_mp.filter(0.1, 30)
    raw_or.filter(0.1, 30)
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
    # Save an epochs_rd to start from at each iteration on the decoding the reorders
    epochs_rd_init = epochs_rd.copy()
    
    # reorder random raw as ordered
    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    ordered_events = list(epochs_or.events)
    events_orrd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xorrd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xorrd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    new_sample+=200
    for event in tqdm(ordered_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            if event[2] in [10, 20, 30, 40]:
                orig_nums.append(random_events_numbers[index])
            samp = random_events[index, 0] - first_samp
            raw_Xorrd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_orrd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    raw_Xorrd.append(raw_Xrd[:, -200:])  # end the reorderd random with the 2 last seconds of the random raw
    orig_nums_orrd = np.array(orig_nums)[1:-1]  # removing the first and last trials
    events_orrd = np.array(events_orrd)[1:-1]  # removing the first and last trials
    raw_Xorrd = np.concatenate(raw_Xorrd, axis=1)
    raw_orrd = mne.io.RawArray(raw_Xorrd, raw_rd.info)

    # reorder random raw as midplus
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
            if event[2] in [10, 20, 30, 40]:
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

    # reorder random raw as midminus
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
            if event[2] in [10, 20, 30, 40]:
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

    # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
    # create an epoch from the reordered raw random 
    epochs_orrd = mne.Epochs(raw_orrd, events_orrd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)[1:-1]
    # keep same trials in epochs_rd and epochs_orrd
    epochs_rd = epochs_rd_init[np.sort(orig_nums_orrd)]
    # keep only the same number of trials in ordered
    epochs_or = epochs_or[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xor = epochs_or.get_data()
    yor = epochs_or.events[:, 2]
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

    # Run cross validation for the midminus (and reorder-midminus)
    # create an epoch from the reordered raw random 
    epochs_mmrd = mne.Epochs(raw_mmrd, events_mmrd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)[1:-1]
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
    cv_rd_to_rd_scores = list()
    cv_rd_to_mm_scores = list()
    cv_rd_to_mmrd_scores = list()
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

    # Run cross validation for the midplus (and reorder-midplus)
    # create an epoch from the reordered raw random 
    epochs_mprd = mne.Epochs(raw_mprd, events_mprd,
                             event_id=[10, 20, 30, 40],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)[1:-1]
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

# create arrays with cross-validated scores
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)
all_cv_rd_to_orrd_scores = np.array(all_cv_rd_to_orrd_scores)
all_cv_rd_to_mp_scores = np.array(all_cv_rd_to_mp_scores)
all_cv_rd_to_mprd_scores = np.array(all_cv_rd_to_mprd_scores)
all_cv_rd_to_mm_scores = np.array(all_cv_rd_to_mm_scores)
all_cv_rd_to_mmrd_scores = np.array(all_cv_rd_to_mmrd_scores)

