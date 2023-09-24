import os.path as op
import os
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

path_data = '/Users/romainquentin/Desktop/data/MEG_demarchi/MEG'
path_results = '/Users/romainquentin/Desktop/data/MEG_demarchi/results'
# list all data files for each condition
MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
MEG_mms = sorted([f for f in os.listdir(path_data) if 'midminus' in f])
MEG_mps = sorted([f for f in os.listdir(path_data) if 'midplus' in f])
MEG_ors = sorted([f for f in os.listdir(path_data) if 'ordered' in f])
# Initialize list of scores (no cross-validation)
all_rd_to_mm_scores = list()
all_rd_to_mp_scores = list()
all_rd_to_or_scores = list()
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_cv_rd_to_mm_scores = list()
all_cv_rd_to_mp_scores = list()
all_cv_rd_to_or_scores = list()

# Start the main loop analysis - Loop on participant
for meg_rd, meg_mm, meg_mp, meg_or in zip(MEG_rds, MEG_mms, MEG_mps, MEG_ors):
    # results folder where to save the scores for one participant
    participant = meg_or[:12]
    results_folder = op.join(path_results, participant, 'initial_reproduction')
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
    raw_mp = mne.io.read_raw_fif(meg_mp, preload=True)
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
    epochs_rd = mne.Epochs(raw_rd, events_rd,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=-1, tmax=1, baseline=None, preload=True)
    epochs_rd.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    epochs_mm = mne.Epochs(raw_mm, events_mm,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=-1, tmax=1, baseline=None, preload=True)
    epochs_mm.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    epochs_mp = mne.Epochs(raw_mp, events_mp,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=-1, tmax=1, baseline=None, preload=True)
    epochs_mp.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    epochs_or = mne.Epochs(raw_or, events_or,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=-1, tmax=1, baseline=None, preload=True)
    epochs_or.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # get the X and Y for each condition in numpy array
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xmm = epochs_mm.get_data()
    ymm = epochs_mm.events[:, 2]
    Xmp = epochs_mp.get_data()
    ymp = epochs_mp.events[:, 2]
    Xor = epochs_or.get_data()
    yor = epochs_or.events[:, 2]
    # remove omission and following trials
    om_rd = np.where(np.isin(yrd, [10, 20, 30, 40]))[0]
    om_fo_rd = np.sort(np.concatenate([om_rd, om_rd+1]))
    if om_fo_rd[-1] == len(yrd):
        om_fo_rd = np.delete(om_fo_rd, -1)
    Xrd = np.delete(Xrd, om_fo_rd, axis=0)
    yrd = np.delete(yrd, om_fo_rd, axis=0)
    om_mm = np.where(np.isin(ymm, [10, 20, 30, 40]))[0]
    om_fo_mm = np.sort(np.concatenate([om_mm, om_mm+1]))
    if om_fo_mm[-1] == len(ymm):
        om_fo_mm = np.delete(om_fo_mm, -1)
    Xmm = np.delete(Xmm, om_fo_mm, axis=0)
    ymm = np.delete(ymm, om_fo_mm, axis=0)
    om_mp = np.where(np.isin(ymp, [10, 20, 30, 40]))[0]
    om_fo_mp = np.sort(np.concatenate([om_mp, om_mp+1]))
    if om_fo_mp[-1] == len(ymp):
        om_fo_mp = np.delete(om_fo_mp, -1)
    Xmp = np.delete(Xmp, om_fo_mp, axis=0)
    ymp = np.delete(ymp, om_fo_mp, axis=0)
    om_or = np.where(np.isin(yor, [10, 20, 30, 40]))[0]
    om_fo_or = np.sort(np.concatenate([om_or, om_or+1]))
    if om_fo_or[-1] == len(yor):
        om_fo_or = np.delete(om_fo_or, -1)
    Xor = np.delete(Xor, om_fo_or, axis=0)
    yor = np.delete(yor, om_fo_or, axis=0) 
    
    # train on all random trials (no cross-validation)
    clf.fit(Xrd, yrd)
    # test on all midminus trials
    rd_to_mm_scores = clf.score(Xmm, ymm)
    # test on all midplus trials
    rd_to_mp_scores = clf.score(Xmp, ymp)
    # test on all ordered trials
    rd_to_or_scores = clf.score(Xor, yor)
    # save scores (no cross-validation)
    np.save(op.join(results_folder, 'rd_to_mm_scores.npy'), rd_to_mm_scores)
    np.save(op.join(results_folder, 'rd_to_mp_scores.npy'), rd_to_mp_scores)
    np.save(op.join(results_folder, 'rd_to_or_scores.npy'), rd_to_or_scores)
    # keep these non-crossvalidated scores in a list
    all_rd_to_mm_scores.append(rd_to_mm_scores)
    all_rd_to_mp_scores.append(rd_to_mp_scores)
    all_rd_to_or_scores.append(rd_to_or_scores)
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_mm_scores = list()
    cv_rd_to_mp_scores = list()
    cv_rd_to_or_scores = list()
    cv = StratifiedKFold(5)
    for ((train_rd, test_rd),
         (train_mm, test_mm),
         (train_mp, test_mp),
         (train_or, test_or)) in zip(cv.split(Xrd, yrd),
                                     cv.split(Xmm, ymm),
                                     cv.split(Xmp, ymp),
                                     cv.split(Xor, yor)):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_rd_score = clf.score(Xrd[test_rd], yrd[test_rd])
        cv_rd_to_mm_score = clf.score(Xmm[test_mm], ymm[test_mm])
        cv_rd_to_mp_score = clf.score(Xmp[test_mp], ymp[test_mp])
        cv_rd_to_or_score = clf.score(Xor[test_or], yor[test_or])
        cv_rd_to_rd_scores.append(cv_rd_to_rd_score)
        cv_rd_to_mm_scores.append(cv_rd_to_mm_score)
        cv_rd_to_mp_scores.append(cv_rd_to_mp_score)
        cv_rd_to_or_scores.append(cv_rd_to_or_score)
    cv_rd_to_rd_scores = np.array(cv_rd_to_rd_scores)
    cv_rd_to_mm_scores = np.array(cv_rd_to_mm_scores)
    cv_rd_to_mp_scores = np.array(cv_rd_to_mp_scores)
    cv_rd_to_or_scores = np.array(cv_rd_to_or_scores)

    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_rd_scores.npy'), cv_rd_to_rd_scores)
    np.save(op.join(results_folder, 'cv_rd_to_mm_scores.npy'), cv_rd_to_mm_scores)
    np.save(op.join(results_folder, 'cv_rd_to_mp_scores.npy'), cv_rd_to_mp_scores)
    np.save(op.join(results_folder, 'cv_rd_to_or_scores.npy'), cv_rd_to_or_scores)

    all_cv_rd_to_rd_scores.append(cv_rd_to_rd_scores)
    all_cv_rd_to_mm_scores.append(cv_rd_to_mm_scores)
    all_cv_rd_to_mp_scores.append(cv_rd_to_mp_scores)
    all_cv_rd_to_or_scores.append(cv_rd_to_or_scores)
# create arrays with non cross-validated scores
all_rd_to_mm_scores = np.array(all_rd_to_mm_scores)
all_rd_to_mp_scores = np.array(all_rd_to_mp_scores)
all_rd_to_or_scores = np.array(all_rd_to_or_scores)

# create arrays with cross-validated scores
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_mm_scores = np.array(all_cv_rd_to_mm_scores)
all_cv_rd_to_mp_scores = np.array(all_cv_rd_to_mp_scores)
all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)

# plt.matshow(all_scores.mean(0), origin='lower', extent=[-1, 1, -1, 1])
# plt.matshow(all_gscores.mean(0), origin='lower', extent=[-1, 1, -1, 1], vmin=0.24, vmax=0.26)
