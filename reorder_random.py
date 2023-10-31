import os.path as op
import os
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

path_data = '/Users/romainquentin/Desktop/data/MEG_demarchi/MEG'
path_results = '/Users/romainquentin/Desktop/data/MEG_demarchi/results'
# list all data files for each condition
MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
MEG_ors = sorted([f for f in os.listdir(path_data) if 'ordered' in f])
# Initialize list of scores (no cross-validation)
all_rd_to_or_scores = list()  # train on random and test on ordered
all_rd_to_rerd_scores = list()  # train on random and test on random reordered
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_cv_rd_to_or_scores = list()
all_cv_rd_to_rerd_scores = list()
# define tmin and tmax
tmin, tmax = -0.4, 0.4
# Start the main loop analysis - Loop on participant
for meg_rd, meg_or in zip(MEG_rds, MEG_ors):
    # results folder where to save the scores for one participant
    participant = meg_or[:12]
    results_folder = op.join(path_results, participant, 'reorder_random')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Add the path to each data file
    meg_rd = op.join(path_data, meg_rd)
    meg_or = op.join(path_data, meg_or)
    # Read raw files
    raw_rd = mne.io.read_raw_fif(meg_rd, preload=True)
    raw_or = mne.io.read_raw_fif(meg_or, preload=True)
    # Get events 
    events_rd = mne.find_events(raw_rd, shortest_event=1)
    events_or = mne.find_events(raw_or, shortest_event=1)
    # Filter the raw
    raw_rd.filter(0.1, 30)
    raw_or.filter(0.1, 30)
    # Create epochs
    raw_rd.pick_types(meg=True, eog=False, ecg=False,
                  ias=False, stim=False, syst=False)
    raw_or.pick_types(meg=True, eog=False, ecg=False,
                  ias=False, stim=False, syst=False)
    epochs_rd = mne.Epochs(raw_rd, events_rd,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_rd.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    epochs_or = mne.Epochs(raw_or, events_or,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_or.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    # # get the X and Y for each condition in numpy array
    # Xrd = epochs_rd.get_data()
    # yrd = epochs_rd.events  
    # Xor = epochs_or.get_data()
    # yor = epochs_or.events # keep all columns to get timing for reordering random raw data
    # remove omission and following trials in random trials
    om_rd = np.where(np.isin(epochs_rd.events, [10, 20, 30, 40]))[0]
    om_fo_rd = np.sort(np.concatenate([om_rd, om_rd+1]))
    if om_fo_rd[-1] == len(epochs_rd.events):
        om_fo_rd = np.delete(om_fo_rd, -1)
    epochs_rd.drop(om_fo_rd)
    # remove omission and following trials in ordered trials
    om_or = np.where(np.isin(epochs_or.events, [10, 20, 30, 40]))[0]
    om_fo_or = np.sort(np.concatenate([om_or, om_or+1]))
    if om_fo_or[-1] == len(epochs_or.events):
        om_fo_or = np.delete(om_fo_or, -1)
    epochs_or.drop(om_fo_or)    

    # reorder random raw
    # get events for random and ordered and initialize new events for reordered data
    random_events = epochs_rd.events.copy()
    ordered_events = list(epochs_or.events)
    events_rerd = list()  # the reordered events (based on yor)
    # prepare raw data
    raw_Xrd = raw_rd.get_data()
    raw_Xrerd = list()  # the reordered X (based on yor)
    new_sample = 0  # keep track of the current sample to create the reordered events
    raw_Xrerd.append(raw_Xrd[:, :200])  # start the reorderd random with the 2 first seconds of the random raw
    first_samp = raw_rd.first_samp
    # keep the original trial numbers in the random (for correct cross-validation and also comparison with the same not-reordered random trials)
    random_events_numbers = np.arange(len(random_events))
    orig_nums = list()
    # raw_Xrerd = raw_Xrd[:, :200]  # start the reorderd random with the 2 first seconds of the random raw 
    new_sample+=200
    for event in tqdm(ordered_events):
        if event[2] in random_events[:, 2]:
            index = random_events[:, 2].tolist().index(event[2])
            orig_nums.append(random_events_numbers[index])
            # raw_Xrerd = np.concatenate((raw_Xrerd, raw_Xrd[:, random_events[index, 0]:random_events[index, 0]+33]), axis=1)
            samp = random_events[index, 0] - first_samp
            raw_Xrerd.append(raw_Xrd[:, samp:samp+33])
            random_events = np.delete(random_events, index, axis=0)
            random_events_numbers = np.delete(random_events_numbers, index, axis=0)
            events_rerd.append([new_sample, 0, event[2]])
            new_sample+=33
        else:
            pass
    orig_nums = np.array(orig_nums)[1:-1]  # removing the first and last trials
    events_rerd = np.array(events_rerd)[1:-1]  # removing the first and last trials
    raw_Xrerd = np.concatenate(raw_Xrerd, axis=1)
    raw_rerd = mne.io.RawArray(raw_Xrerd, raw_rd.info)

    # create an epoch from the reordered raw random 
    epochs_rerd = mne.Epochs(raw_rerd, events_rerd,
                             event_id=[1, 2, 3, 4],
                             tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # keep same trials in epochs_rd and epochs_rerd
    epochs_rd = epochs_rd[orig_nums]
    # keep only the same number of trials in ordered
    epochs_or = epochs_or[:len(epochs_rd)]
    # get the X and Y for each condition in numpy array
    Xor = epochs_or.get_data()
    yor = epochs_or.events[:, 2]
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    Xrerd = epochs_rerd.get_data()
    yrerd = epochs_rerd.events[:, 2]

    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)

    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv_rd_to_or_scores = list()
    cv_rd_to_rerd_scores = list()
    cv = StratifiedKFold(5)
    for train_rd, test_rd in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        cv_rd_to_rd_score = clf.score(Xrd[test_rd], yrd[test_rd])
        cv_rd_to_or_score = clf.score(Xor[test_rd], yor[test_rd])
        test_rerd = np.isin(orig_nums, test_rd)  # why sum(test_rerd) != len(test_rd) 
        cv_rd_to_rerd_score = clf.score(Xrerd[test_rerd], yor[test_rerd])
        cv_rd_to_rd_scores.append(cv_rd_to_rd_score)
        cv_rd_to_or_scores.append(cv_rd_to_or_score)
        cv_rd_to_rerd_scores.append(cv_rd_to_rerd_score)
    cv_rd_to_rd_scores = np.array(cv_rd_to_rd_scores)
    cv_rd_to_or_scores = np.array(cv_rd_to_or_scores)
    cv_rd_to_rerd_scores = np.array(cv_rd_to_rerd_scores)

    # save scores (cross-validation)
    np.save(op.join(results_folder, 'cv_rd_to_rd_scores.npy'), cv_rd_to_rd_scores)
    np.save(op.join(results_folder, 'cv_rd_to_or_scores.npy'), cv_rd_to_or_scores)
    np.save(op.join(results_folder, 'cv_rd_to_rerd_scores.npy'), cv_rd_to_rerd_scores)

    all_cv_rd_to_rd_scores.append(cv_rd_to_rd_scores.mean(0))
    all_cv_rd_to_or_scores.append(cv_rd_to_or_scores.mean(0))
    all_cv_rd_to_rerd_scores.append(cv_rd_to_rerd_scores.mean(0))
# create arrays with cross-validated scores
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)
all_cv_rd_to_rerd_scores = np.array(all_cv_rd_to_rerd_scores)

plt.figure()
plt.matshow(all_cv_rd_to_rd_scores.mean(0), origin='lower', extent=[tmin, tmax, tmin, tmax], vmin=0.23, vmax=0.27)
plt.figure()
plt.matshow(all_cv_rd_to_or_scores.mean(0), origin='lower', extent=[tmin, tmax, tmin, tmax], vmin=0.23, vmax=0.27)
plt.figure()
plt.matshow(all_cv_rd_to_rerd_scores.mean(0), origin='lower', extent=[tmin, tmax, tmin, tmax], vmin=0.23, vmax=0.27)
