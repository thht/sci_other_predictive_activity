import os.path as op
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from config import MEG_randoms, MEG_ordereds

path_data = '/Users/romainquentin/Desktop/data/MEG_demarchi/MEG'

all_scores = list()
all_gscores = list()
all_gscores_nocv = list()
for meg_random, meg_ordered in zip(MEG_randoms, MEG_ordereds):
    meg_random = op.join(path_data, meg_random)
    meg_ordered = op.join(path_data, meg_ordered)

    raw_random = mne.io.read_raw_fif(meg_random, preload=True)
    events_random = mne.find_events(raw_random, shortest_event=1)
    raw_ordered = mne.io.read_raw_fif(meg_ordered, preload=True)
    events_ordered = mne.find_events(raw_ordered, shortest_event=1)

    raw_random.filter(0.1, 30)
    raw_ordered.filter(0.1, 30)

    epochs_random = mne.Epochs(raw_random, events_random,
                               event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                               tmin=-1, tmax=1, baseline=None, preload=True, decim=1)
    epochs_random.pick_types(meg=True, eog=False, ecg=False,
                             ias=False, stim=False, syst=False)
    epochs_ordered = mne.Epochs(raw_ordered, events_ordered,
                                event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                                tmin=-1, tmax=1, baseline=None, preload=True, decim=1)
    epochs_ordered.pick_types(meg=True, eog=False, ecg=False,
                              ias=False, stim=False, syst=False)

    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)

    # scores = list()
    # gscores = list()
    Xr = epochs_random.get_data()
    yr = epochs_random.events[:, 2]
    Xo = epochs_ordered.get_data()
    yo = epochs_ordered.events[:, 2]
    # get omissions + following trials
    om_random = np.where(np.isin(yr, [10, 20, 30, 40]))[0]
    om_ordered = np.where(np.isin(yo, [10, 20, 30, 40]))[0]
    om_and_fo_random = np.sort(np.concatenate([om_random, om_random+1]))
    om_and_fo_ordered = np.sort(np.concatenate([om_ordered, om_ordered+1]))
    if om_and_fo_random[-1] == len(yr):
        om_and_fo_random = np.delete(om_and_fo_random, -1)
    if om_and_fo_ordered[-1] == len(yo):
        om_and_fo_ordered = np.delete(om_and_fo_ordered, -1)
    # remove omission and following trials
    Xr = np.delete(Xr, om_and_fo_random, axis=0)
    yr = np.delete(yr, om_and_fo_random, axis=0)
    Xo = np.delete(Xo, om_and_fo_ordered, axis=0)
    yo = np.delete(yo, om_and_fo_ordered, axis=0)

    clf.fit(Xr, yr)
    gscores = clf.score(Xo, yo)
    all_gscores_nocv.append(gscores)

    scores = list()
    gscores = list()
    cv = StratifiedKFold(5)
    for train, test in cv.split(Xr, yr):
        clf.fit(Xr[train], yr[train])
        score = clf.score(Xr[test], yr[test])
        test_ordered = np.random.choice(np.arange(0, len(yo)), len(test))
        gscore = clf.score(Xo[test_ordered], yo[test_ordered])
        scores.append(score)
        gscores.append(gscore)
    scores = np.array(scores)
    gscores = np.array(gscores)
    all_scores.append(scores.mean(0))
    all_gscores.append(gscores.mean(0))
all_scores = np.array(all_scores)
all_gscores = np.array(all_gscores)
all_gscores_nocv = np.array(all_gscores_nocv)

plt.matshow(all_scores.mean(0), origin='lower', extent=[-1, 1, -1, 1])
plt.matshow(all_gscores.mean(0), origin='lower', extent=[-1, 1, -1, 1], vmin=0.24, vmax=0.26)

np.save('../results/test_decod_cv/all_gscores.npy', all_gscores)
np.save('../results/test_decod_cv/all_scores.npy', all_scores)
