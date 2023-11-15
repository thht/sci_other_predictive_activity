import os.path as op
import os
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

path_data = '/Users/romainquentin/Desktop/data/MEG_demarchi/MEG'
path_results = '/Users/romainquentin/Desktop/data/MEG_demarchi/results'
path_figures = '/Users/romainquentin/Desktop/data/MEG_demarchi/figures'
# list all data files for each condition
MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_predicts = list()
all_ytrue = list()
# Start the main loop analysis - Loop on participant
for meg_rd in MEG_rds:
    # results folder where to save the scores for one participant
    participant = meg_rd[:12]
    results_folder = op.join(path_results, participant, 'decoding_curves_and_predictions')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Add the path to each data file
    meg_rd = op.join(path_data, meg_rd)
    # Read raw files
    raw_rd = mne.io.read_raw_fif(meg_rd, preload=True)
    # Get events 
    events_rd = mne.find_events(raw_rd, shortest_event=1)
    # Filter the raw
    raw_rd.filter(0.1, 30)
    # Create epochs
    epochs_rd = mne.Epochs(raw_rd, events_rd,
                           event_id=[1, 2, 3, 4, 10, 20, 30, 40],
                           tmin=-0.3, tmax=0.7, baseline=None, preload=True)
    epochs_rd.pick_types(meg=True, eog=False, ecg=False,
                         ias=False, stim=False, syst=False)
    # Initialize classifier 
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # get the X and Y for each condition in numpy array
    Xrd = epochs_rd.get_data()
    yrd = epochs_rd.events[:, 2]
    # remove omission and following trials
    om_rd = np.where(np.isin(yrd, [10, 20, 30, 40]))[0]
    om_fo_rd = np.sort(np.concatenate([om_rd, om_rd+1]))
    if om_fo_rd[-1] == len(yrd):
        om_fo_rd = np.delete(om_fo_rd, -1)
    Xrd = np.delete(Xrd, om_fo_rd, axis=0)
    yrd = np.delete(yrd, om_fo_rd, axis=0)    
    # Train and test with cross-validation
    cv_rd_to_rd_scores = list()
    cv = StratifiedKFold(5)
    predicts = np.zeros((len(yrd), Xrd.shape[-1]))
    for (train_rd, test_rd) in cv.split(Xrd, yrd):
        clf.fit(Xrd[train_rd], yrd[train_rd])
        predicts[test_rd] = clf.predict(Xrd[test_rd])
        cv_rd_to_rd_score = clf.score(Xrd[test_rd], yrd[test_rd])
        cv_rd_to_rd_scores.append(cv_rd_to_rd_score)
    cv_rd_to_rd_scores = np.array(cv_rd_to_rd_scores)
    # Save the sliding decoding curves + predictions for confusion matrix
    np.save(op.join(results_folder, 'cv_rd_to_rd_scores.npy'), cv_rd_to_rd_scores)
    np.save(op.join(results_folder, 'predictions.npy'), predicts)
    np.save(op.join(results_folder, 'ytrue.npy'), yrd)
    
    all_cv_rd_to_rd_scores.append(cv_rd_to_rd_scores.mean(0))
    all_predicts.append(predicts)
    all_ytrue.append(yrd)
all_predicts = np.array(all_predicts)
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_ytrue = np.array(all_ytrue)

times = epochs_rd.times
plt.plot(times, all_cv_rd_to_rd_scores.mean(0))

all_conf_mat = list()
for yrd, predicts in zip(all_ytrue, all_predicts):
    conf_mat = confusion_matrix(yrd, predicts[:, 41], normalize='true')
    all_conf_mat.append(conf_mat)
all_conf_mat = np.array(all_conf_mat)

plt.matshow(all_conf_mat.mean(0))
for (x, y), value in np.ndenumerate(all_conf_mat.mean(0)):
    plt.text(x, y, f"{value:.2f}", va="center", ha="center")
plt.xlabel('predicted labels')
plt.ylabel('true labels')
plt.savefig(op.join(path_figures, 'conf_matrix.png'))