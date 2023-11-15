%matplotlib inline
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import EpochsArray
from mne.io.meas_info import create_info
from mne.decoding import SlidingEstimator, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from base import create_seq

path_data = '/Users/romainquentin/Desktop/data/MEG_demarchi/MEG'

### Simulate raw data
sfreq = 100  # Hz
n_trials = 4000
ITI = 0.333  # s
n_chan, n_samples = 275, n_trials * ITI * sfreq
# Get sequences of trials
yr = create_seq(n_trials, structure=3)
yo = create_seq(n_trials, structure=0)


