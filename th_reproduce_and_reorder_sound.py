#%% imports

import os.path as op
import os, sys
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator, LinearModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import scipy.signal

from th import is_subarray

from base import corresp
from base import events_simple_pred
from base import cond2code
from base import events_omission, events_sound
from base import reorder
from base import getFiltPat

#%% set variables

os.environ['SCRATCH'] = 'scratch'
#os.environ['DEMARCHI_DATA_PATH'] ='/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi'
os.environ['TEMP_DATA_DEMARCHI'] = 'SCRATCH/memerr/demarchi'

#path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'
#path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
path_data = 'data_synced/upstream'
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'
path_results = 'data_nogit/results'
# list all data files for each condition
# DWARNING: this assumes same ordering of files (participants)
#MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
#MEG_mms = sorted([f for f in os.listdir(path_data) if 'midminus' in f])
#MEG_mps = sorted([f for f in os.listdir(path_data) if 'midplus' in f])
#MEG_ors = sorted([f for f in os.listdir(path_data) if 'ordered' in f])
## Start the main loop analysis - Loop on participant
#for meg_rd, meg_mm, meg_mp, meg_or in zip(MEG_rds, MEG_mms, MEG_mps, MEG_ors):
# Initialize list of scores (with cross_validation)

# Import the argparse module

# Assign the arguments to the variables
extract_filters_patterns = 1
nfolds = 5
force_refilt = False
shuffle_cv = False
sids_to_use = [0]



# define tmin and tmax
tmin, tmax = -0.7, 0.7
events_all = events_sound + events_omission
del_processed = 1  
cut_fl = 0 
reord_narrow_test = 0 
#gen_est_verbose = True
gen_est_verbose = False # def True
dur = 200
nsamples = 33

#%% gather subject IDs
# parse directory names from the data directory
rows = [ dict(zip(['subj','block','cond','path'], v[:-15].split('_') + [v])  ) for v in os.listdir(path_data)]
df0 = pd.DataFrame(rows)
df = df0.copy()
df['sid'] = df['subj'].apply(lambda x: corresp[x])

# which subject IDs to use
df = df.query('sid.isin(@sids_to_use)')

# check we have complete data (all conditions for every subject)
grp = df.groupby(['subj'])
assert grp.size().min() == grp.size().max()
assert grp.size().min() == 4

#%% load and filter data
# iterating over subjects (if we selected one, then process one subject)
group_items = list(grp.groups.items())
#for g,inds in group_items:
g, inds = group_items[0]
subdf = df.loc[inds]

# get paths to datasets for each entropy condition per subject
subdf= subdf.set_index('cond')
subdf = subdf.drop(columns=['subj','block','sid'])
meg_rd = subdf.to_dict('index')['random']['path']
meg_or = subdf.to_dict('index')['ordered']['path']
print(meg_rd, meg_or)

# results folder where to save the scores for one participant
ps = [p[:12] for p in [meg_rd, meg_or] ]
assert len(set(ps) ) == 1

participant = meg_or[:12]
print('------------------------------------------------------')
print('---------------------- Starting participant', participant)
print('------------------------------------------------------')
results_folder = op.join(path_results, participant, 'reorder_random')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

cond2epochs = {}
cond2raw   = {}

# load or recalc filtered epochs
p0 = op.join( os.path.expandvars('$TEMP_DATA_DEMARCHI') , meg_rd[:-15] )
if op.exists(op.join(p0, 'flt_rd-epo.fif')) and (not force_refilt):
    print('!!!!!   Loading precomputed filtered epochs from ',p0)
    #epochs_rd = mne.read_epochs( op.join(p0, 'flt_rd-epo.fif'))
    #epochs_or = mne.read_epochs( op.join(p0, 'flt_or-epo.fif'))
    raw_rd = mne.io.read_raw_fif(op.join(p0,'flt_rd-raw.fif'), preload=True)
    raw_rd.pick_types(meg=True, eog=False, ecg=False,
                    ias=False, stim=False, syst=False)

    for cond,condcode in cond2code.items():
        s = condcode
        cond2epochs[cond] = mne.read_epochs( op.join(p0, f'flt_{s}-epo.fif')) 

        raw_ = mne.io.read_raw_fif(op.join(p0,f'flt_{s}-raw.fif'), preload=True) 
        raw_.pick_types(meg=True, eog=False, ecg=False,
                        ias=False, stim=False, syst=False)
        cond2raw[cond] = raw_

else:
    print('!!!!!   (Re)compute filtered raws from ',p0)
    for cond,condcode in cond2code.items():
        fnf = op.join(path_data, subdf.to_dict('index')[cond]['path'] )
        # Read raw files
        raw = mne.io.read_raw_fif(fnf, preload=True)
        print('Filtering ')
        raw.filter(0.1, 30, n_jobs=-1)
        if not op.exists(p0):
            os.makedirs(p0)
        raw.save( op.join(p0, f'flt_{condcode}-raw.fif'), overwrite = True )
        # Get events
        events = mne.find_events(raw, shortest_event=1)
        raw.pick_types(meg=True, eog=False, ecg=False,
                        ias=False, stim=False, syst=False)
        cond2raw[cond] = raw

        # Create epochs
        epochs = mne.Epochs(raw, events,
                            event_id=events_all,
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)
        epochs.save( op.join(p0, f'flt_{condcode}-epo.fif'), overwrite=True)
        cond2epochs[cond] = epochs

    raw_or = cond2raw['ordered']
    raw_rd = cond2raw['random']


#%% remove omission and following trials in random trials
lens_ext = []
cond2counts = {}
# cycle over four entropy conditions
for cond,epochs in cond2epochs.items():
    # just in case save numbers before removing omission trials
    lens_ext += [(cond+'_keepomission',len(epochs))  ]
    cond2counts[cond+'_keepomission'] = Counter(epochs.events[:,2])

    # get indices of omission events
    om = np.where(np.isin(epochs.events, events_omission))[0]
    # take next indices after them and sort indices
    om_fo = np.sort(np.concatenate([om, om+1]))
    # if the last one is not an index, remove it
    if om_fo[-1] == len(epochs.events):
        om_fo = np.delete(om_fo, -1)
    # remove these indices from random epochs
    cond2epochs[cond] = epochs.drop(om_fo)

    cond2counts[cond] = Counter(cond2epochs[cond].events[:,2])


################################################################
#%% reorder random as ...
################################################################

epochs_rd_init = cond2epochs['random'].copy()

cond2epochs_reord = {}
cond2orig_nums_reord = {}

cond2epochs_sp_reord = {}
cond2orig_nums_sp_reord = {}

reorder_pars = dict(del_processed= del_processed, cut_fl=cut_fl, tmin=tmin, tmax=tmax, dur=dur, nsamples=nsamples)
# cycle over four entropy conditions (targets of reordering)
for cond,epochs in cond2epochs.items():
    # original random events
    random_events = epochs_rd_init.events.copy()
    # target events
    events0 = epochs.events.copy()
    
    # reorder random events to another entropy condition
    epochs_reord0, orig_nums_reord0 = reorder(random_events, events0, raw_rd, **reorder_pars) 
    cond2epochs_reord[cond] = epochs_reord0
    cond2orig_nums_reord[cond] = orig_nums_reord0

    cond2counts[cond+'_reord'] = Counter(cond2epochs_reord[cond].events[:,2])

    #######################################################
    ##############   reorder simple prediction
    #######################################################

    # first we transform events from the current entropy condtion into it's "simple prediction" (most probable next event) verion 
    events = events_simple_pred(epochs.events.copy(), cond2code[cond])
    # then we do the reorderig like before, but in this case the target events are the transformed events, not the true ones
    epochs_reord, orig_nums_reord = reorder(random_events, events, raw_rd, **reorder_pars) 
    cond2epochs_sp_reord[cond] = epochs_reord
    cond2orig_nums_sp_reord[cond] = orig_nums_reord

    cond2counts[cond+'_sp_reord'] = Counter(cond2epochs_sp_reord[cond].events[:,2])

#%% check reordered data
# Findings: As expected and specified by the reorder function, the reordered data in all condition
# matches the data from the random condition. The function does what it is supposed to do.

# crop_tmin, crop_tmax = 0.1, 0.2

# original_random_epochs = epochs_rd_init.copy()
# original_random_epochs.crop(crop_tmin, crop_tmax)
# original_random_epochs_array = original_random_epochs.get_data()
# all_conditions = list(cond2epochs.keys())

# dicts_for_pandas = []

# for cur_condition in tqdm(all_conditions):
#     this_cond = {
#         'original_epochs': cond2epochs[cur_condition].copy().crop(crop_tmin, crop_tmax).get_data(),
#         'reordered_epochs': cond2epochs_reord[cur_condition].copy().crop(crop_tmin, crop_tmax).get_data(),
#         'reordered_sp_epochs': cond2epochs_sp_reord[cur_condition].copy().crop(crop_tmin, crop_tmax).get_data()
#     }

#     for kind, data in tqdm(this_cond.items()):
#         this_matches = np.zeros((data.shape[0], ), dtype=bool)
#         for idx, this_data in enumerate(data):
#             is_in_there = np.any([np.all(this_data == cur_random_data) for cur_random_data in original_random_epochs_array])
#             this_matches[idx] = is_in_there

#         tmp_dict = {
#             'condition': cur_condition,
#             'kind': kind,
#             'matches': this_matches
#         }
        
#         dicts_for_pandas.append(tmp_dict)

# df_matches = pd.DataFrame(dicts_for_pandas)

#%% some further analysis
#df_matches['all_match'] = df_matches['matches'].apply(lambda x: np.all(x))
    


#%% go on.


# save counts of all classes to process later (not in this script)
fnf = op.join(results_folder, f'cond2counts.npz' )
print('Saving ',fnf)
np.savez(fnf , cond2counts )

###################################################################
########################     CV
###################################################################
#%% setup classification
print("------------   Starting CV")
cv = StratifiedKFold(nfolds, shuffle=shuffle_cv)

# we need to know minimum number of trials to use it always (they don't actually differ that much but it reduces headache with folds correspondance)
lens = [ len(ep) for ep in cond2epochs.values() ]
lens += [ len(ep) for ep in cond2epochs_reord.values() ]
lens += [ len(ep) for ep in cond2epochs_sp_reord.values() ]
minl = np.min(lens)
print('epochs lens = ',lens, ' minl = ',minl)

lens_ext += [ (cond,len(ep) ) for cond,ep in cond2epochs.items() ]
lens_ext += [ (cond+'_reord',len(ep) ) for cond,ep in cond2epochs_reord.items() ]
lens_ext += [ (cond+'sp_reord',len(ep) ) for cond,ep in cond2epochs_sp_reord.items() ]

# helper function, check if the key is in the dict, if not -- creates it, otherwise add to it
def dadd(d,k,v):
    if k in d:
        d[k] += [v]
    else:
        d[k] = [v]

# Initialize classifier
if extract_filters_patterns:
    clf = LinearModel(LinearDiscriminantAnalysis() ); 
else:
    clf = make_pipeline(LinearDiscriminantAnalysis())
clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=gen_est_verbose)
#condcond2scores = {} # tuples of cond and reord 2 scores

first_cut_sample = int(np.ceil(-tmin * 100)) + 2  # start 2 samples later and end two samples earlier...
last_cut_sample = first_cut_sample + nsamples - 4

prestim_last_cut_sample = int(np.ceil(-tmin * 100)) - 2
prestim_first_cut_sample = prestim_last_cut_sample - nsamples + 4

test_n_channels = 20

list_for_df = []

# cycle over entropies
for cond,epochs in cond2epochs.items():
    print(f"-----  CV for {cond}")

    # keep only the same number of trials for all conditions
    epochs = epochs[:minl]  
    # get the X and Y for each condition in numpy array
    X = epochs.get_data()
    y_sp_ = events_simple_pred(epochs.events.copy() , cond2code[cond])
    y_sp = y_sp_[:, 2] 

    #----------
    epochs_reord = cond2epochs_reord[cond][:minl]
    orig_nums_reord = cond2orig_nums_reord[cond] 
    # TODO: find way to use both sp and not sp, reord and not

    # keep same trials in epochs_rd and epochs_reord
    # TH: I think this is the trick that they do to avoid overfitting.
    # They take the random epochs but reorder them so that their order
    # matches the one in the reordered data.
    # If this is true, then the post stimulus part of all the epochs in 
    # Xrd1 and Xreord must match completely and be in the same order....
    # We're going to check this below...
    epochs_rd1 = epochs_rd_init[orig_nums_reord][:minl]
    Xrd1 = epochs_rd1.get_data()
    yrd1 = epochs_rd1.events[:, 2]

    epochs_sp_reord = cond2epochs_sp_reord[cond]
    #orig_nums_sp_reord = cond2orig_nums_sp_reord[cond] 
    #
    #epochs_rd2 = epochs_rd_init[orig_nums_sp_reord][:minl]
    #Xrd2 = epochs_rd2.get_data()
    #yrd2 = epochs_rd2.events[:, 2]

    y0_ = epochs.events.copy()[:minl]
    y0 = y0_[:, 2] 

    Xreord = epochs_reord.get_data()[:minl]
    yreord_ = epochs_reord.events
    yreord = yreord_[:, 2]
    yreord_sp = events_simple_pred(yreord_, cond2code[cond])[:, 2]

    #Xsp_reord = epochs_sp_reord.get_data()[:minl]
    ysp_reord_ = epochs_sp_reord.events
    ysp_reord = ysp_reord_[:, 2]

    # get short entropy condition code to generate save filenames
    s = cond2code[cond]
    scores = {} # score type 2 score

    # For this to work, all poststim data of Xrd1 and Xreord must match exactly
    test_Xrd1 = Xrd1[:, :, first_cut_sample:last_cut_sample]
    test_Xreord = Xreord[:, :, first_cut_sample:last_cut_sample]

    print(f'Poststim part of Xrd1 and Xreord matches: {np.all(test_Xrd1 == test_Xreord)}')

    filters  = []
    patterns = []

    n_fold = 1
    for train_rd, test_rd in cv.split(Xrd1, yrd1):
        print(f"##############  Starting {cond} fold")
        print('Lens of train and test are :',len(train_rd), len(test_rd) )
        
        # Let's check for potential overfitting because the pre stim data seen during testing
        # might have already been seen during training

        train_data = np.hstack(Xrd1[train_rd, :, first_cut_sample-2:last_cut_sample+2])
        test_data = Xreord[test_rd]

        prestim_match = np.zeros((test_n_channels, test_data.shape[0]), dtype=bool)
        poststim_match = np.zeros((test_n_channels, test_data.shape[0]), dtype=bool)

        for idx_channel in tqdm(list(range(test_n_channels))):
            prestim_testdata = test_data[:, idx_channel, prestim_first_cut_sample:prestim_last_cut_sample]
            poststim_testdata = test_data[:, idx_channel, first_cut_sample:last_cut_sample]

            prestim_match[idx_channel, :] = [is_subarray(train_data[idx_channel, :], pt) for pt in prestim_testdata]
            poststim_match[idx_channel, :] = [is_subarray(train_data[idx_channel, :], pt) for pt in poststim_testdata]

        print(f'prestim: {np.sum(np.all(prestim_match, axis=0))}, poststim: {np.sum(np.all(poststim_match, axis=0))}')
        dict_for_df = {
            'condition': cond,
            'fold': n_fold,
            'n_test_epochs': len(test_rd),
            'prestim_matches_raw': prestim_match,
            'poststim_matches_raw': poststim_match
        }

        list_for_df.append(dict_for_df)

        n_fold += n_fold

        continue
        
        # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
        # Train and test with cross-validation
        clf.fit(Xrd1[train_rd], yrd1[train_rd])  # fit on random

        # to plot patterns later... not very useful in the end, they are too inconsistent
        if extract_filters_patterns:
            filters_, patterns_ = getFiltPat(clf)
            filters  += [filters_]
            patterns += [patterns_]


        # fit on random, test on random
        cv_rd_to_rd_score = clf.score(Xrd1[test_rd], yrd1[test_rd])
        # fit on random, test on order
        cv_rd_to__score = clf.score(X[test_rd], y0[test_rd])
        # fit on random, test on order simple pred
        cv_rd_to_sp_score = clf.score(X[test_rd], y_sp[test_rd])

        # TODO: use assert with sample indices
        #assert len( set( yrd1[train_rd] == yrd1[test_rd]) ) == 0
        #assert len( set( yrd1[train_rd] == y_sp[test_rd]) ) == 0 
        #assert len( set( yrd1[train_rd] == y0[test_rd])   ) == 0 

        # DQ: is it good to restrict test number so much?
        if reord_narrow_test:
            test_reord = np.isin(orig_nums_reord, test_rd)  # why sum(test_reord) != len(test_rd)
            print('{} test_rd among orig_nums_reord. Total = {} '.format( len(test_reord), len(test_rd) ) )
            cv_rd_to_reord_score = clf.score(Xreord[test_reord], yreord[test_reord])
        else:
            # Check for overlap in trials
            train_data = Xrd1[train_rd]
            test_data = Xreord[test_rd]
            train_data_cut = train_data[:, :, first_cut_sample:last_cut_sample]
            test_data_cut = test_data[:, :, first_cut_sample:last_cut_sample]

            test_in_train_data = np.zeros((test_data_cut.shape[0], ), dtype=bool)

            for idx, cur_data in enumerate(test_data_cut):
                is_in_there = np.any([np.all(cur_data == cur_train_data) for cur_train_data in train_data_cut])
                test_in_train_data[idx] = is_in_there

            print(f'Found {np.sum(test_in_train_data)} overlapping trials for condition {cond}.')
            cv_rd_to_reord_score = clf.score(Xreord[test_rd], yreord[test_rd])
            cv_rd_to_reord_sp_score = clf.score(Xreord[test_rd], yreord_sp[test_rd])

            # not used so far
            cv_rd_to_sp_reord_score = clf.score(Xreord[test_rd], ysp_reord[test_rd])

        dadd(scores,'rd_to_rd',cv_rd_to_rd_score      )
        dadd(scores,f'rd_to_{s}',cv_rd_to__score        )
        dadd(scores,f'rd_to_{s}_sp',cv_rd_to_sp_score        )

        dadd(scores,f'rd_to_{s}_reord',cv_rd_to_reord_score   )
        dadd(scores,f'rd_to_{s}_reord_sp',cv_rd_to_reord_sp_score   )
        dadd(scores,f'rd_to_{s}_sp_reord',cv_rd_to_sp_reord_score   )
        #'cv'
    continue
    filters_rd,patterns_rd = np.array(filters), np.array(patterns)

    # train on non-random and test on same or reord (to make "self" plots)
    filters  = []
    patterns = []
    # train on NOT (only) random and test on itself
    for train, test in cv.split(X, y0):
        print(f"##############  Starting {cond} fold")
        clf.fit(X[train], y0[train])  
        if extract_filters_patterns:
            filters_, patterns_ = getFiltPat(clf)
            filters  += [filters_]
            patterns += [patterns_]

        cv__to__score = clf.score(X[test], y0[test])
        cv__to_reord_score = clf.score(Xreord[test], yreord[test])
        dadd(scores,f'{s}_to_{s}', cv__to__score )
        dadd(scores,f'{s}_to_{s}_reord', cv__to_reord_score )
    filters_cond,patterns_cond = np.array(filters), np.array(patterns)

    filters  = []
    patterns = []
    for train, test in cv.split(Xreord, yreord):
        print(f"##############  Starting {cond} fold reord")
        # Check if this needs to be newly initialized
        clf.fit(Xreord[train], yreord[train])
        if extract_filters_patterns:
            filters_, patterns_ = getFiltPat(clf)
            filters  += [filters_]
            patterns += [patterns_]

        # This could be a problem. But only for random
        cv_reord_to__score = clf.score(X[test], y0[test])
        cv_reord_to_reord_score = clf.score(Xreord[test], yreord[test])
        dadd(scores,f'{s}_reord_to_{s}', cv_reord_to__score )
        dadd(scores,f'{s}_reord_to_{s}_reord', cv_reord_to_reord_score )
    filters_cond_reord,patterns_cond_reord = np.array(filters), np.array(patterns)

    if extract_filters_patterns:
        for name,(filters_,patterns_) in zip( ['fit_rd0', f'fit_{cond}',f'fit_{cond}_reord'], 
                [ (filters_rd,patterns_rd), (filters_cond,patterns_cond), (filters_cond_reord,patterns_cond_reord) ] ) :
            fnf = op.join(results_folder, f'cv_{name}_filters.npy' )
            print('Saving ',fnf)
            np.save(fnf , filters_ )     # folds x times x classes x channels 

            fnf = op.join(results_folder, f'cv_{name}_patterns.npy' )
            print('Saving ',fnf)
            np.save(fnf , patterns_ )    # folds x times x classes x channels

    #   # train on non-random simplepred and test on same or reord
    #   for train, test in cv.split(X, y_sp):
    #       print(f"##############  Starting {cond} fold sp2")
    #       clf.fit(X[train], y_sp[train])  
    #       cv__to_sp_score = clf.score(X[test], y_sp[test])
    #       cv__to_reord_score = clf.score(Xsp_reord[test], ysp_reord[test])
    #       dadd(scores,f'{s}_sp_to_{s}_sp', cv__to_sp_score )
    #       dadd(scores,f'{s}_sp_to_{s}_reord', cv__to_reord_score )

    #   # train on non-random simplepred and test on same or reord
    #   for train, test in cv.split(Xsp_reord, ysp_reord):
    #       print(f"##############  Starting {cond} fold sp2")
    #       clf.fit(X[train], y_sp[train])  
    #       cv1 = clf.score(X[test], y_sp[test])
    #       cv2 = clf.score(Xsp_reord[test], ysp_reord[test])
    #       dadd(scores,f'{s}_sp_reord_to_{s}_sp', cv1 )
    #       dadd(scores,f'{s}_sp_reord_to_{s}_sp_reord', cv2 )

    #for train_rd, test_rd in cv.split(Xrd2, yrd2):
    #    print("##############  Starting ordered")
    #    print('Lens of train and test are :',len(train_rd), len(test_rd) )
    #    # Train and test with cross-validation
    #    clf.fit(Xrd2[train_rd], yrd2[train_rd])  # fit on random


    #for train, test in cv.split(X, y_sp):
    #    clf.fit(X[train], y_sp[train])  
    #    cv__to__score = clf.score(X[test], y_sp[test])

    # TODO: add or to or

    # save everything
    for k,v in scores.items():
        scores[k] = np.array(v)
        fnf = op.join(results_folder, f'cv_{k}_scores.npy' )
        print('Saving ',fnf)
        np.save(fnf , v )

    #cv_rd_to_rd_scores = np.array(cv_rd_to_rd_scores)
    #cv_rd_to__scores = np.array(cv_rd_to__scores)
    #cv_rd_to_reord_scores = np.array(cv_rd_to_reord_scores)
    ## save scores (cross-validation)
    #np.save(op.join(results_folder, 'cv_rd_to_rd__sp_scores.npy'), cv_rd_to_rd_scores)

    #np.save(op.join(results_folder, f'cv_{s}_to_{s}__sp_scores.npy'), ?)

    #s = cond2code[cond]
    #np.save(op.join(results_folder, f'cv_rd_to_{s}__sp_scores.npy'), cv_rd_to__scores)

    #s += 'rd'
    #np.save(op.join(results_folder, f'cv_rd_to_{s}__sp_scores.npy'), cv_rd_to_reord_scores)
    import gc; gc.collect()


# %% analyize
df = pd.DataFrame(list_for_df)
