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
from base import events_simple_pred

path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
#path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'
path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results_test'
# list all data files for each condition
# DWARNING: this assumes same ordering of files (participants)
#MEG_rds = sorted([f for f in os.listdir(path_data) if 'random' in f])
#MEG_mms = sorted([f for f in os.listdir(path_data) if 'midminus' in f])
#MEG_mps = sorted([f for f in os.listdir(path_data) if 'midplus' in f])
#MEG_ors = sorted([f for f in os.listdir(path_data) if 'ordered' in f])
## Start the main loop analysis - Loop on participant
#for meg_rd, meg_mm, meg_mp, meg_or in zip(MEG_rds, MEG_mms, MEG_mps, MEG_ors):
# Initialize list of scores (with cross_validation)
#all_cv_rd_to_rd_scores = list()
#all_cv_rd_to_or_scores = list()
#all_cv_rd_to_orrd_scores = list()
#all_cv_rd_to_mm_scores = list()
#all_cv_rd_to_mmrd_scores = list()
#all_cv_rd_to_mp_scores = list()
#all_cv_rd_to_mprd_scores = list()
# define tmin and tmax
from base import events_omission, events_sound
tmin, tmax = -0.7, 0.7
events_all = events_sound + events_omission
del_processed = 1  # when reorder. =1 means Romain version
force_refilt = 0
cut_fl = 0 # True in 1st and False in last Romain ver
shuffle_cv = False # def = False
reord_narrow_test = 0 # True in 1st and False in last Romain ver 
#gen_est_verbose = True
gen_est_verbose = False # def True
dur = 200
nsamples = 33

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

cond2code = dict(zip(['random','midminus','midplus','ordered'],['rd','mm','mp','or']))

for g,inds in grp.groups.items():
    subdf = df.loc[inds]

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

    p0 = op.join( os.path.expandvars('$SCRATCH/memerr/demarchi') , meg_rd[:-15] )
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
        # Add the path to each data file
        meg_rd1 = op.join(path_data, meg_rd)
        meg_or1 = op.join(path_data, meg_or)
        # Read raw files
        raw_rd = mne.io.read_raw_fif(meg_rd1, preload=True)
        raw_or = mne.io.read_raw_fif(meg_or1, preload=True)
        # Filter the raw
        print('Filtering ')
        raw_rd.filter(0.1, 30, n_jobs=-1)
        raw_or.filter(0.1, 30, n_jobs=-1)

        if not op.exists(p0):
            os.makedirs(p0)
        raw_rd.save( op.join(p0, 'flt_rd-raw.fif'), overwrite = True )
        raw_or.save( op.join(p0, 'flt_or-raw.fif'), overwrite = True )

        # Get events
        events_rd = mne.find_events(raw_rd, shortest_event=1)
        events_or = mne.find_events(raw_or, shortest_event=1)

        # Create epochs
        raw_rd.pick_types(meg=True, eog=False, ecg=False,
                      ias=False, stim=False, syst=False)
        raw_or.pick_types(meg=True, eog=False, ecg=False,
                      ias=False, stim=False, syst=False)

        epochs_rd = mne.Epochs(raw_rd, events_rd,
                               event_id=events_all,
                               tmin=tmin, tmax=tmax, baseline=None, preload=True)
        epochs_or = mne.Epochs(raw_or, events_or,
                               event_id=events_all,
                               tmin=tmin, tmax=tmax, baseline=None, preload=True)

        epochs_rd.save( op.join(p0, 'flt_rd-epo.fif'), overwrite=True)
        epochs_or.save( op.join(p0, 'flt_or-epo.fif'), overwrite=True)


    # remove omission and following trials in random trials
    for cond,epochs in cond2epochs.items():
        # get indices of omission events
        om = np.where(np.isin(epochs.events, events_omission))[0]
        # take next indices after them and sort indices
        om_fo = np.sort(np.concatenate([om, om+1]))
        # if the last one is not an index, remove it
        if om_fo[-1] == len(epochs.events):
            om_fo = np.delete(om_fo, -1)
        # remove these indices from random epochs
        cond2epochs[cond] = epochs.drop(om_fo)

    epochs_rd_init = cond2epochs['random'].copy()

    ################################################################
    # reorder random as ...
    ################################################################
    from base import reorder

    cond2epochs_reord = {}
    cond2orig_nums_reord = {}

    cond2epochs_sp_reord = {}
    cond2orig_nums_sp_reord = {}

    reorder_pars = dict(del_processed= del_processed, cut_fl=cut_fl, tmin=tmin, tmax=tmax, dur=dur, nsamples=nsamples)
    for cond,epochs in cond2epochs.items():
        random_events = epochs_rd_init.events.copy()
        events0 = epochs.events.copy()
        
        epochs_reord0, orig_nums_reord0 = reorder(random_events, events0, raw_rd, **reorder_pars) 
        cond2epochs_reord[cond] = epochs_reord0
        cond2orig_nums_reord[cond] = orig_nums_reord0

        #-----------------
        events = events_simple_pred(epochs.events.copy())
        epochs_reord, orig_nums_reord = reorder(random_events, events, raw_rd, **reorder_pars) 
        cond2epochs_sp_reord[cond] = epochs_reord
        cond2orig_nums_sp_reord[cond] = orig_nums_reord

    ###################################################################
    ########################     CV
    ###################################################################
    print("------------   Starting CV")
    cv = StratifiedKFold(5, shuffle=shuffle_cv)

    lens = [ len(ep) for ep in cond2epochs.values() ]
    lens += [ len(ep) for ep in cond2epochs_reord.values() ]
    lens += [ len(ep) for ep in cond2epochs_sp_reord.values() ]
    minl = np.min(lens)
    print('epochs lens = ',lens, ' minl = ',minl)

    # helper function
    def dadd(d,k,v):
        if k in d:
            d[k] += [v]
        else:
            d[k] = [v]

    # Initialize classifier
    clf = make_pipeline(LinearDiscriminantAnalysis())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=gen_est_verbose)
    condcond2scores = {} # tuples of cond and reord 2 scores
    for cond,epochs in cond2epochs.items():
        print(f"-----  CV for {cond}")

        # keep only the same number of trials in ordered
        epochs = epochs[:minl]  
        # get the X and Y for each condition in numpy array
        X = epochs.get_data()
        y_sp_ = events_simple_pred(epochs.events.copy() )
        y_sp = y_sp_[:, 2] 

        #----------
        epochs_reord = cond2epochs_reord[cond]
        orig_nums_reord = cond2orig_nums_reord[cond] 
        # TODO: find way to use both sp and not sp, reord and not

        # keep same trials in epochs_rd and epochs_reord
        epochs_rd1 = epochs_rd_init[orig_nums_reord][:minl]
        Xrd1 = epochs_rd1.get_data()
        yrd1 = epochs_rd1.events[:, 2]

        epochs_sp_reord = cond2epochs_sp_reord[cond]
        orig_nums_sp_reord = cond2orig_nums_sp_reord[cond] 
        #
        epochs_rd2 = epochs_rd_init[orig_nums_sp_reord][:minl]
        Xrd2 = epochs_rd2.get_data()
        yrd2 = epochs_rd2.events[:, 2]

        y0_ = epochs.events.copy()
        y0 = y0_[:, 2] 

        Xreord = epochs_reord.get_data()
        yreord_ = epochs_reord.events
        yreord = yreord_[:, 2]

        Xreord_sp = epochs_sp_reord.get_data()
        yreord_sp_ = epochs_sp_reord.events
        yreord_sp = yreord_sp_[:, 2]

        s = cond2code[cond]
        scores = {}
        for train_rd, test_rd in cv.split(Xrd1, yrd1):
            print(f"##############  Starting {cond} fold")
            print('Lens of train and test are :',len(train_rd), len(test_rd) )
            # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
            # Train and test with cross-validation
            clf.fit(Xrd1[train_rd], yrd1[train_rd])  # fit on random
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
                cv_rd_to_reord_score = clf.score(Xreord[test_rd], yreord[test_rd])
                cv_rd_to_reord_sp_score = clf.score(Xreord[test_rd], yreord_sp[test_rd])

            #cv_rd_to_rd_scores.append(cv_rd_to_rd_score)
            #cv_rd_to__scores.append(cv_rd_to__score)
            #cv_rd_to_reord_scores.append(cv_rd_to_reord_score)
            dadd(scores,'rd_to_rd',cv_rd_to_rd_score      )
            dadd(scores,f'rd_to_{s}',cv_rd_to__score        )
            dadd(scores,f'rd_to_{s}_sp',cv_rd_to_sp_score        )

            dadd(scores,f'rd_to_{s}_reord',cv_rd_to_reord_score   )
            dadd(scores,f'rd_to_{s}_sp_reord',cv_rd_to_reord_sp_score   )
            #'cv'


        # train on non-random and test on same or reord
        for train, test in cv.split(X, y0):
            print(f"##############  Starting {cond} fold sp")
            clf.fit(X[train], y0[train])  
            cv__to__score = clf.score(X[test], y0[test])
            cv__to_reord_score = clf.score(Xreord[test], yreord[test])
            dadd(scores,f'{s}_to_{s}', cv__to__score )
            dadd(scores,f'{s}_to_{s}_reord', cv__to_reord_score )

        # train on non-random simplepred and test on same or reord
        for train, test in cv.split(X, y_sp):
            print(f"##############  Starting {cond} fold sp2")
            clf.fit(X[train], y_sp[train])  
            cv__to_sp_score = clf.score(X[test], y_sp[test])
            cv__to_reord_score = clf.score(Xreord_sp[test], yreord_sp[test])
            dadd(scores,f'{s}_to_{s}_sp_fromsp', cv__to_sp_score )
            dadd(scores,f'{s}_to_{s}_reord_fromsp', cv__to_reord_score )

        #for train_rd, test_rd in cv.split(Xrd2, yrd2):
        #    print("##############  Starting ordered")
        #    print('Lens of train and test are :',len(train_rd), len(test_rd) )
        #    # Train and test with cross-validation
        #    clf.fit(Xrd2[train_rd], yrd2[train_rd])  # fit on random


        #for train, test in cv.split(X, y_sp):
        #    clf.fit(X[train], y_sp[train])  
        #    cv__to__score = clf.score(X[test], y_sp[test])

        # TODO: add or to or

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

