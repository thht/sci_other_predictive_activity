from plus_slurm import Job
import os.path as op
import os
import numpy as np
import pandas as pd
import importlib
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator, LinearModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from joblib import dump

import mne
from collections import Counter

from base import events_omission, events_sound, cond2code, reorder, events_simple_pred

def dadd(d,k,v):
    if k in d:
        d[k] += [v]
    else:
        d[k] = [v]


class ClassifyJob(Job):
    def run(self, subject_id: str, fold_fun: str, remove_overlap: bool):
        path_data = 'data_synced/upstream'

        sklearn_modelselect = importlib.import_module('sklearn.model_selection')
        Fold_Class = getattr(sklearn_modelselect, fold_fun)

        nfolds = 5
        shuffle_cv = False

        sample_counter_start = {
            'random': 0,
            'midminus': 1000000,
            'midplus': 2000000,
            'ordered': 300000
        }

        tmin, tmax = -0.7, 0.7
        events_all = events_sound + events_omission
        del_processed = 1  
        cut_fl = 0 

        dur = 200
        nsamples = 33

        path_results = 'data_nogit/results_classify'

        # parse directory names from the data directory
        rows = [ dict(zip(['subj','block','cond','path'], v[:-15].split('_') + [v])  ) for v in os.listdir(path_data)]
        df0 = pd.DataFrame(rows)
        df = df0.copy()
        df_orig = df

        tmp = [subject_id]

        df = df_orig.query('subj.isin(@tmp)')

        # check we have complete data (all conditions for every subject)
        grp = df.groupby(['subj'])
        assert grp.size().min() == grp.size().max()
        assert grp.size().min() == 4

        #% load and filter data
        # iterating over subjects (if we selected one, then process one subject)
        group_items = list(grp.groups.items())
        #for g,inds in group_items:
        g, inds = group_items[0]
        subdf = df.loc[inds]

        # get paths to datasets for each entropy condition per subject
        subdf= subdf.set_index('cond')
        subdf = subdf.drop(columns=['subj','block'])
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

        results_folder = op.join(path_results, f'ro_{remove_overlap}', fold_fun, participant, 'reorder_random')
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        cond2epochs = {}
        cond2raw   = {}

        for cond,condcode in cond2code.items():
            fnf = op.join(path_data, subdf.to_dict('index')[cond]['path'] )
            # Read raw files
            raw = mne.io.read_raw_fif(fnf, preload=True)
            print('Filtering ')
            raw.filter(0.1, 30, n_jobs=-1)

            # create sample counter channel
            counter_channel = np.array([np.arange(raw.n_times)]) + sample_counter_start[cond]
            counter_info = mne.create_info(['Counter'], sfreq=raw.info['sfreq'])
            counter_data = mne.io.RawArray(counter_channel, counter_info)
            raw.add_channels([counter_data], force_update_info=True)
            # Get events
            events = mne.find_events(raw, shortest_event=1)
            raw.pick_types(meg=True, eog=False, ecg=False,
                            ias=False, stim=False, syst=False, misc=True)
            cond2raw[cond] = raw

            # Create epochs
            epochs = mne.Epochs(raw, events,
                                event_id=events_all,
                                tmin=tmin, tmax=tmax, baseline=None, preload=True)
            cond2epochs[cond] = epochs

        raw_or = cond2raw['ordered']
        raw_rd = cond2raw['random']

                #%% check the Counter channel for consistency...

        # in raw_rd, the difference must always be one...
        assert np.all(np.diff(raw_rd.get_data('Counter')) == 1)

        # in all epochs, the difference must be >= 1
        for cond, epochs in cond2epochs.items():
            assert(np.all(np.diff(epochs.get_data('Counter')) >= 1))

        #% remove omission and following trials in random trials
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
            # for idx in range(epochs.events.shape[0]):
            #     if epochs.events[idx, 2] >= 10:
            #         epochs.events[idx, 2] /= 10

            cond2counts[cond] = Counter(cond2epochs[cond].events[:,2])

        # in all epochs, the difference must be >= 1
        for cond, epochs in cond2epochs.items():
            assert(np.all(np.diff(epochs.get_data('Counter')) >= 1))

        ################################################################
        #% reorder random as ...
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

        # in all epochs, the difference must be >= 1
        for cond, epochs in cond2epochs.items():
            assert(np.all(np.diff(epochs.get_data('Counter')) >= 1))

        training_window = (epochs.time_as_index(0)[0] + 1, epochs.time_as_index(0.33)[0] - 1)
        testing_windows = {
            'prepre': (epochs.time_as_index(-0.66)[0] + 1, epochs.time_as_index(-0.33)[0] - 1),
            'pre': (epochs.time_as_index(-0.33)[0] + 1, epochs.time_as_index(0)[0] - 1),
            'post': training_window,
            'postpost': (epochs.time_as_index(0.33)[0] + 1, epochs.time_as_index(0.66)[0] - 1)
        }

        # we need to know minimum number of trials to use it always (they don't actually differ that much but it reduces headache with folds correspondance)
        lens = [ len(ep) for ep in cond2epochs.values() ]
        lens += [ len(ep) for ep in cond2epochs_reord.values() ]
        lens += [ len(ep) for ep in cond2epochs_sp_reord.values() ]
        minl = np.min(lens)
        print('epochs lens = ',lens, ' minl = ',minl)

        lens_ext += [ (cond,len(ep) ) for cond,ep in cond2epochs.items() ]
        lens_ext += [ (cond+'_reord',len(ep) ) for cond,ep in cond2epochs_reord.items() ]
        lens_ext += [ (cond+'sp_reord',len(ep) ) for cond,ep in cond2epochs_sp_reord.items() ]

        clf = make_pipeline(LinearDiscriminantAnalysis())
        clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy')

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

            filters  = []

            cv = Fold_Class(n_splits=nfolds, shuffle=shuffle_cv)
            for train_rd, test_rd in cv.split(Xrd1, yrd1):
                print(f"##############  Starting {cond} fold")
                print('Lens of train and test are :',len(train_rd), len(test_rd) )
                # Run cross validation for the ordered (and reorder-order) (and keep the score on the random too only here)
                # Train and test with cross-validation
                train_data_all = Xrd1[train_rd]
                train_data_for_classifier = train_data_all[:, :-1, :]
                
                clf.fit(train_data_for_classifier, yrd1[train_rd])  # fit on random
                Xrd1_testing_all = Xrd1[test_rd]
                Xrd1_testing_for_classifier = Xrd1_testing_all[:, :-1, :]
                Xrd1_testing_for_match = Xrd1_testing_all[:, -1, :]
                X_testing_all = X[test_rd]
                X_testing_for_classifier = X_testing_all[:, :-1, :]
                X_testing_for_match = X_testing_all[:, -1, :]
                X_reord_testing_all = Xreord[test_rd]
                X_reord_testing_for_classifier = X_reord_testing_all[:, :-1, :]
                X_reord_testing_for_match = X_reord_testing_all[:, -1, :]

                yrd1_testing = yrd1[test_rd]
                y0_testing = y0[test_rd]
                y_reord_testing = yreord[test_rd]
                y_reord_sp_testing = yreord_sp[test_rd]
                ysp_reord_testing = ysp_reord[test_rd]
                y_sp_testing = y_sp[test_rd]

                train_data_for_match = np.hstack(train_data_all[:, -1, training_window[0]:training_window[1]] )

                Xrd1_matches = np.any(np.isin(Xrd1_testing_for_match, train_data_for_match), axis=1)
                X_matches = np.any(np.isin(X_testing_for_match, train_data_for_match), axis=1)
                X_reord_matches = np.any(np.isin(X_reord_testing_for_match, train_data_for_match), axis=1)

                print(f'Found {np.sum(Xrd1_matches)} matches in Xrd1')
                print(f'Found {np.sum(X_matches)} matches in X')
                print(f'Found {np.sum(X_reord_matches)} matches in X_reord')

                if remove_overlap:
                    Xrd1_testing_for_classifier = Xrd1_testing_for_classifier[~Xrd1_matches]
                    yrd1_testing = yrd1[test_rd][~Xrd1_matches]
                    X_testing_for_classifier = X_testing_for_classifier[~X_matches]
                    yrd1_testing = yrd1_testing[~Xrd1_matches]
                    y0_testing = y0_testing[~X_matches]
                    y_sp_testing = y_sp_testing[~X_matches]
                    X_reord_testing_for_classifier = X_reord_testing_for_classifier[~X_reord_matches]
                    y_reord_testing = y_reord_testing[~X_reord_matches]
                    y_reord_sp_testing = y_reord_sp_testing[~X_reord_matches]
                    ysp_reord_testing = ysp_reord_testing[~X_reord_matches]

                # fit on random, test on random
                cv_rd_to_rd_score = clf.score(Xrd1_testing_for_classifier, yrd1_testing)
                # fit on random, test on order
                cv_rd_to__score = clf.score(X_testing_for_classifier, y0_testing)
                # fit on random, test on order simple pred
                cv_rd_to_sp_score = clf.score(X_testing_for_classifier, y_sp_testing)

                cv_rd_to_reord_score = clf.score(X_reord_testing_for_classifier, y_reord_testing)
                cv_rd_to_reord_sp_score = clf.score(X_reord_testing_for_classifier, y_reord_sp_testing)

                # not used so far
                cv_rd_to_sp_reord_score = clf.score(X_reord_testing_for_classifier, ysp_reord_testing)

                dadd(scores,'rd_to_rd',cv_rd_to_rd_score      )
                dadd(scores,f'rd_to_{s}',cv_rd_to__score        )
                dadd(scores,f'rd_to_{s}_sp',cv_rd_to_sp_score        )

                dadd(scores,f'rd_to_{s}_reord',cv_rd_to_reord_score   )
                dadd(scores,f'rd_to_{s}_reord_sp',cv_rd_to_reord_sp_score   )
                dadd(scores,f'rd_to_{s}_sp_reord',cv_rd_to_sp_reord_score   )

            for train, test in cv.split(X, y0):
                print(f"##############  Starting {cond} fold")
                train_data_all = X[train]
                train_data_for_classifier = train_data_all[:, :-1, :]
                train_data_for_match = np.hstack(train_data_all[:, -1, training_window[0]:training_window[1]] )

                clf.fit(train_data_for_classifier, y0[train])

                X_testing_all = X[test]
                X_testing_for_classifier = X_testing_all[:, :-1, :]
                X_testing_for_match = X_testing_all[:, -1, :]
                Xreord_testing_all = Xreord[test]
                Xreord_testing_for_classifier = Xreord_testing_all[:, :-1, :]
                Xreord_testing_for_match = Xreord_testing_all[:, -1, :]
                yreord_testing = yreord[test]
                y0_testing = y0[test]

                X_matches = np.any(np.isin(X_testing_for_match, train_data_for_match), axis=1)
                X_reord_matches = np.any(np.isin(Xreord_testing_for_match, train_data_for_match), axis=1)

                print(f'Found {np.sum(X_matches)} matches in X')
                print(f'Found {np.sum(X_reord_matches)} matches in X_reord')

                if remove_overlap:
                    X_testing_for_classifier = X_testing_for_classifier[~X_matches]
                    y0_testing = y0_testing[~X_matches]
                    Xreord_testing_for_classifier = Xreord_testing_for_classifier[~X_reord_matches]
                    yreord_testing = yreord_testing[~X_reord_matches]


                cv__to__score = clf.score(X_testing_for_classifier, y0_testing)
                cv__to_reord_score = clf.score(Xreord_testing_for_classifier, yreord_testing)
                dadd(scores,f'{s}_to_{s}', cv__to__score )
                dadd(scores,f'{s}_to_{s}_reord', cv__to_reord_score )

            for train, test in cv.split(Xreord, yreord):
                print(f"##############  Starting {cond} fold reord")
                train_data_all = Xreord[train]
                train_data_for_classifier = train_data_all[:, :-1, :]
                train_data_for_match = np.hstack(train_data_all[:, -1, training_window[0]:training_window[1]] )

                clf.fit(train_data_for_classifier, yreord[train])

                X_testing_all = X[test]
                X_testing_for_classifier = X_testing_all[:, :-1, :]
                X_testing_for_match = X_testing_all[:, -1, :]
                Xreord_testing_all = Xreord[test]
                Xreord_testing_for_classifier = Xreord_testing_all[:, :-1, :]
                Xreord_testing_for_match = Xreord_testing_all[:, -1, :]
                yreord_testing = yreord[test]
                y0_testing = y0[test]

                X_matches = np.any(np.isin(X_testing_for_match, train_data_for_match), axis=1)
                X_reord_matches = np.any(np.isin(Xreord_testing_for_match, train_data_for_match), axis=1)

                print(f'Found {np.sum(X_matches)} matches in X')
                print(f'Found {np.sum(X_reord_matches)} matches in X_reord')

                if remove_overlap:
                    X_testing_for_classifier = X_testing_for_classifier[~X_matches]
                    y0_testing = y0_testing[~X_matches]
                    Xreord_testing_for_classifier = Xreord_testing_for_classifier[~X_reord_matches]
                    yreord_testing = yreord_testing[~X_reord_matches]


                cv_reord_to__score = clf.score(X_testing_for_classifier, y0_testing)
                cv_reord_to_reord_score = clf.score(Xreord_testing_for_classifier, yreord_testing)
                dadd(scores,f'{s}_reord_to_{s}', cv_reord_to__score )
                dadd(scores,f'{s}_reord_to_{s}_reord', cv_reord_to_reord_score )

            for k,v in scores.items():
                scores[k] = np.array(v)
                fnf = op.join(results_folder, f'cv_{k}_scores.npy' )
                print('Saving ',fnf)
                np.save(fnf , v )