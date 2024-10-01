from .job import Job
import os.path as op
import os
import numpy as np
import pandas as pd

from joblib import dump

import mne
from sklearn.model_selection import StratifiedKFold, KFold
from collections import Counter


from base import corresp
from base import events_simple_pred
from base import cond2code
from base import events_omission, events_sound
from base import reorder


class FindOverfittingJob(Job):
    job_data_folder = '01_overfitting'

    def run(self, subject_id):
        os.environ['SCRATCH'] = 'scratch'
        #os.environ['DEMARCHI_DATA_PATH'] ='/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi'
        os.environ['TEMP_DATA_DEMARCHI'] = 'SCRATCH/memerr/demarchi_test_sample_based'

        #path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
        #path_results = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/results'
        #path_data = os.path.expandvars('$DEMARCHI_DATA_PATH') + '/MEG'
        path_data = 'data_synced/upstream'
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

        sample_counter_start = {
            'random': 0,
            'midminus': 1000000,
            'midplus': 2000000,
            'ordered': 300000
        }



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
        df_orig = df

        tmp = [subject_id]

        df = df_orig.query('subj.isin(@tmp)')

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
                            ias=False, stim=False, syst=False, misc=True)

            for cond,condcode in cond2code.items():
                s = condcode
                cond2epochs[cond] = mne.read_epochs( op.join(p0, f'flt_{s}-epo.fif')) 

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

                # create sample counter channel
                counter_channel = np.array([np.arange(raw.n_times)]) + sample_counter_start[cond]
                counter_info = mne.create_info(['Counter'], sfreq=raw.info['sfreq'])
                counter_data = mne.io.RawArray(counter_channel, counter_info)
                raw.add_channels([counter_data], force_update_info=True)
                raw.save( op.join(p0, f'flt_{condcode}-raw.fif'), overwrite = True )
                # Get events
                events = mne.find_events(raw, shortest_event=1)
                raw.pick_types(meg=True, eog=False, ecg=False,
                                ias=False, stim=False, syst=False, misc=True)
                cond2raw[cond] = raw

                # Create epochs
                epochs = mne.Epochs(raw, events,
                                    event_id=events_all,
                                    tmin=tmin, tmax=tmax, baseline=None, preload=True)
                epochs.save( op.join(p0, f'flt_{condcode}-epo.fif'), overwrite=True)
                cond2epochs[cond] = epochs

            raw_or = cond2raw['ordered']
            raw_rd = cond2raw['random']


        #%% check the Counter channel for consistency...

        # in raw_rd, the difference must always be one...
        assert np.all(np.diff(raw_rd.get_data('Counter')) == 1)

        # in all epochs, the difference must be >= 1
        for cond, epochs in cond2epochs.items():
            assert(np.all(np.diff(epochs.get_data('Counter')) >= 1))

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
            # for idx in range(epochs.events.shape[0]):
            #     if epochs.events[idx, 2] >= 10:
            #         epochs.events[idx, 2] /= 10

            cond2counts[cond] = Counter(cond2epochs[cond].events[:,2])

        # in all epochs, the difference must be >= 1
        for cond, epochs in cond2epochs.items():
            assert(np.all(np.diff(epochs.get_data('Counter')) >= 1))


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

        # in all epochs, the difference must be >= 1
        for cond, epochs in cond2epochs.items():
            assert(np.all(np.diff(epochs.get_data('Counter')) >= 1))
        
        #%% go on.


        # save counts of all classes to process later (not in this script)

        ###################################################################
        ########################     CV
        ###################################################################
        #%% setup classification
        print("------------   Starting CV")

        cvs = {
            'normal': KFold(nfolds, shuffle=shuffle_cv),
            'stratified': StratifiedKFold(nfolds, shuffle=shuffle_cv)
        }

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

        list_for_df = []

        # cycle over entropies
        for cv_label, cv in cvs.items():
            for testing_window_label, testing_window in testing_windows.items():
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

                    n_fold = 1
                    for train_rd, test_rd in cv.split(Xrd1, yrd1):
                        print(f"##############  Starting {cond} fold")
                        print('Lens of train and test are :',len(train_rd), len(test_rd) )
                        
                        # Let's check for potential overfitting because the pre stim data seen during testing
                        # might have already been seen during training

                        train_data_nonstacked = Xrd1[train_rd, -1, training_window[0]:training_window[1]] 
                        train_data = np.hstack(train_data_nonstacked)
                        test_data = Xreord[test_rd, -1, testing_window[0]:testing_window[1]]

                        matches = np.isin(test_data, train_data)

                        # This should also be zero but is a benchmark
                    
                        dict_for_df = {
                            'subject': participant,
                            'cv': cv_label,
                            'testing_window': testing_window_label,
                            'condition': cond,
                            'fold': n_fold,
                            'n_test_epochs': len(test_rd),
                            'matches': matches,
                            'matching_trials': np.sum(np.any(matches, axis=1))
                        }

                        print(dict_for_df)

                        list_for_df.append(dict_for_df)

                        n_fold += 1

        df_results = pd.DataFrame(list_for_df)
        dump(df_results, self.full_output_path)
