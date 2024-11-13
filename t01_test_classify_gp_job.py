#%% imports
from cluster_jobs.c03_classify_gp import ClassifyGPJob
from cluster_jobs.c02_classify import ClassifyJob
from pathlib import Path

#%% set vars
all_subjects = list(set([x.name[:12] for x in Path('data_synced/gp_data').glob('*.fif')]))
subject = all_subjects[0]

fold_fun = 'StratifiedKFold'
remove_overlap = True

#%% get job
job = ClassifyGPJob(subject_id=subject, fold_fun=fold_fun, remove_overlap=remove_overlap)
#job = ClassifyJob(subject_id=subject, fold_fun=fold_fun, remove_overlap=remove_overlap)

job.run_private()