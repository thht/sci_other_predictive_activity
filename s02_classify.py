#%% imports
from plus_slurm import JobCluster, PermuteArgument
from pathlib import Path
from cluster_jobs.c02_classify import ClassifyJob

#%% set vars
all_subjects = set([x.name[:12] for x in Path('data_synced/upstream').glob('*.fif')])

jobs = JobCluster(required_ram='32G', request_cpus=1, request_time=5*60)

#%% setup jobs
jobs.add_job(ClassifyJob, subject_id=PermuteArgument(all_subjects),
             fold_fun=PermuteArgument(['KFold', 'StratifiedKFold']),
             remove_overlap=PermuteArgument([True, False]),
)

jobs.submit()