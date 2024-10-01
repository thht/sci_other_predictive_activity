#%% imports
from plus_slurm import JobCluster, PermuteArgument
from pathlib import Path
from cluster_jobs.c01_find_overfitting import FindOverfittingJob

#%% set vars
all_subjects = set([x.name[:12] for x in Path('data_synced/upstream').glob('*.fif')])

jobs = JobCluster(required_ram='32G', request_cpus=2, request_time=30)

#%% setup jobs
jobs.add_job(FindOverfittingJob, subject_id=PermuteArgument(all_subjects))

jobs.submit()