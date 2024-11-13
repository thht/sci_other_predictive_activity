#%% imports
from plus_slurm import JobCluster, PermuteArgument
from pathlib import Path
from cluster_jobs.c03_classify_gp import ClassifyGPJob

#%% set vars
all_subjects = set([x.name[:12] for x in Path('data_synced/gp_data').glob('*.fif')])

jobs = JobCluster(required_ram='64G', request_cpus=2, request_time=5*60)

#%% setup jobs
jobs.add_job(ClassifyGPJob, subject_id=PermuteArgument(all_subjects),
             fold_fun=PermuteArgument(['KFold', 'StratifiedKFold']),
             remove_overlap=PermuteArgument([True, False]),
)

jobs.submit()