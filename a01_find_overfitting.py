#%% imports
from joblib import load
import pandas as pd
import numpy as np
from pathlib import Path

#%% set vars
src_folder = Path('data_nogit/cluster_results/01_overfitting')
all_files = list(src_folder.rglob('*.dat'))

#%% load data
all_dfs = [load(x) for x in all_files]
# %% concat
df = pd.concat(all_dfs)

# %%
data_columns = ['matching_trials']
df_by_condition = df[['condition', 'n_test_epochs', 'testing_window', 'cv', 'subject'] + data_columns].groupby(['condition', 'testing_window', 'cv', 'subject']).sum()
df_by_condition = df_by_condition.reset_index()
df_by_condition['matching_trials_percent'] = 100 * (df_by_condition['matching_trials'] / df_by_condition['n_test_epochs'])

df_all = df_by_condition.drop('subject', axis=1).groupby(['cv', 'condition', 'testing_window']).mean()
df_all = df_all.reset_index()
# %%
