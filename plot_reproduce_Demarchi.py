import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt


path_results = '/Users/romainquentin/Desktop/data/MEG_demarchi/results'
results_folder = 'initial_reproduction'
# list all participants
participants = [f for f in os.listdir(path_results) if f[:1] == '1']
# Initialize list of scores (no cross-validation)
all_rd_to_mm_scores = list()
all_rd_to_mp_scores = list()
all_rd_to_or_scores = list()
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_cv_rd_to_mm_scores = list()
all_cv_rd_to_mp_scores = list()
all_cv_rd_to_or_scores = list()

# Loop accross participants
for participant in participants:
    print(participant)
    # Append scores (no cross validation)
    all_rd_to_mm_scores.append(np.load(op.join(path_results, participant, results_folder, 'rd_to_mm_scores.npy')))
    all_rd_to_mp_scores.append(np.load(op.join(path_results, participant, results_folder, 'rd_to_mp_scores.npy')))
    all_rd_to_or_scores.append(np.load(op.join(path_results, participant, results_folder, 'rd_to_or_scores.npy')))
    # Append scores (with cross validation)
    all_cv_rd_to_rd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_rd_scores.npy')))
    all_cv_rd_to_mm_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mm_scores.npy')))
    all_cv_rd_to_mp_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mp_scores.npy')))
    all_cv_rd_to_or_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_or_scores.npy')))
# create arrays with non cross-validated scores
all_rd_to_mm_scores = np.array(all_rd_to_mm_scores)
all_rd_to_mp_scores = np.array(all_rd_to_mp_scores)
all_rd_to_or_scores = np.array(all_rd_to_or_scores)
# create arrays with cross-validated scores
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_mm_scores = np.array(all_cv_rd_to_mm_scores)
all_cv_rd_to_mp_scores = np.array(all_cv_rd_to_mp_scores)
all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)

# Plot on big windows
fig, axs = plt.subplots(4, 1)
vmin, vmax = 0.24, 0.26

all_gat_p_values = gat_stats(np.array(all_scores) - chance)
all_sig = np.array(all_gat_p_values < 0.05)

axs[0].matshow(all_cv_rd_to_rd_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
axs[1].matshow(all_cv_rd_to_mm_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
axs[2].matshow(all_cv_rd_to_mp_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
axs[3].matshow(all_cv_rd_to_or_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
plt.show()




