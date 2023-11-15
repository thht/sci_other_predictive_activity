import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt
from base import gat_stats
from tqdm import tqdm


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
times = np.linspace(-1, 1, all_cv_rd_to_or_scores.shape[-1])
fig, axs = plt.subplots(4, 1)
vmin, vmax = 0.24, 0.26
chance = 0.25
# get permutations clusters
# all_sig = list()
# for scores in tqdm([all_cv_rd_to_or_scores, all_cv_rd_to_mp_scores, all_cv_rd_to_mm_scores, all_cv_rd_to_rd_scores]):
#     print('iteration')
#     gat_p_values = gat_stats(np.array(scores) - chance)
#     sig = np.array(gat_p_values < 0.05)
#     all_sig.append(sig)
# plot the 4 conditions
axs[0].matshow(all_cv_rd_to_rd_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
axs[1].matshow(all_cv_rd_to_mm_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
axs[2].matshow(all_cv_rd_to_mp_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
axs[3].matshow(all_cv_rd_to_or_scores.mean((0, 1)), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
plt.show()

# Plot on small windows
wh_x = np.where((times > -0.3) & (times < 0.7))[0]
wh_y = np.where((times > 0) & (times < 0.33))[0]
# cut the windows scores according to Demarchi et al.
ixgrid = np.ix_(wh_y, wh_x)
all_cv_rd_to_rd_scores = all_cv_rd_to_rd_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)  # mean on the second dimension because we kept scores on each fold 
all_cv_rd_to_mm_scores = all_cv_rd_to_mm_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
all_cv_rd_to_mp_scores = all_cv_rd_to_mp_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)
all_cv_rd_to_or_scores = all_cv_rd_to_or_scores[:, :, ixgrid[0], ixgrid[1]].mean(1)

fig, axs = plt.subplots(4, 1)
vmin, vmax = 0.24, 0.26
chance = 0.25
# get permutations clusters
all_sig = list()
for scores in tqdm([all_cv_rd_to_or_scores, all_cv_rd_to_mp_scores, all_cv_rd_to_mm_scores, all_cv_rd_to_rd_scores]):
    gat_p_values = gat_stats(np.array(scores) - chance)
    sig = np.array(gat_p_values < 0.05)
    all_sig.append(sig)
# plot the 4 conditions
fig, axs = plt.subplots(4, 1)
axs[3].matshow(all_cv_rd_to_rd_scores.mean(0), origin='lower', extent=[-0.3, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[3].contour(xx, yy, all_sig[3], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[2].matshow(all_cv_rd_to_mm_scores.mean(0), origin='lower', extent=[-0.3, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[2].contour(xx, yy, all_sig[2], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[1].matshow(all_cv_rd_to_mp_scores.mean(0), origin='lower', extent=[-0.3, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[1].contour(xx, yy, all_sig[1], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[0].contour(xx, yy, all_sig[0], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[0].matshow(all_cv_rd_to_or_scores.mean(0), origin='lower', extent=[-0.3, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
plt.show()

#plot differences between ordered and random
vmin, vmax = -0.01, 0.01
chance = 0
scores = all_cv_rd_to_or_scores - all_cv_rd_to_rd_scores
gat_p_values = gat_stats(np.array(scores) - chance)
sig = np.array(gat_p_values < 0.05)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
plt.matshow(scores.mean(0), origin='lower', extent=[-0.3, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
plt.contour(xx, yy, sig, colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
plt.show()



