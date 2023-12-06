import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt
from base import gat_stats
from tqdm import tqdm
from scipy.stats import spearmanr

plt.rcParams['axes.titlesize'] = 10

#condition = 'omission'  # 'omission' or 'sound' 
condition = 'sound'  # 'omission' or 'sound' 

#path_data = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/MEG'
path_results = '/p/project/icei-hbp-2022-0017/demarchi/data_demarchi/MEG_demarchi/results'
if condition == "omission":
    results_folder = f'reorder_random_{condition}'; s0='_omission'
elif condition == "sound":
    results_folder = 'reorder_random'; s0=''
path_fig = '/p/project/icei-hbp-2022-0017/demarchi/output_plots'

# list all participants
participants = [f for f in os.listdir(path_results) if f[:1] == '1']
# Initialize list of scores (with cross_validation)
all_cv_rd_to_rd_scores = list()
all_cv_rd_to_mm_scores = list()
all_cv_rd_to_mp_scores = list()
all_cv_rd_to_or_scores = list()
all_cv_rd_to_mmrd_scores = list()
all_cv_rd_to_mprd_scores = list()
all_cv_rd_to_orrd_scores = list()

# Loop accross participants
for participant in participants:
    print(participant)
    # Append scores (with cross validation)
    all_cv_rd_to_rd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_rd_scores.npy')))
    all_cv_rd_to_mm_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mm_scores.npy')))
    all_cv_rd_to_mp_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mp_scores.npy')))
    all_cv_rd_to_or_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_or_scores.npy')))
    all_cv_rd_to_mmrd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mmrd_scores.npy')))
    all_cv_rd_to_mprd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_mprd_scores.npy')))
    all_cv_rd_to_orrd_scores.append(np.load(op.join(path_results, participant, results_folder, 'cv_rd_to_orrd_scores.npy')))
# create arrays with cross-validated scores
all_cv_rd_to_rd_scores = np.array(all_cv_rd_to_rd_scores)
all_cv_rd_to_mm_scores = np.array(all_cv_rd_to_mm_scores)
all_cv_rd_to_mp_scores = np.array(all_cv_rd_to_mp_scores)
all_cv_rd_to_or_scores = np.array(all_cv_rd_to_or_scores)
all_cv_rd_to_mmrd_scores = np.array(all_cv_rd_to_mmrd_scores)
all_cv_rd_to_mprd_scores = np.array(all_cv_rd_to_mprd_scores)
all_cv_rd_to_orrd_scores = np.array(all_cv_rd_to_orrd_scores)
# cut the figure to plot a smaller time window
times = np.linspace(-0.7, 0.7, all_cv_rd_to_or_scores.shape[-1])
wh_x = np.where((times >= -0.7) & (times <= 0.7))[0]
wh_y = np.where((times >= 0) & (times <= 0.33))[0]
ixgrid = np.ix_(wh_y, wh_x)
# average accross cross-validation if cross-validated (sound condition)
if condition == 'sound':
    all_cv_rd_to_rd_scores = all_cv_rd_to_rd_scores.mean(1) 
    all_cv_rd_to_mm_scores = all_cv_rd_to_mm_scores.mean(1)
    all_cv_rd_to_mp_scores = all_cv_rd_to_mp_scores.mean(1)
    all_cv_rd_to_or_scores = all_cv_rd_to_or_scores.mean(1)
    all_cv_rd_to_mmrd_scores = all_cv_rd_to_mmrd_scores.mean(1)
    all_cv_rd_to_mprd_scores = all_cv_rd_to_mprd_scores.mean(1)
    all_cv_rd_to_orrd_scores = all_cv_rd_to_orrd_scores.mean(1)

all_cv_rd_to_rd_scores = all_cv_rd_to_rd_scores[: ,ixgrid[0], ixgrid[1]]
all_cv_rd_to_mm_scores = all_cv_rd_to_mm_scores[: ,ixgrid[0], ixgrid[1]]
all_cv_rd_to_mp_scores = all_cv_rd_to_mp_scores[: ,ixgrid[0], ixgrid[1]]
all_cv_rd_to_or_scores = all_cv_rd_to_or_scores[: ,ixgrid[0], ixgrid[1]]
all_cv_rd_to_mmrd_scores = all_cv_rd_to_mmrd_scores[: ,ixgrid[0], ixgrid[1]]
all_cv_rd_to_mprd_scores = all_cv_rd_to_mprd_scores[: ,ixgrid[0], ixgrid[1]]
all_cv_rd_to_orrd_scores = all_cv_rd_to_orrd_scores[: ,ixgrid[0], ixgrid[1]]

# compute spearman correlation as in Demarchi et al. across entropies
# Initialize spearman rho results
rhos = np.zeros([33, 33, 141])
for i in tqdm(range(rhos.shape[0])):
    for row in range(rhos.shape[1]):
        for column in range(rhos.shape[2]):
            corr_values = [all_cv_rd_to_rd_scores[i, row, column], all_cv_rd_to_mm_scores[i, row, column], all_cv_rd_to_mp_scores[i, row, column], all_cv_rd_to_or_scores[i, row, column]]
            rhos[i, row, column], _ = spearmanr([0, 1, 2, 3], corr_values)


# Plot main figure of Demarchi et al. (Fig3a)
vmin, vmax = 0.23, 0.27
chance = 0.25
# get permutations clusters
all_sig = list()
for scores in tqdm([all_cv_rd_to_or_scores, all_cv_rd_to_mp_scores, all_cv_rd_to_mm_scores, all_cv_rd_to_rd_scores, rhos]):
    print('iteration')
    if scores.all() == rhos.all():
        gat_p_values = gat_stats(np.array(scores))
    else:
        gat_p_values = gat_stats(np.array(scores) - chance)
    sig = np.array(gat_p_values < 0.05)
    all_sig.append(sig)
# plot the 4 conditions
fig, axs = plt.subplots(5, 1, figsize=(7,8), constrained_layout=True)
axs[3].matshow(all_cv_rd_to_rd_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[3].contour(xx, yy, all_sig[3], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[3].title.set_text('Random')
axs[2].matshow(all_cv_rd_to_mm_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[2].contour(xx, yy, all_sig[2], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[2].title.set_text('Midminus')
axs[1].matshow(all_cv_rd_to_mp_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[1].contour(xx, yy, all_sig[1], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[1].title.set_text('Midplus')
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[0].matshow(all_cv_rd_to_or_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
axs[0].contour(xx, yy, all_sig[0], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[0].title.set_text('Ordered')
axs[4].matshow(rhos.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=-0.6, vmax=0.6)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[4].contour(xx, yy, all_sig[4], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[4].title.set_text('Correlation across entropies')
#plt.savefig(op.join'/Users/romainquentin/Desktop/data/MEG_demarchi/figures/main_fig_demarchi_%s.png' % condition)
plt.savefig(path_fig + f'/Rmain_fig_demarchi{s0}.png')

# compute spearman correlation as in Demarchi et al. across entropies
# Initialize spearman rho results
rhos = np.zeros([33, 33, 141])
for i in tqdm(range(rhos.shape[0])):
    for row in range(rhos.shape[1]):
        for column in range(rhos.shape[2]):
            corr_values = [all_cv_rd_to_rd_scores[i, row, column], all_cv_rd_to_mmrd_scores[i, row, column], all_cv_rd_to_mprd_scores[i, row, column], all_cv_rd_to_orrd_scores[i, row, column]]
            rhos[i, row, column], _ = spearmanr([0, 1, 2, 3], corr_values)

# Plot main figure of Demarchi et al. (Fig3a) with reordered events from only random 
vmin, vmax = 0.23, 0.27
chance = 0.25
# get permutations clusters
all_sig = list()
for scores in tqdm([all_cv_rd_to_orrd_scores, all_cv_rd_to_mprd_scores, all_cv_rd_to_mmrd_scores, all_cv_rd_to_rd_scores, rhos]):
    print('iteration')
    if scores.all() == rhos.all():
        gat_p_values = gat_stats(np.array(scores))
    else:
        gat_p_values = gat_stats(np.array(scores) - chance)
    sig = np.array(gat_p_values < 0.05)
    all_sig.append(sig)
# plot the 4 conditions
fig, axs = plt.subplots(5, 1, figsize=(7,8), constrained_layout=True)
axs[3].matshow(all_cv_rd_to_rd_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[3].contour(xx, yy, all_sig[3], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[3].title.set_text('Random')
axs[2].matshow(all_cv_rd_to_mmrd_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[2].contour(xx, yy, all_sig[2], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[2].title.set_text('Midminus')
axs[1].matshow(all_cv_rd_to_mprd_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[1].contour(xx, yy, all_sig[1], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[1].title.set_text('Midplus')
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[0].matshow(all_cv_rd_to_orrd_scores.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
axs[0].contour(xx, yy, all_sig[0], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[0].title.set_text('Ordered')
axs[4].matshow(rhos.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=-0.6, vmax=0.6)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[4].contour(xx, yy, all_sig[4], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[4].title.set_text('Correlation across entropies')
#plt.savefig('/Users/romainquentin/Desktop/data/MEG_demarchi/figures/main_fig_reorder_demarchi_%s.png' % condition)
plt.savefig(path_fig + f'/Rmain_fig_reorder_demarchi{s0}.png')


# Plot main figure of Demarchi et al. (Fig3a) with differences between classic and reordered trials
vmin, vmax = -0.1, 0.1
chance = 0
# get permutations clusters
all_sig = list()
diff_rd_to_orrd = all_cv_rd_to_or_scores - all_cv_rd_to_orrd_scores
diff_rd_to_mprd = all_cv_rd_to_mp_scores - all_cv_rd_to_mprd_scores
diff_rd_to_mmrd = all_cv_rd_to_mm_scores - all_cv_rd_to_mmrd_scores
diff_rd_to_rd = all_cv_rd_to_rd_scores - all_cv_rd_to_rd_scores

# compute spearman correlation as in Demarchi et al. across entropies
# Initialize spearman rho results
rhos = np.zeros([33, 33, 141])
for i in tqdm(range(rhos.shape[0])):
    for row in range(rhos.shape[1]):
        for column in range(rhos.shape[2]):
            corr_values = [diff_rd_to_rd[i, row, column], diff_rd_to_mmrd[i, row, column], diff_rd_to_mprd[i, row, column], diff_rd_to_orrd[i, row, column]]
            rhos[i, row, column], _ = spearmanr([0, 1, 2, 3], corr_values)
for scores in tqdm([diff_rd_to_orrd, diff_rd_to_mprd, diff_rd_to_mmrd, diff_rd_to_rd, rhos]):
    print('iteration')
    if scores.all() == rhos.all():
        gat_p_values = gat_stats(np.array(scores))
    else:
        gat_p_values = gat_stats(np.array(scores) - chance)
    sig = np.array(gat_p_values < 0.05)
    all_sig.append(sig)
# plot the 4 conditions
fig, axs = plt.subplots(5, 1, figsize=(7,8), constrained_layout=True)
axs[3].matshow(diff_rd_to_rd.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[3].contour(xx, yy, all_sig[3], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[3].title.set_text('Random')
axs[2].matshow(diff_rd_to_mmrd.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[2].contour(xx, yy, all_sig[2], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[2].title.set_text('Midminus')
axs[1].matshow(diff_rd_to_mprd.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[1].contour(xx, yy, all_sig[1], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[1].title.set_text('Midplus')
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[0].matshow(diff_rd_to_orrd.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=vmin, vmax=vmax)
axs[0].contour(xx, yy, all_sig[0], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[0].title.set_text('Ordered')
axs[4].matshow(rhos.mean(0), origin='lower', extent=[-0.7, 0.7, 0, 0.33], vmin=-0.6, vmax=0.6)
xx, yy = np.meshgrid(times[wh_x], times[wh_y], copy=False, indexing='xy')
axs[4].contour(xx, yy, all_sig[4], colors='k', levels=[0],
            linestyles='dashed', linewidths=1)
axs[4].title.set_text('Correlation across entropies')
#plt.savefig('/Users/romainquentin/Desktop/data/MEG_demarchi/figures/main_fig_diff_with_reorder_demarchi_%s.png' % condition)
plt.savefig(path_fig + f'/Rmain_fig_diff_with_reorder_demarchi{s0}.png')

print('Finished successfully')
