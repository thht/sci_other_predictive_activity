from os.path import join as pjoin
from mne import EvokedArray
import numpy as np

from base import dadd

from base import corresp
from base import cond2code

participants = corresp.keys()

name2pat = {}
for participant in participants:
    #for cond in ['rd','mm','mp','or']:
    for cond in cond2code.keys():
        for name in ['fit_rd0',f'fit_{cond}', f'fit_{cond}_reord' ]:
            fnf = pjoin(path_results,participant,'reorder_random',
                        f'cv_{name}_patterns.npy')
            if os.path.exists(fnf):
                r = np.load(fnf)
                dadd(name2pat,name, r)
            else:
                print('MISSING ', fnf)
#t/19830114RFTM/reorder_random/cv_fit_rd_filters.npy

for k,v in name2pat.items():
    name2pat[k] = np.array(name2pat[k])
print(name2pat['fit_rd0'].shape)


# load one epochs file to get info
meg_rd = '19830114RFTM_block04_random_nosss_ds10.fif'
p0 = pjoin( os.path.expandvars('$SCRATCH/memerr/demarchi') , meg_rd[:-15] ) 
epochs = mne.read_epochs( pjoin(p0,'flt_rd-epo.fif') )


times_show = np.linspace(0,0.33,10)

for name, pats in name2pat.items():
    pat_fit_rd = name2pat[name].mean(1).mean(2).mean(0)
    print(name, name2pat[name].shape, pat_fit_rd.shape)

    # Extract and plot spatial filters and spatial patterns
    coef =  pat_fit_rd    
    # Plot
    evoked = EvokedArray(coef.T, epochs.info, tmin=epochs.tmin)
    fig = evoked.plot_topomap(times=times_show)
    fig.suptitle(f"MEG {name}")
