import numpy as np
import pandas as pd
import seaborn as sns


def create_seq(n_trials, structure):
    """
    This function create a sequence with n_trials and a specific structure from random to structure (see Demarchi et al. 2019)
    :param n_trials: number of trials
    :param structure: integer (from 0 to 3, 0 being random)
    :return: dataFrame
    """
    sounds = [0, 1, 2, 3]
    trials = np.arange(n_trials)
    if structure == 0:
        data = pd.DataFrame({'trials': trials,
                             'sound': np.random.randint(4, size=n_trials)})
    elif structure == 3:
        data = pd.DataFrame({'trials': trials,
                             'sound': np.zeros(len(trials), dtype=int)})
        for trial in trials:
            # Initialize the first key of each block (random)
            if trial == 0:
                data.sound.at[trial] = np.random.choice(sounds)
            else:
                rnd = np.random.random()
                if data.sound[trial-1] == sounds[0]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[1]
                    else:
                        data.sound.at[trial] = sounds[0]
                elif data.sound[trial-1] == sounds[1]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[2]
                    else:
                        data.sound.at[trial] = sounds[1]
                elif data.sound[trial-1] == sounds[2]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[3]
                    else:
                        data.sound.at[trial] = sounds[2]
                elif data.sound[trial-1] == sounds[3]:
                    if rnd < 3/4.:
                        data.sound.at[trial] = sounds[0]
                    else:
                        data.sound.at[trial] = sounds[3]
    return data