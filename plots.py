import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import arviz as az
from gplot import gpl
import posterior_draws


class Plot:

    def __init__(self):
        self.pathname = None
        self.op_file = None

    def plot_posterior_draws(self):
        musims = self.get_npy_data(self.pathname, self.op_file)
        logps1 = self.get_npy_data(self.pathname, f'logps_p{gpl.section_id}_0')
        logps2 = self.get_npy_data(self.pathname, f'logps_p{gpl.section_id}_1')

        logps = np.concatenate((logps1, logps2))

        params, logp, mubest = posterior_draws.find_best_fit(logps)

        t, mutrue, vlps, x = posterior_draws.load_section_data()

        self.plot_results(x, mutrue, musims, mubest, params)
        self.save_figs()

    def get_npy_data(self, p, f):
        data = np.load(os.path.join(p, f'{f}.npy'))

        return data

    def plot_results(self, x, mt, musims, mubest, params):
        abest, bbest, Dcbest, mu0best = params
        x = np.transpose(x)

        plt.plot(x, musims, color='indianred', alpha=0.01)
        plt.plot(x, mt, 'k.', label='observed')
        plt.plot(x, mubest, color='red', label=f'best fit\n'
                                               f'a={abest.round(4)}\n'
                                               f'b={bbest.round(4)}\n'
                                               f'$D_c$={Dcbest.round(3)}\n'
                                               f'$\mu_0$={mu0best.round(3)}')

        plt.xlabel('load point displacement ($\mu$m)')
        plt.ylabel('$\mu$')
        plt.title(f'Posterior draws: Sample {gpl.section_id}')
        plt.legend()

    def save_figs(self):
        # check if folder exists, make one if it doesn't
        name = gpl.get_musim_storage_folder()
        print(f'find figures and .out file here: {name}')
        w = plt.get_fignums()
        print('w = ', w)
        for i in plt.get_fignums():
            print('i = ', i)
            plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300, bbox_inches='tight')

plotresults = Plot()