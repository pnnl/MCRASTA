import os
import numpy as np
import pandas as pd


class Globals:
    def __init__(self):
        self.samplename = 'p5866'
        self.mintime = None
        self.maxtime = None
        self.mindisp = None
        self.maxdisp = None
        self.section_id = '5866001'
        self.k = 0.0005
        self.lc = 125
        self.rootpath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles')
        self.vel_windowlen = None
        self.filter_windowlen = None
        self.q = None
        self.ndr = 500000
        self.nch = 4
        self.ntune = None
        self.ncores = 4
        # self.sim_name = f'out_{self.ndr}d{self.nch}ch_{self.section_id}'
        self.sim_name = f'out_{self.ndr}d{self.nch}ch'
        self.mu_sim = None
        self.aminbmode = None
        self.threshold = 8
        self.nrstep = 4
        self.nrplot = 500000

    def make_path(self, *args):
        return os.path.join(self.rootpath, *args)

    def get_output_storage_folder(self):
        p = self.make_path('postprocess_out', self.samplename, self.sim_name)

        isExisting = os.path.exists(p)

        if isExisting is False:
            print(f'directory does not exist, creating new directory --> {p}')
            os.makedirs(p)
            return p
        elif isExisting is True:
            print(f'directory exists, all outputs will be saved to existing directory and any existing files will be '
                  f'overwritten --> {p}')
            return p

    def get_prior_parameters(self):
        # prior parameters for a, b, Dc, mu0 (in that order)
        mus = [-5, -5, 3, -1]
        sigmas = [0.8, 0.8, 0.3, 0.3]

        return mus, sigmas

    def set_vch(self, vlps):
        vch = np.max(vlps)
        return vch

    def set_disp_bounds(self, x):
        self.mindisp = x[0]
        self.maxdisp = x[-1]

    def store_mu_sim(self, m):
        print('aslkdjgah', id(self))

        if self.mu_sim is not None:
            self.mu_sim.append(m)
        else:
            self.mu_sim = [m]

    def save_mu_sim(self):
        print('saving', id(self))
        p = os.path.join(self.rootpath, f'{os.getpid()}.y_preds.npy')
        np.save(p, np.array(self.mu_sim))


gpl = Globals()
