import os
import sys
import numpy as np
import json


class Globals:
    def __init__(self):
        self.samplename = 'p5894'
        self.mintime = None
        self.maxtime = None
        self.mindisp = None
        self.maxdisp = None
        self.section_id = '5894001'
        self.k = 0.00153
        self.lc = 125
        self.rootpath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles')
        self.vel_windowlen = None
        self.filter_windowlen = None
        self.q = None
        self.ndr = 100000
        self.nch = 4
        self.ntune = None
        self.ncores = 4
        # self.sim_name = f'out_{self.ndr}d{self.nch}ch_{self.section_id}'
        self.sim_name = f'out_{self.ndr}d{self.nch}ch_{self.section_id}'
        self.mu_sim = None
        self.aminbmode = None
        self.threshold = 0.1
        self.nrstep = 1
        self.nrplot = self.nch * self.ndr / self.nrstep   # nch * ndr / nrstep
        self.tcrit = None
        self.vmax = None
        self.dirname = f'out_{self.ndr}d{self.nch}ch'
        self.msname = f'mu_simsp{self.section_id}'

    def make_path(self, *args):
        return os.path.join(self.rootpath, *args)

    def idata_location(self):
        if sys.platform == 'win32':
            # print(gpl.make_path('mcmc_out', 'linux_runs_all', gpl.samplename, gpl.sim_name))
            return self.make_path('mcmc_out', 'linux_runs_all', self.samplename, self.sim_name)
        else:
            return self.make_path('mcmc_out', self.samplename, self.sim_name)

    def get_musim_storage_folder(self):
        p = self.make_path('musim_out', self.samplename, self.sim_name)

        isExisting = os.path.exists(p)

        if isExisting is False:
            print(f'directory does not exist, creating new directory --> {p}')
            os.makedirs(p)
            return p
        elif isExisting is True:
            print(f'directory exists, all outputs will be saved to existing directory and any existing files will be '
                  f'overwritten --> {p}')
            return p

    def get_postprocess_storage_folder(self):
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
        mus = [-5, -5, 4, -1]
        sigmas = [0.8, 0.8, 0.5, 0.3]

        return mus, sigmas

    def set_vch(self, vlps):
        self.vmax = np.max(vlps)

    def set_disp_bounds(self, x):
        self.mindisp = x[0]
        self.maxdisp = x[-1]

    def read_from_json(self, dirpath):
        jpath = os.path.join(dirpath, 'out.json')
        with open(jpath, 'r') as rfile:
            js = json.load(rfile)
            # print(js)
            self.samplename = self.samplename
            self.mintime = js.get('time_start')
            self.maxtime = js.get('time_end')
            self.mindisp = js.get('x_start')
            self.maxdisp = js.get('x_end')
            self.section_id = js.get('section_ID')
            self.k = js.get('k')
            self.lc = js.get('lc')
            self.vel_windowlen = js.get('dvdt_window_len')
            self.filter_windowlen = js.get('filter_window_len')
            self.q = js.get('q')
            self.ndr = js.get('n_draws')
            self.nch = js.get('n_chains')
            self.ntune = js.get('n_tune')
            vref = js.get('vref')
            self.threshold = js.get('threshold')

            priors_info = js.get('prior_mus_sigmas', 'priors info not available')
            mus = priors_info[0]
            sigmas = priors_info[1]

            return vref, mus, sigmas


gpl = Globals()
