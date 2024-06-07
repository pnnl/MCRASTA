import sys
import yaml
import os
import json


class ConfigPlot:
    variable_names = ('a', 'b', 'Dc', 'mu0', 's')

    def __init__(self, path=None):
        self.samplename = None
        self.section_id = None
        self.ndr = None
        self.nch = None
        self.rootpath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles')
        self.mcmc_out_dir = None
        self.postprocess_out_dir = None
        self.mintime = None
        self.maxtime = None
        self.mindisp = None
        self.maxdisp = None
        self.k = None
        self.lc = None
        self.vel_windowlen = None
        self.filter_windowlen = None
        self.q = None
        self.ntune = None
        self.vref = None
        self.threshold = None
        self.mus = None
        self.sigmas = None
        self.alphas = None
        self.betas = None
        self.dist_types = None
        self.nrstep = None
        self.sim_name = None
        self.num_posterior_draws = None
        self.ab_pairlim = None

        if path is None:
            path = 'configplot.yaml'

        self._load_from_file(path)

    def _load_from_file(self, path):
        if not os.path.isfile(path):
            print(f'Warning, {path} is not a path')
            sys.exit()

        with open(path, 'r') as rfile:
            cfg = yaml.safe_load(rfile)
            attrs = ('samplename', 'section_id', 'ndr', 'nch', 'nrstep',
                     'num_posterior_draws', 'output_postprocess_dir', 'ab_pairlim')
            for a in attrs:
                if a not in cfg:
                    print(f'Warning: {a} not in cfg')
                    print(cfg)
                    sys.exit()
                setattr(self, a, cfg.get(a))

            for v in self.variable_names:
                v = f'{v}lims'
                setattr(self, v, PlotLims(cfg.get(v)))

        self.sim_name = f"out_{self.ndr}d{self.nch}ch_{self.section_id}"

        self.mcmc_out_dir = self.make_path('mcmc_out', self.samplename, self.sim_name)

        self._create_directory(cfg['output_postprocess_dir'])

        self.set_vars()

        # load plot axis lims

    def _create_directory(self, dname):
        p = self.make_path(dname, self.samplename, self.sim_name)

        if not os.path.exists(p):
            print(f'Creating output directory: {p}')
            os.makedirs(p)

        # print(f'Directory exists. Any existing files from previous runs will be '
        #       f'overwritten: {p}')

        self.postprocess_out_dir = p

    def get_plot_lims(self, figtype=None):
        prpolims = []
        postlims = []
        for v in self.variable_names:
            lim = getattr(self, f'{v}lims')
            prpolims.append(lim.pr_po)
            postlims.append(lim.post)

        if figtype is not None:
            if figtype == 'pr_po':
                return prpolims
            if figtype == 'post':
                return postlims

        return prpolims, postlims

    def set_vars(self):
        jpath = os.path.join(self.mcmc_out_dir, 'out.json')
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
            self.vref = js.get('vref')
            self.threshold = js.get('threshold')

            priors_info = js.get('prior_mus_sigmas', 'priors info not available')
            self.mus = priors_info[0]
            self.sigmas = priors_info[1]
            self.alphas = priors_info[2]
            self.betas = priors_info[3]
            self.dist_types = priors_info[4]
            # self.dist_types = priors_info[2]

    def make_path(self, *args):
        return os.path.join(self.rootpath, *args)

    # def idata_location(self):
    #     if sys.platform == 'win32':
    #         # print(gpl.make_path('mcmc_out', 'linux_runs_all', gpl.samplename, gpl.sim_name))
    #         self.idata_path = self.make_path('mcmc_out', 'linux_runs_all', self.samplename, self.sim_name)
    #     else:
    #         self.idata_path = self.make_path('mcmc_out', self.samplename, self.sim_name)


class PlotLims:

    def __init__(self, obj):
        try:
            self.pr_po = obj['pr_po']
            self.post = obj['post']
        except KeyError as e:
            print('Using default axis limits')
            pass


cplot = ConfigPlot()
