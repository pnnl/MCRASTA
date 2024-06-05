import sys
import yaml
import os


class Config:
    variable_names = ('a', 'b', 'Dc', 'mu0', 's')

    def __init__(self, path=None):
        self.samplename = 'p5894'
        self.mintime = 21342.39
        self.maxtime = 21754.89
        self.mindisp = None
        self.maxdisp = None
        self.section_id = '5894001'
        self.k = 0.00153
        self.lc = 125
        self.rootpath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles')
        self.vel_windowlen = 10
        self.filter_windowlen = 20
        self.q = 5
        self.ndr = 100
        self.nch = 2
        self.ntune = 2
        self.ncores = 4
        self.sim_name = f'out_{self.ndr}d{self.nch}ch_{self.section_id}'
        self.mu_sim = None
        self.threshold = 0.1
        self.input_data_dir = None
        self.input_data_fname = None
        self.output_mcmc_dir = None
        self.output_postprocess_dir = None
        self.mcmc_out_dir = None

        if path is None:
            path = 'config.yaml'

        self._load_from_file(path)

    def _load_from_file(self, path):
        if not os.path.isfile(path):
            print(f'Warning, {path} is not a path')
            sys.exit()

        with open(path, 'r') as rfile:
            cfg = yaml.safe_load(rfile)
            attrs = ('samplename', 'section_id', 'mintime', 'maxtime',
                     'k', 'lc', 'vel_windowlen', 'filter_windowlen', 'q',
                     'ndr', 'nch', 'ntune', 'ncores', 'threshold',
                     'input_data_dir', 'input_data_fname', 'output_mcmc_dir',
                     )
            for a in attrs:
                if a not in cfg:
                    print(f'Warning: {a} not in cfg')
                    print(cfg)
                    sys.exit()
                setattr(self, a, cfg.get(a))

            # load priors
            for p in self.variable_names:
                p = f'{p}_prior'
                setattr(self, p, Prior(cfg.get(p)))

        self._create_directory(cfg['output_mcmc_dir'])

    def _create_directory(self, dname):
        p = self.make_path(dname, self.samplename, self.sim_name)

        if not os.path.exists(p):
            print(f'Creating output directory: {p}')
            os.makedirs(p)

        print(f'Directory exists. Any existing files from previous runs will be '
              f'overwritten: {p}')

        self.mcmc_out_dir = p

    def set_disp_bounds(self, x):
        self.mindisp = x[0]
        self.maxdisp = x[-1]

    def get_prior_parameters(self):
        # prior parameters for a, b, Dc, mu0, s (in that order)
        mus = []
        sigmas = []
        dist_types = []

        for a in self.variable_names:
            prior = getattr(self, f'{a}_prior')
            mus.append(prior.mu)
            sigmas.append(prior.sigma)
            dist_types.append(prior.dist_type)

        return mus, sigmas, dist_types

    def make_path(self, *args):
        return os.path.join(self.rootpath, *args)


class Prior:

    def __init__(self, obj):
        try:
            self.dist_type = obj['dist_type']
            self.mu = obj['mu']
            self.sigma = obj['sigma']
        except KeyError as e:
            print('Invalid prior.')
            print(e)
            sys.exit()


cfig = Config()
