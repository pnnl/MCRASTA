import sys
import yaml
import os


class Config:
    variable_names = ('a', 'b', 'Dc', 'mu0')

    def __init__(self, path=None):
        self.samplename = None
        self.mintime = None
        self.maxtime = None
        self.mindisp = None
        self.maxdisp = None
        self.section_id = None
        self.k = None
        self.lc = None
        self.rootpath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles')
        self.vel_windowlen = None
        self.filter_windowlen = None
        self.q = None
        self.ndr = None
        self.nch = None
        self.ntune = None
        self.ncores = None
        self.sim_name = None
        self.mu_sim = None
        self.threshold = None
        self.input_data_dir = None
        self.input_data_fname = None
        self.output_mcmc_dir = None
        self.mcmc_out_dirname = None

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
                     'input_data_dir', 'input_data_fname', 'output_mcmc_dirname',
                     )
            for a in attrs:
                if a not in cfg:
                    print(f'Warning: {a} not in cfg')
                    print(cfg)
                    sys.exit()
                setattr(self, a, cfg.get(a))
            self.sim_name = f'out_{self.ndr}d{self.nch}ch_{self.section_id}'

            # load priors
            for p in self.variable_names:
                p = f'{p}_prior'
                setattr(self, p, Prior(cfg.get(p)))

            self._create_directory(cfg['output_mcmc_dirname'])

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
