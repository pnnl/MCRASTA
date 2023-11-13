import os


class Globals:
    def __init__(self):
        self.samplename = 'p5760'
        self.mindisp = 6.894
        self.maxdisp = 8.874
        self.section_id = 5760002
        self.k = 0.00194
        self.lc = 125
        self.rootpath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles')
        self.vel_windowlen = 100
        self.filter_windowlen = 3
        self.q = 2
        self.ndr = 100000
        self.nch = 4
        self.ntune = 20000
        self.ncores = 4
        self.sim_name = f'out_{self.ndr}d{self.nch}ch_{self.section_id}'


    def make_path(self, *args):
        return os.path.join(self.rootpath, *args)

    def get_output_storage_folder(self):
        p = self.make_path('mcmc_out', self.samplename, self.sim_name)

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
        mus = [-5, -5, 3, -1]
        sigmas = [0.8, 0.8, 0.3, 0.3]

        return mus, sigmas




