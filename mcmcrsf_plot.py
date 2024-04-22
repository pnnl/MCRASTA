import json
import math
import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from plotrsfmodel import staterelations, rsf, plot
import sys
from scipy.stats import lognorm, mode, skew, kurtosis
from scipy import signal
import seaborn as sns
import globals
import arviz.labels as azl
from random import sample
from multiprocessing import Process
from gplot import gpl

# 1. read in .npy file
# 2. calculate log(p) for each simulation
# 3. calculate ensemble statistics
# 4. plot simulated mus

home = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out')
# idata_location = gpl.make_path('mcmc_out', 'linux_runs_all', gpl.samplename, gpl.sim_name)


def manual_bci(x, mt, msims):
    # calculate probability distribution of simulated mus at each displacement
    hs = []
    for i in range(len(x)):
        h = plt.hist(msims[:, i], bins=100, range=[0, 1], density=True)
        hs.append(h)
    plt.show()


def get_npy_data():
    data = gpl.make_path('musim_out', f'{gpl.msname}.npy')
    return np.load(data)


def load_section_data():
    section_data = pd.read_csv(os.path.join(gpl.idata_location, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy().round(2)
    mutrue = df['mutrue'].to_numpy().round(3)
    vlps = df['vlps'].to_numpy().round(2)
    x = df['x'].to_numpy().round(2)

    return times, mutrue, vlps, x


def calc_ensemble_stats(x, msims):
    msims[msims < 0] = np.nan
    msims[msims > 1] = np.nan
    msims[msims == np.inf] = np.nan
    msims[msims == -np.inf] = np.nan

    means = np.nanmean(msims, axis=0)
    stdevs = np.nanstd(msims, axis=0)

    sig_upper = means + stdevs
    sig_lower = means - stdevs

    plt.plot(x, means, 'r')
    plt.plot(x, sig_lower, 'c-')
    plt.plot(x, sig_upper, 'c-')


def find_best_fits(x, mt, msims):
    resids = np.transpose(mt) - msims
    rsq = resids ** 2
    srsq = rsq.sum(axis=1)
    logps = - 1 / 2 * srsq

    # plt.plot(logps, '.')
    # plt.ylim([-10, 0])
    # plt.show()

    bestlogps = [i for i in range(len(logps)) if logps[i] > -0.01]
    bestmsims = msims[bestlogps, :]

    sortedi = np.argsort(np.abs(logps))
    # ci3max = np.round(0.03*len(sortedi)).astype(int)
    # ci97max = np.round(0.97*len(sortedi)).astype(int)
    # ci3_i = sortedi[0:ci3max]
    # ci97_i = sortedi[ci3max:ci97max]
    #
    # ci3_ms = msims[ci3_i, :]
    # ci97_ms = msims[ci97_i, :]
    ms_best = msims[sortedi[0], :]
    plt.gcf()
    plt.plot(x, mt, 'r')
    plt.plot(x, np.transpose(ms_best), 'b')
    plt.ylim([0.3, 0.5])
    plt.show()


def main():
    msims = get_npy_data()
    msims.round(2)
    t, mutrue, vlps, x = load_section_data()

    calc_ensemble_stats(x, msims)

    # manual_bci(x, mutrue, msims)
    find_best_fits(x, mutrue, msims)
    print('done')


if __name__ == '__main__':
    main()