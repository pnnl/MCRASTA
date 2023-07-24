import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from rsfmodel import staterelations, rsf, plot
import pytensor
import sys
import h5py
import scipy as sp
from scipy.signal import savgol_filter

home = os.path.expanduser('~')
dirname = 'out_501d2ch'
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', dirname)
idataname = f'{dirname}_idata'


def load_inference_data(dirpath, name):
    fullname = os.path.join(dirpath, name)
    trace = az.from_netcdf(fullname)

    print('trace = ', trace)

    # to extract simulated mu values for realizations
    # stacked_pp = az.extract(trace.posterior_predictive)
    # print(f'stacked = {stacked_pp}')
    # musims = stacked_pp.simulator.values
    #
    # plt.plot(musims)
    # plt.show()
    return trace


def sample_posterior_predcheck(idata):
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)


def plot_posterior_predictive(idata):
    az.plot_ppc(idata)


def plot_trace(idata):
    az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'], kind="rank_vlines")
    plt.show()


def main():
    idata = load_inference_data(dirpath, idataname)

    sample_posterior_predcheck(idata)
    # plot_trace(idata)


if __name__ == '__main__':
    main()
