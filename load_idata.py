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
dirname = 'out_2d2ch'
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', dirname)
idataname = f'{dirname}_idata'


def load_inference_data(dirpath, name):
    fullname = os.path.join(dirpath, name)
    trace = az.from_netcdf(fullname)

    print('trace = ', trace)

    # to extract simulated mu values for realizations
    stacked_pp = az.extract(trace.posterior_predictive)
    print(f'stacked = {stacked_pp}')
    musims = stacked_pp.simulator.values

    plt.plot(musims)
    plt.show()


def main():
    load_inference_data(dirpath, idataname)


if __name__ == '__main__':
    main()
