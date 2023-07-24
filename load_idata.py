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
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', 'mcmc_out', dirname)
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


def save_trace(idata, dirpath, idataname):
    idata.to_netcdf(os.path.join(dirpath, f'{idataname}_pp'))


def sample_posterior_predcheck(idata):
    print('sampling posterior predictive')
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)


def plot_posterior_predictive(idata):
    az.plot_ppc(idata)


def plot_trace(idata):
    az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'], kind="rank_vlines")
    plt.show()


def get_trace_variables(idata):
    modelvals = az.extract(idata.posterior)
    print('modelvals = ', modelvals)

    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    print(f'a shape = {a.shape}')

    return a, b, Dc, mu0


def get_constants(vlps):
    k = 0.0015
    vref = vlps[0]

    return k, vref


def generate_rsf_data(times, vlps, a, b, Dc, mu0):
    # runs rsfmodel.py to generate synthetic friction data
    k, vref = get_constants(vlps)

    # Size of dataset
    size = len(times)
    print(f'size of dataset = {size}')

    model = rsf.Model()

    # pre-allocate array, only do the first 100 right now

    # Set model initial conditions
    model.mu0 = mu0  # Friction initial (at the reference velocity)
    model.a = a  # Empirical coefficient for the direct effect
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.b = b  # Empirical coefficient for the evolution effect
    state1.Dc = Dc  # Critical slip distance

    model.state_relations = [state1]  # Which state relation we want to use

    # We want to solve for 40 seconds at 100Hz
    model.time = times

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    # Run the model!
    model.solve()

    mu = model.results.friction
    theta = model.results.states

    print(model.results)

    # change model results to noisy result, so I can still use the plots easily
    model.results.friction = mutrue


def load_section_data(dirpath):
    section_data = pd.read_csv(os.path.join(dirpath, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times']
    mutrue = df['mutrue']
    vlps = df['vlps']
    x = df['x']

    return times, mutrue, vlps, x


def main():
    idata = load_inference_data(dirpath, idataname)
    times, mutrue, vlps, x = load_section_data(dirpath)

    a, b, Dc, mu0 = get_trace_variables(idata)

    generate_rsf_data(times, vlps, a, b, Dc, mu0)


    # sample_posterior_predcheck(idata)
    # save_trace(idata, dirpath, idataname)
    # plot_trace(idata)


if __name__ == '__main__':
    main()
