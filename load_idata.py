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
from multiprocessing import Pool

home = os.path.expanduser('~')
dirname = 'out_501d2ch'
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', 'mcmc_out', dirname)
idataname = f'{dirname}_idata'

um_to_mm = 0.001


def load_inference_data(dirpath, name):
    fullname = os.path.join(dirpath, name)
    trace = az.from_netcdf(fullname)

    # print('trace = ', trace)

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
    # print('modelvals = ', modelvals)

    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    # all = np.vstack((a, b, Dc, mu0))
    # allt = np.transpose(all)
    # m_reals = pd.DataFrame(allt, columns=['a', 'b', 'Dc', 'mu0'])
    # print('m_reals = ', m_reals)

    return a, b, Dc, mu0


def get_constants(vlps):
    k = 0.0015
    vref = vlps[0]

    return k, vref


def generate_rsf_data():
    idata = load_inference_data(dirpath, idataname)
    a, b, Dc, mu0 = get_trace_variables(idata)
    times, mutrue, vlps, x = load_section_data(dirpath)
    # runs rsfmodel.py to generate synthetic friction data
    k, vref = get_constants(vlps)

    nr = 16
    # nr = 10
    nobs = len(times)
    # print('num reals = ', nr)
    # print('num obs = ', nobs)

    # pre-allocate array
    mu_sims = np.ones((nobs, nr))
    print(f'mu_sims.shape = {mu_sims.shape}')

    # set up rsf model
    model = rsf.Model()
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    model.state_relations = [state1]  # Which state relation we want to use

    # We want to solve for 40 seconds at 100Hz
    model.time = times

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    # iterate over markov chain parameter (m) estimates and calculate each simulated mu value
    # need to iterate over nr rows, that's it
    print('this takes a long time for large number of realizations')
    # for i in np.arange(nr):
    print(f'solving for realization')
    # Set model initial conditions
    model.mu0 = mu0   # Friction initial (at the reference velocity)
    print('model mu0 = ', model.mu0)
    model.a = a          # Empirical coefficient for the direct effect
    state1.b = b        # Empirical coefficient for the evolution effect
    state1.Dc = Dc    # Critical slip distance

    # Run the model!
    model.solve()

    mu_sim = model.results.friction
    mu_sims[:, nr] = mu_sim

    return mu_sims


def plot_simulated_mus(x, times, mu_sims, mutrue, nr):
    plt.figure(1)
    plt.plot(x*um_to_mm, mutrue, '.', label='observed', alpha=0.5)
    plt.plot(x*um_to_mm, mu_sims, 'b-', alpha=0.2)
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title('Simulated mu values, {nr} realizations')
    plt.show()


def load_section_data(dirpath):
    section_data = pd.read_csv(os.path.join(dirpath, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def main():
    idata = load_inference_data(dirpath, idataname)
    times, mutrue, vlps, x = load_section_data(dirpath)

    m_reals = get_trace_variables(idata)
    mu_sims = generate_rsf_data()

    return times, vlps, m_reals

    # mu_sims = generate_rsf_data(times, vlps, m_reals)

    # plot_simulated_mus(x, times, mu_sims, mutrue, nr=len(mu_sims[0:]))

    # sample_posterior_predcheck(idata)
    # save_trace(idata, dirpath, idataname)
    # plot_trace(idata)

def dummy_fn(x):
    y = np.sqrt(x)
    return y

def run_complex_operations(operation, input, pool):
    pool.map(operation, input)


processes_count = 4
input = range(4)
if __name__ == '__main__':
    main()
    # processes_pool = Pool(processes_count)
    # # times, vlps, ms = main()
    #
    # run_complex_operations(generate_rsf_data, input, processes_pool)
    # print('done')




