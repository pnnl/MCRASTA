import json
import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

import config
from rsfmodel import staterelations, rsf, plot
import pytensor as pt
import sys
import h5py
import scipy as sp
from scipy.signal import savgol_filter
from datetime import datetime
import time
import seaborn as sns
from config import cfig

um_to_mm = 0.001

pt.config.optimizer = 'fast_compile'
rng = np.random.normal()
np.random.seed(1234)
az.style.use("arviz-darkgrid")


# GENERAL SCRIPT SETUP
# most functions below this are fetching folders, filenames, etc.
# need to configure better
def get_time(name):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'{name} time = {current_time}')

    codetime = time.time()

    return codetime


# this only runs when starting new sample
def preplot(df):
    t = df['time_s']
    x = df['vdcdt_um']

    plt.plot(x * um_to_mm)
    plt.title('x')
    plt.ylabel('displacement (mm)')
    plt.show()

    sys.exit()


# POST MODEL-RUN OPERATIONS AND PLOTTING FUNCTIONS
def save_trace(idata):
    # saves trace for post-processing
    out_name = f'{cfig.sim_name}_idata'
    p = cfig.mcmc_out_dir
    idata.to_netcdf(os.path.join(p, f'{out_name}'))


def plot_trace(idata):
    az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'])


def save_stats(idata):
    summary = az.summary(idata, kind='stats')
    print(f'summary: {summary}')
    summary.to_csv(os.path.join(cfig.mcmc_out_dir, 'idata.csv'))

    return summary


def post_processing(idata, times, vlps, mutrue, x):
    # save dataset for later use
    df_data = pd.DataFrame(np.column_stack((times, x, vlps, mutrue)), columns=['times', 'x', 'vlps', 'mutrue'])
    df_data.to_csv(os.path.join(cfig.mcmc_out_dir, 'section_data.csv'))

    # plot pymc posterior trace
    plot_trace(idata)

    print('post processing complete')


def save_figs(out_folder):
    # check if folder exists, make one if it doesn't
    name = out_folder
    print(f'find figures and .out file here: {name}')
    w = plt.get_fignums()
    print('w = ', w)
    for i in plt.get_fignums():
        print('i = ', i)
        plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300)


def write_model_info(time_elapsed, vref, vsummary, file_name, times):
    fname = os.path.join(cfig.mcmc_out_dir, 'out.txt')

    samplerstrs = ['SAMPLER INFO', 'num draws', 'num chains', 'tune', 'prior mus and sigmas', 'runtime (s)']
    modelstrs = ['MODEL INFO', 'constants', 'k', 'vref', 'section min displacement', 'section max displacement',
                 'characteristic length', 'velocity calc window length', 'filter window length', 'downsampling rate']
    summarystr = ['SAMPLE VARS SUMMARY']
    strlist = [samplerstrs, modelstrs, summarystr]

    samplervals = ['', cfig.ndr, cfig.nch, cfig.ntune, cfig.get_prior_parameters(), time_elapsed]
    modelvals = ['', '', cfig.k, vref, cfig.mindisp, cfig.maxdisp, cfig.lc, cfig.vel_windowlen,
                 cfig.filter_windowlen, cfig.q]
    summaryvals = [vsummary]
    vallist = [samplervals, modelvals, summaryvals]

    with open(fname, mode='w') as f:
        f.write(f'SAMPLE: {file_name}\n')
        f.write(f'section_ID: {cfig.section_id}\n')
        f.write(f'from t = {times[0]} to t= {times[-1]} seconds\n')
        f.write(f'from x = {cfig.mindisp} to x = {cfig.maxdisp} mm\n')
        for strings, vals in zip(strlist, vallist):
            # f.writelines(f'{strings}: {vals}')
            for string, val in zip(strings, vals):
                f.write(f'{string}: {val}\n')

    payload = {'sample': file_name,
               'section_ID': cfig.section_id,
               'time_start': cfig.mintime,
               'time_end': cfig.maxtime,
               'x_start': cfig.mindisp,
               'x_end': cfig.maxdisp,
               'n_draws': cfig.ndr,
               'n_chains': cfig.nch,
               'n_tune': cfig.ntune,
               'prior_mus_sigmas': cfig.get_prior_parameters(),
               'runtime_s': time_elapsed,
               'k': cfig.k,
               'vref': vref,
               'lc': cfig.lc,
               'dvdt_window_len': cfig.vel_windowlen,
               'filter_window_len': cfig.filter_windowlen,
               'q': cfig.q,
               'threshold': cfig.threshold
               }

    with open(os.path.join(cfig.mcmc_out_dir, 'out.json'), mode='w') as wfile:
        json.dump(payload, wfile)


# plot_obs_data_processing(...) only used for testing data processing
def plot_obs_data_processing(x, mu1, mu2, mu3, xog):
    plt.figure(1)
    plt.plot(xog * um_to_mm, mu1, '.', label='raw', alpha=0.3)
    plt.plot(xog * um_to_mm, mu2, '.', label='filtered', alpha=0.4)
    plt.plot(x, mu3, '.', label='filtered + downsampled', alpha=0.3)
    plt.xlim([x[0], x[-1]])
    plt.ylim([np.min(mu3) - 0.02, np.max(mu3) + 0.02])
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title('Observed data section, p5756')
    plt.legend()
    save_figs('figs')

    sys.exit()


# DATA PROCESSING
# from Jeff - calculate derivative dx/dt (=dy/dx)
def calc_derivative(y, x, window_len=None):
    # returns dydx
    if window_len is not None:
        print(f'calculating derivative using SG filter and window length {window_len}')
        # smooth
        # x_smooth = smooth(x,window_len=params['window_len'],window='flat')
        # y_smooth = smooth(y,window_len=params['window_len'],window='flat')
        # dydx = np.gradient(y_smooth,x_smooth)
        dxdN = savgol_filter(x,
                             window_length=window_len,
                             polyorder=3,
                             deriv=1)
        dydN = savgol_filter(y,
                             window_length=window_len,
                             polyorder=3,
                             deriv=1)
        dydx = dydN / dxdN
        dydx_smooth = savgol_filter(dydx,
                                    window_length=window_len,
                                    polyorder=1)
        dydx_smooth[dydx_smooth < 0] = 0.0001
        return dydx_smooth
    else:
        print(f'calculating derivative using gradient because window_len= {window_len}')
        dydx = np.gradient(y, x)
        dydx[dydx < 0] = 0
        return dydx


# imports observed data, sends it through series of processing steps
def get_obs_data():
    data_path = cfig.make_path('data', 'FORGE_DataShare', f'{cfig.samplename}',
                                    f'{cfig.samplename}_proc.hdf5')

    # f = h5py.File(data_path, 'r')

    # data_path = os.path.join(cfig.input_data_dir, cfig.samplename, cfig.input_data_fname)
    print(f'Pulling experimental data from: {data_path}')
    # f = h5py.File(data_path, 'r')

    # read in data from hdf file, print column names
    df, names = read_hdf(data_path)

    # comment this in when deciding which displacement sections to use
    # preplot(df)

    # first remove any mu < 0 data from experiment
    df = df[(df['mu'] > 0)]

    # convert to numpy arrays
    t = df['time_s'].to_numpy()
    mu = df['mu'].to_numpy()
    x = df['vdcdt_um'].to_numpy()

    # filters and downsamples data
    f_ds, mu_f = downsample_dataset(mu, t, x)

    # sections data
    sectioned_data, start_idx, end_idx = section_data(f_ds)

    # need to check that time and displacement values are monotonically increasing after being processed
    t = sectioned_data[:, 1]
    x = sectioned_data[:, 2]
    print('checking that time and displacement series are monotonic')
    print(f'times monotonic: {isMonotonic(t)}')
    print(f'x monotonic: {isMonotonic(x)}')

    # remove non-monotonically increasing time indices if necessary
    cleaned_data = remove_non_monotonic(t, x, sectioned_data, axis=0)

    # data for pymc
    mutrue = cleaned_data[:, 0]
    t = cleaned_data[:, 1]
    x = cleaned_data[:, 2]

    # calculate loading velocities = dx/dt
    vlps = calc_derivative(x, t, window_len=cfig.vel_windowlen)

    cfig.set_disp_bounds(x)
    plotx = x * um_to_mm
    # plot raw data section with filtered/downsampled for reference
    df_raw = df[(df['vdcdt_um'] > cfig.mindisp) & (df['vdcdt_um'] < cfig.maxdisp)]
    plt.figure(1)
    plt.plot(df_raw['vdcdt_um'] * um_to_mm, df_raw['mu'], '.', alpha=0.5, label='raw data')
    plt.plot(plotx, mutrue, '.', alpha=0.8, label='downsampled, filtered, sectioned data')
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title('Observed data section (def get_obs_data)')
    plt.ylim([np.min(mutrue) - 0.01, np.max(mutrue) + 0.01])
    plt.legend()
    # plt.show()

    return mutrue, t, vlps, x


def isMonotonic(A):
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))


def remove_non_monotonic(times, x, data, axis=0):
    nmi = []
    if not np.all(np.diff(times) >= 0):
        print('time series can become non-monotonic after downsampling which is an issue for the sampler')
        print('now removing non-monotonic t indices from (t, mu, x) dataset')
        print(f'input downsampled data shape = {data.shape}')
        # Find the indices where the array is not monotonically increasing
        nmi_t = np.where(np.diff(times) < 0)[0]
        nmi.append(nmi_t)
        # print(f'non monotonic time indices = {non_monotonic_indices}')

    if not np.all(np.diff(x) >= 0):
        print('displacement series is non-monotonic')
        print('now removing non-monotonic x indices from (t, mu, x) dataset')
        print(f'input downsampled data shape = {data.shape}')
        nmi_x = np.where(np.diff(x) < 0)[0]
        nmi.append(nmi_x)

    if nmi:
        # Remove the non-monotonic data points
        cleaned_data = np.delete(data, nmi, axis)
        print('removed bad data? should be True')
        print(isMonotonic(cleaned_data[:, 1]))
        return cleaned_data

    # Array is already monotonically increasing, return it as is
    print('Array is already monotonically increasing, returning as is')
    return data


# reads in data
def read_hdf(fullpath):
    filename = fullpath
    names = []
    df = pd.DataFrame()
    with h5py.File(filename, 'r') as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]
        # loop on names:
        for name in f.keys():
            # print(name)
            names.append(name)
        # loop on names and H5 objects:
        for name, h5obj in f.items():
            if isinstance(h5obj, h5py.Group):
                print(f'{name} is a Group')
            elif isinstance(h5obj, h5py.Dataset):
                # return a np.array using dataset object:
                arr1 = h5obj[:]
                # return a np.array using dataset name:
                arr2 = f[name][:]
                df[f'{name}'] = arr1

    return df, names


def downsample_dataset(mu, t, x):
    # low pass filter
    mu_f = savgol_filter(mu, window_length=cfig.filter_windowlen, polyorder=3, mode='mirror')

    # stack time and mu arrays to sample together
    f_data = np.column_stack((mu_f, t, x))

    # downsamples to every qth sample after applying low-pass filter along columns
    f_ds = sp.signal.decimate(f_data, cfig.q, ftype='fir', axis=0)

    # FOR P5760 ONLY - no downsampling
    # f_ds = f_data

    return f_ds, mu_f


# section_data(...) slices friction data into model-able sections
def section_data(data):
    df0 = pd.DataFrame(data)
    # changing column names
    df = df0.set_axis(['mu', 't', 'x'], axis=1)

    # cut off first 100 points to avoid sectioning mistakes
    df = df.iloc[100:]

    start_idx = np.argmax(df['t'] > cfig.mintime)
    end_idx = np.argmax(df['t'] > cfig.maxtime)

    df_section = df.iloc[start_idx:end_idx]

    return df_section.to_numpy(), start_idx, end_idx


# generate_rsf_data(...) is a synthetic data generator - only used when troubleshooting
def generate_rsf_data(times, vlps, a, b, Dc, mu0):
    # runs rsfmodel.py to generate synthetic friction data
    k, vref = get_constants(vlps)
    print('STARTING SYNTHETIC PARAMETERS - ANSWERS')
    print(f'a={a}')
    print(f'b={b}')
    print(f'Dc={Dc}')
    print(f'mu0={mu0}')

    # Size of dataset
    size = len(times)
    print(f'size of dataset = {size}')

    model = rsf.Model()

    # Set model initial conditions
    model.mu0 = mu0  # Friction initial (at the reference velocity)
    model.a = a  # Empirical coefficient for the direct effect
    model.k = cfig.k  # Normalized System stiffness (friction/micron)
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

    # plus noise
    mutrue = mu + (1 / 100) * np.random.normal(np.mean(mu), 0.1, (len(mu),))

    return mutrue, size


# MCMC MODEL SETUP FUNCTIONS
# constants used in rsf model
def get_constants(vlps):
    k = cfig.k
    vref = vlps[0]

    return k, vref


# MCMC priors
def get_priors():
    mus, sigmas, dist_types = cfig.get_prior_parameters()
    labels = ['a', 'b', 'Dc', 'mu0', 's']
    # s = pm.HalfNormal('s', sigma=0.01)

    priors = []

    for l, m, sig, d in zip(labels, mus, sigmas, dist_types):
        if d == 'LogNormal':
            pr = pm.LogNormal(l, mu=m, sigma=sig)
            priors.append(pr)
        if d == 'HalfNormal':
            pr = pm.HalfNormal(l, sigma=sig)
            priors.append(pr)

    return priors

    # return [pm.LogNormal(l, mu=m, sigma=s) for l, m, s in zip(labels, mus, sigmas)], s


def check_priors(a, b, Dc, mu0, s, mus, sigmas):
    vpriors = pm.draw([a, b, Dc, mu0, s], draws=cfig.ndr)
    names = ['a', 'b', 'Dc', 'mu0', 's']

    for i, name in enumerate(names):
        print(f'{name} input mu, sigma = {mus[i]}, {sigmas[i]}')
        print(f'{name} prior min,max = {np.min(vpriors[i])}, {np.max(vpriors[i])}')
        print(f'{name} prior mode = {(sp.stats.mode(vpriors[i])).mode}')
        plt.figure(1000)
        sns.kdeplot(vpriors[i], label=f'{name}', common_norm=False, bw_method=0.1)
        plt.xlim(-0.1, 100)
        plt.title('prior distributions')
        plt.legend()
    plt.show()
    sys.exit()


def get_vmax_lc(vlps):
    # define characteristic length and velocity
    # characteristic length = max grain size in gouge = 125 micrometers for most
    # characteristic velocity = max loading velocity
    lc = cfig.lc
    vmax = np.max(vlps)

    return lc, vmax


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    # define characteristic length and velocity for nondimensionalizing

    lc, vmax = get_vmax_lc(vlps)

    # then remove dimensions
    k0 = cfig.k * cfig.lc
    vlps0 = (vlps / vmax)
    vref0 = vref / vmax

    t0 = times * vmax / lc
    t0 = t0 - t0[0]

    return k0, vlps0, vref0, t0


# MAIN - CALLS ALL FUNCTIONS AND IMPLEMENTS MCMC MODEL RUN
def main():
    print('MCMC RATE AND STATE FRICTION MODEL')
    # so I can figure out how long it's taking when I inevitably forget to check
    comptime_start = get_time('start')

    # observed data
    mutrue, times, vlps, x = get_obs_data()
    vmax = np.max(vlps)

    if np.any(vlps < 0):
        print('NEGATIVE VELOCITIES - FIX!')

    k, vref = get_constants(vlps)
    print(f'k = {k}; vref = {vref}')

    from Loglikelihood import Loglike

    # use PyMC to sampler from log-likelihood
    with pm.Model() as mcmcmodel:
        # priors on stochastic parameters and non-dimensionalized constants
        a, b, Dc, mu0, s = get_priors()
        k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

        # create loglikelihood Op (wrapper for numerical solution to work with pymc)
        loglike = Loglike(t0, vlps0, k0, vref0, mutrue, vmax)

        # convert parameters to be estimated to tensor vector
        theta = pt.tensor.as_tensor_variable([a, b, Dc, mu0, s])

        # use a Potential for likelihood function
        pm.Potential("likelihood", loglike(theta))

        # mcmc sampler parameters
        tune = cfig.ntune
        draws = cfig.ndr
        chains = cfig.nch
        cores = cfig.ncores

        # initvals = {'a': 0.005, 'b': 0.005, 'Dc': 50, 'mu0': 0.41}

        print(f'num draws = {draws}; num chains = {chains}')
        print('starting sampler')
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=pm.Metropolis(),
                          discard_tuned_samples=True)
        print(f'inference data = {idata}')

        # save model parameter stats
        vsummary = save_stats(idata)

        # save the trace
        save_trace(idata)

        # post-processing plots bare minimum results, save figs saves figures
        post_processing(idata, times, vlps, mutrue, x)
        save_figs(cfig.mcmc_out_dir)

    comptime_end = get_time('end')
    time_elapsed = comptime_end - comptime_start
    print(f'time elapsed = {time_elapsed}')

    write_model_info(time_elapsed=time_elapsed,
                     vref=vref,
                     vsummary=vsummary,
                     file_name=f'{cfig.samplename}_proc.hdf5',
                     times=times)

    # plt.show()

    print('simulation complete')


if __name__ == '__main__':
    main()
