import json
import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from rsfmodel import staterelations, rsf, plot
import pytensor as pt
import pytensor.tensor as tt
import sys
import h5py
import scipy as sp
from scipy.signal import savgol_filter
from datetime import datetime
import time
import seaborn as sns
import globals
from globals import myglobals
import cProfile


p = myglobals.get_output_storage_folder()

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

    plt.plot(x * um_to_mm, df['mu'])
    plt.title('mu')
    plt.xlabel('displacement (mm)')

    sys.exit()


def check_file_exist(p, filename):
    isExisting = os.path.exists(os.path.join(p, filename))
    if isExisting is False:
        print(f'file does not exist, returning file name --> {filename}')
        return filename
    elif isExisting is True:
        print(f'file does exist, overwriting --> {filename}')
        return filename


# POST MODEL-RUN OPERATIONS AND PLOTTING FUNCTIONS
def save_trace(idata):
    # saves trace for post-processing
    out_name = f'{myglobals.sim_name}_idata'
    p = myglobals.get_output_storage_folder()
    name = check_file_exist(p, out_name)
    idata.to_netcdf(os.path.join(p, f'{name}'))


def plot_trace(idata):
    az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'])


def save_stats(idata):
    summary = az.summary(idata, kind='stats')
    print(f'summary: {summary}')
    summary.to_csv(os.path.join(p, 'idata.csv'))

    return summary


def post_processing(idata, times, vlps, mutrue, x):
    # save dataset in case needed later
    df_data = pd.DataFrame(np.column_stack((times, x, vlps, mutrue)), columns=['times', 'x', 'vlps', 'mutrue'])
    p = myglobals.get_output_storage_folder()
    df_data.to_csv(os.path.join(p, 'section_data.csv'))

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
    fname = os.path.join(p, 'out.txt')

    samplerstrs = ['SAMPLER INFO', 'num draws', 'num chains', 'tune', 'prior mus and sigmas', 'runtime (s)']
    modelstrs = ['MODEL INFO', 'constants', 'k', 'vref', 'section min displacement', 'section max displacement',
                 'characteristic length', 'velocity calc window length', 'filter window length', 'downsampling rate']
    summarystr = ['SAMPLE VARS SUMMARY']
    strlist = [samplerstrs, modelstrs, summarystr]

    samplervals = ['', myglobals.ndr, myglobals.nch, myglobals.ntune, myglobals.get_prior_parameters(), time_elapsed]
    modelvals = ['', '', myglobals.k, vref, myglobals.mindisp, myglobals.maxdisp, myglobals.lc, myglobals.vel_windowlen,
                 myglobals.filter_windowlen, myglobals.q]
    summaryvals = [vsummary]
    vallist = [samplervals, modelvals, summaryvals]

    with open(fname, mode='w') as f:
        f.write(f'SAMPLE: {file_name}\n')
        f.write(f'section_ID: {myglobals.section_id}\n')
        f.write(f'from t = {times[0]} to t= {times[-1]} seconds\n')
        f.write(f'from x = {myglobals.mindisp} to x = {myglobals.maxdisp} mm\n')
        for strings, vals in zip(strlist, vallist):
            # f.writelines(f'{strings}: {vals}')
            for string, val in zip(strings, vals):
                f.write(f'{string}: {val}\n')

    payload = {'sample': file_name,
               'section_ID': myglobals.section_id,
               'time_start': myglobals.mintime,
               'time_end': myglobals.maxtime,
               'x_start': myglobals.mindisp,
               'x_end': myglobals.maxdisp,
               'n_draws': myglobals.ndr,
               'n_chains': myglobals.nch,
               'n_tune': myglobals.ntune,
               'prior_mus_sigmas': myglobals.get_prior_parameters(),
               'runtime_s': time_elapsed,
               'k': myglobals.k,
               'vref': vref,
               'lc': myglobals.lc,
               'dvdt_window_len': myglobals.vel_windowlen,
               'filter_window_len': myglobals.filter_windowlen,
               'q': myglobals.q,
               'threshold': myglobals.threshold
               }

    with open(os.path.join(p, 'out.json'), mode='w') as wfile:
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
                             polyorder=1,
                             deriv=1)
        dydN = savgol_filter(y,
                             window_length=window_len,
                             polyorder=1,
                             deriv=1)
        dydx = dydN / dxdN

        dydx_smooth = savgol_filter(dydx,
                                    window_length=window_len,
                                    polyorder=1)
        return dydx_smooth
    else:
        print(f'calculating derivative using gradient because window_len= {window_len}')
        dydx = np.gradient(y, x)
        return dydx


# imports observed data, sends it through series of processing steps
def get_obs_data():
    data_path = myglobals.make_path('data', 'FORGE_DataShare', f'{myglobals.samplename}',
                                    f'{myglobals.samplename}_proc.hdf5')
    print(f'getting data from: {data_path}')
    f = h5py.File(data_path, 'r')

    # read in data from hdf file, print column names
    df, names = read_hdf(data_path)

    # comment this in when deciding which displacement sections to use
    # preplot(df, names)

    # first remove any mu < 0 data from experiment
    df = df[(df['mu'] > 0)]

    # convert to numpy arrays
    t = df['time_s'].to_numpy()
    mu = df['mu'].to_numpy()
    x = df['vdcdt_um'].to_numpy()

    # calculate loading velocities = dx/dt
    vlps = calc_derivative(x, t, window_len=myglobals.vel_windowlen)

    # filters and downsamples data
    f_ds, mu_f = downsample_dataset(mu, t, vlps, x)

    # sections data - make this into a loop to run multiple sections one after another
    sectioned_data, start_idx, end_idx = section_data(f_ds)

    # need to check that time vals are monotonically increasing after being processed
    t = sectioned_data[:, 1]
    print('checking that time series is monotonic after processing')
    print(isMonotonic(t))

    # remove non-monotonically increasing time indices if necessary
    cleaned_data = remove_non_monotonic(t, sectioned_data, axis=0)

    # data for pymc
    mutrue = cleaned_data[:, 0]
    times = cleaned_data[:, 1]
    vlps = cleaned_data[:, 2]
    x = cleaned_data[:, 3]

    myglobals.set_disp_bounds(x)
    plotx = x * um_to_mm
    # plot raw data section with filtered/downsampled for reference
    df_raw = df[(df['vdcdt_um'] > myglobals.mindisp) & (df['vdcdt_um'] < myglobals.maxdisp)]
    plt.figure(1)
    plt.plot(df_raw['vdcdt_um'] * um_to_mm, df_raw['mu'], '.', alpha=0.2, label='raw data')
    plt.plot(plotx, mutrue, '.', alpha=0.8, label='downsampled, filtered, sectioned data')
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title('Observed data section (def get_obs_data)')
    plt.ylim([np.min(mutrue) - 0.01, np.max(mutrue) + 0.01])
    plt.legend()
    # plt.show()

    return mutrue, times, vlps, x


def isMonotonic(A):
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))


def remove_non_monotonic(times, data, axis=0):
    # this may have only been an issue before removing mu values < 0 from the dataset, keeping it in just in case
    if not np.all(np.diff(times) >= 0):
        print('time series can become non-monotonic after downsampling which is an issue for the sampler')
        print('now removing non-monotonic t and mu values from dataset')
        print(f'input downsampled data shape = {data.shape}')
        # Find the indices where the array is not monotonically increasing
        non_monotonic_indices = np.where(np.diff(times) < 0)[0]
        # print(f'non monotonic time indices = {non_monotonic_indices}')

        # Remove the non-monotonic data points
        cleaned_data = np.delete(data, non_monotonic_indices, axis)
        print('removed bad data? should be True')
        print(isMonotonic(cleaned_data[:, 1]))
        return cleaned_data

    # Array is already monotonically increasing, return it as is
    print('Array is already monotonically increasing, returning as is')
    return data


# reads in data
def read_hdf(fullpath):
    filename = fullpath
    print(f'reading file: {filename}')
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


def downsample_dataset(mu, t, vlps, x):
    # low pass filter
    mu_f = savgol_filter(mu, window_length=myglobals.filter_windowlen, polyorder=1, mode='mirror')

    # stack time and mu arrays to sample together
    f_data = np.column_stack((mu_f, t, vlps, x))

    # downsamples to every qth sample after applying low-pass filter along columns
    f_ds = sp.signal.decimate(f_data, myglobals.q, ftype='fir', axis=0)

    # FOR P5760 ONLY - no downsampling
    # f_ds = f_data

    return f_ds, mu_f


# section_data(...) slices friction data into model-able sections
def section_data(data):
    df0 = pd.DataFrame(data)
    # changing column names
    df = df0.set_axis(['mu', 't', 'vlps', 'x'], axis=1)

    # cut off first 100 points to avoid sectioning mistakes
    df = df.iloc[100:]

    start_idx = np.argmax(df['t'] > myglobals.mintime)
    end_idx = np.argmax(df['t'] > myglobals.maxtime)

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
    model.k = myglobals.k  # Normalized System stiffness (friction/micron)
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
    k = myglobals.k
    vref = vlps[0]

    return k, vref


# MCMC priors
def get_priors():
    mus, sigmas = myglobals.get_prior_parameters()

    labels = ['a', 'b', 'Dc', 'mu0']

    # a = pm.LogNormal('a', mu=mus[0], sigma=sigmas[0])
    # b = pm.LogNormal('b', mu=mus[1], sigma=sigmas[1])
    # Dc = pm.LogNormal('Dc', mu=mus[2], sigma=sigmas[2])
    # mu0 = pm.LogNormal('mu0', mu=mus[3], sigma=sigmas[3])
    #
    # check_priors(a, b, Dc, mu0, mus, sigmas)

    s = pm.HalfNormal('s', sigma=1)
    return [pm.LogNormal(l, mu=m, sigma=s) for l, m, s in zip(labels, mus, sigmas)], s

    # return a, b, Dc, mu0, mus, sigmas


def check_priors(a, b, Dc, mu0, mus, sigmas):
    vpriors = pm.draw([a, b, Dc, mu0], draws=500000)
    names = ['a', 'b', 'Dc', 'mu0']

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
    lc = myglobals.lc
    vmax = np.max(vlps)

    return lc, vmax


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    # define characteristic length and velocity for nondimensionalizing

    lc, vmax = get_vmax_lc(vlps)

    # then remove dimensions
    k0 = myglobals.k * myglobals.lc
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
    np.save('timetest.npy', times)
    vmax = myglobals.set_vch(vlps)

    k, vref = get_constants(vlps)
    print(f'k = {k}; vref = {vref}')

    from Loglikelihood import Loglike

    # use PyMC to sampler from log-likelihood
    with pm.Model() as mcmcmodel:
        # priors on stochastic parameters and non-dimensionalized constants
        [a, b, Dc, mu0], s = get_priors()
        k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

        # create loglikelihood Op (wrapper for numerical solution to work with pymc)
        loglike = Loglike(t0, vlps0, k0, vref0, mutrue, vmax)

        # convert parameters to be estimated to tensor vector
        theta = pt.tensor.as_tensor_variable([a, b, Dc, mu0, s])

        # use a Potential for likelihood function
        pm.Potential("likelihood", loglike(theta))

        # mcmc sampler parameters
        tune = myglobals.ntune
        draws = myglobals.ndr
        chains = myglobals.nch
        cores = myglobals.ncores

        # initvals = {'a': 0.005, 'b': 0.005, 'Dc': 50, 'mu0': 0.41}

        print(f'num draws = {draws}; num chains = {chains}')
        print('starting sampler')
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=pm.Metropolis(),
                          discard_tuned_samples=True)
        print(f'inference data = {idata}')

        # create storage directory
        myglobals.get_output_storage_folder()

        # save model parameter stats
        vsummary = save_stats(idata)

        # save the trace
        save_trace(idata)

        # post-processing plots bare minimum results, save figs saves figures
        post_processing(idata, times, vlps, mutrue, x)
        save_figs(p)

    comptime_end = get_time('end')
    time_elapsed = comptime_end - comptime_start
    print(f'time elapsed = {time_elapsed}')

    write_model_info(time_elapsed=time_elapsed,
                     vref=vref,
                     vsummary=vsummary,
                     file_name=f'{myglobals.samplename}_proc.hdf5',
                     times=times)

    plt.show()

    print('simulation complete')


if __name__ == '__main__':
    main()
