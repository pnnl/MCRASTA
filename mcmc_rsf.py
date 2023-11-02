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
def preplot(df, colnames):
    t = df['time_s']
    x = df['vdcdt_um']

    plt.plot(x * um_to_mm, df['mu'])
    plt.title('mu')
    plt.xlabel('displacement (mm)')
    plt.show()

    sys.exit()


def check_file_exist(dirpath, filename):
    isExisting = os.path.exists(os.path.join(dirpath, filename))
    if isExisting is False:
        print(f'file does not exist, returning file name --> {filename}')
        return filename
    elif isExisting is True:
        print(f'file does exist, overwriting --> {filename}')
        return filename


def get_storage_folder(dirname, samplename):
    global dirpath
    print('checking if storage directory exists')
    homefolder = os.path.expanduser('~')
    outfolder = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', f'{samplename}')

    dirpath = os.path.join(homefolder, outfolder, dirname)
    isExisting = os.path.exists(dirpath)
    if isExisting is False:
        print(f'directory does not exist, creating new directory --> {dirpath}')
        os.makedirs(dirpath)
        return dirpath
    elif isExisting is True:
        print(f'directory exists, all outputs will be saved to existing directory and any existing files will be '
              f'overwritten --> {dirpath}')
        return dirpath


def get_sim_name(draws, chains):
    global sim_name
    sim_name = f'out_{draws}d{chains}ch'
    return sim_name


# POST MODEL-RUN OPERATIONS AND PLOTTING FUNCTIONS
def save_trace(idata):
    # saves trace for post-processing
    out_name = f'{sim_name}_idata'
    name = check_file_exist(dirpath, out_name)
    idata.to_netcdf(os.path.join(dirpath, f'{name}'))


def plot_trace(idata):
    az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'])


def save_stats(idata, root):
    summary = az.summary(idata, kind='stats')
    print(f'summary: {summary}')
    summary.to_csv(os.path.join(root, 'idata.csv'))

    return summary


def post_processing(idata, times, vlps, mutrue, x):
    # save dataset in case needed later
    df_data = pd.DataFrame(np.column_stack((times, x, vlps, mutrue)), columns=['times', 'x', 'vlps', 'mutrue'])
    df_data.to_csv(os.path.join(dirpath, 'section_data.csv'))

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


def write_model_info(draws, chains, tune, prior_mus, prior_sigmas, time_elapsed, k, vref, vsummary, ppsummary, file_name, times):
    fname = os.path.join(dirpath, 'out.txt')

    samplerstrs = ['SAMPLER INFO', 'num draws', 'num chains', 'tune', 'prior mus', 'prior sigmas', 'runtime (s)']
    modelstrs = ['MODEL INFO', 'constants', 'k', 'vref']
    summarystr = ['SAMPLE VARS SUMMARY', 'POST PRED SAMPLE SUMMARY']
    strlist = [samplerstrs, modelstrs, summarystr]

    samplervals = ['', draws, chains, tune, prior_mus, prior_sigmas, time_elapsed]
    modelvals = ['', '', k, vref]
    summaryvals = [vsummary, 'none']
    vallist = [samplervals, modelvals, summaryvals]

    with open(fname, mode='w') as f:
        f.write(f'SAMPLE: {file_name}\n')
        f.write(f'from t = {times[0]} to t= {times[-1]} seconds\n')
        for strings, vals in zip(strlist, vallist):
            # f.writelines(f'{strings}: {vals}')
            for string, val in zip(strings, vals):
                f.write(f'{string}: {val}\n')


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
def calc_derivative(y, x, window_len=10):
    # returns dydx
    if window_len is not None:
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
        return dydx_smooth
    else:
        dydx = np.gradient(y, x)
        return dydx


# imports observed data, sends it through series of processing steps
def get_obs_data(samplename):
    homefolder = os.path.expanduser('~')
    path = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'data', 'FORGE_DataShare', f'{samplename}')
    name = f'{samplename}_proc.hdf5'
    sample_name = name
    fullpath = os.path.join(homefolder, path, name)
    print(f'getting data from: {fullpath}')
    f = h5py.File(os.path.join(homefolder, path, name), 'r')

    # read in data from hdf file, print column names
    df, names = read_hdf(fullpath)

    # comment this in when deciding which displacement sections to use
    # preplot(df, names)

    # first remove any mu < 0 data from experiment
    df = df[(df['mu'] > 0)]

    # convert to numpy arrays
    t = df['time_s'].to_numpy()
    mu = df['mu'].to_numpy()
    x = df['vdcdt_um'].to_numpy()

    # calculate loading velocities = dx/dt
    vlps = calc_derivative(x, t)

    plt.figure(10)
    plt.plot(x, vlps)
    plt.show()
    sys.exit()

    # filters and downsamples data
    f_ds, mu_f = downsample_dataset(mu, t, vlps, x)

    # sections data
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

    # plot raw data section with filtered/downsampled for reference
    df_raw = df[(df['vdcdt_um'] > 5.5 / um_to_mm) & (df['vdcdt_um'] < 6.78 / um_to_mm)]
    plt.figure(1)
    plt.plot(df_raw['vdcdt_um'] * um_to_mm, df_raw['mu'], '.', alpha=0.2, label='raw data')
    plt.plot(x * um_to_mm, mutrue, '.', alpha=0.8, label='downsampled, filtered, sectioned data')
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title('Observed data section (def get_obs_data)')
    plt.ylim([np.min(mutrue) - 0.01, np.max(mutrue) + 0.01])
    plt.legend()
    plt.show()

    return mutrue, times, vlps, x, sample_name


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
            print(name)
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
    mu_f = savgol_filter(mu, window_length=3, polyorder=2, mode='mirror')

    # stack time and mu arrays to sample together
    f_data = np.column_stack((mu_f, t, vlps, x))

    # downsamples to every qth sample after applying low-pass filter along columns
    q = 2
    f_ds = sp.signal.decimate(f_data, q, ftype='fir', axis=0)

    # FOR P5760 ONLY - no downsampling
    f_ds = f_data

    t_ds = f_ds[:, 1]
    mu_ds = f_ds[:, 0]
    x_ds = f_ds[:, 3]

    # # plot series as sanity check
    # plt.plot(x, mu, '.-', label='original data')
    # plt.plot(x, mu_f, '.-', label='filtered data')
    # plt.plot(x_ds, mu_ds, '.-', label='downsampled data')
    # plt.xlabel('disp (mm)')
    # plt.ylabel('mu')
    # plt.title('def downsample_dataset')
    # plt.legend()
    # plt.show()
    # sys.exit()

    return f_ds, mu_f


# section_data(...) slices friction data into model-able sections
def section_data(data):
    df0 = pd.DataFrame(data)
    # changing column names
    df = df0.set_axis(['mu', 't', 'vlps', 'x'], axis=1)

    start_idx = np.argmax(df['x'] > 5.5 / um_to_mm)
    end_idx = np.argmax(df['x'] > 6.78 / um_to_mm)

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

    # plus noise
    mutrue = mu + (1 / 100) * np.random.normal(np.mean(mu), 0.1, (len(mu),))

    return mutrue, size


# MCMC MODEL SETUP FUNCTIONS
# constants used in rsf model
def get_constants(vlps):
    k = 0.00194
    vref = vlps[0]

    return k, vref


# MCMC priors
def get_priors(vref, times):
    mus = [-5, -5, 3, -1]
    sigmas = [0.8, 0.8, 0.3, 0.3]

    a = pm.LogNormal('a', mu=mus[0], sigma=sigmas[0])
    b = pm.LogNormal('b', mu=mus[1], sigma=sigmas[1])
    Dc = pm.LogNormal('Dc', mu=mus[2], sigma=sigmas[2])
    mu0 = pm.LogNormal('mu0', mu=mus[3], sigma=sigmas[3])

    check_priors(a, b, Dc, mu0, mus, sigmas)

    return a, b, Dc, mu0, mus, sigmas


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


# forward RSF model - from Leeman (2016), uses the RSF toolkit from GitHub. rsf.py; state_relations.py; plot.py
# returns simulated mu value for use in pymc
def mcmc_rsf_sim(theta, t, v, k, vref, vmax):
    # unpack parameters
    a, b, Dc, mu0 = theta
    l0, vwrong = get_vmax_l0(v)

    # initialize rsf model
    model = rsf.Model()

    # Size of dataset
    model.datalen = len(t)

    # Set initial conditions
    model.mu0 = mu0  # Friction initial (at the reference velocity)
    model.a = a  # Empirical coefficient for the direct effect
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = v[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.b = b  # Empirical coefficient for the evolution effect
    state1.Dc = Dc  # Critical slip distance
    # all other parameters are already nondimensionalized, but the state parameter is nd'd in staterelations.py,
    # so we need to pass characteristic velocity (vmax) and length (l0) into the fwd model
    state1.vmax = vmax
    state1.l0 = l0

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t   # nondimensionalized time
    lp_velocity = v

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity

    # Run the model!
    model.solve()

    mu_sim = model.results.friction

    return mu_sim


def get_vmax_l0(vlps):
    # define characteristic length and velocity
    # characteristic length = max grain size in gouge = 125 micrometers for most
    # characteristic velocity = max loading velocity
    l0 = 125
    vmax = np.max(vlps)

    return l0, vmax


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    # define characteristic length and velocity for nondimensionalizing
    l0, vmax = get_vmax_l0(vlps)

    # then remove dimensions
    k0 = k * l0
    vlps0 = vlps / vmax
    vref0 = vref / vmax

    t0 = times * vmax / l0
    t0 = t0 - t0[0]

    return k0, vlps0, vref0, t0


# Custom LogLikelihood function for use in pymc - runs forward model with sample draws
def log_likelihood(theta, times, vlps, k, vref, data, vmax):
    if type(theta) == list:
        theta = theta[0]
    (
        a,
        b,
        Dc,
        mu0,
    ) = theta

    y_pred = mcmc_rsf_sim(theta, times, vlps, k, vref, vmax)
    resids = (data - y_pred)
    # print('resid = ', resids)
    logp = -1 / 2 * (np.sum(resids ** 2))
    # print(f'logp = {logp}')

    return logp


# wrapper classes to theano-ize log likelihood
class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, times, vlps, k, vref, data, vmax):
        self.data = data
        self.times = times
        self.vlps = vlps
        self.k = k
        self.vref = vref
        self.vmax = vmax

    def perform(self, node, inputs, outputs):
        logp = log_likelihood(inputs, self.times, self.vlps, self.k, self.vref, self.data, self.vmax)
        outputs[0][0] = np.array(logp)


# MAIN - CALLS ALL FUNCTIONS AND IMPLEMENTS MCMC MODEL RUN
def main():
    print('MCMC RATE AND STATE FRICTION MODEL')
    # so I can figure out how long it's taking when I inevitably forget to check
    comptime_start = get_time('start')
    samplename = 'p5760'

    # observed data
    mutrue, times, vlps, x, file_name = get_obs_data(samplename)
    vmax = np.max(vlps)

    k, vref = get_constants(vlps)
    print(f'k = {k}; vref = {vref}')

    # use PyMC to sampler from log-likelihood
    with pm.Model() as mcmcmodel:
        # priors on stochastic parameters and non-dimensionalized constants
        a, b, Dc, mu0, prior_mus, prior_sigmas = get_priors(vref, times)
        k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

        # create loglikelihood Op (wrapper for numerical solution to work with pymc)
        loglike = Loglike(t0, vlps0, k0, vref0, mutrue, vmax)

        # convert parameters to be estimated to tensor vector
        theta = pt.tensor.as_tensor_variable([a, b, Dc, mu0])

        # use a Potential for likelihood function
        pm.Potential("likelihood", loglike(theta))

        # mcmc sampler parameterss
        tune = 20000
        draws = 500000
        chains = 4
        cores = 4

        print(f'num draws = {draws}; num chains = {chains}')
        print('starting sampler')
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=pm.Metropolis(), discard_tuned_samples=False)
        print(f'inference data = {idata}')

        # create storage directory
        get_sim_name(draws, chains)
        get_storage_folder(sim_name, samplename)

        # save model parameter stats
        vsummary = save_stats(idata, dirpath)

        # save the trace
        save_trace(idata)

        # post-processing plots bare minimum results, save figs saves figures
        post_processing(idata, times, vlps, mutrue, x)
        save_figs(dirpath)

    comptime_end = get_time('end')
    time_elapsed = comptime_end - comptime_start
    print(f'time elapsed = {time_elapsed}')

    write_model_info(draws=draws,
                     chains=chains,
                     tune=tune,
                     prior_mus=prior_mus,
                     prior_sigmas=prior_sigmas,
                     time_elapsed=time_elapsed,
                     k=k,
                     vref=vref,
                     vsummary=vsummary,
                     ppsummary=None,
                     file_name=file_name,
                     times=times)

    plt.show()

    print('simulation complete')


if __name__ == '__main__':
    main()
