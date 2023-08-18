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


def preplot(df, colnames):
    t = df['time_s']
    x = df['vdcdt_um']

    plt.plot(x * um_to_mm, df['mu'])
    plt.title('mu')
    plt.xlabel('displacement (mm)')
    # for i, col in enumerate(colnames):
    #     plt.figure(i)
    #     plt.plot(t, df[f'{col}'])
    #     plt.title(f'{col}')
    #     plt.xlabel('time (s)')
    #
    # lpdisp = df['vdcdt_um']*um_to_mm
    # plt.figure(i+1)
    # plt.plot(lpdisp, df['mu'])
    plt.show()

    sys.exit()


def check_file_exist(dirpath, filename):
    isExisting = os.path.exists(os.path.join(dirpath, filename))
    if isExisting is False:
        print(f'file does not exist, returning file name --> {filename}')
        return filename
    elif isExisting is True:
        print(f'file does exist, rename new output for now, eventually delete previous --> {filename}')
        oldname = filename
        newname = f'{oldname}_a'
        return newname


def get_storage_folder(dirname):
    global dirpath
    print('checking if storage directory exists')
    homefolder = os.path.expanduser('~')
    outfolder = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out')
    # name = sim_name

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
def sample_posterior_predcheck(idata):
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)


def save_trace(idata):
    # save trace for easier debugging if needed
    out_name = f'{sim_name}_idata'
    name = check_file_exist(dirpath, out_name)
    idata.to_netcdf(os.path.join(dirpath, f'{name}'))


def plot_trace(idata):
    ax = az.plot_trace(idata, var_names=['a', 'b', 'Dc_nd', 'mu0'])


def plot_posterior_predictive(idata):
    az.plot_ppc(idata)


def save_stats(idata, root):
    summary = az.summary(idata, kind='stats')
    print(f'summary: {summary}')
    summary.to_csv(os.path.join(root, 'idata.csv'))

    return summary


def post_processing(idata, times, vlps, mutrue, x):
    # save dataset in case needed later
    df_data = pd.DataFrame(np.column_stack((times, x, vlps, mutrue)), columns=['times', 'x', 'vlps', 'mutrue'])
    df_data.to_csv(os.path.join(dirpath, 'section_data.csv'))

    # to extract simulated mu values for realizations
    # stacked_pp = az.extract(idata.posterior_predictive)
    # print(f'stacked = {stacked_pp}')
    # musims = stacked_pp.simulator.values

    # df_musims = pd.DataFrame(musims)
    # df_musims['t'] = times
    #
    # # now save them
    # df_musims.to_csv(os.path.join(dirpath, 'musims.csv'))
    # print(f'simulated mu values = {musims}')
    # print(f'shape of posterior predictive dataset = {musims.shape}')

    # plot trace and then posterior predictive plot
    plot_trace(idata)
    # plot_posterior_predictive(idata)

    # modelvals = az.extract(idata.posterior)
    # # print('modelvals = ', modelvals)
    #
    # a = modelvals.a.values
    # b = modelvals.b.values
    # Dc = modelvals.Dc.values
    # mu0 = modelvals.mu0.values

    # mu_sims = []
    # for i in np.arange(0, len(a), 1):
    #     mu_sim, bs = generate_rsf_data(times, vlps, a[i], b[i], Dc[i], mu0[i])
    #     mu_sims.append(mu_sim)
    #
    # musims = np.array(mu_sims)
    # musims = np.transpose(musims)
    #
    #
    # # now plot simulated mus with true mu
    # t = times
    # plt.figure(500)
    # plt.plot(t, mutrue, 'k.', label='observed', alpha=0.7)
    # plt.plot(t, musims, 'b-', alpha=0.3)
    # plt.xlabel('time (s)')
    # plt.ylabel('mu')
    # plt.title('Observed and simulated friction values')
    # plt.legend()
    # plt.show()

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


def write_model_info(draws, chains, tune, prior_mus, prior_sigmas, time_elapsed, k, vref, vsummary, ppsummary, sample_name, times):
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
        f.write(f'SAMPLE: {sample_name}\n')
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
def calc_derivative(y, x, window_len=100):
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
def get_obs_data():
    # global sample_name, mutrue, vlps, times, x
    homefolder = os.path.expanduser('~')
    path = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'data', 'FORGE_DataShare', 'p5894')
    name = 'p5894_proc.hdf5'
    sample_name = name
    fullpath = os.path.join(homefolder, path, name)
    print(f'getting data from: {fullpath}')
    f = h5py.File(os.path.join(homefolder, path, name), 'r')
    # print(list(f.keys()))

    # read in data from hdf file, print column names
    df, names = read_hdf(fullpath)
    # print(names)

    # preplot(df, names)

    # first remove any mu < 0 data from end of experiment
    idx = np.argmax(df['mu'] < 0)
    df = df.iloc[0:idx]

    # convert to numpy arrays
    t = df['time_s'].to_numpy()
    mu = df['mu'].to_numpy()
    x = df['vdcdt_um'].to_numpy()

    # calculate loading velocities = dx/dt
    vlps = calc_derivative(x, t)

    # filters and downsamples data
    f_ds, mu_f = downsample_dataset(mu, t, vlps, x)

    # sections data
    sectioned_data, start_idx, end_idx = section_data(f_ds)

    # print(f'sectioned data shape = {sectioned_data.shape}')

    # need to check that time vals are monotonically increasing after being processed
    t = sectioned_data[:, 1]
    print('is time series monotonic after processing??')
    print(isMonotonic(t))

    # remove non-monotonically increasing time indices if necessary
    cleaned_data = remove_non_monotonic(t, sectioned_data, axis=0)

    # data for pymc
    mutrue = cleaned_data[:, 0]
    times = cleaned_data[:, 1]
    vlps = cleaned_data[:, 2]
    x = cleaned_data[:, 3]

    return mutrue, times, vlps, x, sample_name


def isMonotonic(A):
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))


def remove_non_monotonic(times, data, axis=0):
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
        # print("Keys: %s" % f.keys())
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
                # print(f'{name} is a Dataset')
                # return a np.array using dataset object:
                arr1 = h5obj[:]
                # print(type(arr1))
                # return a np.array using dataset name:
                arr2 = f[name][:]
                # compare arr1 to arr2 (should always return True):
                # print(np.array_equal(arr1, arr2))
                df[f'{name}'] = arr1

    # print('df = ', df)

    return df, names


def downsample_dataset(mu, t, vlps, x):
    # low pass filter
    mu_f = savgol_filter(mu, 50, 2, mode='mirror')
    # print(f'mu_f.shape = {mu_f.shape}')

    # stack time and mu arrays to sample together
    f_data = np.column_stack((mu_f, t, vlps, x))
    # print(f't_muf.shape = {f_data.shape}')

    # downsamples to every qth sample after applying low-pass filter along columns
    q = 11
    f_ds = sp.signal.decimate(f_data, q, ftype='fir', axis=0)
    # print(f'number samples in downsampled series = {f_ds.shape}')
    t_ds = f_ds[:, 1]
    mu_ds = f_ds[:, 0]
    x_ds = f_ds[:, 3]

    # plot series as sanity check
    # plt.plot(x, mu, '.-', label='original data')
    # plt.plot(x, mu_f, '.-', label='filtered data')
    # plt.plot(x_ds, mu_ds, '.-', label='downsampled data')
    # plt.xlabel('disp (mm)')
    # plt.ylabel('mu')
    # plt.legend()
    # # plt.show()

    return f_ds, mu_f


# section_data(...) slices friction data into model-able sections
def section_data(data):
    df0 = pd.DataFrame(data)
    # print(f'dataframe col names = {list(df0)}')
    df = df0.set_axis(['mu', 't', 'vlps', 'x'], axis=1)
    # print(f'new dataframe col names = {list(df)}')

    start_idx = np.argmax(df['x'] > 18 / um_to_mm)
    end_idx = np.argmax(df['x'] > 20 / um_to_mm)

    df_section = df.iloc[start_idx:end_idx]

    # print(f'original shape = {df.shape}')
    # print(f'section shape = {df_section.shape}')

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
    # mutrue = mu + (1 / 100) * np.random.normal(np.mean(mu), 0.1, (len(mu),))
    #
    # # change model results to noisy result, so I can still use the plots easily
    # model.results.friction = mutrue
    #
    # thetatrue = theta

    # plt.figure(100)
    # plot.dispPlot(model)
    #
    # plt.figure(101)
    # plot.timePlot(model)
    # 
    # plt.figure(102)
    # plt.hist(mutrue_mincon)
    # plt.show()

    # return mutrue, size


# MCMC MODEL SETUP FUNCTIONS
# constants used in rsf model
def get_constants(vlps):
    k = 0.0015
    vref = vlps[0]

    return k, vref


def lognormal_mode_to_parameters(desired_modes):
    sigmas = []
    mus = []
    for desired_mode in desired_modes:
        sigma = np.sqrt(np.log(1 + (desired_mode ** 2)))
        mu = np.log(desired_mode) - (sigma ** 2) / 2
        sigmas.append(sigma)
        mus.append(mu)
    return mus, sigmas


# MCMC priors
def get_priors(vref, times):
    desired_modes = (6, 6, 3.2, 0.5)
    mus, sigmas = lognormal_mode_to_parameters(desired_modes)

    # keep mus, overwrite sigmas to make priors wider
    sigmas = [4, 4, 2, 1]

    a = pm.LogNormal('a', mu=mus[0], sigma=sigmas[0])
    b = pm.LogNormal('b', mu=mus[1], sigma=sigmas[1])
    Dc_nd = pm.LogNormal('Dc_nd', mu=mus[2], sigma=sigmas[2])
    mu0 = pm.LogNormal('mu0', mu=mus[3], sigma=sigmas[3])

    # #print(a)
    # vpriors = pm.draw([a, b, Dc_nd, mu0], draws=100)
    # sns.kdeplot(vpriors[0], color='b', label='A prior', common_norm=False, bw_method=0.1)
    # #sns.kdeplot(vpriors, common_norm=False, bw_method=0.1)
    # sns.kdeplot(vpriors[1], color='r', label='B prior')
    # plt.legend()
    # # plt.show()

    # priors = [a, b, Dc_nd, mu0]

    return a, b, Dc_nd, mu0, mus, sigmas


# forward RSF model - from Leeman (2016) and uses the RSF toolkit from GitHub. rsf.py; state_relations.py; plot.py
# returns simulated mu value for use in pymc
def mcmc_rsf_sim(theta, t, v, k, vref):
    # unpack parameters
    a, b, Dc_nd, mu0, sigma = theta

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
    state1.Dc = Dc_nd  # Critical slip distance

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t
    lp_velocity = v

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity

    # Run the model!
    model.solve()

    mu_sim = model.results.friction

    return mu_sim


def nondimensionalize_parameters(vlps, vref, times):
    time_total = times[-1] - times[0]
    times_nd = times / times[0]
    # Dc_nd = pm.Deterministic('Dc_nd', Dc / (time_total * vref))
    vlps_nd = vlps / vref
    vref_nd = vref / np.mean(vlps)

    test = np.argwhere(vlps < 0)


    return times_nd, vlps_nd, vref_nd




# LogLikelihood
def log_likelihood(theta, times, vlps, k, vref, data):
    if type(theta) == list:
        theta = theta[0]
    (
        a,
        b,
        Dc,
        mu0,
        sigma,
    ) = theta

    y_pred = mcmc_rsf_sim(theta, times, vlps, k, vref)
    resids = (data - y_pred)
    # print('resid = ', resids)
    logp = -1 / 2 * np.sum(resids ** 2)
    # print(f'logp = {logp}')

    return logp


# wrapper classes to theano-ize log likelihood
class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, times, vlps, k, vref, data):
        self.data = data
        self.times = times
        self.vlps = vlps
        self.k = k
        self.vref = vref

    def perform(self, node, inputs, outputs):
        logp = log_likelihood(inputs, self.times, self.vlps, self.k, self.vref, self.data)
        outputs[0][0] = np.array(logp)


# MAIN - CALLS ALL FUNCTIONS AND IMPLEMENTS MCMC MODEL RUN
def main():
    print('MCMC RATE AND STATE FRICTION MODEL')
    # so I can figure out how long it's taking when I inevitably forget to check
    comptime_start = get_time('start')

    # observed data
    mutrue, times, vlps, x, sample_name = get_obs_data()

    k, vref = get_constants(vlps)
    print(f'k = {k}; vref = {vref}')
    sigma = 0.001  # standard deviation of measurements - change to actual eventually

    # use PyMC to sampler from log-likelihood
    with pm.Model() as mcmcmodel:
        # priors on stochastic parameters, constants
        a, b, Dc_nd, mu0, prior_mus, prior_sigmas = get_priors(vref, times)
        # a, b, Dc_nd, mu0 = priors

        times_nd, vlps_nd, vref_nd = nondimensionalize_parameters(vlps, vref, times)

        # create loglikelihood Op (wrapper for numerical solution to work with pymc)
        loglike = Loglike(times_nd, vlps_nd, k, vref_nd, mutrue)

        # convert parameters to be estimated to tensor vector
        theta = pt.tensor.as_tensor_variable([a, b, Dc_nd, mu0, sigma])

        # use a Potential for likelihood function
        pm.Potential("likelihood", loglike(theta))

        # seq. mcmc sampler parameterss
        tune = 10000
        draws = 50006
        chains = 2
        cores = 4

        print(f'num draws = {draws}; num chains = {chains}')
        print('starting sampler')
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=pm.Metropolis(), discard_tuned_samples=False)
        print(f'inference data = {idata}')

        # create storage directory
        get_sim_name(draws, chains)
        get_storage_folder(sim_name)

        # save model parameter stats
        vsummary = save_stats(idata, dirpath)

        # save the trace
        save_trace(idata)

        # sample the posterior for validation (can only use if using gradient-based solver)
        # sample_posterior_predcheck(idata)

        # print and save new idata stats that includes posterior predictive check
        # summary_pp = save_stats(idata, dirpath)
        # print(f'idata summary: {summary_pp}')

        # post-processing takes results and makes plots, save figs saves figures
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
                     sample_name=sample_name,
                     times=times)

    plt.show()

    print('simulation complete')


if __name__ == '__main__':
    main()
