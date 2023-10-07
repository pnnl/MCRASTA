import math
import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from rsfmodel import staterelations, rsf, plot
import mcmc_rsf
import pytensor
import sys
import h5py
import scipy as sp
from scipy.signal import savgol_filter
from scipy.stats import lognorm, mode, skew, kurtosis
import seaborn as sns


home = os.path.expanduser('~')
nr = 100004
dirname = f'out_{nr}d2ch'
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', 'mcmc_out', dirname)
idataname = f'{dirname}_idata'

um_to_mm = 0.001


def get_storage_folder(dirname):
    print('checking if storage directory exists')
    homefolder = os.path.expanduser('~')
    outfolder = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'postprocess_out')
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


def load_inference_data(dirpath, name):
    fullname = os.path.join(dirpath, name)
    trace = az.from_netcdf(fullname)

    return trace


def save_trace(idata, dirpath, idataname):
    idata.to_netcdf(os.path.join(dirpath, f'{idataname}_pp'))


def sample_posterior_predcheck(idata):
    print('sampling posterior predictive')
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)


def plot_posterior_predictive(idata):
    az.plot_ppc(idata)


def plot_trace(idata):
    ax = az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'], combined=False)
    ax[0][0].set_xlim(0, 0.08)
    ax[1][0].set_xlim(0, 0.12)
    ax[2][0].set_xlim(0, 50)

    ax2 = az.plot_posterior(idata, var_names=['a', 'b', 'Dc', 'mu0'], point_estimate='mode')
    print(ax2)
    ax2[0].set_xlim(0, 0.08)
    ax2[1].set_xlim(0, 0.12)
    ax2[2].set_xlim(0, 50)


def plot_pairs(idata):
    # plot_kwargs = {'linewidths': 0.2}
    marginal_kwargs = {'color': 'teal'}
    # kde_kwargs = {'hdi_probs': [0.95]}
    ax = az.plot_pair(
        idata,
        var_names=['a', 'b', 'Dc', 'mu0'],
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs={'color': 'teal', 'alpha': 0.2},
        # kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs
    )
    #xlims = [50, 50, 25, 1]

    #1plt.gca().set_xlim(0, xmax)
    # ax[0][0].set_xlim(0, 100)  # set the x limits for the first row first col e.g upper left
    print('pairs ax = ', ax)

    # sys.exit()
    ax[0][0].set_xlim(0, 0.08)
    ax[1][0].set_xlim(0, 0.12)
    ax[2][0].set_ylim(0, 50)
    ax[1][1].set_xlim(0, 0.12)
    ax[2][2].set_xlim(0, 50)


def get_model_vals(idata):
    modelvals = az.extract(idata.posterior, combined=False)

    return modelvals


def get_trace_variables_allchains(modelvals):
    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    return a, b, Dc, mu0


def get_trace_variables(modelvals, chain):
    a = modelvals.a.values[chain, :]
    b = modelvals.b.values[chain, :]
    Dc = modelvals.Dc.values[chain, :]
    mu0 = modelvals.mu0.values[chain, :]

    return a, b, Dc, mu0


def get_constants(vlps):
    k = 0.0015
    vref = vlps[0]

    return k, vref


def redimensionalize_Dc_nd(Dc_nd, times, vref):
    time_total = times[-1] - times[0]
    Dc = Dc_nd * (time_total * vref)

    return Dc


def generate_rsf_data(nr, vars, mutrue_nd):
    a, b, Dc, mu0 = vars

    print(mu0)

    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)

    nobs = len(times)

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

    times_nd = times * k * vref

    model.tc = 1 / (k * vref)   # nondimensionalizing parameter to multiply things by.

    model.time = times_nd

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    logps = []
    # need to iterate over nr rows, that's it
    print('this takes a long time for large number of realizations')
    for i in np.arange(nr):
        print(f'solving for realization')
        # Set model initial conditions
        model.mu0 = mu0[i]  # Friction initial (at the reference velocity)
        # print('model mu0 = ', model.mu0)
        model.a = a[i]  # Empirical coefficient for the direct effect
        state1.b = b[i]  # Empirical coefficient for the evolution effect
        state1.Dc = Dc[i]  # Critical slip distance

        # Run the model!
        model.solve()

        mu_sim = model.results.friction
        state_sim = model.results.states
        print(f'state sim = {state_sim}')

        resids = mutrue_nd - mu_sim
        logp = -1/2 * np.sum(resids ** 2)
        logps.append(logp)

        mu_sims[:, i] = mu_sim

        if logp == np.max(logps):
            map_vars = a[i], b[i], Dc[i], mu0[i]
            map_mu_sim = mu_sim

    return mu_sims, logps, map_vars, map_mu_sim


def plot_simulated_mus(x, times, mu_sims, mutrue, nr, chain):
    plt.figure(chain)
    plt.plot(x * um_to_mm, mu_sims, alpha=0.1)
    plt.plot(x * um_to_mm, mutrue, '.', label='observed')
    # plt.plot(x * um_to_mm, mu_95, 'r-', label='95% cred. interval')
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title(f'Simulated mu values, {nr} realizations')
    plt.show()
    sys.exit()


def load_section_data(dirpath):
    section_data = pd.read_csv(os.path.join(dirpath, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def get_credible_int_bounds(idata, chain):
    hdi_data = az.hdi(idata, hdi_prob=0.95, coords={'chain': [chain]})

    aci = hdi_data.a.data / 1000
    bci = hdi_data.b.data / 1000
    Dc_ndci = hdi_data.Dc_nd.data / 1000
    mu0ci = hdi_data.mu0.data

    return [aci, bci, Dc_ndci, mu0ci]


def get_credible_intervals(a, b, Dc, mu0, aci, bci, Dcci, mu0ci):
    # a.sort()
    # b.sort()
    # Dc.sort()
    # mu0.sort()

    a95 = a[np.where((a >= aci[0]) & (a <= aci[1]))]
    b95 = b[np.where((b >= bci[0]) & (b <= bci[1]))]
    Dc95 = Dc[np.where((Dc >= Dcci[0]) & (Dc <= Dcci[1]))]
    mu095 = mu0[np.where((mu0 >= mu0ci[0]) & (mu0 <= mu0ci[1]))]

    return a95, b95, Dc95, mu095


def original_trace_all_chains(modelvals, times, vref):
    a, b, Dc, mu0 = get_trace_variables_allchains(modelvals)

    plt.plot(a)
    plt.show()
    # Dc = redimensionalize_Dc_nd(Dc_nd, times, vref)
    # datadict = {'a': a, 'b': b, 'Dc': Dc_nd, 'mu0': mu0}
    datadict = {'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    new_idata = az.convert_to_inference_data(datadict)

    plot_pairs(new_idata)
    plot_trace(new_idata)



def save_figs(out_folder):
    # check if folder exists, make one if it doesn't
    name = out_folder
    print(f'find figures and .out file here: {name}')
    w = plt.get_fignums()
    print('w = ', w)
    for i in plt.get_fignums():
        print('i = ', i)
        plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300)


# Create a custom lognormal mode function to estimate the parameters needed for lognormal distr
def lognormal_mode_to_parameters(desired_modes):
    sigmas = []
    mus = []
    for desired_mode in desired_modes:
        sigma = np.sqrt(np.log(1 + (desired_mode ** 2)))
        mu = np.log(desired_mode) - (sigma ** 2) / 2
        sigmas.append(sigma)
        mus.append(mu)
    return mus, sigmas


def plot_priors_posteriors(*posts):
    # get info for priors
    # desired_modes = (8, 4, 5.2, 0.3)
    # mus, sigmas = lognormal_mode_to_parameters(desired_modes)

    # define priors same as in mcmc_rsf.py - get this info from out file
    mus = [0, 0, 0, 1.5]
    sigmas = [0.7, 0.7, 1, 0.2]

    a = pm.LogNormal('a', mu=mus[0], sigma=sigmas[0])
    b = pm.LogNormal('b', mu=mus[1], sigma=sigmas[1])
    Dc_nd = pm.LogNormal('Dc_nd', mu=mus[2], sigma=sigmas[2])
    mu0 = pm.LogNormal('mu0', mu=mus[3], sigma=sigmas[3])

    # take same number of draws as in mcmc_rsf.py
    vpriors = pm.draw([a, b, Dc_nd, mu0], draws=60000)

    # Dc_redim = redimensionalize_Dc_nd(vpriors[2], times, vref)
    # vpriors_scaled = (vpriors[0]/1000, vpriors[1]/1000, Dc_redim/1000, vpriors[3]/100)

    # plot priors with posteriors
    xlims = [5, 5, 400, 10]

    for i, (prior, post, label, xmax) in enumerate(zip(vpriors, posts, ('a', 'b', 'dc', 'mu0'), xlims)):
        plt.figure(10+i)
        # sns.histplot(prior, kde=True)
        # sns.kdeplot(prior, color='b', label=f'{label} prior', common_norm=False, bw_method=0.1)
        sns.kdeplot(post, color='g', label=f'{label} post', common_norm=False)
        # plt.gca().set_xlim(0, xmax)
        plt.legend()


def plot_priors(a, b, Dc, mu0, mus, sigmas, desired_modes):
    datas = a, b, Dc, mu0
    dataname = ['a', 'b', 'Dc', 'mu0']
    num_bins = 100

    i = 10
    for data, name in zip(datas[0:2], dataname[0:2]):
        plt.figure(i+1)
        kde = sns.kdeplot(data, color='b', label=f'{name} prior', common_norm=False, bw_method=0.1)
        plt.xlim(-0.02, 0.05)
        y_max = kde.get_lines()[0].get_ydata().max()
        kde.get_lines()[0].set_ydata(kde.get_lines()[0].get_ydata() / y_max)
        i += 1
        plt.legend()

    i = 20
    for data, name in zip(datas[2:], dataname[2:]):
        # hist, bins = np.histogram(data, bins=num_bins, density=True)
        plt.figure(i+1)
        kde = sns.kdeplot(data, color='b', label=f'{name} prior', common_norm=False, bw_method=0.1)
        y_max = kde.get_lines()[0].get_ydata().max()
        kde.get_lines()[0].set_ydata(kde.get_lines()[0].get_ydata() / y_max)
        # plt.ylim(0, 1)
        plt.legend()
        # plt.hist(data, bins=num_bins, density=True, alpha=0.6, color='blue')
        # plt.plot(bins, pdf_vals, 'r', label=f'{name} prior PDF')
        i += 1


def get_posteriors(modelvals, chain):
    a, b, Dc, mu0 = get_trace_variables(modelvals, chain)

    return a, b, Dc, mu0


def plot_posteriors(a, b, Dc, mu0):
    datas = a, b, Dc, mu0
    dataname = ['a', 'b', 'Dc', 'mu0']

    i = 10
    for data, name in zip(datas[0:2], dataname[0:2]):
        plt.figure(i+1)
        kde = sns.kdeplot(data, color='g', label=f'{name} posterior', common_norm=False, bw_method=0.1)
        y_max = kde.get_lines()[0].get_ydata().max()
        kde.get_lines()[0].set_ydata(kde.get_lines()[0].get_ydata() / y_max)
        plt.xlim(-0.02, 0.05)
        i += 1
        plt.legend()

    i = 20
    for data, name in zip(datas[2:], dataname[2:]):
        plt.figure(i+1)
        kde = sns.kdeplot(data, color='g', label=f'{name} posterior', common_norm=False, bw_method=0.1)
        y_max = kde.get_lines()[0].get_ydata().max()
        kde.get_lines()[0].set_ydata(kde.get_lines()[0].get_ydata() / y_max)
        # plt.ylim(0, 1)
        # hist, bins = np.histogram(data, bins=num_bins, density=True)
        # pdf_vals = lognorm.pdf(bins, s=sigma, scale=np.exp(mu / 1000))
        # plt.hist(data, bins=num_bins, density=True, alpha=0.6, color='blue')
        # plt.plot(bins, pdf_vals, 'r', label=f'{name} posterior PDF')
        i += 1
        plt.legend()
    # plt.show()


def get_modes(modelvals, chain):
    a, b, Dc_nd, mu0 = get_trace_variables(modelvals, chain)

    amode = mode(a, axis=None, keepdims=False)
    bmode = mode(b, axis=None, keepdims=False)
    Dcndmode = mode(Dc_nd, axis=None, keepdims=False)
    mu0mode = mode(mu0, axis=None, keepdims=False)

    return amode, bmode, Dcndmode, mu0mode


def redimensionalize_parameters(a, b, Dc, mu0, tc):
    ard = a / tc
    brd = b / tc
    # Dcrd = Dc / tc
    Dcrd = Dc
    mu0rd = mu0 / tc

    return ard, brd, Dcrd, mu0rd


def calc_logp(mutrue, mu_sims, nr):
    # nr = 1000
    logps = []
    for real in np.arange(nr):
        resids = (mutrue - mu_sims[real, :])
        logp = -1 / 2 * np.sum(resids ** 2)
        logps.append(logp)

    plt.plot(logps)
    plt.show()
    return logps


def main():
    out_folder = get_storage_folder(dirname)
    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)

    idata = load_inference_data(dirpath, idataname)
    # az.plot_trace(idata, var_names=['a', 'b', 'Dc_nd', 'mu0'])

    modelvals = get_model_vals(idata)

    # plots trace pairs and samples
    original_trace_all_chains(modelvals, times, vref)

    numchains = 2
    for chain in np.arange(numchains):
        # get posteriors and plot them
        # constant for redimensionalizing variables
        tc = 1 / (k * vref)

        # get posteriors from model trace
        apost, bpost, Dcpost, mu0post = get_posteriors(modelvals, chain)

        # redimensionalize parameters
        # ard, brd, Dcrd, mu0rd = redimensionalize_parameters(apost, bpost, Dcpost, mu0post, tc)

        # plot the dimensionalized priors and posteriors for comparison when necessary
        # plot_priors_posteriors(ard, brd, Dcrd, mu0rd)

        vars_all = apost, bpost, Dcpost, mu0post
        # vars_all = ard, brd, Dcrd, mu0rd
        #
        mutrue_nd = mutrue * tc

        mu_sims, logps, map_vars, map_mu_sim = generate_rsf_data(nr, vars_all, mutrue_nd)

        plt.figure(70)
        plt.plot(times, mutrue_nd, '.', alpha=0.2, label='observed')
        plt.plot(times, map_mu_sim, label='max logp solution')

        # plot_simulated_mus(x, times, mu_sims, mutrue_nd, len(apost), chain)

    save_figs(out_folder)


if __name__ == '__main__':
    main()
