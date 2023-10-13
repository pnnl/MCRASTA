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
nr = 500000
nch = 4
dirname = f'out_{nr}d{nch}ch'
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


def plot_trace(idata, chain):
    ax = az.plot_trace(idata, var_names=['a', 'b', 'Dc', 'mu0'], combined=False)
    ax[0][0].set_xlim(0, 0.08)
    ax[1][0].set_xlim(0, 0.12)
    ax[2][0].set_xlim(0, 100)

    ax2 = az.plot_posterior(idata, var_names=['a', 'b', 'Dc', 'mu0'], point_estimate='mode')
    print(ax2)
    ax2[0].set_xlim(0, 0.08)
    ax2[1].set_xlim(0, 0.12)
    ax2[2].set_xlim(0, 100)


def plot_posterior_distributions(modelvals, chain, modes):
    tracevals = get_trace_variables(modelvals, chain)
    names = ['a', 'b', 'Dc', 'mu0']
    colors = ['b', 'g', 'k', 'm']

    i=200
    for traceval, name, c, m in zip(tracevals, names, colors, modes):
        counts, bins = np.histogram(traceval, bins='doane', density=True)
        plt.figure(i)
        plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, color=c, label=f'chain {chain}, mode={m}')
        plt.title(f'probability density, {name}')
        plt.legend()
        i += 1


def plot_pairs(idata, chain=None):
    # plot_kwargs = {'linewidths': 0.2}
    marginal_kwargs = {'color': 'teal'}
    kde_kwargs = {'hdi_probs': [0.50, 0.70, 0.90, 0.95]}
    ax = az.plot_pair(
        idata,
        var_names=['a', 'b', 'Dc', 'mu0'],
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs={'color': 'teal', 'alpha': 0.6},
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
    )

    #1plt.gca().set_xlim(0, xmax)
    # ax[0][0].set_xlim(0, 100)  # set the x limits for the first row first col e.g upper left
    print('pairs ax = ', ax)

    # sys.exit()
    ax[0][0].set_xlim(0, 0.08)
    ax[1][0].set_xlim(0, 0.12)
    ax[2][0].set_ylim(0, 100)
    ax[1][1].set_xlim(0, 0.12)
    ax[2][2].set_xlim(0, 100)


def get_model_vals(idata):
    modelvals = az.extract(idata.posterior, combined=False)

    return modelvals


def get_trace_variables_allchains(modelvals):
    a = modelvals.a.values[:, 0::1000]
    b = modelvals.b.values[:, 0::1000]
    Dc = modelvals.Dc.values[:, 0::1000]
    mu0 = modelvals.mu0.values[:, 0::1000]

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


def generate_one_realization(modelvars):
    a, b, Dc, mu0 = modelvars

    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)
    l0, vmax = get_vmax_l0(vlps)

    # k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

    # set up rsf model
    model = rsf.Model()
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.vmax = 1
    state1.l0 = 1

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = times - times[0]

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    print(f'solving for single realization')
    # Set model initial conditions
    model.mu0 = mu0  # Friction initial (at the reference velocity)
    # print('model mu0 = ', model.mu0)
    model.a = a  # Empirical coefficient for the direct effect
    state1.b = b  # Empirical coefficient for the evolution effect
    state1.Dc = Dc  # Critical slip distance

    # Run the model!
    model.solve()

    mu_sim = model.results.friction
    state_sim = model.results.states

    resids = mutrue - mu_sim
    logp = -1/2 * np.sum(resids ** 2)

    return mu_sim, logp


def generate_rsf_data(nr, modelvars):
    a, b, Dc, mu0 = modelvars

    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)
    l0, vmax = get_vmax_l0(vlps)

    k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

    nobs = len(t0)

    # set up rsf model
    model = rsf.Model()
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.vmax = 1
    state1.l0 = 1

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t0

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    # number of realizations we want to look at
    nrplot = 500
    nrstep = 1000

    # pre-allocate array
    mu_sims = np.ones((nobs, nrplot))
    print(f'mu_sims.shape = {mu_sims.shape}')

    logps = []
    # need to iterate over nr rows, that's it
    print('this takes a long time for large number of realizations')
    print(f'only plotting every {nrstep}th realization')
    j = 0
    for i in range(0, nr, nrstep):
        print(f'solving for realization {i}')
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

        resids = mutrue - mu_sim
        logp = -1/2 * np.sum(resids ** 2)
        logps.append(logp)

        mu_sims[:, j] = mu_sim

        if logp == np.nanmax(logps):
            map_vars = a[i], b[i], Dc[i], mu0[i]
            map_mu_sim = mu_sim
            maxlogp = logp

        j += 1

    return mu_sims, logps, map_vars, map_mu_sim, maxlogp


def get_vmax_l0(vlps):
    l0 = 125
    vmax = np.max(vlps)

    return l0, vmax

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


def original_trace_all_chains(modelvals, times, vref):
    a, b, Dc, mu0 = get_trace_variables_allchains(modelvals)

    datadict = {'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    new_idata = az.convert_to_inference_data(datadict)

    plot_pairs(new_idata, chain=None)
    plot_trace(new_idata, chain=None)


def plot_chain_trace(modelvals, chain):
    a, b, Dc, mu0 = get_trace_variables(modelvals, chain)
    # Dc = redimensionalize_Dc_nd(Dc_nd, times, vref)
    # datadict = {'a': a, 'b': b, 'Dc': Dc_nd, 'mu0': mu0}
    datadict = {'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    new_idata = az.convert_to_inference_data(datadict)

    plot_pairs(new_idata, chain)
    plot_trace(new_idata, chain)



def save_figs(out_folder):
    # check if folder exists, make one if it doesn't
    name = out_folder
    print(f'find figures and .out file here: {name}')
    w = plt.get_fignums()
    print('w = ', w)
    for i in plt.get_fignums():
        print('i = ', i)
        plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300)


def plot_priors_posteriors(*posts):
    # define priors same as in mcmc_rsf.py - get this info from out file
    mus = [-4, -4, 4, -1]
    sigmas = [0.5, 0.5, 0.1, 0.2]

    a = pm.LogNormal.dist(mu=mus[0], sigma=sigmas[0])
    b = pm.LogNormal.dist(mu=mus[1], sigma=sigmas[1])
    Dc_nd = pm.LogNormal.dist(mu=mus[2], sigma=sigmas[2])
    mu0 = pm.LogNormal.dist(mu=mus[3], sigma=sigmas[3])

    # take same number of draws as in mcmc_rsf.py
    vpriors = pm.draw([a, b, Dc_nd, mu0], draws=100000)

    for i, (prior, post, label) in enumerate(zip(vpriors, posts, ('a', 'b', 'dc', 'mu0'))):
        plt.figure(10+i)
        # sns.histplot(prior, kde=True)
        sns.kdeplot(prior, color='b', label=f'{label} prior', common_norm=False, bw_method=0.1)
        sns.kdeplot(post, color='g', label=f'{label} post', common_norm=False)
        # plt.gca().set_xlim(0, xmax)
        plt.legend()


def get_posteriors(modelvals, chain):
    a, b, Dc, mu0 = get_trace_variables(modelvals, chain)

    return a, b, Dc, mu0


def get_modes(modelvals, chain):
    a, b, Dc_nd, mu0 = get_trace_variables(modelvals, chain)

    amode = mode(a, axis=None, keepdims=False)
    bmode = mode(b, axis=None, keepdims=False)
    Dcndmode = mode(Dc_nd, axis=None, keepdims=False)
    mu0mode = mode(mu0, axis=None, keepdims=False)

    return amode, bmode, Dcndmode, mu0mode


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    l0, vmax = get_vmax_l0(vlps)

    k0 = k * l0
    vlps0 = vlps / vmax
    vref0 = vref / vmax
    t0 = times - times[0]

    return k0, vlps0, vref0, t0


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


def calc_expected_vals(modelvar):
    n = len(modelvar)
    muhat = np.sum(np.log(modelvar))/n
    sigmahat = np.sqrt((np.sum((modelvar - muhat)**2))/(n-1))

    return muhat, sigmahat


def main():
    out_folder = get_storage_folder(dirname)
    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)

    idata = load_inference_data(dirpath, idataname)
    # az.plot_trace(idata, var_names=['a', 'b', 'Dc_nd', 'mu0'])

    modelvals = get_model_vals(idata)

    # plots trace pairs and samples
    original_trace_all_chains(modelvals, times, vref)

    # plot observed data on this figure before chains are plotted to avoid plotting it 4 times
    plt.figure(70)
    plt.plot(times, mutrue, '.', alpha=0.2, label='observed')

    numchains = nch

    for chain in np.arange(numchains):
        # get posteriors and plot them
        # get posteriors from model trace
        apost, bpost, Dcpost, mu0post = get_posteriors(modelvals, chain)
        modes = get_modes(modelvals, chain)
        amode, bmode, Dcmode, mu0mode = modes

        print(f'MODES calcd by scipy, chain = {chain}')
        print(f'a: {amode[0]}')
        print(f'b: {bmode[0]}')
        print(f'Dc: {Dcmode[0]}')
        print(f'mu0: {mu0mode[0]}')

        # plot the dimensionalized priors and posteriors for comparison when necessary
        plot_priors_posteriors(apost, bpost, Dcpost, mu0post)
        plot_posterior_distributions(modelvals, chain, modes)

        vars_all = apost, bpost, Dcpost, mu0post
        names = ['a', 'b', 'Dc', 'mu0']

        for modelvar, name in zip(vars_all, names):
            print(f'model parameter = {name}')
            mi = np.min(modelvar)
            mx = np.max(modelvar)
            print(f'min = {mi}')
            print(f'max = {mx}')

        modes = [amode[0], bmode[0], Dcmode[0], mu0mode[0]]
        mu_sim, logp = generate_one_realization(modes)
        print(f'logp = {logp}')
        plt.figure(80)
        plt.plot(times, mutrue, '.', alpha=0.2, label='observed')
        plt.plot(times, mu_sim, label=f'chain={chain}; logp={logp}')
        plt.legend()

        plot_flag = 'yes'
        if plot_flag == 'yes':
            mu_sims, logps, map_vars, map_mu_sim, maxlogp = generate_rsf_data(nr, vars_all)

            ahat, bhat, Dchat, mu0hat = map_vars
            plt.figure(70)
            # plt.plot(times, mutrue, '.', alpha=0.2, label='observed')
            plt.plot(times, map_mu_sim, label=f'chain={chain}; max logp = {maxlogp} \n a={ahat}; b={bhat}; '
                                              f'Dc={Dchat}; mu0={mu0hat}')
            plt.xlabel('time (s)')
            plt.ylabel('mu')
            plt.title('best-fit solutions')
            plt.legend(fontsize='small')

            plt.figure(71)
            plt.plot(logps, '.')
            plt.ylabel('logp values')
            plt.xlabel('realization no.')
            plt.title('logp vals')
        elif plot_flag == 'no':
            print('skipping plotting observed data with realizations')



    save_figs(out_folder)


if __name__ == '__main__':
    main()
