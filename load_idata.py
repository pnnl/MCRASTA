import math
import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
from rsfmodel import staterelations, rsf, plot
import sys
from scipy.stats import lognorm, mode, skew, kurtosis
from scipy import signal
import seaborn as sns
import globals

myglobals = globals.Globals()

# az.style.use("arviz-darkgrid")

home = os.path.expanduser('~')
nr = myglobals.ndr
nch = myglobals.nch
samplename = 'p5760'
section = '003'
sampleid = f'5760{section}'
dirname = f'out_{nr}d{nch}ch_{sampleid}'
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', samplename, dirname)
idataname = f'{dirname}_idata'

# nrstep = interval between processed samples to avoid correlated samples (and/or to just work with less data/make it more interpretable)
nrstep = 500
# nrplot = number of total realizations we'll look at
nrplot = 1000

um_to_mm = 0.001


def get_storage_folder(dirname):
    print('checking if storage directory exists')
    homefolder = os.path.expanduser('~')
    outfolder = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'postprocess_out', samplename)
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

    ax2 = az.plot_posterior(idata, var_names=['a', 'b', 'Dc', 'mu0'], point_estimate='mode', round_to=3)
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
        plt.hist(bins[:-1], bins, density=True, stacked=True, weights=counts, alpha=0.5, color=c, label=f'chain {chain}, mode={round(m, 4)}')
        plt.title(f'probability density, {name}')
        plt.legend(fontsize='small')
        i += 1


def plot_pairs(idata, chain=None):
    # plot_kwargs = {'linewidths': 0.2}
    marginal_kwargs = {'color': 'teal'}
    kde_kwargs = {'hdi_probs': [0.89, 0.94]}
    ax = az.plot_pair(
        idata,
        var_names=['a_min_b', 'Dc', 'mu0'],
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs={'color': 'teal', 'alpha': 0.6},
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
    )

    #1plt.gca().set_xlim(0, xmax)
    # ax[0][0].set_xlim(0, 100)  # set the x limits for the first row first col e.g upper left
    # print('pairs ax = ', ax)
    #
    # # sys.exit()
    # ax[0][0].set_xlim(0, 0.08)
    # ax[1][0].set_xlim(0, 0.12)
    # ax[2][0].set_ylim(0, 100)
    # ax[1][1].set_xlim(0, 0.12)
    # ax[2][2].set_xlim(0, 100)


def get_model_vals(idata):
    modelvals = az.extract(idata.posterior, combined=False)

    return modelvals


def get_trace_variables_allchains(modelvals):
    a_min_b = modelvals.a_min_b.values[:, 0::nrstep]
    a = modelvals.a.values[:, 0::nrstep]
    b = modelvals.b.values[:, 0::nrstep]
    Dc = modelvals.Dc.values[:, 0::nrstep]
    mu0 = modelvals.mu0.values[:, 0::nrstep]

    return a_min_b, a, b, Dc, mu0


def get_thinned_idata(modelvals):
    a = modelvals.a.values[:, 0::nrstep]
    b = modelvals.b.values[:, 0::nrstep]
    Dc = modelvals.Dc.values[:, 0::nrstep]
    mu0 = modelvals.mu0.values[:, 0::nrstep]

    datadict = {'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    new_idata = az.convert_to_inference_data(datadict)

    return new_idata


def get_trace_variables(modelvals, chain):
    a = modelvals.a.values[chain, 0::nrstep]
    b = modelvals.b.values[chain, 0::nrstep]
    Dc = modelvals.Dc.values[chain, 0::nrstep]
    mu0 = modelvals.mu0.values[chain, 0::nrstep]

    return a, b, Dc, mu0


def get_constants(vlps):
    k = myglobals.k
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

    # dimensional variables output from mcmc_rsf.py
    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)
    lc, vmax = get_vmax_l0(vlps)

    k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

    nobs = len(t0)

    # set up rsf model
    model = rsf.Model()
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.vmax = vmax
    state1.lc = myglobals.lc

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t0

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    # pre-allocate array
    mu_sims = np.ones((nobs, nrplot))
    print(f'mu_sims.shape = {mu_sims.shape}')

    logps = []
    # need to iterate over nr rows, that's it
    print('this takes a long time for large number of realizations')
    print(f'only plotting every {nrstep}th realization')
    j = 0
    for i in np.arange(nrplot):
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
    l0 = myglobals.lc
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
    a_min_b, a, b, Dc, mu0 = get_trace_variables_allchains(modelvals)

    datadict = {'a_min_b': a_min_b, 'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    new_idata = az.convert_to_inference_data(datadict)

    plot_pairs(new_idata, chain=None)
    # plot_trace(new_idata, chain=None)


def plot_chain_trace(modelvals, chain):
    a, b, Dc, mu0 = get_trace_variables(modelvals, chain)
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
    mus, sigmas = myglobals.get_prior_parameters()

    a = pm.LogNormal.dist(mu=mus[0], sigma=sigmas[0])
    b = pm.LogNormal.dist(mu=mus[1], sigma=sigmas[1])
    Dc_nd = pm.LogNormal.dist(mu=mus[2], sigma=sigmas[2])
    mu0 = pm.LogNormal.dist(mu=mus[3], sigma=sigmas[3])

    # take same number of draws as in mcmc_rsf.py
    vpriors = pm.draw([a, b, Dc_nd, mu0], draws=500000)

    xmaxs = [0.05, 0.05, 60, 1.25]

    for i, (prior, post, label, xmax) in enumerate(zip(vpriors, posts, ('a', 'b', 'dc', 'mu0'), xmaxs)):
        plt.figure(10+i)
        # sns.histplot(prior, kde=True)
        line1 = sns.kdeplot(prior, color='b', common_norm=False, bw_method=0.1)
        line2 = sns.kdeplot(post, color='g', common_norm=False)
        plt.gca().set_xlim(0, xmax)
        # plt.gca().set_legend([line1, line2], ['priors', 'posteriors'])


def get_posteriors(modelvals, chain):
    a, b, Dc, mu0 = get_trace_variables(modelvals, chain)

    return a, b, Dc, mu0


def get_modes(modelvals, chain):
    a, b, Dc_nd, mu0 = get_trace_variables(modelvals, chain)

    amode = az.plots.plot_utils.calculate_point_estimate('mode', a)
    bmode = az.plots.plot_utils.calculate_point_estimate('mode', b,)
    Dcndmode = az.plots.plot_utils.calculate_point_estimate('mode', Dc_nd)
    mu0mode = az.plots.plot_utils.calculate_point_estimate('mode', mu0)

    return amode, bmode, Dcndmode, mu0mode


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    k0 = myglobals.k * myglobals.lc
    vlps0 = vlps / vmax
    vref0 = vref / vmax
    t0 = times * vmax / myglobals.lc
    t0 = t0 - t0[0]

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


def autocorr(x, step):
    corr = signal.correlate(x, x, mode='full')
    corr = corr[np.argmax(corr):]
    corr /= np.max(corr)
    laglen = len(corr)
    lags = np.arange(laglen)

    return lags, corr


def plot_model_autocorrelations(warmupvals, modelvals):
    aw, bw, Dcw, mu0w = warmupvals

    steptry = 150

    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    afull = np.concatenate((aw, a), axis=1)
    bfull = np.concatenate((bw, b), axis=1)
    Dcfull = np.concatenate((Dcw, Dc), axis=1)
    mu0full = np.concatenate((mu0w, mu0), axis=1)

    parameters = afull[:, 0::steptry], bfull[:, 0::steptry], Dcfull[:, 0::steptry], mu0full[:, 0::steptry]
    names = ['a', 'b', 'Dc', 'mu0']

    i = 500
    for j, (p, name) in enumerate(zip(parameters, names)):
        lags, corr = autocorr(p[j, :], steptry)
        plt.figure(i)
        plt.plot(lags, corr, '.')
        # plt.ylim(-0.5, 1)
        plt.xlabel('lag')
        plt.title(f'autocorrelated posterior draws: {name}')
        i += 1
    plt.show()
    sys.exit()


def get_warmup_vals(idata):
    warmupvals = az.extract(idata.warmup_posterior, combined=False)

    aw = warmupvals.a.values
    bw = warmupvals.b.values
    Dcw = warmupvals.Dc.values
    mu0w = warmupvals.mu0.values

    return aw, bw, Dcw, mu0w


def plot_a_minus_b(idata, vlps, vref, nrstep):
    modelvals = az.extract(idata.posterior, combined=True)
    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    a_min_b = a-b
    datadict = {'a_min_b': a_min_b, 'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    ab_idata = az.convert_to_inference_data(datadict, group='posterior')
    hdi_prob = 0.89

    plt.figure(300)
    ax = az.plot_posterior(ab_idata, var_names=['a_min_b'], point_estimate='mode', round_to=4, hdi_prob=hdi_prob)
    ax.set_xlim(-0.05, 0.05)
    ax.set_title(f'(a-b) posterior distribution, {samplename}')
    mab = az.plots.plot_utils.calculate_point_estimate('mode', a_min_b)

    plt.figure(301)
    ax1 = az.plot_posterior(idata, var_names=['a'], point_estimate='mode', round_to=4, hdi_prob=hdi_prob)
    ax1.set_title(f'a posterior distribution, {samplename}')
    ax1.set_xlim(0, 0.04)

    plt.figure(302)
    ax2 = az.plot_posterior(idata, var_names=['b'], point_estimate='mode', round_to=4, hdi_prob=hdi_prob)
    ax2.set_title(f'b posterior distribution, {samplename}')
    ax2.set_xlim(0, 0.05)

    plt.figure(303)
    ax3 = az.plot_posterior(idata, var_names=['Dc'], point_estimate='mode', round_to=4, hdi_prob=hdi_prob)
    ax3.set_title(f'Dc posterior distribution, {samplename}')
    ax3.set_xlim(0, 60)

    plt.figure(304)
    ax4 = az.plot_posterior(idata, var_names=['mu0'], point_estimate='mode', round_to=4, hdi_prob=hdi_prob)
    ax4.set_title(f'mu0 posterior distribution, {samplename}')

    marginal_kwargs = {'color': 'purple'}
    kde_kwargs = {'hdi_probs': [0.89, 0.94]}
    ax = az.plot_pair(
        ab_idata,
        var_names=['a_min_b', 'Dc', 'mu0'],
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs={'color': 'purple', 'alpha': 0.6},
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
    )

    # plt.figure(300)
    # plt.plot(a_min_b, '.', alpha=0.2)
    #
    # colors = ['k--', 'g--']
    # for prob, color in zip(hdi_probs, colors):
    #     hdi = az.hdi(ab_idata, hdi_prob=prob)
    #     y0 = hdi.a_min_b.data[0]*np.ones_like(x)
    #     y1 = hdi.a_min_b.data[1]*np.ones_like(x)
    #     y2 = mab*np.ones_like(x)
    #     plt.plot(x, y0, color, label=f'ci={prob*100}%, [{hdi.a_min_b.data[0]}, {hdi.a_min_b.data[1]}]')
    #     plt.plot(x, y1, color)
    #     plt.plot(x, y2, 'r', label=f'mode={mab}')
    #     plt.title('(a-b) ')
    #     plt.legend()
    # plt.show()
    # sys.exit()

    # plt.show()

    return ab_idata


def plot_hdi_mu_sims(mu_sims):
    az.plot_hdi(mu_sims, mu_sims)


def main():
    out_folder = get_storage_folder(dirname)
    times, mutrue, vlps, x = load_section_data(dirpath)
    k, vref = get_constants(vlps)

    idata = load_inference_data(dirpath, idataname)

    ab_idata = plot_a_minus_b(idata, vlps, vref, nrstep)

    # warmup_posterior_vals = get_warmup_vals(idata)
    # aw, bw, Dcw, mu0w = warmup_posterior_vals
    modelvals = get_model_vals(idata)
    modelvals_ab = get_model_vals(ab_idata)

    # plot_model_autocorrelations(warmup_posterior_vals, modelvals)
    # sys.exit()

    # plots pairs for a-b, Dc, mu0
    original_trace_all_chains(modelvals_ab, times, vref)

    # plots original trace
    thinned_idata = get_thinned_idata(modelvals)
    plot_trace(thinned_idata, chain=None)

    # plot observed data on this figure before chains are plotted to avoid plotting it 4 times
    # fig, ax = plt.subplots(num=70)
    # p1, = ax.plot(times, mutrue, '.', alpha=0.2, label='observed')
    plt.figure(70)
    xax = x
    plt.plot(xax, mutrue, '.', alpha=0.2, label='observed')
    numchains = nch

    for chain in np.arange(numchains):
        # get posteriors and plot them
        # get posteriors from model trace
        apost, bpost, Dcpost, mu0post = get_posteriors(modelvals, chain)
        modes = get_modes(modelvals, chain)
        amode, bmode, Dcmode, mu0mode = modes

        # plot the priors and posteriors for comparison when necessary
        plot_priors_posteriors(apost, bpost, Dcpost, mu0post)
        plot_posterior_distributions(modelvals, chain, modes)

        vars_all = apost, bpost, Dcpost, mu0post
        names = ['a', 'b', 'Dc', 'mu0']

        # this is to double-check sampler is sampling appropriate range
        for modelvar, name in zip(vars_all, names):
            print(f'model parameter = {name}')
            mi = np.min(modelvar)
            mx = np.max(modelvar)
            print(f'min = {mi}')
            print(f'max = {mx}')

        # this generates the rsf data using parameter draws, calc logp vals, and plots the best fit with observed
        # data for each chain
        plot_flag = 'yes'
        if plot_flag == 'yes':
            # variables are nondimensionalized in this function for comparison to observed data
            mu_sims, logps, map_vars, map_mu_sim, maxlogp = generate_rsf_data(nr, vars_all) # generates rsf sim

            # plot_hdi_mu_sims(mu_sims)

            ahat, bhat, Dchat, mu0hat = map_vars    # parameter vals which resulted in highest logp val

            aminb_hat = ahat - bhat

            # plot all the stuff
            plt.figure(70)
            plt.plot(xax, map_mu_sim, label=f'chain {chain}; max logp = {round(maxlogp, 4)} \n '
                                                          f'a-b={round(aminb_hat, 4)} \n '
                                                          f'a={round(ahat, 4)}; '
                                                          f'b={round(bhat, 4)}; '
                                                          f'Dc={round(Dchat, 2)}; '
                                                          f'mu0={round(mu0hat, 3)}')
            plt.legend()

            plt.figure(72)
            plt.plot(x * um_to_mm, vlps, 'r--', label='velocity (um/s)')
            plt.xlabel('loadpoint displacement (mm)')
            # twin3 = ax.twinx()
            # p3, = twin3.plot(x * um_to_mm, vlps * um_to_mm, 'r--', label='velocity (mm/s)')
            # twin3.set(ylabel='velocity (mm/s)')

            # ax.set_title('best-fit solutions')
            # ax.legend(handles=[p1, p2], fontsize='x-small')
            plt.legend()
            plt.show()

            # plot logp vals as sanity check
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
