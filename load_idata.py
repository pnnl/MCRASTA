import json
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
import arviz.labels as azl
from random import sample
from gplot import gpl

# az.style.use("arviz-darkgrid")

home = os.path.expanduser('~')

samplename = 'p5756'
nr = 500000
nch = 4
section = '001'
sampleid = f'5756{section}'
dirname = f'out_{nr}d{nch}ch_{sampleid}'
# dirname = f'~out_{nr}d{nch}ch'
dirpath = os.path.join(home, 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out', samplename, dirname)
idataname = f'{dirname}_idata'

# nrstep = interval between processed samples to avoid correlated samples (and/or to just work with less data/make it
# more interpretable)
nrstep = 500
# nrplot = number of total realizations we'll look at
nrplot = 1000

um_to_mm = 0.001


def read_from_json(dirpath):
    jpath = os.path.join(dirpath, 'out.json')
    with open(jpath, 'r') as rfile:
        js = json.load(rfile)
        print(js)
        gpl.samplename = samplename
        gpl.mintime = js.get('time_start')
        gpl.maxtime = js.get('time_end')
        gpl.mindisp = js.get('x_start')
        gpl.maxdisp = js.get('x_end')
        gpl.section_id = js.get('section_ID')
        gpl.k = js.get('k')
        gpl.lc = js.get('lc')
        gpl.vel_windowlen = js.get('dvdt_window_len')
        gpl.filter_windowlen = js.get('filter_window_len')
        gpl.q = js.get('q')
        gpl.ndr = js.get('n_draws')
        gpl.nch = js.get('n_chains')
        gpl.ntune = js.get('n_tune')
        vref = js.get('vref')

        priors_info = js.get('prior_mus_sigmas', 'priors info not available')
        mus = priors_info[0]
        sigmas = priors_info[1]

        return mus, sigmas, vref


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
    labeller = azl.MapLabeller(var_name_map={'a_min_b': 'a-b', 'Dc': r'$D_{c}$ ($\mu$m)', 'mu0': r'$\mu_{0}$'})
    ax = az.plot_pair(
        idata,
        var_names=['a_min_b', 'Dc', 'mu0'],
        kind=['scatter', 'kde'],
        marginals=True,
        scatter_kwargs={'color': 'teal', 'alpha': 0.6},
        labeller=labeller,
        point_estimate='mode',
        # kde_kwargs=kde_kwargs,
        # marginal_kwargs=marginal_kwargs,
        textsize=18,
    )

    ax[0][0].set_xlim(-0.02, 0.02)
    ax[1][1].set_xlim(0, 45)


def get_model_vals(idata):
    modelvals = az.extract(idata.posterior, combined=True)

    return modelvals


def get_trace_variables_allchains(modelvals):
    a_min_b = modelvals.a_min_b.values[0::nrstep]
    a = modelvals.a.values[0::nrstep]
    b = modelvals.b.values[0::nrstep]
    Dc = modelvals.Dc.values[0::nrstep]
    mu0 = modelvals.mu0.values[0::nrstep]

    return a_min_b, a, b, Dc, mu0


def get_thinned_idata(modelvals):
    a = modelvals.a.values[0::nrstep]
    b = modelvals.b.values[0::nrstep]
    Dc = modelvals.Dc.values[0::nrstep]
    mu0 = modelvals.mu0.values[0::nrstep]

    datadict = {'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0}
    new_idata = az.convert_to_inference_data(datadict)

    return new_idata


def get_trace_variables(modelvals, chain):
    a = modelvals.a.values[0::nrstep]
    b = modelvals.b.values[0::nrstep]
    Dc = modelvals.Dc.values[0::nrstep]
    mu0 = modelvals.mu0.values[0::nrstep]

    return a, b, Dc, mu0


def get_constants(vlps):
    k = 0.00194
    vref = vlps[0]

    return k, vref


def generate_rsf_data(nrplot, modelvars):
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
    state1.lc = gpl.lc

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
    l0 = gpl.lc
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
        plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300, bbox_inches='tight')


def plot_priors_posteriors(*posts):
    # define priors same as in mcmc_rsf.py - get this info from out file
    mus, sigmas = gpl.get_prior_parameters()

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
    k0 = gpl.k * gpl.lc
    vlps0 = vlps / vmax
    vref0 = vref / vmax
    t0 = times * vmax / gpl.lc
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


def save_data(mu_sims, logps, map_vars, map_mu_sims, maxlogps, out_folder):
    mu_sims = np.array(mu_sims)
    logps = np.array(logps)
    map_vars = np.array(map_vars)
    map_mu_sims = np.array(map_mu_sims)
    maxlogps = np.array(maxlogps)

    rdim = mu_sims.shape[0] * mu_sims.shape[2]
    cdim = mu_sims.shape[1]

    mu_sims = mu_sims.reshape(rdim, cdim)
    logps = logps.reshape(rdim,)
    map_vars = map_vars.reshape(16,)
    map_mu_sims = map_mu_sims.reshape(4, cdim)
    maxlogps = maxlogps.reshape(4,)

    df1 = pd.DataFrame(mu_sims)
    df2 = pd.DataFrame(logps)
    df3 = pd.DataFrame(map_vars)
    df4 = pd.DataFrame(map_mu_sims)
    df5 = pd.DataFrame(maxlogps)

    dfs = [df1, df2, df3, df4, df5]
    names = ['mu_sims', 'logps', 'map_vars', 'map_mu_sims', 'maxlogps']


    for df, name in zip(dfs, names):
        p = os.path.join(out_folder, f'{name}.h5')
        df.to_hdf(p, key='df', mode='w')
        # with open(p, 'w') as wfile:
        #     json.dump(df, wfile)
        # df.to_csv(os.path.join(out_folder, f'{name}.csv'))


# def plot_data(ax, lw=2, title="Hudson's Bay Company Data"):
#     ax.plot(data.year, data.lynx, color="b", lw=lw, marker="o", markersize=12, label="Lynx (Data)")
#     ax.plot(data.year, data.hare, color="g", lw=lw, marker="+", markersize=14, label="Hare (Data)")
#     ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
#     ax.set_xlim([1900, 1920])
#     ax.set_ylim(0)
#     ax.set_xlabel("Year", fontsize=14)
#     ax.set_ylabel("Pelts (Thousands)", fontsize=14)
#     ax.set_xticks(data.year.astype(int))
#     ax.set_xticklabels(ax.get_xticks(), rotation=45)
#     ax.set_title(title, fontsize=16)
#     return ax
#
#
# def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
#     cols = ["alpha", "beta", "gamma", "delta", "xto", "yto"]
#     row = trace_df.iloc[row_idx, :][cols].values
#
#     # alpha, beta, gamma, delta, Xt0, Yt0
#     time = np.arange(1900, 1921, 0.01)
#     theta = row
#     x_y = odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,))
#     plot_model(ax, x_y, time=time, lw=lw, alpha=alpha)
#
#
#
# def plot_inference(
#     ax,
#     trace,
#     num_samples=25,
#     title="Hudson's Bay Company Data and\nInference Model Runs",
#     plot_model_kwargs=dict(lw=1, alpha=0.2),
# ):
#     trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
#     plot_data(ax, lw=0)
#     for row_idx in range(num_samples):
#         plot_model_trace(ax, trace_df, row_idx, **plot_model_kwargs)
#         generate_rsf_data(nr, modelvars)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1, 0.5))
#     ax.set_title(title, fontsize=16)


def plot_chisquare_interval(logps, mu_sims, mutrue, x):
    mu_sims = np.array(mu_sims)
    logps = np.abs(logps)
    logps = np.array(logps)
    rdim = mu_sims.shape[0]
    cdim = mu_sims.shape[1]

    # mu_sims = mu_sims.reshape(rdim, cdim)
    # logps = logps.reshape(rdim,)

    ensemble_modes = []
    ensemble_means = []
    hdi_lower = []
    hdi_upper = []
    # hdi_data = []
    hdi_data = np.zeros((rdim, 2))
    for i in np.arange(rdim):
        r = mu_sims[i, :]
        ensemble_mode = az.plots.plot_utils.calculate_point_estimate('mode', r, skipna=True)
        ensemble_mean = az.plots.plot_utils.calculate_point_estimate('mean', r, skipna=True)
        hdi = az.hdi(r, hdi_prob=0.89, skipna=True)
        # hdi_data.append(hdi)
        hdi_data[i, :] = hdi
        ensemble_modes.append(ensemble_mode)
        ensemble_means.append(ensemble_mean)
        hdi_lower.append(hdi[0])
        hdi_upper.append(hdi[1])


    # hdi_data = np.array(hdi_data)
    plt.figure(601)
    # plt.plot(x, hdi_lower, 'b-')
    # plt.plot(x, hdi_upper, 'c-')
    az.plot_hdi(x, y=None, hdi_data=hdi_data, color='cyan')
    # plt.plot(x, ensemble_modes, 'k.')
    plt.plot(x, mutrue, 'b.', alpha=0.2)
    plt.ylim([np.min(hdi_data[:, 0]) - 0.01, np.max(hdi_data[:, 1]) + 0.01])
    plt.xlabel('loadpoint displacement ($\mu$m)')
    plt.ylabel('$\mu$')
    plt.title('Posterior Predictive Check')
    plt.show()


    # data = np.column_stack((logps, mu_sims))
    # df = pd.DataFrame(data)
    # dfsort = df.sort_values(by=0)
    #
    # limit = round(0.11 * nrplot*gpl.nch)
    # mu_sort = dfsort.iloc[:, 1:]
    # mu_sort = np.array(mu_sort)
    #
    # plt.figure(600)
    # for i in np.arange(limit):
    #     plt.plot(x, mu_sort[i, :], 'b-', alpha=0.05)
    #
    # plt.plot(x, mutrue, 'k.', alpha=0.2, label='observed')
    # plt.legend()


def draw_from_posteriors(idata, mutrue, x):
    # draw values from the 89% credible interval for each parameter
    # then generate rsf data for draws

    modelvals = get_model_vals(idata)
    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    k = 10000

    rsa = np.random.choice(a, k)
    rsb = np.random.choice(b, k)
    rsDc = np.random.choice(Dc, k)
    rsmu0 = np.random.choice(mu0, k)

    vars_all = rsa, rsb, rsDc, rsmu0
    mu_sims, logps, map_vars, map_mu_sim, maxlogp = generate_rsf_data(k, vars_all)  # generates rsf sim
    # plt.plot(mu_sims, 'b-', alpha=0.01)
    # plt.plot(mutrue, '.')
    # plt.show()

    plot_chisquare_interval(logps, mu_sims, mutrue, x)


def main():
    # k, lc, priors_info = read_from_json(dirpath)
    out_folder = get_storage_folder(dirname)
    times, mutrue, vlps, x = load_section_data(dirpath)

    vref, mus, sigmas = read_from_json(dirpath)

    # k, vref = get_constants(vlps)

    idata = load_inference_data(dirpath, idataname)
    fig, ax = plt.subplots(num=100)

    draw_from_posteriors(idata, mutrue, x)
    ab_idata = plot_a_minus_b(idata, vlps, vref, nrstep)

    # warmup_posterior_vals = get_warmup_vals(idata)
    # aw, bw, Dcw, mu0w = warmup_posterior_vals
    modelvals = get_model_vals(idata)
    modelvals_ab = get_model_vals(ab_idata)

    # plots pairs for a-b, Dc, mu0
    original_trace_all_chains(modelvals_ab, times, vref)

    # plots original trace
    thinned_idata = get_thinned_idata(modelvals)
    plot_trace(thinned_idata, chain=None)

    # plot observed data on this figure before chains are plotted to avoid plotting it 4 times
    # fig, ax = plt.subplots(num=70)
    # p1, = ax.plot(times, mutrue, '.', alpha=0.2, label='observed')
    fig, axs = plt.subplots(2, 1, sharex='all', num=70, gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0.05)
    xax = x
    axs[0].plot(xax * um_to_mm, mutrue, '.', alpha=0.2, label='observed')
    axs[0].set(ylabel=r'$\mu$', ylim=[np.min(mutrue) - 0.01, np.max(mutrue) + 0.01])

    map_mu_sims = []
    mu_sims_all = []
    logps_all = []
    map_vars_all = []
    maxlogps = []

    for chain in np.arange(nch):
        # get posteriors and plot them
        # get posteriors from model trace
        apost, bpost, Dcpost, mu0post = get_posteriors(modelvals, chain)
        modes = get_modes(modelvals, chain)

        # plot the priors and posteriors for comparison when necessary
        # plot_priors_posteriors(apost, bpost, Dcpost, mu0post)
        # plot_posterior_distributions(modelvals, chain, modes)

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

        plot_flag = 'no'
        if plot_flag == 'yes':
            # necessary variables are nondimensionalized in this function for comparison to observed data
            mu_sims, logps, map_vars, map_mu_sim, maxlogp = generate_rsf_data(nr, vars_all) # generates rsf sim
            map_mu_sims.append(map_mu_sim)
            mu_sims_all.append(mu_sims)
            logps_all.append(logps)
            map_vars_all.append(map_vars)
            maxlogps.append(maxlogp)

            # plot_hdi_mu_sims(mu_sims)

            ahat, bhat, Dchat, mu0hat = map_vars    # parameter vals which resulted in highest logp val

            aminb_hat = ahat - bhat

            # plot all the stuff
            axs[0].plot(xax * um_to_mm, map_mu_sim, label=f'chain {chain};'
                                                          # f'max logp = {round(maxlogp, 4)} \n '
                                                          f'a-b={round(aminb_hat, 4)} \n '
                                                          # f'a={round(ahat, 4)}; '
                                                          # f'b={round(bhat, 4)}; '
                                                          f'Dc={round(Dchat, 2)}; '
                                                          f'mu0={round(mu0hat, 3)}')

            axs[1].plot(xax * um_to_mm, vlps, 'k')
            axs[1].set(ylabel=r'Velocity ($\mu$m/s)')
            plt.xlabel(r'Loadpoint Displacement ($\mu$m)')

            pos = axs[0].get_position()
            axs[0].set_position([pos.x0, pos.y0, pos.width*0.9, pos.height])
            axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='x-small')

            pos1 = axs[1].get_position()
            axs[1].set_position([pos1.x0, pos1.y0, pos1.width*0.9, pos1.height])

            # plot logp vals as sanity check
            plt.figure(71)
            logps = np.sort(logps)
            plt.plot(logps, '.')
            plt.ylabel('logp values')
            plt.xlabel('realization no.')
            plt.title('logp vals')
        elif plot_flag == 'no':
            print('skipping plotting observed data with realizations')

    # save_data(mu_sims_all, logps_all, map_vars_all, map_mu_sims, maxlogps, out_folder)
    save_figs(out_folder)


if __name__ == '__main__':
    main()
