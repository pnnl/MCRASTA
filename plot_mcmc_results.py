import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns
import arviz.labels as azl
from configplot import cplot

home = os.path.expanduser('~')

um_to_mm = 0.001
Dclabel = r'$D_c$ ($\mu$m)'
mu0label = r'$\mu_0$'
fontsize = 12


def write_model_info(modes, hdis):
    headers = ['sampleID', 'aminbmode', 'aminbhdi', 'Dcmode', 'Dchdi', 'mu0mode', 'mu0hdi',
               'amode', 'ahdi', 'bmode', 'bhdi', 'smode', 'shdi']
    fname = os.path.join(cplot.postprocess_out_dir, 'posterior_stats.csv')

    amode, bmode, Dcmode, mu0mode, aminbmode, smode = modes
    ahdi, bhdi, Dchdi, mu0hdi, aminbhdi, shdi = hdis

    col = [f'{cplot.section_id}', aminbmode, aminbhdi, Dcmode, Dchdi, mu0mode, mu0hdi,
            amode, ahdi, bmode, bhdi, smode, shdi]
    df = pd.DataFrame(col)

    row = df.T

    row.to_csv(fname, ',', header=False, index=False, index_label=False, mode='a')


def load_inference_data():
    p = os.path.join(cplot.mcmc_out_dir, f'{cplot.sim_name}_idata')
    trace = az.from_netcdf(p)

    return trace


def plot_trace(idata):
    chain_prop = {'color': ['rosybrown', 'firebrick', 'red', 'maroon'], 'linestyle': ['solid', 'dotted', 'dashed', 'dashdot']}
    backend_kwargs = {'layout': 'tight'}
    plot_kwargs = {'textsize': 16}
    labeller = azl.MapLabeller(
        var_name_map={'a': 'a',
                      'b': 'b',
                      'Dc': r'$D_{c}$ ($\mu$m)',
                      'mu0': r'$\mu_{0}$',
                      's': r'$\sigma$'})
    ax = az.plot_trace(idata,
                       var_names=['a', 'b', 'Dc', 'mu0', 's'],
                       labeller=labeller,
                       combined=False,
                       plot_kwargs=plot_kwargs,
                       backend_kwargs=backend_kwargs,
                       chain_prop=chain_prop,
                       )

    kwargs = {'color': 'firebrick'}
    ax2 = az.plot_posterior(idata, var_names=['a', 'b', 'Dc', 'mu0', 's'], point_estimate='mode', round_to=3, **kwargs)
    ax2[0][0].set_xlim(0, 0.08)
    ax2[0][1].set_xlim(0, 0.12)
    ax2[0][2].set_xlim(0, 180)


def plot_pairs(idata, modes, chain=None):
    # plot_kwargs = {'linewidths': 0.2}
    marginal_kwargs = {'color': 'teal', 'textsize': 18}
    kde_kwargs = {'hdi_probs': [0.10, 0.50, 0.75, 0.89, 0.94]}
    # reference_values = {0}
    # reference_values_kwargs = {'label': 'mode'}
    labeller = azl.MapLabeller(var_name_map={'a_min_b': 'a-b', 'Dc': r'$D_{c}$ ($\mu$m)', 'mu0': r'$\mu_{0}$'})
    ax = az.plot_pair(
        idata,
        var_names=['a_min_b', 'Dc', 'mu0'],
        kind=['scatter', 'kde'],
        marginals=True,
        scatter_kwargs={'color': 'teal', 'alpha': 0.1},
        labeller=labeller,
        point_estimate='mode',
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
        textsize=18,
    )

    ax[0][0].set_xlim(-0.03, 0.03)
    ax[1][1].set_xlim(0, 180)


def get_model_vals(idata, combined=True):
    modelvals = az.extract(idata.posterior, combined=combined)

    return modelvals


def get_posterior_data(modelvals, return_aminb=False, thin_data=False):
    if thin_data is False:
        cplot.nrstep = 1
    elif thin_data is True:
        cplot.nrstep = cplot.nrstep

    a = modelvals.a.values[0::cplot.nrstep]
    b = modelvals.b.values[0::cplot.nrstep]
    Dc = modelvals.Dc.values[0::cplot.nrstep]
    mu0 = modelvals.mu0.values[0::cplot.nrstep]
    s = modelvals.s.values[0::cplot.nrstep]

    if return_aminb is True:
        a_min_b = modelvals.a_min_b.values[0::cplot.nrstep]
        return a_min_b, a, b, Dc, mu0, s
    elif return_aminb is False:
        return a, b, Dc, mu0, s


def load_section_data():
    section_data = pd.read_csv(os.path.join(cplot.mcmc_out_dir, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def plot_pairs_thinned_idata(modelvals):
    a_min_b, a, b, Dc, mu0, s = get_posterior_data(modelvals, return_aminb=True, thin_data=True)

    datadict = {'a_min_b': a_min_b, 'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0, 's': s}
    new_idata = az.convert_to_inference_data(datadict)

    modes = get_modes(modelvals)

    plot_pairs(new_idata, modes, chain=None)


def save_figs(out_folder):
    # check if folder exists, make one if it doesn't
    name = out_folder
    print(f'find figures and .out file here: {name}')
    w = plt.get_fignums()
    print('w = ', w)
    for i in plt.get_fignums():
        print('i = ', i)
        plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300, bbox_inches='tight')


def plot_priors_posteriors(modelvals):
    posts = get_posterior_data(modelvals, return_aminb=False, thin_data=False)

    # define priors same as in mcrasta.py - get this info from json file
    mus = cplot.mus
    sigmas = cplot.sigmas
    xmaxs = [0.05, 0.05, 180, 1.25, 5]

    a = pm.LogNormal.dist(mu=mus[0], sigma=sigmas[0])
    b = pm.LogNormal.dist(mu=mus[1], sigma=sigmas[1])
    Dc = pm.LogNormal.dist(mu=mus[2], sigma=sigmas[2])
    mu0 = pm.LogNormal.dist(mu=mus[3], sigma=sigmas[3])

    # s = pm.HalfNormal.dist(sigma=sigmas[4])
    s = pm.HalfNormal.dist(sigma=0.01)

    # take same number of draws as in mcrasta.py
    vpriors = pm.draw([a, b, Dc, mu0, s], draws=cplot.ndr * cplot.nch)

    for i, (prior, post, label, xmax) in enumerate(zip(vpriors, posts, ('a', 'b', f'{Dclabel}', f'{mu0label}', r'$\sigma$'), xmaxs)):
        sns.displot(data=(prior, post), kind='kde')
        plt.gca().set_xlim(0, xmax)
        plt.title('Prior and Posterior PDFs')
        plt.xlabel(f'{label}')
        plt.ylabel('Probability Density')
        plt.grid(True)

    # plt.show()


def get_modes(modelvals):
    aminb, a, b, Dc, mu0, s = get_posterior_data(modelvals, return_aminb=True, thin_data=False)

    amode = az.plots.plot_utils.calculate_point_estimate('mode', a)
    bmode = az.plots.plot_utils.calculate_point_estimate('mode', b, )
    Dcmode = az.plots.plot_utils.calculate_point_estimate('mode', Dc)
    mu0mode = az.plots.plot_utils.calculate_point_estimate('mode', mu0)
    aminbmode = az.plots.plot_utils.calculate_point_estimate('mode', aminb)
    smode = az.plots.plot_utils.calculate_point_estimate('mode', s)

    ahdi = az.hdi(a, hdi_prob=0.89)
    bhdi = az.hdi(b, hdi_prob=0.89)
    Dchdi = az.hdi(Dc, hdi_prob=0.89)
    mu0hdi = az.hdi(mu0, hdi_prob=0.89)
    aminbhdi = az.hdi(aminb, hdi_prob=0.89)
    shdi = az.hdi(s, hdi_prob=0.89)

    hdis = [ahdi, bhdi, Dchdi, mu0hdi, aminbhdi, shdi]

    return [amode, bmode, Dcmode, mu0mode, aminbmode, smode], hdis


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    k0 = cplot.k * cplot.lc
    vlps0 = vlps / vmax
    vref0 = vref / vmax
    t0 = times * vmax / cplot.lc
    t0 = t0 - t0[0]

    return k0, vlps0, vref0, t0


def plot_a_minus_b(idata):
    modelvals = az.extract(idata.posterior, combined=True)
    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values
    s = modelvals.s.values

    labeller = azl.MapLabeller(
        var_name_map={'a_min_b': 'a-b',
                      'Dc': r'$D_{c}$ ($\mu$m)',
                      'mu0': r'$\mu_{0}$',
                      's': r'$\sigma$'}
                            )
    color = 'firebrick'

    a_min_b = a - b
    datadict = {'a_min_b': a_min_b, 'a': a, 'b': b, 'Dc': Dc, 'mu0': mu0, 's': s}
    ab_idata = az.convert_to_inference_data(datadict, group='posterior')
    hdi_prob = 0.89

    num = plt.gcf().number
    plt.figure(num + 1)
    ax = az.plot_posterior(ab_idata,
                           var_names=['a_min_b'],
                           point_estimate='mode',
                           round_to=4,
                           hdi_prob=hdi_prob,
                           color=color)
    ax.set_xlim(-0.05, 0.05)
    ax.set_title(f'(a-b) posterior distribution, {cplot.samplename}')
    mab = az.plots.plot_utils.calculate_point_estimate('mode', a_min_b)
    cplot.aminbmode = mab

    plt.figure(num + 2)
    ax1 = az.plot_posterior(idata,
                            var_names=['a'],
                            point_estimate='mode',
                            round_to=4,
                            hdi_prob=hdi_prob,
                            color=color)
    ax1.set_title(f'a posterior distribution, {cplot.samplename}')
    ax1.set_xlim(0, 0.04)
    ma = az.plots.plot_utils.calculate_point_estimate('mode', a)


    plt.figure(num + 3)
    ax2 = az.plot_posterior(idata,
                            var_names=['b'],
                            point_estimate='mode',
                            round_to=4,
                            hdi_prob=hdi_prob,
                            color=color)
    ax2.set_title(f'b posterior distribution, {cplot.samplename}')
    ax2.set_xlim(0, 0.05)

    plt.figure(num + 4)
    ax3 = az.plot_posterior(idata,
                            var_names=['Dc'],
                            point_estimate='mode',
                            round_to=4,
                            hdi_prob=hdi_prob,
                            color=color)
    ax3.set_title(f'$D_c$ ($\mu$m) posterior distribution, {cplot.samplename}')
    ax3.set_xlim(0, 150)
    # plt.show()

    plt.figure(num + 5)
    ax4 = az.plot_posterior(idata,
                            var_names=['mu0'],
                            point_estimate='mode',
                            round_to=4,
                            hdi_prob=hdi_prob,
                            color=color)
    ax4.set_title(f'$\mu_0$ posterior distribution, {cplot.samplename}')

    plt.figure(num + 6)
    ax4 = az.plot_posterior(idata,
                            var_names=['s'],
                            point_estimate='mode',
                            round_to=4,
                            hdi_prob=hdi_prob,
                            color=color)
    ax4.set_title(f'$\sigma$ posterior distribution, {cplot.samplename}')

    fill_kwargs = {'alpha': 0.5}
    marginal_kwargs = {'color': color, 'quantiles': [0.11, 0.89], 'fill_kwargs': fill_kwargs}
    kde_kwargs = {'hdi_probs': [0.10, 0.25, 0.50, 0.75, 0.89, 0.94]}
    ax = az.plot_pair(
        ab_idata,
        var_names=['a_min_b', 'Dc', 'mu0', 's'],
        kind=["scatter", "kde"],
        marginals=True,
        scatter_kwargs={'color': 'firebrick', 'alpha': 0.01},
        point_estimate='mode',
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
        labeller=labeller,
        textsize=18,
    )

    ax[0][0].set_xlim(-0.03, 0.03)
    ax[1][1].set_xlim(0, 180)
    ax[2][2].set_xlim(0.3, 0.53)

    return ab_idata


def plot_observed_and_vlps(mutrue, vlps, xax):
    num = plt.gcf().number + 1
    fig, axs = plt.subplots(2, 1, sharex='all', num=num, gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0.05)
    axs[0].plot(xax * um_to_mm, mutrue, 'k.', alpha=0.7, label='observed data', markersize=1)
    axs[0].set(ylabel=r'$\mu$', ylim=[np.min(mutrue) - 0.01, np.max(mutrue) + 0.01])

    axs[1].plot(xax * um_to_mm, vlps, '.', color='darkred', markersize=1)
    axs[1].set(ylabel=r'Velocity (m/s)')
    plt.xlabel(r'Loadpoint Displacement (mm)')

    pos = axs[0].get_position()
    axs[0].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    axs[0].legend(loc='upper right', fontsize='x-small')

    pos1 = axs[1].get_position()
    axs[1].set_position([pos1.x0, pos1.y0, pos1.width * 0.9, pos1.height])

    # plt.show()


def main():
    # load observed section data and mcmc inference data
    times, mutrue, vlps, x = load_section_data()
    idata = load_inference_data()

    ess = az.ess(idata)
    az.plot_ess(idata)
    print(f'ESS: {ess}')

    rhat = az.rhat(idata)
    print(f'rhat: {rhat}')
    az.plot_forest(idata, r_hat=True, ess=True)

    # plt.show()

    modelvals = get_model_vals(idata, combined=True)
    plot_priors_posteriors(modelvals)

    # first plot: mcmc trace with all original data
    plot_trace(idata)

    # this plots posteriors and pair plot for (a-b) dataset
    # instead of a and b individually
    ab_idata = plot_a_minus_b(idata)
    modelvals = get_model_vals(ab_idata, combined=True)

    modes, hdis = get_modes(modelvals)

    write_model_info(modes, hdis)

    # plots thinned data pairs for a-b, Dc, mu0
    plot_pairs_thinned_idata(modelvals)

    # plot observed data section and velocity steps
    plot_observed_and_vlps(mutrue, vlps, xax=x)

    # save all figures
    save_figs(cplot.postprocess_out_dir)


if __name__ == '__main__':
    main()
