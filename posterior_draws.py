import os
import sys
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gplot import gpl
import plot_mcmc_results as pmr
import psutil
# from plotrsfmodel import rsf, staterelations
from rsfmodel import rsf, staterelations
from multiprocessing import Process, Queue, Pool

''' this script takes random draws from 
the posterior distribution, runs the forward model for each set.
It imports logp data and finds the best fit parameter set, then
plots the best fit with the simulated friction values '''

home = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'mcmcrsf_xfiles', 'mcmc_out')


def get_npy_data(p, f):
    data = np.load(os.path.join(p, f'{f}.npy'))

    return data


def load_section_data():
    section_data = pd.read_csv(os.path.join(gpl.idata_location(), 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def get_constants(vlps):
    k = gpl.k
    vref = vlps[0]

    return k, vref


def get_vmax_l0(vlps):
    l0 = gpl.lc
    vmax = np.max(vlps)

    return l0, vmax


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    k0 = gpl.k * gpl.lc
    vlps0 = vlps / vmax
    vref0 = vref / vmax
    t0 = times * vmax / gpl.lc
    t0 = t0 - t0[0]

    return k0, vlps0, vref0, t0


def generate_rsf_data(inputs):
    gpl.read_from_json(gpl.idata_location())
    # print(f'self.threshold = {gpl.threshold}')
    a, b, Dc, mu0, s = inputs

    # dimensional variables output from mcmc_rsf.py
    times, mutrue, vlps, x = load_section_data()
    k, vref = get_constants(vlps)
    lc, vmax = get_vmax_l0(vlps)

    if np.any(vlps < 0):
        print('velocities less than 0, check')

    k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

    # set up rsf model
    model = rsf.Model()
    model.k = k0  # Normalized System stiffness (friction/micron)
    model.v = vlps0[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref0  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.vmax = vmax
    state1.lc = gpl.lc

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t0

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps0

    model.mu0 = mu0
    model.a = a
    state1.b = b
    state1.Dc = Dc / gpl.lc

    model.solve(threshold=gpl.threshold)

    mu_sim = model.results.friction

    # resids = np.transpose(mutrue) - mu_sim
    # rsq = resids ** 2
    # srsq = np.nansum(rsq)
    # logp = np.abs(- 1 / 2 * srsq)

    return mu_sim


# def plot_results(x, mt, musims, mubest, params):
#     abest, bbest, Dcbest, mu0best = params
#     x = np.transpose(x)
#
#     plt.plot(x, musims, color='indianred', alpha=0.01)
#     plt.plot(x, mt, 'k.', label='observed')
#     plt.plot(x, mubest, color='red', label=f'best fit\n'
#                                            f'a={abest.round(4)}\n'
#                                            f'b={bbest.round(4)}\n'
#                                            f'$D_c$={Dcbest.round(3)}\n'
#                                            f'$\mu_0$={mu0best.round(3)}')
#
#     plt.xlabel('load point displacement ($\mu$m)')
#     plt.ylabel('$\mu$')
#     plt.title(f'Posterior draws: Sample {gpl.section_id}')
#     plt.legend()
#
#     # plt.show()


def write_best_estimates(bvars, lpbest):
    a, b, Dc, mu0 = bvars
    p = gpl.get_musim_storage_folder()
    fname = os.path.join(p, 'param_estimates_best.txt')

    paramstrings = ['a', 'b', 'Dc', 'mu0', 'lpbest']
    paramvals = [a, b, Dc, mu0, lpbest]

    with open(fname, mode='w') as f:
        for string, val in zip(paramstrings, paramvals):
            f.write(f'{string}: {val}\n')


def get_model_values():
    # p = os.path.join(gpl.idata_location(), f'out_500000d4ch_{gpl.section_id}_idata')

    p = os.path.join(gpl.idata_location(), f'{gpl.sim_name}_idata')
    idata = az.from_netcdf(p)
    modelvals = az.extract(idata.posterior, combined=True)

    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values
    s = modelvals.s.values

    return a, b, Dc, mu0, s


def draw_from_posteriors(ndraws=1000):
    # draw values from the 89% credible interval for each parameter
    # then generate rsf data for draws

    a, b, Dc, mu0, s = get_model_values()

    modelvals = np.column_stack((a, b, Dc, mu0, s))

    draws = modelvals[np.random.choice(modelvals.shape[0], ndraws, replace=False), :]

    return draws


def save_stats(ens_mean, ens_stdev, bestfit):
    p = gpl.get_postprocess_storage_folder()
    np.save(os.path.join(p, 'ens_mean'), ens_mean)
    np.save(os.path.join(p, 'ens_stdev'), ens_stdev)
    np.save(os.path.join(p, 'bestfit'), bestfit)


# def save_figs():
#     # check if folder exists, make one if it doesn't
#     name = gpl.get_musim_storage_folder()
#     print(f'find figures and .out file here: {name}')
#     w = plt.get_fignums()
#     print('w = ', w)
#     for i in plt.get_fignums():
#         print('i = ', i)
#         plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    t, mutrue, vlps, x = load_section_data()
    parent_dir = gpl.get_musim_storage_folder()

    num_draws = 4000
    # a, b, Dc, mu0 = get_model_values(idata)
    drawed_vars = draw_from_posteriors(ndraws=num_draws)

    ad = drawed_vars[:, 0]
    bd = drawed_vars[:, 1]
    Dcd = drawed_vars[:, 2]
    mu0d = drawed_vars[:, 3]
    sd = drawed_vars[:, 4]

    pathname = os.path.join(parent_dir, f'musim_rd_p{gpl.section_id}')

    with Pool(processes=20, maxtasksperchild=1) as pool:
        outputs = pool.map(generate_rsf_data, zip(ad, bd, Dcd, mu0d, sd))
        op = np.array(outputs)
        np.save(pathname, op)

    print('done')
    print(f'saved files are here: {pathname}')
    # next(pathname, op)
