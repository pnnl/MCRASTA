import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import arviz as az
from gplot import gpl
import pandas as pd
from rsfmodel import rsf, staterelations

um_to_mm = 0.001

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
    # gpl.read_from_json(gpl.idata_location())
    # print(f'self.threshold = {gpl.threshold}')
    a, b, Dc, mu0, s = inputs

    # dimensional variables output from mcrasta.py
    times, mutrue, vlps, x = load_section_data()
    k, vref = get_constants(vlps)
    lc, vmax = get_vmax_l0(vlps)

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


def find_best_fit(logps):
    a, b, Dc, mu0, s = get_model_values()

    sortedi = np.argsort(np.abs(logps))

    abest = a[sortedi[0]]
    bbest = b[sortedi[0]]
    Dcbest = Dc[sortedi[0]]
    mu0best = mu0[sortedi[0]]
    sbest = s[sortedi[0]]
    logpbest = logps[sortedi[0]]

    num = 100
    plt.figure(num=num)
    plt.plot(Dc[sortedi], logps[sortedi], '.', alpha=0.1)
    plt.xlabel('sorted Dc')
    plt.ylabel('sorted logps')

    plt.figure(num=num+1)
    plt.plot(s[sortedi], logps[sortedi], '.', alpha=0.1)
    plt.xlabel('sorted sigma')
    plt.ylabel('sorted logps')

    aminb = a[sortedi] - b[sortedi]
    plt.figure(num=num + 2)
    plt.plot(aminb, logps[sortedi], '.', alpha=0.1)
    plt.xlabel('sorted (a-b)')
    plt.ylabel('sorted logps')

    plt.figure(num=num+3)
    plt.plot(mu0[sortedi], logps[sortedi], '.', alpha=0.1)
    plt.xlabel('sorted mu0')
    plt.ylabel('sorted logps')
    plt.show()


    inputs = abest, bbest, Dcbest, mu0best, sbest
    mu_best = generate_rsf_data(inputs)

    return [abest, bbest, Dcbest, mu0best, sbest], logpbest, mu_best


def get_model_values():
    p = os.path.join(gpl.idata_location(), f'{gpl.sim_name}_idata')
    idata = az.from_netcdf(p)

    acc_rate = np.mean(idata.sample_stats['accepted'])
    print(acc_rate)

    modelvals = az.extract(idata.posterior, combined=True)

    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values
    s = modelvals.s.values

    return a, b, Dc, mu0, s


def get_npy_data(p, f):
    data = np.load(os.path.join(p, f'{f}.npy'))

    return data


def plot_results(x, mt, musims, params, mubest):
    abest, bbest, Dcbest, mu0best, sbest = params
    x = np.transpose(x)

    plt.plot(x * um_to_mm, musims.T, color='firebrick', alpha=0.02)
    plt.plot(x * um_to_mm, mt.T, 'k.', label='observed')
    # plt.plot(x * um_to_mm, mubest.T, color='lightseagreen', label=f'best fit\n'
    #                                        f'a={abest}\n'
    #                                        f'b={bbest}\n'
    #                                        f'$D_c$={Dcbest}\n'
    #                                        f'$\mu_0$={mu0best}')
    plt.plot(x * um_to_mm, mubest.T, color='lightseagreen', label=f'best fit\n'
                                           f'a={abest.round(4)}\n'
                                           f'b={bbest.round(4)}\n'
                                           f'$D_c$={Dcbest.round(3)}\n'
                                           f'$\mu_0$={mu0best.round(3)}\n'
                                           f'$\sigma$={sbest.round(3)}')

    plt.xlabel('Loadpoint displacement (mm)')
    plt.ylabel('$\mu$')
    plt.title(f'Posterior draws: Sample p{gpl.section_id}')
    plt.ylim(np.mean(mt) - 0.07, np.mean(mt) + 0.07)
    plt.legend(bbox_to_anchor=(1.01, 1))

    # plt.show()


def load_section_data():
    section_data = pd.read_csv(os.path.join(gpl.idata_location(), 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def save_figs():
    # check if folder exists, make one if it doesn't
    name = gpl.get_musim_storage_folder()
    print(f'find figures and .out file here: {name}')
    w = plt.get_fignums()
    print('w = ', w)
    for i in plt.get_fignums():
        print('i = ', i)
        plt.figure(i).savefig(os.path.join(name, f'newfig{i}.png'), dpi=300, bbox_inches='tight')


def main():
    parent_dir = gpl.get_musim_storage_folder()

    msims = get_npy_data(parent_dir, f'musim_rd_p{gpl.section_id}')

    # msims[msims < 0] = np.nan
    # msims[msims > 1.5] = np.nan
    # msims[msims == np.inf] = np.nan
    # msims[msims == -np.inf] = np.nan

    logps = get_npy_data(parent_dir, f'logps_p{gpl.section_id}')

    params, logp, mubest = find_best_fit(logps)

    mubest = generate_rsf_data(params)

    t, mutrue, vlps, x = load_section_data()
    if np.any(vlps < 0):
        print('velocities less than 0, check')

    plot_results(x, mutrue, msims, params, mubest)
    save_figs()


if __name__ == '__main__':
    main()
