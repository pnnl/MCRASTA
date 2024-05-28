import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import arviz as az
from gplot import gpl
import posterior_draws
import pandas as pd
from plotrsfmodel import rsf, staterelations



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
    a, b, Dc, mu0 = inputs

    # dimensional variables output from mcmc_rsf.py
    times, mutrue, vlps, x = load_section_data()
    k, vref = get_constants(vlps)
    lc, vmax = get_vmax_l0(vlps)

    mutrue.astype('float32')
    vlps.astype('float32')

    k0, vlps0, vref0, t0 = nondimensionalize_parameters(vlps, vref, k, times, vmax)

    # set up rsf model
    model = rsf.Model()
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.vmax = vmax.astype('float32')
    state1.lc = gpl.lc

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t0.astype('float32')

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps.astype('float32')

    model.mu0 = mu0
    model.a = a
    state1.b = b
    state1.Dc = Dc

    model.solve(threshold=gpl.threshold)

    mu_sim = model.results.friction.astype('float32')

    # resids = np.transpose(mutrue) - mu_sim
    # rsq = resids ** 2
    # srsq = np.nansum(rsq)
    # logp = np.abs(- 1 / 2 * srsq)

    return mu_sim


def find_best_fit(logps):
    a, b, Dc, mu0 = get_model_values()

    sortedi = np.argsort(logps)

    abest = a[sortedi[0]]
    bbest = b[sortedi[0]]
    Dcbest = Dc[sortedi[0]]
    mu0best = mu0[sortedi[0]]
    logpbest = logps[sortedi[0]]

    # plt.plot(logps[sortedi])
    # plt.ylim(0, 0.5)
    # plt.show()

    inputs = abest, bbest, Dcbest, mu0best
    mu_best = generate_rsf_data(inputs)

    return [abest, bbest, Dcbest, mu0best], logpbest, mu_best


def get_model_values():
    p = os.path.join(gpl.idata_location(), f'{gpl.sim_name}_idata')
    idata = az.from_netcdf(p)
    modelvals = az.extract(idata.posterior, combined=True)

    a = modelvals.a.values
    b = modelvals.b.values
    Dc = modelvals.Dc.values
    mu0 = modelvals.mu0.values

    return a, b, Dc, mu0


def get_npy_data(p, f):
    data = np.load(os.path.join(p, f'{f}.npy'))

    return data


def plot_results(x, mt, musims, mubest, params):
    abest, bbest, Dcbest, mu0best = params
    x = np.transpose(x)

    plt.plot(x, musims.T, color='indianred', alpha=0.02)
    plt.plot(x, mt.T, 'k.', label='observed')
    plt.plot(x, mubest.T, color='red', label=f'best fit\n'
                                           f'a={abest}\n'
                                           f'b={bbest}\n'
                                           f'$D_c$={Dcbest}\n'
                                           f'$\mu_0$={mu0best}')
    # plt.plot(x, mubest.T, color='red', label=f'best fit\n'
    #                                        f'a={abest.round(4)}\n'
    #                                        f'b={bbest.round(4)}\n'
    #                                        f'$D_c$={Dcbest.round(3)}\n'
    #                                        f'$\mu_0$={mu0best.round(3)}')

    plt.xlabel('load point displacement ($\mu$m)')
    plt.ylabel('$\mu$')
    plt.title(f'Posterior draws: Sample {gpl.section_id}')
    plt.ylim(np.mean(mubest) - 0.1, np.mean(mubest) + 0.1)
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
        plt.figure(i).savefig(os.path.join(name, f'fig{i}.png'), dpi=300, bbox_inches='tight')


def main():
    parent_dir = gpl.get_musim_storage_folder()
    # rds = os.path.join(parent_dir, f'musim_rd_p{gpl.section_id}')

    msims = get_npy_data(parent_dir, f'musim_rd_p{gpl.section_id}')

    msims[msims < 0] = np.nan
    msims[msims > 1.5] = np.nan
    msims[msims == np.inf] = np.nan
    msims[msims == -np.inf] = np.nan

    # logps1 = get_npy_data(parent_dir, f'logps_p{gpl.section_id}_0')
    # logps2 = get_npy_data(parent_dir, f'logps_p{gpl.section_id}_1')

    # logps = np.concatenate((logps1, logps2))

    logps = get_npy_data(parent_dir, f'logps_p{gpl.section_id}')

    params, logp, mubest = find_best_fit(logps)

    # params = [0.00629, 0.00655, 62.4856, 0.41]

    # mubest = np.load(r'C:\Users\fich146\PycharmProjects\mcmcrsf_xfiles\postprocess_out\p5894\out_500000d4ch_5894001'
    #                  r'\bestfit.npy')
    t, mutrue, vlps, x = load_section_data()

    plot_results(x, mutrue, msims, mubest, params)
    save_figs()


if __name__ == '__main__':
    main()
