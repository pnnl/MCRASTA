import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import arviz as az
from gplot import gpl
import posterior_draws
import pandas as pd


def find_best_fit(logps):
    a, b, Dc, mu0 = get_model_values()

    sortedi = np.argsort(logps)

    abest = a[sortedi[0]]
    bbest = b[sortedi[0]]
    Dcbest = Dc[sortedi[0]]
    mu0best = mu0[sortedi[0]]
    logpbest = logps[sortedi[0]]

    mu_best = posterior_draws.generate_rsf_data((abest, bbest, Dcbest, mu0best))

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

    plt.plot(x, musims.T, color='indianred', alpha=0.01)
    plt.plot(x, mt.T, 'k.', label='observed')
    plt.plot(x, mubest.T, color='red', label=f'best fit\n'
                                           f'a={abest.round(4)}\n'
                                           f'b={bbest.round(4)}\n'
                                           f'$D_c$={Dcbest.round(3)}\n'
                                           f'$\mu_0$={mu0best.round(3)}')

    plt.xlabel('load point displacement ($\mu$m)')
    plt.ylabel('$\mu$')
    plt.title(f'Posterior draws: Sample {gpl.section_id}')
    plt.legend()


def load_section_data():
    section_data = pd.read_csv(os.path.join(gpl.idata_location(), 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy().round(2)
    mutrue = df['mutrue'].to_numpy().round(3)
    vlps = df['vlps'].to_numpy().round(2)
    x = df['x'].to_numpy().round(2)

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
    rds = os.path.join(parent_dir, f'musim_rd_p{gpl.section_id}')

    msims = get_npy_data(parent_dir, f'musim_rd_p{gpl.section_id}')

    msims[msims < 0] = np.nan
    msims[msims > 1.5] = np.nan
    msims[msims == np.inf] = np.nan
    msims[msims == -np.inf] = np.nan

    logps1 = get_npy_data(parent_dir, f'logps_p{gpl.section_id}_0')
    logps2 = get_npy_data(parent_dir, f'logps_p{gpl.section_id}_1')

    logps = np.concatenate((logps1, logps2))

    params, logp, mubest = find_best_fit(logps)

    t, mutrue, vlps, x = load_section_data()

    plot_results(x, mutrue, msims, mubest, params)
    save_figs()


if __name__ == '__main__':
    main()
