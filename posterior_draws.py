import os
import arviz as az
import numpy as np
import pandas as pd
from configplot import cplot
from rsfmodel import rsf, staterelations
from multiprocessing import Pool

''' this script takes random draws from 
the posterior distribution, runs the forward model for each set.
It imports logp data and finds the best fit parameter set, then
plots the best fit with the simulated friction values '''


def load_section_data():
    section_data = pd.read_csv(os.path.join(cplot.mcmc_out_dir, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def get_constants(vlps):
    k = cplot.k
    vref = vlps[0]

    return k, vref


def get_vmax_l0(vlps):
    l0 = cplot.lc
    vmax = np.max(vlps)

    return l0, vmax


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    k0 = cplot.k * cplot.lc
    vlps0 = vlps / vmax
    vref0 = vref / vmax
    t0 = times * vmax / cplot.lc
    t0 = t0 - t0[0]

    return k0, vlps0, vref0, t0


def generate_rsf_data(inputs):
    a, b, Dc, mu0, s = inputs

    # dimensional variables output from mcrasta.py
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
    state1.lc = cplot.lc

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t0

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps0

    model.mu0 = mu0
    model.a = a
    state1.b = b
    state1.Dc = Dc / cplot.lc

    model.solve(threshold=cplot.threshold)

    mu_sim = model.results.friction

    return mu_sim


def get_model_values():
    p = os.path.join(cplot.mcmc_out_dir, f'{cplot.sim_name}_idata')
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


def main():
    print('START POSTERIOR DRAWS.PY')

    t, mutrue, vlps, x = load_section_data()
    drawed_vars = draw_from_posteriors(ndraws=cplot.num_posterior_draws)

    ad = drawed_vars[:, 0]
    bd = drawed_vars[:, 1]
    Dcd = drawed_vars[:, 2]
    mu0d = drawed_vars[:, 3]
    sd = drawed_vars[:, 4]

    pathname = os.path.join(cplot.postprocess_out_dir, f'musim_rd_p{cplot.section_id}')

    with Pool(processes=20, maxtasksperchild=1) as pool:
        outputs = pool.map(generate_rsf_data, zip(ad, bd, Dcd, mu0d, sd))
        op = np.array(outputs)
        np.save(pathname, op)

    print('end')
    print(f'saved npy file: {pathname}')

    print('END POSTERIOR DRAWS.PY')


if __name__ == '__main__':
    main()
    # t, mutrue, vlps, x = load_section_data()
    # drawed_vars = draw_from_posteriors(ndraws=cplot.num_posterior_draws)
    #
    # ad = drawed_vars[:, 0]
    # bd = drawed_vars[:, 1]
    # Dcd = drawed_vars[:, 2]
    # mu0d = drawed_vars[:, 3]
    # sd = drawed_vars[:, 4]
    #
    # pathname = os.path.join(cplot.postprocess_out_dir, f'musim_rd_p{cplot.section_id}')
    #
    # with Pool(processes=20, maxtasksperchild=1) as pool:
    #     outputs = pool.map(generate_rsf_data, zip(ad, bd, Dcd, mu0d, sd))
    #     op = np.array(outputs)
    #     np.save(pathname, op)
    #
    # print('end')
    # print(f'saved npy file: {pathname}')
