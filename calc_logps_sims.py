import sys
from datetime import datetime
import time
from multiprocessing import Process, Queue, Pool
import numpy as np
from rsfmodel import rsf, staterelations
import matplotlib.pyplot as plt
import os
import arviz as az
import pandas as pd
from configplot import cplot


''' calculates and returns the goodness of fit 
    metric (log-likelihood) for all parameter estimates '''


def determine_threshold(vlps, t):
    vlps0 = vlps / np.max(vlps)
    t0 = t * np.max(vlps) / cplot.lc
    t0 = t0 - t0[0]
    velocity_gradient = np.gradient(vlps0)
    time_gradient = np.gradient(t0)
    acceleration = velocity_gradient / time_gradient

    threshold_line = cplot.threshold * np.ones_like(acceleration)

    n = plt.gcf().number
    plt.figure(n + 1)
    plt.plot(acceleration)
    plt.plot(threshold_line, 'r')
    plt.title('acceleration values to determine threshold used in ode solver')
    plt.ylabel('acceleration')
    print(cplot.threshold)
    plt.show()


def load_section_data():
    section_data = pd.read_csv(os.path.join(cplot.mcmc_out_dir, 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def load_inference_data():
    p = os.path.join(cplot.mcmc_out_dir, f'{cplot.sim_name}_idata')
    trace = az.from_netcdf(p)

    return trace


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


def get_posterior_data(modelvals, thin_data=False):
    if thin_data is True:
        try:
            nrstep = cplot.nrstep
        except(KeyError) as e:
            print(e)
            print('Specify number for nrstep in configplot.yaml or set thin_data=False.')
            print('Plotting full set of results')
            nrstep = 1
    else:
        nrstep = 1

    a = modelvals.a.values[0::nrstep]
    b = modelvals.b.values[0::nrstep]
    Dc = modelvals.Dc.values[0::nrstep]
    mu0 = modelvals.mu0.values[0::nrstep]
    s = modelvals.s.values[0::nrstep]

    return a, b, Dc, mu0, s


def get_model_values(idata):
    modelvals = az.extract(idata.posterior, combined=True)
    a, b, Dc, mu0, s = get_posterior_data(modelvals)

    return a, b, Dc, mu0, s


def generate_rsf_data(inputs):
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

    resids = np.transpose(mutrue) - mu_sim
    rsq = resids ** 2
    # srsq = np.nansum(rsq)
    # logp = np.abs(- 1 / 2 * srsq)
    logp = (-1 / (2 * (s ** 2))) * (np.sum(rsq))

    return logp


def get_dataset():
    # load observed section data and mcmc inference data
    times, mt, vlps, x = load_section_data()
    # print(len(x))
    idat = load_inference_data()

    # 'new' data = I started storing model parameters so I could read them in instead of manually filling them out
    # 'old' data = had to fill in parameters manually
    # if there's no .json in the mcmc results folder, then the data is type 'old'
    # dataset_type = 'new'
    # if dataset_type == 'old':
    #     k, vref = get_constants(vlps)
    # elif dataset_type == 'new':
    #     vref, mus, sigmas = gpl.read_from_json(gpl.idata_location())

    return idat, mt, vlps, times


def get_time(name):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'{name} time = {current_time}')

    codetime = time.time()

    return codetime


def main():
    comptime_start = get_time('start')
    # parent_dir = gpl.get_musim_storage_folder()
    idata, mutrue, vlps, times = get_dataset()
    # gpl.read_from_json(gpl.idata_location())
    # determine_threshold(vlps, times)
    # gpl.set_vch(vlps)
    # set_critical_times(vlps, times, threshold=gpl.threshold)

    a, b, Dc, mu0, s = get_model_values(idata)

    pathname = os.path.join(cplot.postprocess_out_dir, f'logps_p{cplot.section_id}')

    with Pool(processes=30, maxtasksperchild=1) as pool:
        outputs = pool.map(generate_rsf_data, zip(a, b, Dc, mu0, s))

    op = np.array(outputs)
    np.save(pathname, op)

    comptime_end = get_time('end')
    time_elapsed = comptime_end - comptime_start
    print(f'time elapsed = {time_elapsed}')
    print('END')


if __name__ == '__main__':
    main()
    # comptime_start = get_time('start')
    # # parent_dir = gpl.get_musim_storage_folder()
    # idata, mutrue, vlps, times = get_dataset()
    # # gpl.read_from_json(gpl.idata_location())
    # # determine_threshold(vlps, times)
    # # gpl.set_vch(vlps)
    # # set_critical_times(vlps, times, threshold=gpl.threshold)
    #
    # a, b, Dc, mu0, s = get_model_values(idata)
    #
    # pathname = os.path.join(cplot.postprocess_out_dir, f'logps_p{cplot.section_id}')
    #
    # with Pool(processes=30, maxtasksperchild=1) as pool:
    #     outputs = pool.map(generate_rsf_data, zip(a, b, Dc, mu0, s))
    #
    # op = np.array(outputs)
    # np.save(pathname, op)
    #
    # comptime_end = get_time('end')
    # time_elapsed = comptime_end - comptime_start
    # print(f'time elapsed = {time_elapsed}')
    # print('END')
