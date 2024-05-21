import sys
from datetime import datetime
import time
from multiprocessing import Process, Queue, Pool
from gplot import gpl
import plot_mcmc_results as pmr
import itertools
import numpy as np
from plotrsfmodel import rsf, staterelations
import matplotlib.pyplot as plt
import os
import cProfile
import arviz as az
import pandas as pd
import gc


def determine_threshold(vlps, t):
    vlps0 = vlps / np.max(vlps)
    t0 = t * np.max(vlps) / gpl.lc
    t0 = t0 - t0[0]
    velocity_gradient = np.gradient(vlps0)
    time_gradient = np.gradient(t0)
    acceleration = velocity_gradient / time_gradient

    threshold_line = gpl.threshold * np.ones_like(acceleration)

    n = plt.gcf().number
    plt.figure(n + 1)
    plt.plot(acceleration)
    plt.plot(threshold_line, 'r')
    plt.title('acceleration values to determine threshold used in ode solver')
    plt.ylabel('acceleration')
    print(gpl.threshold)
    plt.show()


def load_section_data():
    section_data = pd.read_csv(os.path.join(gpl.idata_location(), 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def load_inference_data():
    p = os.path.join(gpl.idata_location(), f'{gpl.sim_name}_idata')
    trace = az.from_netcdf(p)

    return trace


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


def set_critical_times(vlps, t, threshold):
    """
    Calculates accelearation and thresholds based on that to find areas
    that are likely problematic to integrate.

    Parameters
    ----------
    threshold : float
        When the absolute value of acceleration exceeds this value, the
        time is marked as "critical" for integration.

    Returns
    -------
    critical_times : list
        List of time values at which integration care should be taken.
    """
    velocity_gradient = np.gradient(vlps)
    time_gradient = np.gradient(t)
    acceleration = velocity_gradient / time_gradient
    critical_times = t[np.abs(acceleration) > threshold]

    # nondimen
    tcrit0 = critical_times * gpl.vmax / gpl.lc
    tcrit = tcrit0 - t[0]
    np.round(tcrit, 2)
    np.save('tcrittest.npy', tcrit)
    print('this should only print once')


def get_posterior_data(modelvals, thin_data=False):
    if thin_data is False:
        gpl.nrstep = 1
    elif thin_data is True:
        gpl.nrstep = gpl.nrstep

    a = modelvals.a.values[0::gpl.nrstep]
    b = modelvals.b.values[0::gpl.nrstep]
    Dc = modelvals.Dc.values[0::gpl.nrstep]
    mu0 = modelvals.mu0.values[0::gpl.nrstep]

    return a, b, Dc, mu0


def get_model_values(idata):
    modelvals = az.extract(idata.posterior, combined=True)
    a, b, Dc, mu0 = get_posterior_data(modelvals)

    return a, b, Dc, mu0


def generate_rsf_data(inputs):
    gpl.read_from_json(gpl.idata_location())
    # print(f'self.threshold = {gpl.threshold}')
    a, b, Dc, mu0 = inputs

    # dimensional variables output from mcmc_rsf.py
    times, mutrue, vlps, x = load_section_data()
    k, vref = get_constants(vlps)
    lc, vmax = get_vmax_l0(vlps)

    mutrue.round(2).astype('float32')
    vlps.round(2).astype('float32')

    # time is the only variable that needs to be re-nondimensionalized...?
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

    model.time = np.round(t0, 2).astype('float32')

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps.astype('float32')

    model.mu0 = mu0
    model.a = a
    state1.b = b
    state1.Dc = Dc

    model.solve(threshold=gpl.threshold)

    mu_sim = model.results.friction.astype('float32')
    state_sim = model.results.states

    return mu_sim


def get_dataset():
    # setup output directory
    out_folder = gpl.get_musim_storage_folder()

    # load observed section data and mcmc inference data
    times, mt, vlps, x = load_section_data()
    # print(len(x))
    idat = load_inference_data()

    # 'new' data = I started storing model parameters so I could read them in instead of manually filling them out
    # 'old' data = had to fill in parameters manually
    # if there's no .json in the mcmc results folder, then the data is type 'old'
    dataset_type = 'new'
    if dataset_type == 'old':
        k, vref = get_constants(vlps)
    elif dataset_type == 'new':
        vref, mus, sigmas = gpl.read_from_json(gpl.idata_location())

    return idat, mt, vlps, times


def get_time(name):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'{name} time = {current_time}')

    codetime = time.time()

    return codetime


def parallel_processing(inputs):
    pathname = os.path.join(parent_dir, f'mu_simsp{gpl.section_id}_{k}')

    pool = Pool(processes=25, maxtasksperchild=1)

    outputs = pool.map(generate_rsf_data, zip(at, bt, Dct, mu0t))
    op = np.array(outputs)
    time.sleep(0.01)
    pool.close()
    pool.join()

    np.save(pathname, op)


if __name__ == '__main__':
    comptime_start = get_time('start')
    parent_dir = gpl.get_musim_storage_folder()

    idata, mutrue, vlps, times = get_dataset()
    # gpl.read_from_json(idata_location)
    # determine_threshold(vlps, times)
    gpl.set_vch(vlps)
    # set_critical_times(vlps, times, threshold=gpl.threshold)
    a, b, Dc, mu0 = get_model_values(idata)
    a = np.round(a, 4).astype('float32')
    b = np.round(b, 4).astype('float32')
    Dc = np.round(Dc, 2).astype('float32')
    mu0 = np.round(mu0, 3).astype('float32')

    stepsize = 50000
    for k, i in enumerate(range(0, 500000, stepsize)):
        print(f'***********************************************')
        print(f'SOLVING FWD MODEL FOR SIMS: {i} - {i+stepsize}')
        at = a[i:i+stepsize]
        bt = b[i:i+stepsize]
        Dct = Dc[i:i+stepsize]
        mu0t = mu0[i:i+stepsize]

        parallel_processing(zip(at, bt, Dct, mu0t))


        # pool = Pool(processes=25, maxtasksperchild=1)
        #
        # outputs = pool.map(generate_rsf_data, zip(at, bt, Dct, mu0t))
        # op = np.array(outputs)
        # time.sleep(0.01)
        # pool.close()
        # pool.join()
        # pathname = gpl.make_path('musim_out', f'{gpl.samplename}', f'mu_simsp{gpl.section_id}_{snum}')
        # np.save(pathname, op)
        gc.collect()

    comptime_end = get_time('end')
    time_elapsed = comptime_end - comptime_start
    print(f'time elapsed = {time_elapsed}')
    print('end')
