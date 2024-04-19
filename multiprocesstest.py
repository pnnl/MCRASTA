from datetime import datetime
import time
from multiprocessing import Process, Queue, Pool
from gplot import gpl
import plot_mcmc_results as pmr
import itertools
import numpy as np
from plotrsfmodel import rsf, staterelations

samplename = gpl.samplename
nr = 500000
nch = 4
section = '001'
sampleid = f'5894{section}'
dirname = f'out_{nr}d{nch}ch_{sampleid}'
# dirname = f'out_{nr}d{nch}ch'
# dirname = f'~out_{nr}d{nch}ch'
idata_location = gpl.make_path('mcmc_out', samplename, dirname)
idataname = f'{dirname}_idata'


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

    #nondimen
    tcrit0 = critical_times * gpl.vmax / gpl.lc
    tcrit = tcrit0 - t[0]
    np.round(tcrit, 2)
    np.save('tcrittest.npy', tcrit)
    print('this should only print once')


def get_model_values(idata):
    modelvals = pmr.get_model_vals(idata, combined=True)
    a, b, Dc, mu0 = pmr.get_posterior_data(modelvals, return_aminb=False, thin_data=True)
    return a, b, Dc, mu0


def generate_rsf_data(inputs):
    gpl.read_from_json(idata_location)
    # print(f'self.threshold = {gpl.threshold}')
    a, b, Dc, mu0 = inputs

    tcrit = np.load('tcrittest.npy')

    # dimensional variables output from mcmc_rsf.py
    times, mutrue, vlps, x = pmr.load_section_data(idata_location)
    k, vref = pmr.get_constants(vlps)
    lc, vmax = pmr.get_vmax_l0(vlps)

    # time is the only variable that needs to be re-nondimensionalized...?
    k0, vlps0, vref0, t0 = pmr.nondimensionalize_parameters(vlps, vref, k, times, vmax)

    # set up rsf model
    model = rsf.Model()
    model.k = k  # Normalized System stiffness (friction/micron)
    model.v = vlps[0]  # Initial spmrer velocity, generally is vlp(t=0)
    model.vref = vref  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.vmax = vmax
    state1.lc = gpl.lc

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = np.round(t0, 2)

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    model.mu0 = mu0
    model.a = a
    state1.b = b
    state1.Dc = Dc

    # model.tcrit = inputs[4]

    # need to pass in critical times as a variable, because it's recalculating that gradient every single time
    model.tcrit = tcrit

    model.solve()

    #
    mu_sim = model.results.friction
    state_sim = model.results.states

    return mu_sim


def get_dataset():
    # setup output directory
    out_folder = gpl.get_output_storage_folder()

    # load observed section data and mcmc inference data
    times, mt, vlps, x = pmr.load_section_data(idata_location)
    idat = pmr.load_inference_data(idata_location, idataname)

    # first plot: mcmc trace with all original data
    # pmr.plot_trace(idata, chain=None)

    # 'new' data = I started storing model parameters so I could read them in instead of manually filling them out
    # 'old' data = had to fill in parameters manually
    # if there's no .json in the mcmc results folder, then the data is type 'old'
    dataset_type = 'new'
    if dataset_type == 'old':
        k, vref = pmr.get_constants(vlps)
    elif dataset_type == 'new':
        vref, mus, sigmas = gpl.read_from_json(idata_location)

    return idat, mt, vlps, times


def get_time(name):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'{name} time = {current_time}')

    codetime = time.time()

    return codetime


if __name__ == '__main__':
    comptime_start = get_time('start')
    idata, mutrue, vlps, times = get_dataset()
    gpl.set_vch(vlps)
    set_critical_times(vlps, times, threshold=gpl.threshold)
    a, b, Dc, mu0 = get_model_values(idata)
    at = a[0:10]
    bt = b[0:10]
    Dct = Dc[0:10]
    mu0t = mu0[0:10]

    pool = Pool(processes=1)

    outputs = pool.map(generate_rsf_data, zip(at, bt, Dct, mu0t))
    op = np.array(outputs)
    print(op.shape)
    pool.close()
    pool.join()
    np.save(f'mu_sims{gpl.samplename}', op)

    comptime_end = get_time('end')
    time_elapsed = comptime_end - comptime_start
    print(f'time elapsed = {time_elapsed}')
    print('end')
