from multiprocessing import Process, Queue, Pool
from gplot import gpl
import plot_mcmc_results as pmr
import itertools
import numpy as np
from plotrsfmodel import rsf, staterelations

samplename = 'p5866'
nr = 500000
nch = 4
section = '001'
sampleid = f'5866{section}'
dirname = f'out_{nr}d{nch}ch_{sampleid}'
# dirname = f'out_{nr}d{nch}ch'
# dirname = f'~out_{nr}d{nch}ch'
idata_location = gpl.make_path('mcmc_out', samplename, dirname)
idataname = f'{dirname}_idata'


def get_model_values(idata):
    modelvals = pmr.get_model_vals(idata, combined=True)
    a, b, Dc, mu0 = pmr.get_posterior_data(modelvals, return_aminb=False, thin_data=True)
    return a, b, Dc, mu0


def generate_rsf_data(inputs):
    gpl.read_from_json(idata_location)
    print(f'self.threshold = {gpl.threshold}')
    a, b, Dc, mu0 = inputs

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

    model.time = t0

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = vlps

    model.mu0 = mu0
    model.a = a
    state1.b = b
    state1.Dc = Dc

    model.solve(threshold=gpl.threshold)

    # pre-allocate array
    # nobs = len(t0)
    # # mu_sims = np.ones((nobs, nrplot))
    # # print(f'mu_sims.shape = {mu_sims.shape}')
    #
    # logps = []
    # j = 0

    #
    mu_sim = model.results.friction
    state_sim = model.results.states
    #
    # resids = mutrue - mu_sim
    # logp = -1/2 * np.sum(resids ** 2)
    # logps.append(logp)
    #
    #     # attempt at storing results to save time - seems like it's just too much data
    return mu_sim
    #
    #     # save the max logp, "map" solution, "map" vars
    #     if logp == np.nanmax(logps):
    #         map_vars = a[i], b[i], Dc[i], mu0[i]
    #         map_mu_sim = mu_sim
    #         maxlogp = logp
    #
    #     j += 1
    #
    # return mu_sims, logps, map_vars, map_mu_sim, maxlogp


def get_dataset():
    # setup output directory
    out_folder = gpl.get_output_storage_folder()

    # load observed section data and mcmc inference data
    times, mutrue, vlps, x = pmr.load_section_data(idata_location)
    idata = pmr.load_inference_data(idata_location, idataname)

    # first plot: mcmc trace with all original data
    pmr.plot_trace(idata, chain=None)

    # 'new' data = I started storing model parameters so I could read them in instead of manually filling them out
    # 'old' data = had to fill in parameters manually
    # if there's no .json in the mcmc results folder, then the data is type 'old'
    dataset_type = 'new'
    if dataset_type == 'old':
        k, vref = pmr.get_constants(vlps)
    elif dataset_type == 'new':
        vref, mus, sigmas = gpl.read_from_json(idata_location)

    return idata, mutrue


def rsf_calcs(x, mutrue, idata):
    return pmr.calc_rsf_results(x, mutrue, idata)


if __name__ == '__main__':
    model = rsf.Model
    idata, mutrue = get_dataset()
    a, b, Dc, mu0 = get_model_values(idata)
    # model1 = rsf.Model()
    # model2 = rsf.Model()
    # q1 = Queue()
    # q2 = Queue()
    # pr1 = Process(target=model1.solve(threshold=gpl.threshold), args=(q1,))
    # pr2 = Process(target=model2.solve(threshold=gpl.threshold), args=(q2,))
    # pr1.start()
    # pr2.start()
    # out1 = q1.get()
    # out2 = q2.get()
    # model1.blah = out1
    # model2.blah = out2
    # pr1.join()
    # pr2.join()

    pool = Pool()
    # inputs = zip(a, b, Dc, mu0)
    outputs = pool.map(generate_rsf_data, zip(a, b, Dc, mu0))
    op = np.array(outputs)
    op.reshape(gpl.nrplot, len(mutrue))
    pool.close()
    pool.join()
    print('doneskies')

    # with Pool() as pool:
    #     result = pool.map(rsf_calcs, itertools.izip(x, mutrue, idata))
    # print('Program finished')
