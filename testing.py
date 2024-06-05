import numpy as np
from rsfmodel import rsf, staterelations, plot
import pandas as pd
from gplot import gpl
import os


def load_section_data():
    section_data = pd.read_csv(os.path.join(gpl.idata_location(), 'section_data.csv'))
    df = pd.DataFrame(section_data)
    times = df['times'].to_numpy()
    mutrue = df['mutrue'].to_numpy()
    vlps = df['vlps'].to_numpy()
    x = df['x'].to_numpy()

    return times, mutrue, vlps, x


def solve_rsfmodel(a, b, Dc, mu0):
    # times, mutrue, vlps, x = load_section_data()
    # time = np.load(r'C:\Users\fich146\PycharmProjects\rsfmodel2\timetest.npy')
    # time = np.array(time)

    model = rsf.Model()

    lc = 125
    vmax = 300

    # Set model initial conditions
    model.mu0 = mu0 # Friction initial (at the reference velocity)
    model.a = a # Empirical coefficient for the direct effect
    model.k = 0.00153 * lc # Normalized System stiffness (friction/micron)
    model.v = 3. / vmax # Initial slider velocity, generally is vlp(t=0)
    model.vref = 3. / vmax # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.b = b  # Empirical coefficient for the evolution effect
    state1.Dc = Dc / lc  # Critical slip distance

    model.state_relations = [state1] # Which state relation we want to use

    time = np.arange(0, 40.01, 0.01)
    # We want to solve for 40 seconds at 100Hz
    time = (vmax / lc) * time
    model.time = time - time[0]

    # print(model.time[1] - model.time[0])

    # We want to slide at 1 um/s for 10 s, then at 10 um/s for 31
    lp_velocity = np.ones_like(model.time)
    lp_velocity[10*100:] = 10. # Velocity after 10 seconds is 10 um/s
    lp_velocity[20*100:] = 30.
    lp_velocity[30*100:] = 300.

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity / vmax

    state1.vmax = np.max(lp_velocity)
    state1.lc = 125

    # Run the model!
    model.solve(threshold=2)

    print(f'results: {model.results.friction}')

    # Make the phase plot
    # plot.phasePlot(model)

    # model.results.loadpoint_displacement = np.load(r'C:\Users\fich146\PycharmProjects\mcmc_rsf\lpdisp.npy')
    # # Make a plot in displacement
    # plot.dispPlot(model)
    #
    # # Make a plot in time
    # plot.timePlot(model)

def main():
    mu0s = np.array((0.4, 0.5, 0.4, 0.9))
    avals = np.array((0.006, 0.0005, 0.003, 0.004))
    bvals = np.array((0.006, 0.03, 0.002, 0.001))
    Dcvals = np.array((45, 10000, 2, 3))

    for a, b, Dc, mu0 in zip(avals, bvals, Dcvals, mu0s):
        print(a, b, Dc, mu0)
        solve_rsfmodel(a, b, Dc, mu0)


if __name__ == '__main__':
    main()