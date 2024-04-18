import numpy as np
from rsfmodel import staterelations, rsf, plot
import pytensor.tensor as tt
from globals import myglobals


def mcmc_rsf_sim(theta, t0, v0, k0, vref0, vmax):
    # unpack parameters
    a, b, Dc, mu0 = theta

    # initialize rsf model
    model = rsf.Model()

    # Size of dataset
    model.datalen = len(t0)

    # Set initial conditions
    model.mu0 = mu0  # Friction initial (at the reference velocity)
    model.a = a  # Empirical coefficient for the direct effect
    model.k = k0  # Normalized System stiffness (friction/micron)
    model.v = v0[0]  # Initial slider velocity, generally is vlp(t=0)
    model.vref = vref0  # Reference velocity, generally vlp(t=0)

    state1 = staterelations.DieterichState()
    state1.b = b  # Empirical coefficient for the evolution effect
    state1.Dc = Dc  # Critical slip distance
    # all other parameters are already nondimensionalized, but the state parameter is nd'd in staterelations.py,
    # so we need to pass characteristic velocity (vmax) and length (lc) into the fwd model
    state1.vmax = vmax
    state1.lc = myglobals.lc

    model.state_relations = [state1]  # Which state relation we want to use

    model.time = t0  # nondimensionalized time
    lp_velocity = v0

    # Set the model load point velocity, must be same shape as model.model_time
    model.loadpoint_velocity = lp_velocity

    # Run the model!
    model.solve(threshold=myglobals.threshold)

    mu_sim = model.results.friction

    return mu_sim

class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, times0, vlps0, k0, vref0, data, vmax):
        self.data = data
        self.times = times0
        self.vlps = vlps0
        self.k = k0
        self.vref = vref0
        self.vmax = vmax
        self.y_pred = None

    # Custom LogLikelihood function for use in pymc - runs forward model with sample draws
    def log_likelihood(self, theta):
        if type(theta) == list:
            theta = theta[0]
        (
            a,
            b,
            Dc,
            mu0,
        ) = theta

        y_pred = mcmc_rsf_sim(theta, self.times, self.vlps, self.k, self.vref, self.vmax)
        resids = (self.data - y_pred)
        # myglobals.store_mu_sim(y_pred)
        # YPREDS.append(y_pred)
        logp = -1 / 2 * (np.sum(resids ** 2))

        return logp

    def perform(self, node, inputs, outputs):
        logp = self.log_likelihood(inputs)
        outputs[0][0] = np.array(logp)
        # print(outputs[0][0])