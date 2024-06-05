import numpy as np
from scipy import integrate
from math import exp
from collections import namedtuple


class IntegrationStop(Exception):
    pass


class IncompleteModelError(Exception):
    """
    Special error case for trying to run the model with inadequate information.
    """
    pass


class LoadingSystem(object):
    """
    Contains attributes relating to the external loading system.

    Attributes
    ----------
    k : float
        System stiffness in units of fricton/displacement.
    time : list
        List of time values at which the system will be solved.
    loadpoint_velocity : list
        List of the imposed loadpoint velocities at the corresponding time
        value. Must be same length as time.
    v : float
        Slider velocity
    mu : float
        The current friciton value of the system.
    """

    def __init__(self):
        self.k = None
        self.time = None  # List of times we want answers at
        self.loadpoint_velocity = None  # Matching list of velocities

    def velocity_evolution(self):
        v_contribution = 0
        for state in self.state_relations:
            v_contribution += state.velocity_component(self)
        ratio = (self.mu - self.mu0 - v_contribution) / self.a
        if np.abs(ratio) > 10.0:
            return 'badsample'

        try:
            self.v = self.vref * exp((self.mu - self.mu0 - v_contribution) / self.a)
        except OverflowError as exc:
            pass

    def friction_evolution(self, loadpoint_vel):
        dmudt = self.k * (loadpoint_vel - self.v)
        return self.k * (loadpoint_vel - self.v)


class Model(LoadingSystem):
    """
    Houses the model coefficients and does the integration.

    Attributes
    ----------
    mu0 : float
        Reference friction value at vref.
    a : float
        Rate and state constitutive parameter.
    vref : float
        System reference velocity at which the reference friction is measured.
    state_relations : list
        List of state relations to be used when calculating the model.
    results : namedtuple
        Stores all model outputs.
    """

    def __init__(self):
        LoadingSystem.__init__(self)
        self.mu0 = 0.6
        self.a = None
        self.vref = None
        self.state_relations = []
        self.loadpoint_displacement = None
        self.results = namedtuple("results", ["time", "loadpoint_displacement",
                                              "slider_velocity", "friction",
                                              "states", "slider_displacement"])
        self.new_state = None

    def _integrationStep(self, w, t, system):
        """ Do the calculation for a time-step

        Parameters
        ----------
        w : list
        Current values of integration variables. Friction first, state variables following.
        t : float
        Time at which integration is occurring
        system : model object
        Model that is being solved

        Returns
        -------
        step_results : list
        Results of the integration step. dmu/dt first, followed by dtheta/dt for state variables.
        """
        system.mu = w[0]
        for i, state_variable in enumerate(system.state_relations):
            self.new_state = w[i + 1]
            if np.isinf(self.new_state):
                self.new_state = 100000
            state_variable.state = self.new_state

        flag = system.velocity_evolution()

        if flag == 'badsample':
            raise IntegrationStop()

        # Find the loadpoint_velocity corresponding to the most recent time
        # <= the current time.
        loadpoint_vel = system.loadpoint_velocity[system.time <= t][-1]

        self.dmu_dt = system.friction_evolution(loadpoint_vel)
        step_results = [self.dmu_dt]

        for state_variable in system.state_relations:
            dtheta_dt = state_variable.evolve_state(self)
            step_results.append(dtheta_dt)

        return step_results

    def readyCheck(self):
        """
        Determines if all necessary parameters are set to run the model.
        Will raise appropriate error as necessary.
        """
        # print('forward model: performing ready check')
        if self.a is None:
            raise IncompleteModelError('Parameter a is None')
        elif self.vref is None:
            raise IncompleteModelError('Parameter vref is None')
        elif self.state_relations == []:
            raise IncompleteModelError('No state relations in state_relations')
        elif self.k is None:
            raise IncompleteModelError('Parameter k is None')
        elif self.time is None:
            raise IncompleteModelError('Parameter time is None')
        elif self.loadpoint_velocity is None:
            raise IncompleteModelError('Parameter loadpoint_velocity is not set')

        for state_relation in self.state_relations:
            if state_relation.b is None:
                raise IncompleteModelError('Parameter b is None')
            elif state_relation.Dc is None:
                raise IncompleteModelError('Parameter Dc is None')

        if len(self.time) != len(self.loadpoint_velocity):
            raise IncompleteModelError('Time and loadpoint_velocity lengths do not match')

    def _get_critical_times(self, threshold):
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
        velocity_gradient = np.gradient(self.loadpoint_velocity)
        time_gradient = np.gradient(self.time)
        acceleration = velocity_gradient / time_gradient
        critical_times = self.time[np.abs(acceleration) > threshold]
        return critical_times

    def solve(self, threshold, **kwargs):
        """
        Runs the integrator to actually solve the model and returns a
        named tuple of results.

        Parameters
        ----------
        threshold : float
            Threshold used to determine when integration care should be taken. This threshold is
            in terms of maximum load-point acceleration before time step is marked.

        Returns
        -------
        results : named tuple
            Results of the model
        """
        # print('FORWARD MODEL BEGIN MODEL.SOLVE')
        odeint_kwargs = dict(rtol=1e-3, atol=1e-3, mxstep=1000)
        odeint_kwargs.update(kwargs)

        # Make sure we have everything set before we try to run
        self.readyCheck()

        # Initial conditions at t = 0
        w0 = [self.mu0]
        for state_variable in self.state_relations:
            state_variable.set_steady_state(self)
            w0.append(state_variable.state)

        # Find any critical time points we need to let the integrator know about
        self.critical_times = self._get_critical_times(threshold)

        # Solve it
        try:
            wsol, self.solver_info = integrate.odeint(self._integrationStep, w0, self.time,
                                                      full_output=True, tcrit=self.critical_times,
                                                      args=(self,), **odeint_kwargs)
        except IntegrationStop as e:
            # sends negative infinity array back to pymc (via Loglikelihood) to reject the sample immediately instead
            # of struggling through the rest of the integration
            friction = np.ones_like(self.time)
            states = np.ones_like(self.time)
            self.results.friction = np.inf * -friction
            self.results.states = np.inf * -states
            self.results.time = self.time
            return self.results

        self.results.friction = wsol[:, 0]
        self.results.states = wsol[:, 1:]
        self.results.time = self.time

        return self.results
