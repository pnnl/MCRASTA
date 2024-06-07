
# mcrasta
The Markov chain Monte Carlo Rate and State Friction model (mcrasta) is a Python package that fits rate- and state
friction (RSF) models to experimental data and quantifies parameter uncertainty using Bayesian inference via 
Markov chain Monte Carlo (MCMC). It implements PyMC to build and execute the Bayesian-MCMC model and the Rate 
and State Friction Toolkit (https://github.com/jrleeman/rsfmodel) to solve the forward RSF model, which is then used to 
calculate the likelihood function and update the proposal distribution during MCMC sampling. 

## Setting up the MCMC model
***config.yaml*** and ***config.py:***

***config.yaml*** is the configuration file for running the mcmc model (***mcrasta.py*** and ***Loglikelihood.py***). Initializes data organization and id information including sample name, experimental data folders, output folders; data processing setup variables including Savitzky-Golay filter window lengths and downsample rate; and PyMC model setup parameters including number of draws, chains, tuning draws, and cores. Config.py communicates the config.yaml values to mcrasta.py and Likelihood.py.

***rsfdataviewer.py***:

Plots your observed, filtered, and downsampled datasets so you can fine tune data processing and ODE solver parameters (in config.yaml) before running the full MCMC simulation. 

## Running the MCMC model

***mcrasta.py*** and ***Loglikelihood.py***:

mcrasta.py processes the observed experimental data and sets up the PyMC model framework by defining prior distributions and initializing the MCMC sampler. Observed data is filtered and downsampled, then the resulting displacement and time data is used to calculate loadpoint velocities. PyMC samples distributions for the following parameters: a, b, Dc, mu0, and sigma. Parameters a and b are empirical constants; Dc is the critical slip length; mu0 is the steady-state friction value; and sigma is the experimental uncertainty (parameter in denominator of likelihood function). Loglikelihood.py contains the class Loglike() and the function mcmc_rsf_sim(). The Loglike class is a wrapper function which casts PyMC tensor variables to floating point values to be evaluated by mcmc_rsf_sim(). mcmc_rsf_sim() initializes and runs the forward RSF model (rsfmodel/rsf.py and rsfmodel/staterelations.py) using the sampled (then converted) PyMC tensor variables for each draw in the MCMC model run. More information on how this is done and why it's necessary can be found at the links below.   

https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html

https://www.pymc.io/projects/examples/en/2022.12.0/case_studies/blackbox_external_likelihood_numpy.html#blackbox_external_likelihood_numpy

**rsfmodel**: 

Folder that houses the forward RSF model scripts. These have been modified from their original form (https://github.com/jrleeman/rsfmodel) to handle the wide range of parameter combinations thrown into it by PyMC. Most notable change is addition of IntegrationStop(Exception): in rsf.py. This exception halts integration when the parameters drawn by PyMC result in impossible results (e.g. inf, -inf, nan). This prevents the ODE solver from "spinning its wheels", so to speak, by trying to find a solution using a poor combination of parameters. It also prevents the ODE solver from returning any partially finite/infinite series. All values returned must be finite so PyMC also does not spin its wheels. 

## Processing the results
***configplot.yaml*** and ***configplot.py***:

Another configuration file, but specifically for plotting and processing MCMC results. Differs slightly from config.yaml and config.py and is separate in case one wants to evaluate results for one dataset while running an MCMC model for a different one. Configplot.py communicates the configplot.yaml values to plotting scripts listed below.

***plot_mcmc_results.py***:

Executes various Arviz plotting functions to analyze MCMC model results. Plots include: convergence statistics, MCMC trace plot, prior and posterior distributions for each parameter, pairs plot, and a plot of the observed data and velocity steps.

***calc_logps_sims.py***:

Calculates the log-likelihood for each set of PyMC sample draws. You may be asking yourself, "why is a separate script needed to calculate a value that is already calculated in the MCMC model run?" The answer: because the wrapper function (which makes evaluating our custom "black-box" likelihood function possible) and PyMC can only communicate in frustratingly limited ways. I could not figure out how to save the log-likelihood calculated/used/accepted by the PyMC sampler, so I accepted defeat and decided to calculate them after the fact. The process is sped up significantly using parallel processing. 1,000,000 sample draws are evaluted in under an hour using 20 cores.  
posterior_draws.py:
Reads in results from the MCMC model run and runs a posterior predictive check. That is, takes a random sample of parameter draws, solves the forward RSF model, then saves the results to a file to be used in plots.py. 

***plots.py***:

Plots the results from posterior_draws.py with the observed data and the best-fit (highest likelihood) result from calc_logps_sims.py. This script also utilizes parallel processing to make evaluating results extra speedy. 

***run_plotting_scripts.py***:

Executes all plotting scripts sequentially. 

## Abstract: 

Rate- and state friction (RSF) constitutive equations empirically describe a wide range of observed experimental geomechanical data. Four unknown model parameters (a, b, Dc, mu0) can be solved for using classical inversion techniques (i.e. nonlinear least-squares) which result in a single “best-fit” solution. This approach has disadvantages when applied to nonlinear and underdetermined systems of equations (such as RSF), as it does not rigorously address solution non-uniqueness or parameter uncertainty. Bayesian inference via Markov Chain Monte Carlo (MCMC) overcomes these challenges – the results include a range of probable parameter values which fit RSF laboratory data; an associated uncertainty estimate for each parameter; and information regarding relationships between model parameters. We integrated two open-source Python libraries to develop a software which applies Bayesian-MCMC methods to RSF experimental data.



