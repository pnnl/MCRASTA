
Abstract: Rate- and state friction (RSF) constitutive equations empirically describe a wide range of observed experimental geomechanical data. Four unknown model parameters (a, b, Dc, mu0) can be solved for using classical inversion techniques (i.e. nonlinear least-squares) which result in a single “best-fit” solution. This approach has disadvantages when applied to nonlinear and underdetermined systems of equations (such as RSF), as it does not rigorously address solution non-uniqueness or parameter uncertainty. This limits any subsequent interpretations regarding geomechanical behavior of EGS reservoir rocks. Bayesian inference via Markov Chain Monte Carlo (MCMC) overcomes these challenges – the results include a range of probable parameter values which fit RSF laboratory data; an associated uncertainty estimate for each parameter; and information regarding relationships between model parameters. We integrated two open-source Python libraries to develop a software which applies Bayesian-MCMC methods to RSF experimental data.


Main scripts:

mcmc_rsf.py -- this is the main script that imports data, processes it, then runs the mcmc simulation on it

rsfmodel -- folder that houses scripts for rsf forward model calculations. 
	- cloned from github: https://github.com/jrleeman/rsfmodel
	- with a few edits by Marissa (print statements and some added functions for testing)


Unimportant scripts:

customdist_test.py -- mcmc testing script. needs to be removed from repository.

load_idata.py -- post-mcmc processing script that reads in trace output from mcmc_rsf.py and messes with it. not fully functional. 

nohup.out -- output file from a series of mcmc runs that I didn't mean to include in the repository. will be deleted.

