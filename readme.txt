
Main scripts:

mcmc_rsf.py -- this is the main script that imports data, processes it, then runs the mcmc simulation on it

rsfmodel -- folder that houses scripts for rsf forward model calculations. 
	- cloned from github: https://github.com/jrleeman/rsfmodel
	- with a few edits by Marissa (print statements and some added functions for testing)


Unimportant scripts:

customdist_test.py -- mcmc testing script. needs to be removed from repository.

load_idata.py -- post-mcmc processing script that reads in trace output from mcmc_rsf.py and messes with it. not fully functional. 

nohup.out -- output file from a series of mcmc runs that I didn't mean to include in the repository. will be deleted.

