
The Markov chain Monte Carlo Rate and State Friction model (mcrasta) is a Python package that fits rate- and state
friction (RSF) models to experimental data and quantifies parameter uncertainty using Bayesian inference via 
Markov chain Monte Carlo (MCMC). It implements PyMC to build and execute the Bayesian-MCMC model and the Rate 
and State Friction Toolkit (https://github.com/jrleeman/rsfmodel) to solve the forward RSF model, which is then used to 
calculate the likelihood function and update the proposal distribution during MCMC sampling. 


Abstract: Rate- and state friction (RSF) constitutive equations empirically describe a wide range of observed experimental geomechanical data. Four unknown model parameters (a, b, Dc, mu0) can be solved for using classical inversion techniques (i.e. nonlinear least-squares) which result in a single “best-fit” solution. This approach has disadvantages when applied to nonlinear and underdetermined systems of equations (such as RSF), as it does not rigorously address solution non-uniqueness or parameter uncertainty. Bayesian inference via Markov Chain Monte Carlo (MCMC) overcomes these challenges – the results include a range of probable parameter values which fit RSF laboratory data; an associated uncertainty estimate for each parameter; and information regarding relationships between model parameters. We integrated two open-source Python libraries to develop a software which applies Bayesian-MCMC methods to RSF experimental data.



