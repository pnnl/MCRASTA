from plot_mcmc_results import main as pmr
from calc_logps_sims import main as cls
from posterior_draws import main as post_draws
from plots import main as mainplot


if __name__ == '__main__':
    pmr()
    # cls()
    # post_draws()
    mainplot()
    print('END ALL PLOT SCRIPTS')
