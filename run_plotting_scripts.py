import subprocess
from configplot import cplot

subprocess.run('python plot_mcmc_results.py', shell=True)
subprocess.run('python calc_logps_sims.py', shell=True)
subprocess.run('python posterior_draws.py', shell=True)
subprocess.run('python plots.py', shell=True)

copy_folder1 = cplot.mcmc_out_dir
copy_folder2 = cplot.postprocess_out_dir

# subprocess.run(f'scp -r /home/PycharmProjects/mcrasta_xfiles/{copy_folder1}')
# subprocess.run(f'scp -r /home/PycharmProjects/mcrasta_xfiles/{copy_folder2}')