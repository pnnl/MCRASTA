import numpy as np
import os
from gplot import gpl


def split_large_files():
    chunksize = 100000
    filename = gpl.make_path('musim_out', f'{gpl.msname}_6.npy')
    alldata = np.load(filename)

    dir, oldname = os.path.split(filename)

    total_sims_in_file = alldata.shape[0]

    j = 8
    for i in range(0, total_sims_in_file, chunksize):
        new_name = f'{gpl.msname}_{j}'
        new_path = os.path.join(dir, new_name)
        new_data = alldata[i: i+chunksize, :]
        np.save(new_path, new_data)
        j += 1


def main():
    split_large_files()


if __name__ == '__main__':
    main()
