import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import scipy as sp
from scipy.signal import savgol_filter
from config import cfig

um_to_mm = 0.001


def downsample_dataset(mu, t, x):
    # low pass filter
    mu_f = savgol_filter(mu, window_length=cfig.filter_windowlen, polyorder=3, mode='mirror')

    # stack time and mu arrays to sample together
    f_data = np.column_stack((mu_f, t, x))

    # downsamples to every qth sample after applying low-pass filter along columns
    f_ds = sp.signal.decimate(f_data, cfig.q, ftype='fir', axis=0)

    print(len(f_ds))

    # FOR P5760 ONLY - no downsampling
    # f_ds = f_data

    return f_ds, mu_f


# section_data(...) slices friction data into model-able sections
def section_data(data):
    df0 = pd.DataFrame(data)
    # changing column names
    df = df0.set_axis(['mu', 't', 'x'], axis=1)

    # cut off first 100 points to avoid sectioning mistakes
    df = df.iloc[100:]

    start_idx = np.argmax(df['t'] > cfig.mintime)
    end_idx = np.argmax(df['t'] > cfig.maxtime)

    df_section = df.iloc[start_idx:end_idx]

    return df_section.to_numpy(), start_idx, end_idx


def preplot(df, colnames):
    t = df['time_s']
    x = df['vdcdt_um']

    fig, ax = plt.subplots(num=1)
    ax.plot(x, df['mu'])
    # ax2 = ax.twiny()
    # ax2.plot(t, df['mu'], 'r')
    # ax2.set_xlabel('time (s)')
    ax.set_title('mu')
    ax.set_xlabel('displacement (mm)')
    ax.set_ylabel('mu')

    plt.figure(2)
    plt.plot(t, df['mu'])
    plt.xlabel('time (s)')
    plt.ylabel('mu')
    # plt.show()
    #
    # plt.show()


def read_hdf(fullpath):
    filename = fullpath
    print(f'reading file: {filename}')
    names = []
    df = pd.DataFrame()
    with h5py.File(filename, 'r') as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # loop on names:
        for name in f.keys():
            # print(name)
            names.append(name)
        # loop on names and H5 objects:
        for name, h5obj in f.items():
            if isinstance(h5obj, h5py.Group):
                print(f'{name} is a Group')
            elif isinstance(h5obj, h5py.Dataset):
                # return a np.array using dataset object:
                arr1 = h5obj[:]
                # return a np.array using dataset name:
                arr2 = f[name][:]
                df[f'{name}'] = arr1

    return df, names


def isMonotonic(A):
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))


def remove_non_monotonic(times, x, data, axis=0):
    nmi = []
    if not np.all(np.diff(times) >= 0):
        print('time series can become non-monotonic after downsampling which is an issue for the sampler')
        print('now removing non-monotonic t indices from (t, mu, x) dataset')
        print(f'input downsampled data shape = {data.shape}')
        # Find the indices where the array is not monotonically increasing
        nmi_t = np.where(np.diff(times) < 0)[0]
        nmi.append(nmi_t)
        # print(f'non monotonic time indices = {non_monotonic_indices}')

    if not np.all(np.diff(x) >= 0):
        print('displacement series can become non-monotonic after downsampling which is an issue for derivative calcs')
        print('now removing non-monotonic x indices from (t, mu, x) dataset')
        print(f'input downsampled data shape = {data.shape}')
        nmi_x = np.where(np.diff(x) < 0)[0]
        nmi.append(nmi_x)

    if nmi:
        # Remove the non-monotonic data points
        cleaned_data = np.delete(data, nmi, axis)
        print('removed bad data? should be True')
        print(isMonotonic(cleaned_data[:, 1]))
        return cleaned_data

    # Array is already monotonically increasing, return it as is
    print('Array is already monotonically increasing, returning as is')
    return data


def calc_derivative(y, x, window_len=None):
    # returns dydx
    if window_len is not None:
        print(f'calculating derivative using SG filter and window length {window_len}')
        # smooth
        # x_smooth = smooth(x,window_len=params['window_len'],window='flat')
        # y_smooth = smooth(y,window_len=params['window_len'],window='flat')
        # dydx = np.gradient(y_smooth,x_smooth)
        dxdN = savgol_filter(x,
                             window_length=window_len,
                             polyorder=3,
                             deriv=1)
        # plt.plot(x, dxdN)
        # plt.show()

        dydN = savgol_filter(y,
                             window_length=window_len,
                             polyorder=3,
                             deriv=1)
        dydx = dydN / dxdN

        dydx_smooth = savgol_filter(dydx,
                                    window_length=window_len,
                                    polyorder=1)

        dydx_smooth[dydx_smooth < 0] = 0.0001
        return dydx_smooth
    else:
        print(f'calculating derivative using gradient because window_len= {window_len}')
        dydx = np.gradient(y, x)
        dydx[dydx < 0] = 0
        return dydx


def nondimensionalize_parameters(vlps, vref, k, times, vmax):
    # define characteristic length and velocity for nondimensionalizing
    lc = cfig.lc
    vmax = np.max(vlps)

    # then remove dimensions
    k0 = cfig.k * cfig.lc
    vlps0 = vlps / vmax
    vref0 = vref / vmax

    t0 = times * vmax / lc
    t0 = t0 - t0[0]

    return k0, vlps0, vref0, t0


def determine_threshold(vlps, t):
    vlps0 = vlps / np.max(vlps)
    t0 = t * np.max(vlps) / cfig.lc
    t0 = t0 - t0[0]
    t0 = np.round(t0, 2)
    velocity_gradient = np.gradient(vlps0)
    time_gradient = np.gradient(t0)
    acceleration = velocity_gradient / time_gradient

    critical_times = t0[np.abs(acceleration) > cfig.threshold]

    threshold_line = cfig.threshold * np.ones_like(acceleration)

    n = plt.gcf().number
    plt.figure(n + 1)
    plt.plot(t0, acceleration)
    plt.plot(critical_times, np.zeros_like(critical_times), 'co')
    plt.plot(t0, threshold_line, 'r')
    plt.title('acceleration values to determine threshold used in ode solver')
    plt.ylabel('acceleration')


def get_obs_data(samplename):
    homefolder = os.path.expanduser('~')
    path = os.path.join('PycharmProjects', 'mcmcrsf_xfiles', 'data', 'FORGE_DataShare', f'{samplename}')
    name = f'{samplename}_proc.hdf5'
    sample_name = name
    fullpath = os.path.join(homefolder, path, name)
    print(f'getting data from: {fullpath}')
    f = h5py.File(os.path.join(homefolder, path, name), 'r')

    # read in data from hdf file, print column names
    df, names = read_hdf(fullpath)

    # comment this in when deciding which displacement sections to use
    # preplot(df, names)

    # first remove any mu < 0 data from experiment
    df = df[(df['mu'] > 0)]

    # convert to numpy arrays
    t = df['time_s'].to_numpy()
    mu = df['mu'].to_numpy()
    x = df['vdcdt_um'].to_numpy()

    # filters and downsamples data
    f_ds, mu_f = downsample_dataset(mu, t, x)

    # sections data - make this into a loop to run multiple sections one after another
    sectioned_data, start_idx, end_idx = section_data(f_ds)

    # need to check that time vals are monotonically increasing after being processed
    t = sectioned_data[:, 1]
    x = sectioned_data[:, 2]
    print('checking that time series is monotonic after processing')
    print(isMonotonic(t))
    print(isMonotonic(x))

    # remove non-monotonically increasing time indices if necessary
    cleaned_data = remove_non_monotonic(t, x, sectioned_data, axis=0)

    # data for pymc
    mutrue = cleaned_data[:, 0]
    t = cleaned_data[:, 1]
    x = cleaned_data[:, 2]

    # calculate loading velocities = dx/dt
    vlps = calc_derivative(x, t, window_len=cfig.vel_windowlen)

    plt.plot(t, vlps)
    plt.xlabel('time (s)')
    plt.ylabel('velocity (um/s)')
    # plt.show()

    determine_threshold(vlps, t)

    cfig.set_disp_bounds(x)
    print(cfig.mindisp)
    print(cfig.maxdisp)

    # plot raw data section with filtered/downsampled for reference
    df_raw = df[(df['vdcdt_um'] > cfig.mindisp) & (df['vdcdt_um'] < cfig.maxdisp)]

    xax = x

    fig, ax = plt.subplots()
    ax.plot(df_raw['vdcdt_um'], df_raw['mu'], 'o', alpha=0.5, label='raw data')
    ax.plot(xax, mutrue, '.', alpha=0.8, label='downsampled, filtered, sectioned data')
    plt.xlabel('displacement (mm)')
    plt.ylabel('mu')
    plt.title('Observed data section (def get_obs_data)')
    plt.ylim([np.min(mutrue) - 0.01, np.max(mutrue) + 0.01])
    plt.legend()

    ax2 = ax.twinx()
    ax2.plot(xax, vlps, 'r', label='velocity')
    plt.legend()
    plt.show()

    return mutrue, t, vlps, x, sample_name


def main():
    print('MCMC RATE AND STATE FRICTION MODEL')
    samplename = cfig.samplename

    # observed data
    mutrue, times, vlps, x, file_name = get_obs_data(samplename)


if __name__ == '__main__':
    main()
